"""MaxSim discrimination test (Phase 2.5).

Goal: does token-level MaxSim recover signal that mean-pooling destroys?

Test set: instances where Voyage R@10=1 (gold found) AND Codestral pooled
R@10=0 (gold missed). On these, compare:
  Method A — Pooled cosine (reproduces existing baseline)
  Method B — MaxSim over token-level latents

Decision criterion (set in advance):
  MaxSim R@10 ≥ 0.50  → strong signal; build Phase 3
  MaxSim R@10 0.20-0.50 → mixed; reconsider
  MaxSim R@10 < 0.20  → bet wrong; stop

Usage (cloud H100):
  python scripts/maxsim_test.py \\
      --model-id mistralai/Mamba-Codestral-7B-v0.1 \\
      --dtype bfloat16 \\
      --top-k 20 \\
      --max-test-instances 80

Local smoke (tiny model, no real signal expected):
  python scripts/maxsim_test.py \\
      --model-id state-spaces/mamba-130m-hf \\
      --dtype float32 \\
      --max-test-instances 3 \\
      --restrict-repos sqlfluff/sqlfluff \\
      --out data/results/maxsim_smoke.json
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.eval import ensure_repo_at, load_instances
from src.maxsim import dedup_to_files, maxsim_batch, pooled_score
from src.utils import get_logger, list_eligible_files, project_root

log = get_logger("maxsim_test")

CACHE_DIR = project_root() / "data" / "maxsim_cache"
CHUNK_TOKENS = 1500
ENCODE_BATCH = 4

DECISION_THRESHOLDS = {
    "strong": 0.50,
    "mixed_low": 0.20,
}


# ─────────────────────────────────────────────────────────────────────
# Test set selection
# ─────────────────────────────────────────────────────────────────────

def build_test_set(voyage_path: Path, codestral_path: Path,
                   max_n: int, seed: int = 42,
                   restrict_repos: list[str] | None = None) -> list[dict]:
    """Pick instances where Voyage R@10=1 AND Codestral R@10=0 (both indexable)."""
    v = json.loads(voyage_path.read_text())
    m = json.loads(codestral_path.read_text())
    v_by_id = {r["instance_id"]: r for r in v["results"]}
    m_by_id = {r["instance_id"]: r for r in m["results"]}

    strict, expanded = [], []
    for iid in set(v_by_id) & set(m_by_id):
        rv, rm = v_by_id[iid], m_by_id[iid]
        if not (rv["gold_in_index"] and rm["gold_in_index"]):
            continue
        if restrict_repos and rv["repo"] not in restrict_repos:
            continue
        if rv["recall"]["10"] != 1.0:
            continue
        if rm["recall"]["10"] == 0.0:
            strict.append(iid)
        elif rm["recall"]["10"] < 1.0:
            expanded.append(iid)

    log.info("STRICT (Voyage R@10=1 AND Codestral R@10=0): %d", len(strict))
    log.info("EXPANDED (Voyage R@10=1 AND Codestral R@10<1, not strict): %d",
             len(expanded))

    chosen = strict
    if len(strict) < 30:
        log.warning("strict set < 30, expanding criterion")
        chosen = strict + expanded

    rng = random.Random(seed)
    chosen = sorted(chosen)
    if len(chosen) > max_n:
        chosen = rng.sample(chosen, max_n)
        chosen.sort()

    log.info("test set: %d instances", len(chosen))
    log.info("per-repo: %s", Counter(v_by_id[i]["repo"] for i in chosen).most_common())

    return [{"instance_id": iid,
             "repo": v_by_id[iid]["repo"],
             "voyage_top20": v_by_id[iid]["ranked_top20"],
             "codestral_top20": m_by_id[iid]["ranked_top20"]} for iid in chosen]


# ─────────────────────────────────────────────────────────────────────
# Token-level chunk cache
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RepoChunkPool:
    repo_path: Path
    chunk_files: list[str]      # parallel to tokens
    tokens: list[torch.Tensor]  # one (n_chunk_tokens, D) per chunk


def _chunk_signature(model_id: str, repo_path: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(str(CHUNK_TOKENS).encode())
    h.update(b"token_level")
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        h.update(f.relative_to(repo_path).as_posix().encode())
        h.update(f"{st.st_size}-{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _split_chunks(tokenizer, file_rel: str, content: str) -> list[tuple[str, int, str]]:
    """Returns list of (file_rel, chunk_idx, text)."""
    if not content.strip():
        return []
    enc = tokenizer(content, return_tensors=None, truncation=False,
                    padding=False, add_special_tokens=False)
    ids = enc["input_ids"]
    if not ids:
        return []
    out = []
    for ci, i in enumerate(range(0, len(ids), CHUNK_TOKENS)):
        sub = ids[i: i + CHUNK_TOKENS]
        out.append((file_rel, ci, tokenizer.decode(sub, skip_special_tokens=True)))
    return out


def encode_repo_pool(encoder: MambaEncoder, repo_path: Path,
                     files: list[Path]) -> RepoChunkPool:
    sig = _chunk_signature(encoder.model_id, repo_path, files)
    cache_file = CACHE_DIR / f"{repo_path.name}_{Path(encoder.model_id).name}_{sig}.pt"

    if cache_file.exists():
        log.info("Cache hit %s", cache_file.name)
        blob = torch.load(cache_file, map_location="cpu", weights_only=False)
        return RepoChunkPool(
            repo_path=repo_path,
            chunk_files=blob["chunk_files"],
            tokens=blob["tokens"],
        )

    log.info("Encoding %s (token-level): %d files", repo_path.name, len(files))
    chunks_meta: list[tuple[str, int, str]] = []
    for f in tqdm(files, desc=f"chunk {repo_path.name}", leave=False):
        try:
            content = f.read_text(errors="replace")
        except OSError:
            continue
        rel = f.relative_to(repo_path).as_posix()
        chunks_meta.extend(_split_chunks(encoder.tokenizer, rel, content))

    log.info("Total chunks: %d", len(chunks_meta))
    tokens: list[torch.Tensor] = []
    pbar = tqdm(total=len(chunks_meta), desc="encode (token-level)", leave=False)
    for i in range(0, len(chunks_meta), ENCODE_BATCH):
        batch = [f"# File: {rel}\n\n{txt}" for rel, _, txt in
                 chunks_meta[i: i + ENCODE_BATCH]]
        out = encoder.encode(batch)
        # last_hidden: (B, T, D) on encoder.device
        # Strip pad tokens per row using attention mask, store as cpu tensors.
        for j in range(out.last_hidden.shape[0]):
            mask = out.attention_mask[j].bool()
            row = out.last_hidden[j][mask].detach().to("cpu")
            tokens.append(row.contiguous())
        pbar.update(len(batch))
    pbar.close()

    chunk_files = [m[0] for m in chunks_meta]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"chunk_files": chunk_files, "tokens": tokens}, cache_file)
    size_gb = cache_file.stat().st_size / 1e9
    log.info("Cached %d chunks (%.2f GB) to %s",
             len(chunk_files), size_gb, cache_file.name)
    return RepoChunkPool(repo_path=repo_path, chunk_files=chunk_files, tokens=tokens)


# ─────────────────────────────────────────────────────────────────────
# Eval one instance
# ─────────────────────────────────────────────────────────────────────

K_VALUES = (1, 5, 10, 20)


def recall_at_k(gold: set, ranked: list, k: int) -> float:
    if not gold:
        return 0.0
    return sum(1 for g in gold if g in ranked[:k]) / len(gold)


def reciprocal_rank(gold: set, ranked: list) -> float:
    for i, p in enumerate(ranked, 1):
        if p in gold:
            return 1.0 / i
    return 0.0


def eval_instance(encoder: MambaEncoder, query: str,
                  pool: RepoChunkPool, gold_files: list[str],
                  top_k: int = 20, device: str = None) -> dict:
    device = device or str(encoder.device)
    # Encode query (single chunk; truncates internally to max_length)
    out = encoder.encode([query])
    mask = out.attention_mask[0].bool()
    q_tok = out.last_hidden[0][mask].to(device)

    cand_tok_dev = [t.to(device) for t in pool.tokens]

    t0 = time.time()
    s_pool = pooled_score(q_tok, cand_tok_dev)
    t_pool = time.time() - t0
    pool_ranked = dedup_to_files(s_pool, pool.chunk_files, top_k=top_k)
    pool_paths = [p for p, _ in pool_ranked]

    t0 = time.time()
    s_max = maxsim_batch(q_tok, cand_tok_dev)
    t_max = time.time() - t0
    max_ranked = dedup_to_files(s_max, pool.chunk_files, top_k=top_k)
    max_paths = [p for p, _ in max_ranked]

    gold = set(gold_files)
    return {
        "pooled": {
            "ranked_top20": pool_paths,
            **{f"recall@{k}": recall_at_k(gold, pool_paths, k) for k in K_VALUES},
            "mrr": reciprocal_rank(gold, pool_paths),
            "latency_s": t_pool,
        },
        "maxsim": {
            "ranked_top20": max_paths,
            **{f"recall@{k}": recall_at_k(gold, max_paths, k) for k in K_VALUES},
            "mrr": reciprocal_rank(gold, max_paths),
            "latency_s": t_max,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────

def sanity_pool_matches_baseline(eval_pool_top10: list[str],
                                  baseline_top20: list[str]) -> bool:
    """Method A pooled ranking should overlap heavily with cached baseline."""
    s_eval = set(eval_pool_top10)
    s_base = set(baseline_top20[:10])
    return len(s_eval & s_base) >= 5  # Loose check: >50% overlap


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mistralai/Mamba-Codestral-7B-v0.1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-test-instances", type=int, default=80)
    ap.add_argument("--restrict-repos", nargs="*", default=None,
                    help="Optionally limit test set to a specific repo (smoke)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/results/maxsim_discrimination_test.json")
    ap.add_argument("--budget-hours", type=float, default=4.0,
                    help="Hard cap on runtime; report partial results if exceeded")
    ap.add_argument("--strict-sanity", action="store_true",
                    help="Hard-fail if Method A pooled rankings diverge from the "
                         "cached Codestral baseline. Auto-enabled when --model-id "
                         "matches the encoder that produced the cached baseline.")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    t_start = time.time()
    budget_s = args.budget_hours * 3600

    # Test set
    codestral_path = ROOT / "data/results/mamba_codestral_baseline.json"
    test = build_test_set(
        ROOT / "data/results/voyage_baseline.json",
        codestral_path,
        max_n=args.max_test_instances,
        seed=args.seed,
        restrict_repos=args.restrict_repos,
    )
    if not test:
        log.error("test set empty, aborting")
        return 1

    # Auto-enable strict sanity when caller's encoder matches the baseline's
    baseline_meta = json.loads(codestral_path.read_text())
    baseline_retriever = baseline_meta.get("retriever", "")
    model_short = Path(args.model_id).name
    auto_strict = model_short in baseline_retriever
    strict = args.strict_sanity or auto_strict
    log.info("sanity mode: %s (auto_strict=%s, baseline_retriever=%s)",
             "STRICT (hard fail)" if strict else "informational",
             auto_strict, baseline_retriever)

    # Group by repo
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in test:
        by_repo[t["repo"]].append(t)
    log.info("repos in test set: %s", list(by_repo.keys()))

    # Load full instances dataset to get problem_statement + base_commit + gold_files
    all_instances = {i.instance_id: i for i in load_instances()}

    # Encoder
    log.info("Loading %s (%s)", args.model_id, dtype)
    encoder = MambaEncoder(model_id=args.model_id, dtype=dtype,
                           max_length=CHUNK_TOKENS + 32)
    device = str(encoder.device)
    if "cuda" not in device.lower() and "mps" not in device.lower():
        log.warning("Encoder on %s — slow!", device)

    # Run per repo: encode pool once, then eval each instance
    per_instance_results: list[dict] = []
    sanity_failed = 0
    sanity_checked = 0
    for repo_slug, insts in by_repo.items():
        if time.time() - t_start > budget_s:
            log.warning("Budget cap hit, stopping mid-run")
            break

        # Use the FIRST instance's base_commit (matches our per-repo indexing).
        # Voyage and Codestral baselines indexed at first observed commit too.
        sample_inst_id = insts[0]["instance_id"]
        sample_inst = all_instances[sample_inst_id]
        repo_path = ensure_repo_at(repo_slug, sample_inst.base_commit)
        files = list_eligible_files(repo_path)
        pool = encode_repo_pool(encoder, repo_path, files)

        for t in insts:
            if time.time() - t_start > budget_s:
                log.warning("Budget cap hit mid-instance, stopping")
                break

            inst = all_instances[t["instance_id"]]
            gold = inst.gold_files
            ev = eval_instance(encoder, inst.problem_statement, pool, gold,
                               top_k=args.top_k, device=device)

            # Sanity: does Method A reproduce the cached pooled baseline?
            sanity_checked += 1
            instance_sanity_ok = sanity_pool_matches_baseline(
                ev["pooled"]["ranked_top20"][:10], t["codestral_top20"])
            if not instance_sanity_ok:
                sanity_failed += 1
                log.warning("sanity miss on %s (top10 overlap < 50%% with cached baseline)",
                            t["instance_id"])

            per_instance_results.append({
                "instance_id": t["instance_id"],
                "repo": repo_slug,
                "gold_files": gold,
                "sanity_ok": instance_sanity_ok,
                **ev,
            })

            # Hard-fail when strict + accumulated enough samples + pass rate too low
            if strict and sanity_checked >= 5:
                pass_rate = 1 - sanity_failed / sanity_checked
                if pass_rate < 0.5:
                    log.error("STRICT SANITY ABORT: %d/%d instances diverge from "
                              "cached baseline (pass rate %.1f%% < 50%%). "
                              "Method A should reproduce the baseline ranking when "
                              "the same encoder is used. Likely a harness bug.",
                              sanity_failed, sanity_checked, 100 * pass_rate)
                    return 2

    if sanity_checked:
        sanity_pass_rate = 1 - sanity_failed / sanity_checked
        log.info("Sanity (pooled rank ≈ baseline): %d/%d pass (%.1f%%)",
                 sanity_checked - sanity_failed, sanity_checked,
                 100 * sanity_pass_rate)

    # Aggregate metrics
    def _mean(rows, key1, key2):
        vals = [r[key1][key2] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n": len(per_instance_results),
        "sanity_pass_rate": (sanity_pass_rate
                             if sanity_checked else None),
        "pooled": {f"recall@{k}": _mean(per_instance_results, "pooled", f"recall@{k}")
                   for k in K_VALUES},
        "maxsim": {f"recall@{k}": _mean(per_instance_results, "maxsim", f"recall@{k}")
                   for k in K_VALUES},
    }
    summary["pooled"]["mrr"] = _mean(per_instance_results, "pooled", "mrr")
    summary["maxsim"]["mrr"] = _mean(per_instance_results, "maxsim", "mrr")
    summary["pooled"]["mean_latency_s"] = _mean(per_instance_results, "pooled", "latency_s")
    summary["maxsim"]["mean_latency_s"] = _mean(per_instance_results, "maxsim", "latency_s")

    deltas = {
        f"recall@{k}": summary["maxsim"][f"recall@{k}"] - summary["pooled"][f"recall@{k}"]
        for k in K_VALUES
    }

    # Decision criterion
    msim_r10 = summary["maxsim"]["recall@10"]
    if msim_r10 >= DECISION_THRESHOLDS["strong"]:
        verdict = "STRONG_SIGNAL"
    elif msim_r10 >= DECISION_THRESHOLDS["mixed_low"]:
        verdict = "MIXED_SIGNAL"
    else:
        verdict = "BET_WRONG"
    summary["maxsim_recall@10"] = msim_r10
    summary["verdict"] = verdict
    summary["decision_thresholds"] = DECISION_THRESHOLDS
    summary["delta_maxsim_minus_pooled"] = deltas

    report = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "n_test_instances": len(per_instance_results),
        "summary": summary,
        "per_instance": per_instance_results,
        "wall_seconds": time.time() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Print headline
    print()
    print("=" * 70)
    print("MAXSIM DISCRIMINATION TEST")
    print(f"model: {args.model_id}  dtype: {args.dtype}")
    print(f"test set: {len(per_instance_results)} instances "
          f"(Voyage R@10=1 AND Codestral pooled R@10=0)")
    print("-" * 70)
    print(f"{'metric':<14}{'Pooled':>10}{'MaxSim':>10}{'Δ':>10}")
    for k in K_VALUES:
        p = summary["pooled"][f"recall@{k}"]
        m = summary["maxsim"][f"recall@{k}"]
        print(f"Recall@{k:<7}{p:>10.4f}{m:>10.4f}{m - p:>+10.4f}")
    print(f"{'MRR':<14}{summary['pooled']['mrr']:>10.4f}"
          f"{summary['maxsim']['mrr']:>10.4f}"
          f"{summary['maxsim']['mrr'] - summary['pooled']['mrr']:>+10.4f}")
    print(f"{'Latency (s)':<14}{summary['pooled']['mean_latency_s']:>10.3f}"
          f"{summary['maxsim']['mean_latency_s']:>10.3f}")
    print("-" * 70)
    print(f"Verdict: {verdict}  (R@10={msim_r10:.4f})")
    print(f"  STRONG  ≥ {DECISION_THRESHOLDS['strong']:.2f}: build Phase 3")
    print(f"  MIXED   ≥ {DECISION_THRESHOLDS['mixed_low']:.2f}: reconsider")
    print(f"  WRONG   < {DECISION_THRESHOLDS['mixed_low']:.2f}: stop")
    print("=" * 70)

    # Sample 10 instance details
    print("\nSample 10 instances (gold rank under each method):")
    sample = per_instance_results[: min(10, len(per_instance_results))]
    for r in sample:
        gold = r["gold_files"][0] if r["gold_files"] else "<none>"
        p_rank = r["pooled"]["ranked_top20"].index(gold) + 1 if gold in r["pooled"]["ranked_top20"] else ">20"
        m_rank = r["maxsim"]["ranked_top20"].index(gold) + 1 if gold in r["maxsim"]["ranked_top20"] else ">20"
        print(f"  {r['instance_id']:<35}  gold={gold}")
        print(f"      pooled rank: {p_rank}   top-3: {r['pooled']['ranked_top20'][:3]}")
        print(f"      maxsim rank: {m_rank}   top-3: {r['maxsim']['ranked_top20'][:3]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
