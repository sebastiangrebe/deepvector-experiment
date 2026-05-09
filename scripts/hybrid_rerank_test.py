"""Phase 2.7: Mamba MaxSim + transformer reranker hybrid.

Pipeline:
  Step 1 — MaxSim retrieval, top-100 candidate files per query (reuses
           cached token-level latents from Phase 2.5/2.6 if present)
  Step 2 — For each top-100 candidate, build (issue_text, file_content)
           pair, pass through cross-encoder reranker
  Step 3 — Re-rank by reranker score, measure Recall@10/MRR

Three rerankers tested independently:
  - BAAI/bge-reranker-v2-m3 (568M)
  - jinaai/jina-reranker-v2-base-multilingual (278M)
  - cross-encoder/ms-marco-MiniLM-L-12-v2 (33M)

Decision criterion (pre-registered):
  best hybrid R@10 ≥ 0.75 → competitive end-to-end pipeline
  best hybrid R@10 0.55-0.75 → improvement, not competitive
  best hybrid R@10 ≤ MaxSim → reranker can't recover; need better filter

Usage (cloud H100, latents cached + same instance as Phase 2.5/2.6):
  python scripts/hybrid_rerank_test.py \\
      --model-id mistralai/Mamba-Codestral-7B-v0.1 \\
      --dtype bfloat16 \\
      --max-test-instances 80 \\
      --reranker-top-k 100

Local smoke (mamba-130m + minilm only, sqlfluff):
  python scripts/hybrid_rerank_test.py \\
      --model-id state-spaces/mamba-130m-hf \\
      --dtype float32 \\
      --max-test-instances 5 \\
      --restrict-repos sqlfluff/sqlfluff \\
      --rerankers minilm_l12 \\
      --out data/results/hybrid_rerank_smoke.json
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.eval import ensure_repo_at, load_instances
from src.frozen_methods import dedup_to_files, maxsim_score
from src.rerankers import SPECS, build_reranker
from src.utils import get_logger, list_eligible_files
from scripts.maxsim_test import (build_test_set, encode_repo_pool, K_VALUES,
                                  recall_at_k, reciprocal_rank)

log = get_logger("hybrid_rerank")

DECISION_THRESHOLDS = {"competitive": 0.75, "improvement_low": 0.55}

# Reranker doc truncation: ~6000 chars ≈ 1500 tokens at 4 chars/tok (rough).
# Keeps doc inside reranker context (1024 toks for bge/jina, 512 for minilm).
MAX_DOC_CHARS = 6000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mistralai/Mamba-Codestral-7B-v0.1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-test-instances", type=int, default=80)
    ap.add_argument("--restrict-repos", nargs="*", default=None)
    ap.add_argument("--reranker-top-k", type=int, default=100,
                    help="Number of MaxSim candidates passed to the reranker")
    ap.add_argument("--final-top-k", type=int, default=20)
    ap.add_argument("--rerankers", nargs="*",
                    default=[s.name for s in SPECS],
                    help="Subset of rerankers to run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/results/hybrid_rerank_test.json")
    ap.add_argument("--budget-hours", type=float, default=4.0)
    ap.add_argument("--use-mamba-265-set", action="store_true",
                    help="Expand to all Voyage-indexable instances instead of "
                         "the strict 80-instance discriminating subset")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    t_start = time.time()
    budget_s = args.budget_hours * 3600

    # Test set selection
    voyage_path = ROOT / "data/results/voyage_baseline.json"
    codestral_path = ROOT / "data/results/mamba_codestral_baseline.json"
    test = build_test_set(voyage_path, codestral_path,
                          max_n=args.max_test_instances, seed=args.seed,
                          restrict_repos=args.restrict_repos)
    if not test:
        log.error("test set empty")
        return 1

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in test:
        by_repo[t["repo"]].append(t)

    all_instances = {i.instance_id: i for i in load_instances()}

    # ─── Step 1: MaxSim top-100 candidates per query ───────────────────
    log.info("Step 1: MaxSim retrieval (top-%d candidates)", args.reranker_top_k)
    log.info("Loading encoder %s", args.model_id)
    encoder = MambaEncoder(model_id=args.model_id, dtype=dtype, max_length=1532)
    device = str(encoder.device)

    candidates_by_inst: dict[str, list[tuple[str, float]]] = {}
    file_paths_by_repo: dict[str, dict[str, Path]] = {}

    for repo_slug, insts in by_repo.items():
        if time.time() - t_start > budget_s:
            log.warning("Budget cap hit during retrieval"); break

        sample = all_instances[insts[0]["instance_id"]]
        repo_path = ensure_repo_at(repo_slug, sample.base_commit)
        files = list_eligible_files(repo_path)
        file_paths_by_repo[repo_slug] = {
            f.relative_to(repo_path).as_posix(): f for f in files
        }

        pool = encode_repo_pool(encoder, repo_path, files)
        cand_dev = [t.to(device) for t in pool.tokens]

        for t in tqdm(insts, desc=f"maxsim {repo_slug}", leave=False):
            inst = all_instances[t["instance_id"]]
            out = encoder.encode([inst.problem_statement])
            mask = out.attention_mask[0].bool()
            q_tok = out.last_hidden[0][mask].to(device)

            with torch.no_grad():
                scores = maxsim_score(q_tok, cand_dev)
            ranked = dedup_to_files(scores, pool.chunk_files,
                                    top_k=args.reranker_top_k)
            candidates_by_inst[t["instance_id"]] = ranked

        # Free repo's GPU pool before next repo to avoid OOM accumulation
        del cand_dev, pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Free encoder VRAM before loading rerankers
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ─── Step 2: per-reranker scoring (fast → slow order, save after each) ─
    # Preserve fastest-first ordering from SPECS (minilm → jina → bge) so
    # cheap rerankers finish first if a slow one trips the budget cap.
    name_to_spec = {s.name: s for s in SPECS}
    selected_specs = [name_to_spec[n] for n in
                      [s.name for s in SPECS if s.name in args.rerankers]]
    log.info("Step 2-3: rerankers (in order) %s",
             [s.name for s in selected_specs])

    reranker_results: dict[str, list[dict]] = {s.name: [] for s in selected_specs}
    first_instance_top3: dict[str, dict] = {}   # spec_name → {gold, top3}

    def _flush_partial(args_, summary_so_far: dict, raw_data: dict) -> None:
        """Write partial JSON after each reranker completes."""
        path = args_.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"partial": True, "summary_so_far": summary_so_far, **raw_data},
            indent=2,
        ))

    for spec_idx, spec in enumerate(selected_specs):
        if time.time() - t_start > budget_s:
            log.warning("Budget cap hit before reranker %s", spec.name); break
        rr = build_reranker(spec, dtype=dtype)

        first_iid_logged = False
        for iid, ranked in tqdm(candidates_by_inst.items(),
                                desc=spec.name, leave=False):
            inst = all_instances[iid]
            repo_paths = file_paths_by_repo[inst.repo]
            pairs: list[tuple[str, str]] = []
            paths_in_order: list[str] = []
            for rel_path, _ in ranked:
                p = repo_paths.get(rel_path)
                if p is None:
                    continue
                try:
                    # Raw on-disk file content. NOT from the latent cache.
                    # Rerankers consume raw text, latents are only used in
                    # Step 1 (MaxSim filter).
                    content = p.read_text(errors="replace")[:MAX_DOC_CHARS]
                except OSError:
                    continue
                pairs.append((inst.problem_statement, content))
                paths_in_order.append(rel_path)

            scores = rr.score_pairs(pairs) if pairs else []
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            reranked = [paths_in_order[i] for i in order][: args.final_top_k]

            # Sanity log: first instance, all rerankers — eyeball top-3 vs gold
            if not first_iid_logged:
                first_instance_top3[spec.name] = {
                    "instance_id": iid,
                    "gold_files": inst.gold_files,
                    "top3": reranked[:3],
                }
                first_iid_logged = True

            gold = set(inst.gold_files)
            reranker_results[spec.name].append({
                "instance_id": iid,
                "repo": inst.repo,
                "gold_files": inst.gold_files,
                "maxsim_top20": [p for p, _ in ranked[:20]],
                "reranked_top20": reranked,
                **{f"recall@{k}": recall_at_k(gold, reranked, k) for k in K_VALUES},
                "mrr": reciprocal_rank(gold, reranked),
            })

        # Per-reranker intermediate summary + flush to disk
        rows = reranker_results[spec.name]
        if rows:
            mid = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / len(rows)
                   for k in K_VALUES}
            mid["mrr"] = sum(r["mrr"] for r in rows) / len(rows)
            log.info("[%s/%d] %s → R@10=%.4f MRR=%.4f (n=%d)",
                     spec_idx + 1, len(selected_specs), spec.name,
                     mid["recall@10"], mid["mrr"], len(rows))
            _flush_partial(args, {spec.name: mid}, {
                "rerankers_so_far": reranker_results,
                "first_instance_top3": first_instance_top3,
            })

        del rr
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # MaxSim-only baseline (top-K cut from the same candidate list)
    maxsim_only: list[dict] = []
    for iid, ranked in candidates_by_inst.items():
        inst = all_instances[iid]
        gold = set(inst.gold_files)
        paths = [p for p, _ in ranked[: args.final_top_k]]
        maxsim_only.append({
            "instance_id": iid,
            "repo": inst.repo,
            "gold_files": inst.gold_files,
            **{f"recall@{k}": recall_at_k(gold, paths, k) for k in K_VALUES},
            "mrr": reciprocal_rank(gold, paths),
        })

    # ─── Aggregate + verdict ──────────────────────────────────────────
    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            **{f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / len(rows)
               for k in K_VALUES},
            "mrr": sum(r["mrr"] for r in rows) / len(rows),
        }

    summary = {
        "n": len(maxsim_only),
        "maxsim_only": _agg(maxsim_only),
        "rerankers": {n: _agg(rows) for n, rows in reranker_results.items()},
    }
    best_hybrid_r10 = max(
        (s["recall@10"] for s in summary["rerankers"].values() if "recall@10" in s),
        default=0.0,
    )
    summary["best_hybrid_recall@10"] = best_hybrid_r10
    summary["maxsim_only_recall@10"] = summary["maxsim_only"].get("recall@10", 0.0)

    if best_hybrid_r10 >= DECISION_THRESHOLDS["competitive"]:
        verdict = "COMPETITIVE"
    elif best_hybrid_r10 >= DECISION_THRESHOLDS["improvement_low"]:
        verdict = "IMPROVEMENT_NOT_COMPETITIVE"
    elif best_hybrid_r10 > summary["maxsim_only_recall@10"] + 0.02:
        verdict = "MARGINAL_GAIN"
    else:
        verdict = "FILTER_LIMITED"
    summary["verdict"] = verdict
    summary["decision_thresholds"] = DECISION_THRESHOLDS

    report = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "n_test_instances": summary["n"],
        "summary": summary,
        "first_instance_top3": first_instance_top3,
        "maxsim_only": maxsim_only,
        "rerankers": reranker_results,
        "wall_seconds": time.time() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Print headline
    print()
    print("=" * 90)
    print("HYBRID MAXSIM + RERANKER TEST")
    print(f"model: {args.model_id}  dtype: {args.dtype}")
    print(f"test set: {summary['n']} instances")
    print("-" * 90)
    cols = ["maxsim_only"] + [s.name for s in selected_specs]
    head = "{:<14}".format("metric") + "".join(f"{c:>20}" for c in cols)
    print(head)
    for k in K_VALUES:
        row = "{:<14}".format(f"Recall@{k}")
        for c in cols:
            obj = summary["maxsim_only"] if c == "maxsim_only" else summary["rerankers"][c]
            row += f"{obj.get(f'recall@{k}', 0.0):>20.4f}"
        print(row)
    row = "{:<14}".format("MRR")
    for c in cols:
        obj = summary["maxsim_only"] if c == "maxsim_only" else summary["rerankers"][c]
        row += f"{obj.get('mrr', 0.0):>20.4f}"
    print(row)
    print("-" * 90)
    print(f"Verdict: {verdict}  (best_hybrid_R@10={best_hybrid_r10:.4f}, "
          f"MaxSim-only_R@10={summary['maxsim_only_recall@10']:.4f})")
    print(f"  COMPETITIVE                ≥ {DECISION_THRESHOLDS['competitive']:.2f}")
    print(f"  IMPROVEMENT_NOT_COMPETITIVE ≥ {DECISION_THRESHOLDS['improvement_low']:.2f}")
    print(f"  MARGINAL_GAIN: hybrid > MaxSim+0.02 but below 0.55")
    print(f"  FILTER_LIMITED: rerankers don't recover what MaxSim missed")
    print("=" * 90)

    # Sanity eyeball: first-instance top-3 across rerankers
    if first_instance_top3:
        print()
        print("Sanity (first instance — top-3 per reranker vs gold):")
        for name, info in first_instance_top3.items():
            print(f"  {name:<22} gold={info['gold_files']}")
            for i, p in enumerate(info["top3"], 1):
                hit = " ←GOLD" if p in info["gold_files"] else ""
                print(f"    {i}. {p}{hit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
