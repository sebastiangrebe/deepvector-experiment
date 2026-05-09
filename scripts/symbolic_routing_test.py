"""Phase 2.9 + Phase 3.0: tree-index symbolic routing, optionally with
Codestral-generated query expansion.

Pipeline (per instance):
  1. Extract identifiers from issue text via regex (Phase 2.9 + 3.0)
  2. (Phase 3.0 only) call expand_query_to_identifiers via Codestral
  3. Look up identifiers in the repo's tree-index → primary candidate files
  4. Expand via 1-hop import graph → extended candidate pool
  5. Apply fallback rules: <5 candidates → fall back to full corpus
                          >200 candidates → keep only primary
  6. Run MaxSim restricted to candidate files; report top-K

Always also reports a control "maxsim_full" run (no routing) for the same
queries — used as the strict-sanity reproduction of the Phase 2.5 baseline.

Usage:
  # Phase 2.9 (cloud H100, hot cache)
  python scripts/symbolic_routing_test.py \\
      --phase tree_index_only --model-id mistralai/Mamba-Codestral-7B-v0.1 \\
      --dtype bfloat16 --max-test-instances 80 --top-k 20 \\
      --budget-hours 2 --out data/results/tree_index_test.json

  # Phase 3.0
  python scripts/symbolic_routing_test.py \\
      --phase with_llm_expansion --model-id mistralai/Mamba-Codestral-7B-v0.1 \\
      --dtype bfloat16 --max-test-instances 80 --top-k 20 \\
      --budget-hours 3 --out data/results/llm_expansion_test.json
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
from statistics import median

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.eval import ensure_repo_at, load_instances
from src.llm_expansion import expand_query_to_identifiers
from src.frozen_methods import maxsim_score as maxsim_streaming
from src.maxsim import dedup_to_files
from src.tree_index import RepoTreeIndex, extract_identifiers
from src.utils import get_logger, list_eligible_files
from scripts.maxsim_test import (build_test_set, encode_repo_pool, K_VALUES,
                                  recall_at_k, reciprocal_rank)

log = get_logger("symbolic_routing")

DECISION_THRESHOLDS = {"strong": 0.55, "modest_low": 0.46, "hurts": -0.02}
MAXSIM_BASELINE_R10 = 0.4375  # Phase 2.5 reference
SANITY_TOLERANCE = 0.02


def build_candidate_pool(idx: RepoTreeIndex, identifiers: list[str],
                          *, min_pool: int = 5, max_pool: int = 200,
                          import_hops: int = 1) -> dict:
    """Build candidate pool. Returns a dict with full diagnostic info:

        {
            "candidates": set[str],
            "mode": "primary" | "extended" | "fallback_full",
            "primary": set[str],
            "matched_ids": list[str],   # ids that hit ≥1 file
            "unmatched_ids": list[str], # ids that hit 0 files
            "primary_count": int,
            "extended_count": int,      # |extended| - |primary|
        }
    """
    primary: set[str] = set()
    matched: list[str] = []
    unmatched: list[str] = []
    for name in identifiers:
        hit = idx.lookup(name)
        if hit:
            matched.append(name)
            primary |= hit
        else:
            unmatched.append(name)

    if not primary:
        return {
            "candidates": set(idx.files.keys()), "mode": "fallback_full",
            "primary": set(), "matched_ids": matched, "unmatched_ids": unmatched,
            "primary_count": 0, "extended_count": 0,
        }

    extended = idx.expand_via_imports(primary, hops=import_hops)
    primary_count = len(primary)
    extended_count = len(extended) - primary_count
    if len(extended) > max_pool:
        return {
            "candidates": primary, "mode": "primary",
            "primary": primary, "matched_ids": matched, "unmatched_ids": unmatched,
            "primary_count": primary_count, "extended_count": 0,
        }
    if len(extended) < min_pool:
        return {
            "candidates": set(idx.files.keys()), "mode": "fallback_full",
            "primary": primary, "matched_ids": matched, "unmatched_ids": unmatched,
            "primary_count": primary_count, "extended_count": extended_count,
        }
    return {
        "candidates": extended, "mode": "extended",
        "primary": primary, "matched_ids": matched, "unmatched_ids": unmatched,
        "primary_count": primary_count, "extended_count": extended_count,
    }


def restrict_pool_to_candidates(pool_chunk_files: list[str],
                                 pool_tokens: list[torch.Tensor],
                                 candidate_files: set[str]
                                 ) -> tuple[list[str], list[torch.Tensor]]:
    """Filter the cached chunk pool to only chunks whose file is in candidate_files."""
    keep_files: list[str] = []
    keep_tokens: list[torch.Tensor] = []
    for f, t in zip(pool_chunk_files, pool_tokens):
        if f in candidate_files:
            keep_files.append(f)
            keep_tokens.append(t)
    return keep_files, keep_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["tree_index_only", "with_llm_expansion"])
    ap.add_argument("--model-id", default="mistralai/Mamba-Codestral-7B-v0.1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-test-instances", type=int, default=80)
    ap.add_argument("--restrict-repos", nargs="*", default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-hours", type=float, default=3.0)
    ap.add_argument("--llm-max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    t_start = time.time()
    budget_s = args.budget_hours * 3600

    voyage_path = ROOT / "data/results/voyage_baseline.json"
    codestral_path = ROOT / "data/results/mamba_codestral_baseline.json"
    test = build_test_set(voyage_path, codestral_path,
                          max_n=args.max_test_instances, seed=args.seed,
                          restrict_repos=args.restrict_repos)
    if not test:
        log.error("test set empty"); return 1

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in test:
        by_repo[t["repo"]].append(t)

    log.info("Loading encoder %s", args.model_id)
    encoder = MambaEncoder(model_id=args.model_id, dtype=dtype, max_length=1532)
    device = str(encoder.device)
    if "cuda" not in device.lower() and "mps" not in device.lower():
        log.warning("encoder on %s (slow path)", device)

    all_instances = {i.instance_id: i for i in load_instances()}

    per_instance: list[dict] = []
    n_zero_cand_no_fallback = 0
    pool_size_samples: list[int] = []
    fallback_count = 0
    primary_count = 0
    extended_count = 0
    n_regex_per_query: list[int] = []
    n_llm_per_query: list[int] = []
    llm_failures = 0
    llm_latencies: list[float] = []
    first_llm_logged = 0
    sanity_aborted_msg: str | None = None

    for repo_slug, insts in by_repo.items():
        if time.time() - t_start > budget_s:
            log.warning("budget cap"); break

        sample_inst = all_instances[insts[0]["instance_id"]]
        repo_path = ensure_repo_at(repo_slug, sample_inst.base_commit)
        files = list_eligible_files(repo_path)

        # Tree-index (cached)
        tidx = RepoTreeIndex(repo_path)
        tidx.build()
        log.info("[%s] tree-index: %d files, %d identifiers",
                 repo_slug, tidx.files_count(), tidx.identifiers_count())

        # Sanity check 1: known case (django)
        if repo_slug == "django/django":
            hit = tidx.lookup("UnicodeUsernameValidator")
            expected = "django/contrib/auth/validators.py"
            if expected not in hit:
                sanity_aborted_msg = (f"SANITY: UnicodeUsernameValidator not in "
                                      f"tree-index for django; lookup returned {hit}")
                log.error(sanity_aborted_msg)
                return 2
            log.info("[django] sanity: UnicodeUsernameValidator → %s ✓", expected)

        # Cached token-level pool
        pool = encode_repo_pool(encoder, repo_path, files)

        for t in tqdm(insts, desc=repo_slug, leave=False):
            if time.time() - t_start > budget_s:
                log.warning("budget cap mid-instance"); break

            inst = all_instances[t["instance_id"]]
            issue = inst.problem_statement
            gold_set = set(inst.gold_files)

            regex_ids = extract_identifiers(issue)
            n_regex_per_query.append(len(regex_ids))

            llm_ids: list[str] = []
            llm_raw = ""
            llm_dt = 0.0
            if args.phase == "with_llm_expansion":
                t0 = time.time()
                try:
                    llm_ids, llm_raw = expand_query_to_identifiers(
                        issue, encoder,
                        max_identifiers=20,
                        max_new_tokens=args.llm_max_new_tokens,
                    )
                except Exception as e:
                    log.warning("llm expansion failed: %s: %s", type(e).__name__, e)
                    llm_failures += 1
                llm_dt = time.time() - t0
                llm_latencies.append(llm_dt)
                n_llm_per_query.append(len(llm_ids))
                if first_llm_logged < 3:
                    log.info("[LLM smoke %d] iid=%s  ids=%s",
                             first_llm_logged + 1, t["instance_id"], llm_ids[:10])
                    log.info("[LLM smoke %d] raw=%r", first_llm_logged + 1,
                             llm_raw[:300])
                    first_llm_logged += 1
                if not llm_ids and not llm_raw:
                    llm_failures += 1

            all_ids = regex_ids + [i for i in llm_ids if i not in regex_ids]

            # Per-id matched/unmatched in tree-index (computed once for diagnostics)
            regex_matched = [i for i in regex_ids if tidx.lookup(i)]
            regex_unmatched = [i for i in regex_ids if not tidx.lookup(i)]
            llm_matched = [i for i in llm_ids if tidx.lookup(i)]
            llm_unmatched = [i for i in llm_ids if not tidx.lookup(i)]

            # Encode query once (token-level for MaxSim)
            with torch.no_grad():
                out = encoder.encode([issue])
            mask = out.attention_mask[0].bool()
            q_tok = out.last_hidden[0][mask].to(device)

            # Control: full-corpus MaxSim (sanity reproduction)
            with torch.no_grad():
                full_scores = maxsim_streaming(q_tok, pool.tokens)
            full_ranked = dedup_to_files(full_scores, pool.chunk_files,
                                          top_k=args.top_k)
            full_paths = [p for p, _ in full_ranked]

            # Routed: build candidate pool, restrict, MaxSim
            pool_info = build_candidate_pool(tidx, all_ids)
            cand_set = pool_info["candidates"]
            mode = pool_info["mode"]
            if mode == "primary":
                primary_count += 1
            elif mode == "extended":
                extended_count += 1
            else:
                fallback_count += 1
            if not cand_set:
                n_zero_cand_no_fallback += 1
            pool_size_samples.append(len(cand_set))

            kept_files, kept_tokens = restrict_pool_to_candidates(
                pool.chunk_files, pool.tokens, cand_set)

            t0 = time.time()
            if not kept_tokens:
                routed_paths: list[str] = []
            else:
                with torch.no_grad():
                    routed_scores = maxsim_streaming(q_tok, kept_tokens)
                routed_ranked = dedup_to_files(routed_scores, kept_files,
                                                top_k=args.top_k)
                routed_paths = [p for p, _ in routed_ranked]
            routed_dt = time.time() - t0

            per_instance.append({
                "id": t["instance_id"],
                "repo": repo_slug,
                "issue_head": issue[:200],
                "regex_ids_extracted": regex_ids,
                "regex_ids_matched_in_index": regex_matched,
                "regex_ids_unmatched": regex_unmatched,
                "llm_ids_extracted": llm_ids,
                "llm_ids_matched_in_index": llm_matched,
                "llm_ids_unmatched": llm_unmatched,
                "llm_latency_s": llm_dt,
                "primary_candidates_count": pool_info["primary_count"],
                "extended_candidates_count": pool_info["extended_count"],
                "candidate_pool_size_final": len(cand_set),
                "candidate_pool_mode": mode,
                "fallback_to_full_corpus": (mode == "fallback_full"),
                "gold_in_pool": bool(gold_set & cand_set),
                "gold_files": list(gold_set),
                "routed_top20": routed_paths,
                "routed_recall@1":  recall_at_k(gold_set, routed_paths, 1),
                "routed_recall@5":  recall_at_k(gold_set, routed_paths, 5),
                "routed_recall@10": recall_at_k(gold_set, routed_paths, 10),
                "routed_recall@20": recall_at_k(gold_set, routed_paths, 20),
                "routed_mrr": reciprocal_rank(gold_set, routed_paths),
                "routed_latency_s": routed_dt,
                "full_top20": full_paths,
                "full_recall@1":  recall_at_k(gold_set, full_paths, 1),
                "full_recall@5":  recall_at_k(gold_set, full_paths, 5),
                "full_recall@10": recall_at_k(gold_set, full_paths, 10),
                "full_recall@20": recall_at_k(gold_set, full_paths, 20),
                "full_mrr": reciprocal_rank(gold_set, full_paths),
            })

        del pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    def _mean(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    summary: dict = {
        "phase": args.phase,
        "n": len(per_instance),
        "routed_recall@1":  _mean(per_instance, "routed_recall@1"),
        "routed_recall@5":  _mean(per_instance, "routed_recall@5"),
        "routed_recall@10": _mean(per_instance, "routed_recall@10"),
        "routed_recall@20": _mean(per_instance, "routed_recall@20"),
        "routed_mrr":       _mean(per_instance, "routed_mrr"),
        "full_recall@1":  _mean(per_instance, "full_recall@1"),
        "full_recall@5":  _mean(per_instance, "full_recall@5"),
        "full_recall@10": _mean(per_instance, "full_recall@10"),
        "full_recall@20": _mean(per_instance, "full_recall@20"),
        "full_mrr":       _mean(per_instance, "full_mrr"),
        "median_latency_s": float(median(r["routed_latency_s"]
                                          for r in per_instance) if per_instance else 0.0),
        "median_candidate_pool_size": int(median(pool_size_samples)
                                            if pool_size_samples else 0),
        "instances_with_zero_candidates": n_zero_cand_no_fallback,
        "instances_falling_back_to_full_corpus": fallback_count,
        "candidate_pool_modes": {
            "primary": primary_count, "extended": extended_count,
            "fallback_full": fallback_count,
        },
        "regex_identifiers_per_query": {
            "mean": (sum(n_regex_per_query) / len(n_regex_per_query)
                     if n_regex_per_query else 0.0),
            "min": min(n_regex_per_query) if n_regex_per_query else 0,
            "max": max(n_regex_per_query) if n_regex_per_query else 0,
            "median": int(median(n_regex_per_query)) if n_regex_per_query else 0,
        },
    }

    if args.phase == "with_llm_expansion":
        summary["llm_identifiers_per_query"] = {
            "mean": (sum(n_llm_per_query) / len(n_llm_per_query)
                     if n_llm_per_query else 0.0),
            "min": min(n_llm_per_query) if n_llm_per_query else 0,
            "max": max(n_llm_per_query) if n_llm_per_query else 0,
            "median": int(median(n_llm_per_query)) if n_llm_per_query else 0,
        }
        summary["llm_generation_failures"] = llm_failures
        summary["median_llm_latency_s"] = float(median(llm_latencies)
                                                  if llm_latencies else 0.0)

    # Sanity check 2: full-corpus MaxSim must reproduce Phase 2.5 baseline ±0.02.
    # Only enforced when the encoder matches the cached baseline's encoder
    # AND the test set isn't restricted (otherwise different subset → different baseline).
    full_r10 = summary["full_recall@10"]
    is_codestral = "Mamba-Codestral" in args.model_id
    full_subset = (args.restrict_repos is None
                   and args.max_test_instances >= 80)
    if is_codestral and full_subset and abs(full_r10 - MAXSIM_BASELINE_R10) > SANITY_TOLERANCE:
        sanity_aborted_msg = (f"SANITY: full-corpus MaxSim R@10 {full_r10:.4f} "
                              f"diverges from Phase 2.5 baseline {MAXSIM_BASELINE_R10:.4f} "
                              f"by > {SANITY_TOLERANCE}; harness drift")
        log.error(sanity_aborted_msg)

    # Sanity check 3: zero-candidate-no-fallback rate
    if per_instance:
        zero_no_fb_rate = n_zero_cand_no_fallback / len(per_instance)
        if zero_no_fb_rate > 0.05:
            log.warning("SANITY: %.1f%% zero-candidate-no-fallback rate "
                        "(threshold 5%%)", 100 * zero_no_fb_rate)

    summary["sanity_full_maxsim_reproduces_phase25"] = (
        sanity_aborted_msg is None
        or "diverges from Phase 2.5" not in sanity_aborted_msg)
    if sanity_aborted_msg:
        summary["sanity_aborted_msg"] = sanity_aborted_msg

    # Verdict
    routed_r10 = summary["routed_recall@10"]
    delta = routed_r10 - full_r10
    if routed_r10 >= DECISION_THRESHOLDS["strong"]:
        verdict = "STRONG_LIFT"
    elif routed_r10 >= DECISION_THRESHOLDS["modest_low"]:
        verdict = "MODEST_LIFT"
    elif delta >= DECISION_THRESHOLDS["hurts"]:
        verdict = "NO_LIFT"
    else:
        verdict = "HURTS"
    summary["verdict"] = verdict
    summary["maxsim_baseline_r10"] = full_r10
    summary["routed_minus_maxsim"] = delta
    summary["decision_thresholds"] = DECISION_THRESHOLDS

    report = {
        "phase": args.phase,
        "config": {
            "model_id": args.model_id,
            "dtype": args.dtype,
            "n_test": len(per_instance),
            "top_k": args.top_k,
        },
        "summary": summary,
        "per_instance": per_instance,
        "wall_seconds": time.time() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Print
    print()
    print("=" * 100)
    print(f"SYMBOLIC ROUTING TEST  (phase={args.phase})")
    print(f"model: {args.model_id}  n={len(per_instance)}")
    print("-" * 100)
    print(f"{'metric':<22}{'routed':>14}{'maxsim_full':>14}{'Δ (routed-full)':>20}")
    for k in K_VALUES:
        rv = summary[f"routed_recall@{k}"]
        fv = summary[f"full_recall@{k}"]
        print(f"{'Recall@'+str(k):<22}{rv:>14.4f}{fv:>14.4f}{rv - fv:>+20.4f}")
    print(f"{'MRR':<22}{summary['routed_mrr']:>14.4f}"
          f"{summary['full_mrr']:>14.4f}{summary['routed_mrr'] - summary['full_mrr']:>+20.4f}")
    print(f"{'Median latency (s)':<22}{summary['median_latency_s']:>14.3f}")
    print("-" * 100)
    print(f"Candidate pool:  primary={primary_count}  extended={extended_count}  "
          f"fallback_full={fallback_count}  median_size={summary['median_candidate_pool_size']}")
    print(f"Regex identifiers/query: mean={summary['regex_identifiers_per_query']['mean']:.1f}  "
          f"median={summary['regex_identifiers_per_query']['median']}  "
          f"max={summary['regex_identifiers_per_query']['max']}")
    if args.phase == "with_llm_expansion":
        print(f"LLM identifiers/query: mean={summary['llm_identifiers_per_query']['mean']:.1f}  "
              f"median={summary['llm_identifiers_per_query']['median']}  "
              f"failures={summary['llm_generation_failures']}  "
              f"median_latency={summary['median_llm_latency_s']:.2f}s")
    print(f"sanity_full_maxsim_reproduces_phase25: "
          f"{summary['sanity_full_maxsim_reproduces_phase25']}")
    print("-" * 100)
    print(f"Verdict: {verdict}  (routed R@10={routed_r10:.4f}, "
          f"full MaxSim R@10={full_r10:.4f}, Δ={delta:+.4f})")
    print(f"  STRONG_LIFT  routed R@10 ≥ {DECISION_THRESHOLDS['strong']:.2f}")
    print(f"  MODEST_LIFT  routed R@10 ≥ {DECISION_THRESHOLDS['modest_low']:.2f}")
    print(f"  NO_LIFT      |Δ| ≤ {abs(DECISION_THRESHOLDS['hurts']):.2f}")
    print(f"  HURTS        Δ < {DECISION_THRESHOLDS['hurts']:.2f}")
    print("=" * 100)

    # Sample comparison (paper qualitative analysis)
    def _gold_rank(r, paths_key):
        gold = r["gold_files"][0] if r["gold_files"] else None
        if not gold:
            return None
        paths = r[paths_key]
        return (paths.index(gold) + 1) if gold in paths else ">20"

    rescues = [r for r in per_instance
               if any(g in r["routed_top20"][:10] for g in r["gold_files"])
               and not any(g in r["full_top20"][:10] for g in r["gold_files"])]
    both_hit = [r for r in per_instance
                if any(g in r["routed_top20"][:10] for g in r["gold_files"])
                and any(g in r["full_top20"][:10] for g in r["gold_files"])]
    pool_misses = [r for r in per_instance if not r["gold_in_pool"]]

    sample_categories = [
        ("RESCUE (routed top-10, full missed)", rescues[:2]),
        ("BOTH HIT", both_hit[:2]),
        ("POOL MISS (gold not in candidate pool)", pool_misses[:1]),
    ]

    print()
    print("=" * 100)
    print("QUALITATIVE SAMPLES (paper write-up)")
    print("=" * 100)
    sample_dump = []
    for cat, rs in sample_categories:
        for r in rs:
            print(f"\n[{cat}]  {r['id']}  ({r['repo']})")
            print(f"  issue head: {r['issue_head'][:200]}")
            print(f"  regex IDs ({len(r['regex_ids_extracted'])}): "
                  f"{r['regex_ids_extracted'][:8]}{'...' if len(r['regex_ids_extracted']) > 8 else ''}")
            print(f"  matched in tree-index ({len(r['regex_ids_matched_in_index'])}): "
                  f"{r['regex_ids_matched_in_index'][:8]}")
            if r.get('llm_ids_extracted'):
                print(f"  LLM IDs ({len(r['llm_ids_extracted'])}): "
                      f"{r['llm_ids_extracted'][:8]}")
                print(f"  LLM matched ({len(r['llm_ids_matched_in_index'])}): "
                      f"{r['llm_ids_matched_in_index'][:8]}")
            print(f"  pool: primary={r['primary_candidates_count']}  "
                  f"extended={r['extended_candidates_count']}  "
                  f"final={r['candidate_pool_size_final']}  "
                  f"mode={r['candidate_pool_mode']}")
            print(f"  gold: {r['gold_files']}")
            print(f"  gold rank — routed: {_gold_rank(r, 'routed_top20')}  "
                  f"full: {_gold_rank(r, 'full_top20')}")
            sample_dump.append({"category": cat, **r})
    print("=" * 100)

    # Persist sample dump in the JSON for paper write-up
    report["qualitative_samples"] = sample_dump
    args.out.write_text(json.dumps(report, indent=2))

    return 0 if sanity_aborted_msg is None else 2


if __name__ == "__main__":
    sys.exit(main())
