"""Phase 2.8: multi-granularity matching test.

Reuses Phase 2.5/2.6/2.7 cached token-level latents (same data/maxsim_cache)
and derives G0 (file pool), G1 (chunk pool), G2 (sliding-window pool),
G3 (token-level) from the same forward pass — no re-encoding.

Tests 7 matching methods on the 80-instance discriminating subset and
prints a verdict (STRONG / MODEST / NO_LIFT / HURTS).

Usage (cloud H100):
  python scripts/multigranular_test.py \\
      --model-id mistralai/Mamba-Codestral-7B-v0.1 \\
      --dtype bfloat16 \\
      --max-test-instances 80 \\
      --budget-hours 5

Local smoke (mamba-130m, sqlfluff only):
  python scripts/multigranular_test.py \\
      --model-id state-spaces/mamba-130m-hf \\
      --dtype float32 \\
      --max-test-instances 5 \\
      --restrict-repos sqlfluff/sqlfluff \\
      --out data/results/multigranular_smoke.json
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.eval import ensure_repo_at, load_instances
from src.multigranular import (build_file_representations, route_query,
                                score_pooled_chunk, score_pooled_file,
                                score_func_pool, score_maxsim,
                                score_mg_sum, score_mg_max, score_mg_routed,
                                topk_files)
from src.utils import get_logger, list_eligible_files
from scripts.maxsim_test import (build_test_set, encode_repo_pool, K_VALUES,
                                  recall_at_k, reciprocal_rank,
                                  sanity_pool_matches_baseline)

log = get_logger("multigranular")

DECISION_THRESHOLDS = {"strong": 0.06, "modest_low": 0.02, "hurts": -0.02}
METHOD_NAMES = ["pooled_chunk", "pooled_file", "func_pool", "maxsim",
                "mg_sum", "mg_max", "mg_routed"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mistralai/Mamba-Codestral-7B-v0.1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-test-instances", type=int, default=80)
    ap.add_argument("--restrict-repos", nargs="*", default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--g2-mode", default="sliding_window",
                    choices=["sliding_window", "tree_sitter"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/results/multigranular_test.json")
    ap.add_argument("--budget-hours", type=float, default=5.0)
    ap.add_argument("--strict-sanity", action="store_true")
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
        log.error("test set empty")
        return 1

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in test:
        by_repo[t["repo"]].append(t)

    baseline_meta = json.loads(codestral_path.read_text())
    baseline_retriever = baseline_meta.get("retriever", "")
    auto_strict = Path(args.model_id).name in baseline_retriever
    strict = args.strict_sanity or auto_strict
    log.info("sanity mode: %s (auto_strict=%s)",
             "STRICT" if strict else "informational", auto_strict)

    log.info("Loading encoder %s", args.model_id)
    encoder = MambaEncoder(model_id=args.model_id, dtype=dtype, max_length=1532)
    device = str(encoder.device)

    all_instances = {i.instance_id: i for i in load_instances()}

    per_instance_results: list[dict] = []
    sanity_failed = 0
    sanity_checked = 0
    g2_segments_per_file: list[int] = []
    routing_dist: Counter = Counter()

    for repo_slug, insts in by_repo.items():
        if time.time() - t_start > budget_s:
            log.warning("Budget cap hit"); break

        sample = all_instances[insts[0]["instance_id"]]
        repo_path = ensure_repo_at(repo_slug, sample.base_commit)
        files = list_eligible_files(repo_path)
        pool = encode_repo_pool(encoder, repo_path, files)

        # Derive multi-granularity representations from the same cached pool
        log.info("[%s] building G0/G1/G2/G3 from %d cached chunks",
                 repo_slug, len(pool.chunk_files))
        reprs = build_file_representations(pool.chunk_files, pool.tokens,
                                            g2_mode=args.g2_mode)
        log.info("[%s] %d unique files; G2 segs per file: mean=%.1f min=%d max=%d",
                 repo_slug, len(reprs),
                 sum(len(r.g2_segments) for r in reprs) / max(len(reprs), 1),
                 min(len(r.g2_segments) for r in reprs),
                 max(len(r.g2_segments) for r in reprs))
        g2_segments_per_file.extend(len(r.g2_segments) for r in reprs)

        # Build a mapping from file path → cached pool top-20 (for sanity)
        for t in tqdm(insts, desc=repo_slug, leave=False):
            if time.time() - t_start > budget_s:
                log.warning("Budget cap hit mid-instance"); break

            inst = all_instances[t["instance_id"]]
            gold = inst.gold_files
            gold_set = set(gold)

            out = encoder.encode([inst.problem_statement])
            mask = out.attention_mask[0].bool()
            q_tok = out.last_hidden[0][mask].to(device)

            method_results: dict[str, dict] = {}
            with torch.no_grad():
                # Score each method
                for name in METHOD_NAMES:
                    t0 = time.time()
                    routed_to = None
                    if name == "pooled_chunk":
                        scores = score_pooled_chunk(q_tok, reprs)
                    elif name == "pooled_file":
                        scores = score_pooled_file(q_tok, reprs)
                    elif name == "func_pool":
                        scores = score_func_pool(q_tok, reprs)
                    elif name == "maxsim":
                        scores = score_maxsim(q_tok, reprs)
                    elif name == "mg_sum":
                        scores = score_mg_sum(q_tok, reprs)
                    elif name == "mg_max":
                        scores = score_mg_max(q_tok, reprs)
                    elif name == "mg_routed":
                        scores, routed_to = score_mg_routed(
                            q_tok, inst.problem_statement, reprs)
                        routing_dist[routed_to] += 1
                    dt = time.time() - t0
                    ranked = topk_files(scores, reprs, top_k=args.top_k)
                    paths = [p for p, _ in ranked]
                    rec = {f"recall@{k}": recall_at_k(gold_set, paths, k)
                           for k in K_VALUES}
                    method_results[name] = {
                        "ranked_top20": paths,
                        **rec,
                        "mrr": reciprocal_rank(gold_set, paths),
                        "latency_s": dt,
                    }
                    if routed_to:
                        method_results[name]["routed_to"] = routed_to

            # Sanity check vs cached pooled baseline
            sanity_checked += 1
            instance_sanity_ok = sanity_pool_matches_baseline(
                method_results["pooled_chunk"]["ranked_top20"][:10],
                t["codestral_top20"])
            if not instance_sanity_ok:
                sanity_failed += 1
                log.warning("sanity miss on %s", t["instance_id"])

            per_instance_results.append({
                "instance_id": t["instance_id"],
                "repo": repo_slug,
                "gold_files": gold,
                "sanity_ok": instance_sanity_ok,
                "methods": method_results,
            })

            if strict and sanity_checked >= 5:
                pass_rate = 1 - sanity_failed / sanity_checked
                if pass_rate < 0.5:
                    log.error("STRICT SANITY ABORT: %d/%d diverge",
                              sanity_failed, sanity_checked)
                    return 2

        # Free repo pool before next repo
        del pool, reprs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    def _mean(rows, method, key):
        vals = [r["methods"][method][key] for r in rows
                if method in r["methods"]]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {"n": len(per_instance_results), "methods": {}}
    for name in METHOD_NAMES:
        m = {f"recall@{k}": _mean(per_instance_results, name, f"recall@{k}")
             for k in K_VALUES}
        m["mrr"] = _mean(per_instance_results, name, "mrr")
        lats = [r["methods"][name]["latency_s"] for r in per_instance_results
                if name in r["methods"]]
        m["median_latency_s"] = float(sorted(lats)[len(lats) // 2]) if lats else 0.0
        summary["methods"][name] = m

    summary["sanity_pass_rate"] = (1 - sanity_failed / sanity_checked
                                    if sanity_checked else None)
    summary["g2_segments_per_file"] = {
        "mean": sum(g2_segments_per_file) / max(len(g2_segments_per_file), 1),
        "min": min(g2_segments_per_file) if g2_segments_per_file else 0,
        "max": max(g2_segments_per_file) if g2_segments_per_file else 0,
        "n_files": len(g2_segments_per_file),
        "n_zero_segs": sum(1 for x in g2_segments_per_file if x == 0),
    }
    summary["mg_routed_distribution"] = dict(routing_dist)

    # Rescues / regressions vs maxsim (G3-only)
    rescues = {n: 0 for n in METHOD_NAMES if n != "maxsim"}
    regressions = {n: 0 for n in METHOD_NAMES if n != "maxsim"}
    for r in per_instance_results:
        gold = r["gold_files"][0] if r["gold_files"] else None
        if not gold:
            continue
        ms_top10 = r["methods"]["maxsim"]["ranked_top20"][:10]
        in_ms = gold in ms_top10
        for name in METHOD_NAMES:
            if name == "maxsim":
                continue
            this_top10 = r["methods"][name]["ranked_top20"][:10]
            in_this = gold in this_top10
            if in_this and not in_ms:
                rescues[name] += 1
            if in_ms and not in_this:
                regressions[name] += 1
    summary["rescues_vs_maxsim"] = rescues
    summary["regressions_vs_maxsim"] = regressions

    # Verdict
    maxsim_r10 = summary["methods"]["maxsim"]["recall@10"]
    mg_methods = ["mg_sum", "mg_max", "mg_routed"]
    best_mg_r10 = max(summary["methods"][n]["recall@10"] for n in mg_methods)
    delta = best_mg_r10 - maxsim_r10
    if delta >= DECISION_THRESHOLDS["strong"]:
        verdict = "STRONG_MULTIGRAN"
    elif delta >= DECISION_THRESHOLDS["modest_low"]:
        verdict = "MODEST_MULTIGRAN"
    elif delta >= DECISION_THRESHOLDS["hurts"]:
        verdict = "NO_LIFT"
    else:
        verdict = "HURTS"
    summary["maxsim_recall@10"] = maxsim_r10
    summary["best_mg_recall@10"] = best_mg_r10
    summary["best_mg_minus_maxsim"] = delta
    summary["verdict"] = verdict
    summary["decision_thresholds"] = DECISION_THRESHOLDS

    report = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "g2_mode": args.g2_mode,
        "n_test_instances": len(per_instance_results),
        "summary": summary,
        "per_instance": per_instance_results,
        "wall_seconds": time.time() - t_start,
        "interpretation_notes": [
            ("Per-query min-max normalization makes mg_sum effectively a "
             "4-way ensemble vote rather than a magnitude-calibrated "
             "combination. If mg_max wins but mg_sum does not, the "
             "interpretation is 'one granularity matters per query' "
             "rather than 'combining granularities helps.'"),
            ("G2 is sliding-window pool (window=256, stride=128 within each "
             "chunk), not tree-sitter function-level. The experiment tests "
             "whether mid-granularity in SOME form helps. A NO_LIFT result "
             "on sliding-window does not fully rule out semantic-boundary "
             "(tree-sitter) granularity. A STRONG_MULTIGRAN result would be "
             "sufficient to claim mid-granularity helps."),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Print
    print()
    print("=" * 100)
    print("MULTI-GRANULARITY TEST")
    print(f"model: {args.model_id}  dtype: {args.dtype}  g2_mode: {args.g2_mode}")
    print(f"test set: {len(per_instance_results)} instances")
    print("-" * 100)
    head = "{:<14}".format("metric") + "".join(f"{n:>13}" for n in METHOD_NAMES)
    print(head)
    for k in K_VALUES:
        row = "{:<14}".format(f"Recall@{k}")
        for n in METHOD_NAMES:
            row += f"{summary['methods'][n][f'recall@{k}']:>13.4f}"
        print(row)
    row = "{:<14}".format("MRR")
    for n in METHOD_NAMES:
        row += f"{summary['methods'][n]['mrr']:>13.4f}"
    print(row)
    row = "{:<14}".format("med_lat (s)")
    for n in METHOD_NAMES:
        row += f"{summary['methods'][n]['median_latency_s']:>13.3f}"
    print(row)
    print("-" * 100)
    print(f"G2 segments per file: mean={summary['g2_segments_per_file']['mean']:.1f}  "
          f"min={summary['g2_segments_per_file']['min']}  "
          f"max={summary['g2_segments_per_file']['max']}  "
          f"zero-seg files={summary['g2_segments_per_file']['n_zero_segs']}")
    print(f"mg_routed distribution: {summary['mg_routed_distribution']}")
    print(f"Rescues vs MaxSim: {rescues}")
    print(f"Regressions vs MaxSim: {regressions}")
    print(f"sanity_pass_rate: {summary['sanity_pass_rate']:.3f}" if summary['sanity_pass_rate'] is not None else "sanity_pass_rate: N/A")
    print("-" * 100)
    print(f"Verdict: {verdict}  (best_mg={best_mg_r10:.4f}, MaxSim={maxsim_r10:.4f}, Δ={delta:+.4f})")
    print(f"  STRONG_MULTIGRAN  Δ ≥ {DECISION_THRESHOLDS['strong']:.2f}")
    print(f"  MODEST_MULTIGRAN  Δ ≥ {DECISION_THRESHOLDS['modest_low']:.2f}")
    print(f"  NO_LIFT           Δ ≥ {DECISION_THRESHOLDS['hurts']:.2f}")
    print(f"  HURTS             Δ < {DECISION_THRESHOLDS['hurts']:.2f}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
