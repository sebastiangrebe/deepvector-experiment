"""Phase 2.6: frozen architecture ceiling on Mamba latent state.

Runs 7 frozen matching methods (pooled, maxsim, multi-head H=4/8/16/32,
late-interaction H=8) over the same 80 discriminating instances from
Phase 2.5. Reuses cached token-level latents if present, else re-encodes.

Decision criterion (pre-registered):
  best multi-head method beats MaxSim by ≥5 points R@10 → STRONG headroom
  by 2-5 points R@10                                    → MODEST
  ≤2 points or worse                                    → CEILING (MaxSim is the limit)

Usage (cloud H100, latents cached):
  python scripts/frozen_ceiling_test.py \\
      --model-id mistralai/Mamba-Codestral-7B-v0.1 \\
      --dtype bfloat16 \\
      --max-test-instances 80

Local smoke (mamba-130m, sqlfluff only):
  python scripts/frozen_ceiling_test.py \\
      --model-id state-spaces/mamba-130m-hf \\
      --dtype float32 \\
      --max-test-instances 5 \\
      --restrict-repos sqlfluff/sqlfluff \\
      --out data/results/frozen_ceiling_smoke.json
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

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.eval import ensure_repo_at, load_instances
from src.frozen_methods import all_methods, dedup_to_files
from src.utils import get_logger, list_eligible_files
from scripts.maxsim_test import (build_test_set, encode_repo_pool, K_VALUES,
                                  recall_at_k, reciprocal_rank,
                                  sanity_pool_matches_baseline)

log = get_logger("frozen_ceiling")

DECISION_THRESHOLDS = {"strong": 0.05, "modest_low": 0.02}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mistralai/Mamba-Codestral-7B-v0.1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max-test-instances", type=int, default=80)
    ap.add_argument("--restrict-repos", nargs="*", default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-seed", type=int, default=42,
                    help="Seed for the random orthogonal projection matrix")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/results/frozen_ceiling_test.json")
    ap.add_argument("--budget-hours", type=float, default=4.0)
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
        log.error("test set empty, aborting")
        return 1

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in test:
        by_repo[t["repo"]].append(t)

    baseline_meta = json.loads(codestral_path.read_text())
    baseline_retriever = baseline_meta.get("retriever", "")
    model_short = Path(args.model_id).name
    auto_strict = model_short in baseline_retriever
    strict = args.strict_sanity or auto_strict
    log.info("sanity mode: %s (auto_strict=%s, baseline_retriever=%s)",
             "STRICT (hard fail)" if strict else "informational",
             auto_strict, baseline_retriever)

    log.info("Loading %s (%s)", args.model_id, dtype)
    encoder = MambaEncoder(model_id=args.model_id, dtype=dtype, max_length=1532)
    device = str(encoder.device)
    if "cuda" not in device.lower() and "mps" not in device.lower():
        log.warning("Encoder on %s — slow!", device)

    all_instances = {i.instance_id: i for i in load_instances()}
    methods = all_methods()
    log.info("methods: %s", [m.name for m in methods])

    per_instance_results: list[dict] = []
    sanity_failed = 0
    sanity_checked = 0

    for repo_slug, insts in by_repo.items():
        if time.time() - t_start > budget_s:
            log.warning("Budget cap hit, stopping mid-run")
            break

        sample_inst = all_instances[insts[0]["instance_id"]]
        repo_path = ensure_repo_at(repo_slug, sample_inst.base_commit)
        files = list_eligible_files(repo_path)
        pool = encode_repo_pool(encoder, repo_path, files)
        cand_dev = [t.to(device) for t in pool.tokens]

        for t in tqdm(insts, desc=repo_slug, leave=False):
            if time.time() - t_start > budget_s:
                log.warning("Budget cap hit mid-instance, stopping")
                break

            inst = all_instances[t["instance_id"]]
            gold = inst.gold_files
            gold_set = set(gold)

            # Encode query (same path for all methods)
            out = encoder.encode([inst.problem_statement])
            mask = out.attention_mask[0].bool()
            q_tok = out.last_hidden[0][mask].to(device)

            method_results: dict[str, dict] = {}
            for spec in methods:
                t0 = time.time()
                with torch.no_grad():
                    scores = spec.fn(q_tok, cand_dev, **spec.kwargs)
                dt = time.time() - t0
                ranked = dedup_to_files(scores, pool.chunk_files,
                                        top_k=args.top_k)
                paths = [p for p, _ in ranked]
                method_results[spec.name] = {
                    "ranked_top20": paths,
                    **{f"recall@{k}": recall_at_k(gold_set, paths, k) for k in K_VALUES},
                    "mrr": reciprocal_rank(gold_set, paths),
                    "latency_s": dt,
                }

            # Sanity check on Method 0 (pooled) vs cached baseline
            sanity_checked += 1
            instance_sanity_ok = sanity_pool_matches_baseline(
                method_results["pooled"]["ranked_top20"][:10],
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
                    log.error("STRICT SANITY ABORT: %d/%d diverge "
                              "(pass rate %.1f%% < 50%%). Likely harness bug.",
                              sanity_failed, sanity_checked, 100 * pass_rate)
                    return 2

    # Aggregate per-method
    def _mean(rows, method, key):
        vals = [r["methods"][method][key] for r in rows
                if method in r["methods"]]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {"n": len(per_instance_results), "methods": {}}
    for spec in methods:
        m = {f"recall@{k}": _mean(per_instance_results, spec.name, f"recall@{k}")
             for k in K_VALUES}
        m["mrr"] = _mean(per_instance_results, spec.name, "mrr")
        lats = [r["methods"][spec.name]["latency_s"] for r in per_instance_results]
        m["median_latency_s"] = float(np.median(lats)) if lats else 0.0
        summary["methods"][spec.name] = m

    summary["sanity_pass_rate"] = (
        1 - sanity_failed / sanity_checked) if sanity_checked else None

    # Rescue / regression vs MaxSim
    maxsim_top10 = [{r["instance_id"]: r["methods"]["maxsim"]["ranked_top20"][:10]}
                    for r in per_instance_results]
    rescues = {}
    regressions = {}
    for spec in methods:
        if spec.name == "maxsim":
            continue
        rescued = 0
        regressed = 0
        for r in per_instance_results:
            gold = r["gold_files"][0] if r["gold_files"] else None
            if not gold:
                continue
            ms_top10 = r["methods"]["maxsim"]["ranked_top20"][:10]
            this_top10 = r["methods"][spec.name]["ranked_top20"][:10]
            in_ms = gold in ms_top10
            in_this = gold in this_top10
            if in_this and not in_ms:
                rescued += 1
            if in_ms and not in_this:
                regressed += 1
        rescues[spec.name] = rescued
        regressions[spec.name] = regressed
    summary["rescues_vs_maxsim"] = rescues
    summary["regressions_vs_maxsim"] = regressions

    # Best-of-multi-head ensemble
    mh_names = ["mh_4", "mh_8", "mh_16", "mh_32"]
    ensemble_recalls = {f"recall@{k}": 0.0 for k in K_VALUES}
    for r in per_instance_results:
        gold = set(r["gold_files"])
        # Take best rank across MH methods per query
        best = {}  # path → best score
        for n in mh_names:
            for i, p in enumerate(r["methods"][n]["ranked_top20"]):
                # Convert rank to a pseudo-score (higher = better)
                rank_score = -i
                if p not in best or rank_score > best[p]:
                    best[p] = rank_score
        ranked = [p for p, _ in sorted(best.items(), key=lambda x: -x[1])][: max(K_VALUES)]
        for k in K_VALUES:
            ensemble_recalls[f"recall@{k}"] += recall_at_k(gold, ranked, k)
    n = len(per_instance_results) or 1
    for k in K_VALUES:
        ensemble_recalls[f"recall@{k}"] /= n
    summary["mh_ensemble"] = ensemble_recalls

    # Verdict
    maxsim_r10 = summary["methods"]["maxsim"]["recall@10"]
    best_mh = max(summary["methods"][n]["recall@10"] for n in mh_names)
    delta = best_mh - maxsim_r10
    if delta >= DECISION_THRESHOLDS["strong"]:
        verdict = "STRONG_HEADROOM"
    elif delta >= DECISION_THRESHOLDS["modest_low"]:
        verdict = "MODEST_HEADROOM"
    else:
        verdict = "CEILING"
    summary["maxsim_recall@10"] = maxsim_r10
    summary["best_mh_recall@10"] = best_mh
    summary["best_mh_minus_maxsim"] = delta
    summary["verdict"] = verdict
    summary["decision_thresholds"] = DECISION_THRESHOLDS

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

    print()
    print("=" * 95)
    print("FROZEN CEILING TEST")
    print(f"model: {args.model_id}  dtype: {args.dtype}")
    print(f"test set: {len(per_instance_results)} instances "
          f"(Voyage R@10=1 AND Codestral pooled R@10=0)")
    print("-" * 95)
    cols = ["pooled", "maxsim", "mh_4", "mh_8", "mh_16", "mh_32", "late_int_8"]
    head = "{:<14}".format("metric") + "".join(f"{c:>11}" for c in cols)
    print(head)
    for k in K_VALUES:
        row = "{:<14}".format(f"Recall@{k}")
        for n_ in cols:
            row += f"{summary['methods'][n_][f'recall@{k}']:>11.4f}"
        print(row)
    row = "{:<14}".format("MRR")
    for n_ in cols:
        row += f"{summary['methods'][n_]['mrr']:>11.4f}"
    print(row)
    row = "{:<14}".format("med_lat (s)")
    for n_ in cols:
        row += f"{summary['methods'][n_]['median_latency_s']:>11.3f}"
    print(row)
    print("-" * 95)
    print("Rescues vs MaxSim (R@10 hits gained):", rescues)
    print("Regressions vs MaxSim (R@10 hits lost):", regressions)
    print(f"MH-ensemble (best across H ∈ {{4,8,16,32}}) Recall@10: "
          f"{ensemble_recalls['recall@10']:.4f}")
    print("-" * 95)
    print(f"Verdict: {verdict}  (best_MH={best_mh:.4f}, MaxSim={maxsim_r10:.4f}, Δ={delta:+.4f})")
    print(f"  STRONG  Δ ≥ {DECISION_THRESHOLDS['strong']:.2f}: trained Phase 3 has upside")
    print(f"  MODEST  Δ ≥ {DECISION_THRESHOLDS['modest_low']:.2f}: diminishing returns without training")
    print(f"  CEILING Δ < {DECISION_THRESHOLDS['modest_low']:.2f}: MaxSim is the frozen ceiling")
    print("=" * 95)
    return 0


if __name__ == "__main__":
    sys.exit(main())
