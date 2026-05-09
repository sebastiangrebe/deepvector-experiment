"""Phase 3.1: nested Mamba encoding test (django-only).

Four matching methods on the django subset of the discriminating set
(~30 of 80 instances):

  L0_maxsim   — token-level MaxSim. Phase 2.5 baseline reproduced.
  L1_funcsim  — query mean-pool vs each function-level vector;
                file score = max over its functions.
  L2_filesim  — query mean-pool vs file-level vector (cosine).
  Lcomposite  — per-query, per-file max across normalized L0 / L1 / L2.

Decision criteria (pre-registered):
  STRONG_HIERARCHICAL  best of {L1, L2, comp} ≥ 0.50  → real signal
  MODEST_HIERARCHICAL  best ≥ L0 + 0.012             → small lift
  NO_LIFT              best within ±0.02 of L0
  HURTS                best < L0 - 0.02

Usage:
  python scripts/nested_test.py \\
      --model-id mistralai/Mamba-Codestral-7B-v0.1 \\
      --dtype bfloat16 --top-k 20 --budget-hours 3 \\
      --out data/results/nested_test.json
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.eval import ensure_repo_at, load_instances
from src.frozen_methods import maxsim_score as maxsim_streaming
from src.maxsim import dedup_to_files
from src.nested_encoding import build_nested_index
from src.utils import get_logger, list_eligible_files
from scripts.maxsim_test import (build_test_set, encode_repo_pool, K_VALUES,
                                  recall_at_k, reciprocal_rank,
                                  sanity_pool_matches_baseline)

log = get_logger("nested_test")

DECISION_THRESHOLDS = {"strong": 0.50, "modest_delta": 0.012, "hurts": -0.02}


def _minmax(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    rng = (hi - lo).clamp_min(1e-8)
    return (x - lo) / rng if hi > lo else torch.full_like(x, 0.5)


def _l0_maxsim_per_file(q_tok: torch.Tensor, pool, top_k: int = 20):
    """Run streaming MaxSim against all chunk tokens, dedup chunk → file,
    return (file_paths, scores_dict file→best score). Score dict covers all
    files seen in the chunk pool."""
    scores = maxsim_streaming(q_tok, pool.tokens)
    n = scores.shape[0]
    file_best: dict[str, float] = {}
    order = torch.argsort(scores, descending=True).tolist()
    for row in order:
        f = pool.chunk_files[row]
        s = float(scores[row].item())
        if f not in file_best or s > file_best[f]:
            file_best[f] = s
    return file_best


def _l1_funcsim_per_file(q_pool: torch.Tensor,
                         nested_idx,
                         device: torch.device) -> dict[str, float]:
    """For each file, max cosine of query pool against the file's L1 vectors."""
    fvecs = nested_idx.func_vectors.to(device, dtype=q_pool.dtype)
    fvecs = F.normalize(fvecs, dim=-1)
    qn = F.normalize(q_pool, dim=-1)
    if qn.dim() == 1:
        qn = qn.unsqueeze(0)
    sims = (qn @ fvecs.T).squeeze(0)   # (n_funcs,)
    out: dict[str, float] = {}
    for rel, idxs in nested_idx.file_to_func_idx.items():
        if not idxs:
            continue
        s = float(sims[torch.tensor(idxs, dtype=torch.long)].max().item())
        out[rel] = s
    return out


def _l2_filesim_per_file(q_pool: torch.Tensor,
                         nested_idx,
                         device: torch.device) -> dict[str, float]:
    if not nested_idx.file_vectors:
        return {}
    rels = list(nested_idx.file_vectors.keys())
    M = torch.stack([nested_idx.file_vectors[r] for r in rels]).to(
        device, dtype=q_pool.dtype)
    M = F.normalize(M, dim=-1)
    qn = F.normalize(q_pool.unsqueeze(0) if q_pool.dim() == 1 else q_pool, dim=-1)
    sims = (qn @ M.T).squeeze(0)
    return {r: float(sims[i].item()) for i, r in enumerate(rels)}


def _topk_from_dict(scores: dict[str, float], k: int) -> list[str]:
    return [r for r, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mistralai/Mamba-Codestral-7B-v0.1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-test-instances", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-funcs-per-file", type=int, default=200)
    ap.add_argument("--budget-hours", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    t_start = time.time()
    budget_s = args.budget_hours * 3600

    voyage_path = ROOT / "data/results/voyage_baseline.json"
    codestral_path = ROOT / "data/results/mamba_codestral_baseline.json"
    test = build_test_set(voyage_path, codestral_path,
                          max_n=args.max_test_instances, seed=args.seed,
                          restrict_repos=["django/django"])
    if not test:
        log.error("test set empty"); return 1

    log.info("django-restricted test set: %d instances", len(test))
    all_instances = {i.instance_id: i for i in load_instances()}

    log.info("Loading encoder %s", args.model_id)
    encoder = MambaEncoder(model_id=args.model_id, dtype=dtype, max_length=1532)
    device = encoder.device

    # Build django repo path + cached pool
    sample = all_instances[test[0]["instance_id"]]
    repo_path = ensure_repo_at("django/django", sample.base_commit)
    files = list_eligible_files(repo_path)
    pool = encode_repo_pool(encoder, repo_path, files)

    # Build nested index (L1 + L2 forward passes)
    nested = build_nested_index(encoder, repo_path,
                                 max_funcs_per_file=args.max_funcs_per_file)

    # Sanity (b): L2 path produced finite vectors
    if nested.file_vectors:
        sample_path = next(iter(nested.file_vectors.keys()))
        sv = nested.file_vectors[sample_path]
        log.info("L2 sanity: %s shape=%s mean_abs=%.4f std=%.4f isnan=%s",
                 sample_path, tuple(sv.shape), float(sv.abs().mean()),
                 float(sv.std()), bool(torch.isnan(sv).any()))
        if torch.isnan(sv).any() or sv.abs().mean() < 1e-6:
            log.error("L2 vectors look degenerate; aborting")
            return 2

    # Function count stats
    counts = [len(v) for v in nested.file_to_func_idx.values()]
    log.info("Functions per file: median=%d mean=%.1f min=%d max=%d (n_files=%d)",
             int(median(counts)) if counts else 0,
             sum(counts) / len(counts) if counts else 0,
             min(counts) if counts else 0, max(counts) if counts else 0,
             len(counts))

    per_instance: list[dict] = []
    sanity_checked = 0
    sanity_failed = 0

    for t in tqdm(test, desc="instances"):
        if time.time() - t_start > budget_s:
            log.warning("budget cap"); break
        inst = all_instances[t["instance_id"]]
        gold_set = set(inst.gold_files)
        gold_path = inst.gold_files[0] if inst.gold_files else None

        with torch.no_grad():
            out = encoder.encode([inst.problem_statement])
        mask = out.attention_mask[0].bool()
        q_tok = out.last_hidden[0][mask].to(device)
        q_pool = q_tok.mean(dim=0)

        # L0: streaming MaxSim, then dedup chunk → file
        l0_scores_dict: dict[str, float] = _l0_maxsim_per_file(q_tok, pool,
                                                                top_k=args.top_k)
        l0_top = _topk_from_dict(l0_scores_dict, args.top_k)

        # L1
        l1_scores_dict = _l1_funcsim_per_file(q_pool, nested, device)
        l1_top = _topk_from_dict(l1_scores_dict, args.top_k)

        # L2
        l2_scores_dict = _l2_filesim_per_file(q_pool, nested, device)
        l2_top = _topk_from_dict(l2_scores_dict, args.top_k)

        # Composite: union over files appearing in any level; min-max within
        # this query's score distribution at each level; per-file max across.
        all_files = set(l0_scores_dict) | set(l1_scores_dict) | set(l2_scores_dict)
        ordered = sorted(all_files)

        def _norm_dict(d: dict[str, float]) -> dict[str, float]:
            if not d: return {}
            arr = torch.tensor([d.get(r, float("nan")) for r in ordered])
            mask = ~torch.isnan(arr)
            if mask.sum() < 2:
                return {r: 0.5 for r in ordered}
            vals = arr[mask]
            lo, hi = vals.min(), vals.max()
            rng = (hi - lo).clamp_min(1e-8)
            out = {}
            for i, r in enumerate(ordered):
                if torch.isnan(arr[i]):
                    out[r] = 0.0
                else:
                    out[r] = float((arr[i] - lo) / rng) if hi > lo else 0.5
            return out

        l0_n = _norm_dict(l0_scores_dict)
        l1_n = _norm_dict(l1_scores_dict)
        l2_n = _norm_dict(l2_scores_dict)
        comp = {r: max(l0_n.get(r, 0.0), l1_n.get(r, 0.0), l2_n.get(r, 0.0))
                for r in ordered}
        comp_top = _topk_from_dict(comp, args.top_k)

        # Sanity (c): L0 ranking should overlap cached Codestral pooled
        # baseline by ≥50% on the first 10 files.
        sanity_checked += 1
        if not sanity_pool_matches_baseline(l0_top[:10], t["codestral_top20"]):
            sanity_failed += 1

        per_instance.append({
            "id": t["instance_id"],
            "gold_files": list(gold_set),
            "l0_top20": l0_top,
            "l1_top20": l1_top,
            "l2_top20": l2_top,
            "comp_top20": comp_top,
            "l0_recall@10":   recall_at_k(gold_set, l0_top, 10),
            "l1_recall@10":   recall_at_k(gold_set, l1_top, 10),
            "l2_recall@10":   recall_at_k(gold_set, l2_top, 10),
            "comp_recall@10": recall_at_k(gold_set, comp_top, 10),
            "l0_recall@20":   recall_at_k(gold_set, l0_top, 20),
            "l1_recall@20":   recall_at_k(gold_set, l1_top, 20),
            "l2_recall@20":   recall_at_k(gold_set, l2_top, 20),
            "comp_recall@20": recall_at_k(gold_set, comp_top, 20),
            "l0_mrr":   reciprocal_rank(gold_set, l0_top),
            "l1_mrr":   reciprocal_rank(gold_set, l1_top),
            "l2_mrr":   reciprocal_rank(gold_set, l2_top),
            "comp_mrr": reciprocal_rank(gold_set, comp_top),
        })

    # Aggregate
    def _mean(rows, key):
        vals = [r[key] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n": len(per_instance),
        "l0_recall@10": _mean(per_instance, "l0_recall@10"),
        "l1_recall@10": _mean(per_instance, "l1_recall@10"),
        "l2_recall@10": _mean(per_instance, "l2_recall@10"),
        "comp_recall@10": _mean(per_instance, "comp_recall@10"),
        "l0_recall@20": _mean(per_instance, "l0_recall@20"),
        "l1_recall@20": _mean(per_instance, "l1_recall@20"),
        "l2_recall@20": _mean(per_instance, "l2_recall@20"),
        "comp_recall@20": _mean(per_instance, "comp_recall@20"),
        "l0_mrr": _mean(per_instance, "l0_mrr"),
        "l1_mrr": _mean(per_instance, "l1_mrr"),
        "l2_mrr": _mean(per_instance, "l2_mrr"),
        "comp_mrr": _mean(per_instance, "comp_mrr"),
        "sanity_pass_rate_l0_vs_cached_baseline": (
            1 - sanity_failed / sanity_checked) if sanity_checked else None,
        "n_total_functions": len(nested.func_spans),
        "functions_per_file_median": int(median(counts)) if counts else 0,
        "functions_per_file_mean": (sum(counts) / len(counts)) if counts else 0,
    }

    # Verdict
    l0 = summary["l0_recall@10"]
    best = max(summary["l1_recall@10"], summary["l2_recall@10"], summary["comp_recall@10"])
    delta = best - l0
    if best >= DECISION_THRESHOLDS["strong"]:
        verdict = "STRONG_HIERARCHICAL"
    elif delta >= DECISION_THRESHOLDS["modest_delta"]:
        verdict = "MODEST_HIERARCHICAL"
    elif delta >= DECISION_THRESHOLDS["hurts"]:
        verdict = "NO_LIFT"
    else:
        verdict = "HURTS"
    summary["verdict"] = verdict
    summary["best_nested_minus_l0"] = delta
    summary["decision_thresholds"] = DECISION_THRESHOLDS

    report = {
        "phase": "3.1_nested",
        "config": {"model_id": args.model_id, "dtype": args.dtype,
                    "django_only": True, "n_test": len(per_instance)},
        "summary": summary,
        "per_instance": per_instance,
        "wall_seconds": time.time() - t_start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 90)
    print(f"NESTED ENCODING TEST  (django-only, n={len(per_instance)})")
    print(f"model: {args.model_id}  dtype: {args.dtype}")
    print("-" * 90)
    print(f"{'metric':<22}{'L0_maxsim':>12}{'L1_funcsim':>12}{'L2_filesim':>12}{'Lcomposite':>12}")
    for k in (10, 20):
        print(f"{'Recall@'+str(k):<22}"
              f"{summary[f'l0_recall@{k}']:>12.4f}"
              f"{summary[f'l1_recall@{k}']:>12.4f}"
              f"{summary[f'l2_recall@{k}']:>12.4f}"
              f"{summary[f'comp_recall@{k}']:>12.4f}")
    print(f"{'MRR':<22}"
          f"{summary['l0_mrr']:>12.4f}{summary['l1_mrr']:>12.4f}"
          f"{summary['l2_mrr']:>12.4f}{summary['comp_mrr']:>12.4f}")
    print("-" * 90)
    print(f"sanity (L0 vs cached pooled baseline): "
          f"{summary['sanity_pass_rate_l0_vs_cached_baseline']}")
    print(f"functions: total={summary['n_total_functions']}  "
          f"median/file={summary['functions_per_file_median']}  "
          f"mean/file={summary['functions_per_file_mean']:.1f}")
    print("-" * 90)
    print(f"Verdict: {verdict}  (best={best:.4f}, L0={l0:.4f}, Δ={delta:+.4f})")
    print(f"  STRONG_HIERARCHICAL   best ≥ {DECISION_THRESHOLDS['strong']:.2f}")
    print(f"  MODEST_HIERARCHICAL   Δ ≥ {DECISION_THRESHOLDS['modest_delta']:.3f}")
    print(f"  NO_LIFT               Δ within ±{abs(DECISION_THRESHOLDS['hurts']):.2f}")
    print(f"  HURTS                 Δ < {DECISION_THRESHOLDS['hurts']:.2f}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
