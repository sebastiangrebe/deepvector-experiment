"""SWE-Bench retrieval evaluation harness.

Defines a `Retriever` protocol; runs any retriever over SWE-Bench Lite and
reports Recall@K, MRR, latency, and per-repo breakdowns.

CLI:
    python -m src.eval --retriever voyage --top-k 20
    python -m src.eval --retriever voyage --dry-run
"""
from __future__ import annotations

import os
# Avoid duplicate libomp clash between torch and faiss on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Protocol

from tqdm import tqdm

from src.utils import (
    filter_stats,
    get_logger,
    list_eligible_files,
    project_root,
)

log = get_logger("eval")
ROOT = project_root()
DATASET_PATH = ROOT / "data" / "swe_bench_lite.json"
REPOS_DIR = ROOT / "data" / "repos"
RESULTS_DIR = ROOT / "data" / "results"


# ─────────────────────────────────────────────────────────────────────
# Retriever interface
# ─────────────────────────────────────────────────────────────────────

class Retriever(Protocol):
    name: str

    def index(self, repo_path: Path, files: list[Path]) -> None: ...
    def search(self, query: str, top_k: int) -> list[tuple[Path, float]]: ...


# ─────────────────────────────────────────────────────────────────────
# Patch parsing — extract gold files from `patch` field
# ─────────────────────────────────────────────────────────────────────

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$", re.MULTILINE)
_PLUS_HEADER = re.compile(r"^\+\+\+ b/(?P<path>.+)$", re.MULTILINE)


def gold_files_from_patch(patch: str) -> list[str]:
    """Return repo-relative POSIX paths modified by `patch`. Skip /dev/null."""
    if not patch:
        return []
    seen: list[str] = []
    # Prefer `+++ b/<path>` (post-image). Fall back to `diff --git`.
    for m in _PLUS_HEADER.finditer(patch):
        path = m.group("path").strip()
        if path and path != "/dev/null" and path not in seen:
            seen.append(path)
    if not seen:
        for m in _DIFF_HEADER.finditer(patch):
            path = m.group("b").strip()
            if path and path not in seen:
                seen.append(path)
    return seen


# ─────────────────────────────────────────────────────────────────────
# Repo management
# ─────────────────────────────────────────────────────────────────────

def repo_dir(repo_slug: str) -> Path:
    return REPOS_DIR / repo_slug.replace("/", "__")


def ensure_repo_at(repo_slug: str, base_commit: str) -> Path:
    target = repo_dir(repo_slug)
    if not target.exists():
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo_slug}.git"
        log.info("Cloning %s", url)
        subprocess.check_call(["git", "clone", "--quiet", url, str(target)])
    subprocess.check_call(["git", "-C", str(target), "fetch", "--all", "--quiet"])
    subprocess.check_call(["git", "-C", str(target), "checkout", "--quiet", base_commit])
    return target


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_files: list[str]
    raw: dict = field(repr=False)

    @classmethod
    def from_row(cls, row: dict) -> "Instance":
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row.get("problem_statement", "") or row.get("issue", ""),
            gold_files=gold_files_from_patch(row.get("patch", "")),
            raw=row,
        )


def load_instances(split: str | None = None) -> list[Instance]:
    rows = json.loads(DATASET_PATH.read_text())
    if split:
        rows = [r for r in rows if r.get("_split") == split]
    return [Instance.from_row(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

K_VALUES = (1, 5, 10, 20)


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = ranked[:k]
    hit = sum(1 for g in gold if g in top)
    return hit / len(gold)


def reciprocal_rank(gold: set[str], ranked: list[str]) -> float:
    for i, p in enumerate(ranked, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# Per-instance result
# ─────────────────────────────────────────────────────────────────────

@dataclass
class InstanceResult:
    instance_id: str
    repo: str
    n_gold: int
    n_indexed: int
    gold_files: list[str]
    ranked_top20: list[str]
    recall: dict[int, float]
    mrr: float
    latency_ms: float
    gold_in_index: bool = False  # any gold file present in indexed corpus
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────────────────────────────

def run_eval(
    retriever: Retriever,
    instances: list[Instance],
    top_k: int = 20,
    incremental_save: Path | None = None,
) -> dict:
    log.info("Eval: %d instances, retriever=%s, top_k=%d",
             len(instances), retriever.name, top_k)

    # Index per-repo at the FIRST observed base_commit. Trade-off: file content
    # drift across commits within same repo is ignored. Acceptable for
    # SWE-Bench Lite where commits cluster within months and core files are
    # stable. Cuts indexing cost ~30x vs per-scope.
    by_repo: dict[str, list[Instance]] = defaultdict(list)
    first_commit: dict[str, str] = {}
    for inst in instances:
        by_repo[inst.repo].append(inst)
        first_commit.setdefault(inst.repo, inst.base_commit)

    log.info("Indexing per-repo: %d repos (representative commits)",
             len(by_repo))

    results: list[InstanceResult] = []
    pbar = tqdm(total=len(instances), desc="instances")

    for repo_slug, insts in by_repo.items():
        base_commit = first_commit[repo_slug]
        try:
            repo_path = ensure_repo_at(repo_slug, base_commit)
        except subprocess.CalledProcessError as e:
            log.error("Failed to checkout %s @ %s: %s", repo_slug, base_commit, e)
            for inst in insts:
                results.append(_empty_result(inst, note=f"checkout_fail: {e}"))
                pbar.update(1)
            continue

        files = list_eligible_files(repo_path)
        retriever.index(repo_path, files)
        n_indexed = len(files)
        indexed_set = {f.relative_to(repo_path).as_posix() for f in files}

        for inst in insts:
            t0 = time.time()
            ranked = retriever.search(inst.problem_statement, top_k=top_k)
            dt_ms = (time.time() - t0) * 1000.0
            ranked_paths = [p.relative_to(repo_path).as_posix()
                            for p, _ in ranked]
            gold = set(inst.gold_files)
            rec = {k: recall_at_k(gold, ranked_paths, k) for k in K_VALUES}
            mrr = reciprocal_rank(gold, ranked_paths)
            gold_in = any(g in indexed_set for g in gold)
            results.append(InstanceResult(
                instance_id=inst.instance_id,
                repo=inst.repo,
                n_gold=len(gold),
                n_indexed=n_indexed,
                gold_files=list(gold),
                ranked_top20=ranked_paths[:20],
                recall=rec,
                mrr=mrr,
                latency_ms=dt_ms,
                gold_in_index=gold_in,
            ))
            pbar.update(1)

        if incremental_save:
            _write_results(retriever.name, results, incremental_save)

    pbar.close()

    summary = summarize(retriever.name, results)
    return {"retriever": retriever.name, "summary": summary,
            "results": [_result_to_dict(r) for r in results]}


def _empty_result(inst: Instance, note: str) -> InstanceResult:
    return InstanceResult(
        instance_id=inst.instance_id, repo=inst.repo,
        n_gold=len(inst.gold_files), n_indexed=0,
        gold_files=inst.gold_files, ranked_top20=[],
        recall={k: 0.0 for k in K_VALUES}, mrr=0.0, latency_ms=0.0,
        notes=note,
    )


def _result_to_dict(r: InstanceResult) -> dict:
    d = asdict(r)
    d["recall"] = {str(k): v for k, v in r.recall.items()}
    return d


def _write_results(name: str, results: list[InstanceResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"retriever": name,
               "n_results": len(results),
               "results": [_result_to_dict(r) for r in results]}
    path.write_text(json.dumps(payload, indent=2))


def _agg(results: list[InstanceResult]) -> dict:
    if not results:
        return {"n": 0}
    means = {f"recall@{k}": statistics.mean(r.recall[k] for r in results)
             for k in K_VALUES}
    mrr = statistics.mean(r.mrr for r in results)
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    median_lat = statistics.median(latencies) if latencies else 0.0
    return {"n": len(results), **means, "mrr": mrr,
            "median_latency_ms": median_lat}


def summarize(name: str, results: list[InstanceResult]) -> dict:
    if not results:
        return {"retriever": name, "n": 0}

    raw = _agg(results)
    indexable_results = [r for r in results if r.gold_in_index]
    indexable = _agg(indexable_results)

    by_repo: dict[str, list[InstanceResult]] = defaultdict(list)
    for r in results:
        by_repo[r.repo].append(r)
    repo_breakdown = {}
    for repo, rs in by_repo.items():
        ix = [r for r in rs if r.gold_in_index]
        repo_breakdown[repo] = {
            "n_raw": len(rs),
            "n_indexable": len(ix),
            "raw_recall@10": statistics.mean(r.recall[10] for r in rs),
            "indexable_recall@10": (statistics.mean(r.recall[10] for r in ix)
                                    if ix else None),
        }
    hardest = min(repo_breakdown.items(), key=lambda x: x[1]["raw_recall@10"])
    easiest = max(repo_breakdown.items(), key=lambda x: x[1]["raw_recall@10"])

    return {
        "retriever": name,
        "n_total": len(results),
        "n_indexable": len(indexable_results),
        "raw": raw,
        "indexable": indexable,
        "hardest_repo": {"name": hardest[0],
                         "raw_recall@10": hardest[1]["raw_recall@10"]},
        "easiest_repo": {"name": easiest[0],
                         "raw_recall@10": easiest[1]["raw_recall@10"]},
        "per_repo": repo_breakdown,
    }


def print_summary(summary: dict) -> None:
    raw = summary["raw"]
    ix = summary["indexable"]
    n_total = summary["n_total"]
    n_ix = summary["n_indexable"]

    print()
    print("=" * 78)
    print(f"Method:           {summary['retriever']}")
    print(f"Instances:        n_total={n_total}  n_indexable={n_ix} "
          f"({100*n_ix/n_total:.1f}%)")
    print("-" * 78)
    print(f"{'metric':<22}{'raw (all 323)':>18}{'indexable':>18}")
    print("-" * 78)
    for k in K_VALUES:
        rv = raw.get(f"recall@{k}", 0.0)
        iv = ix.get(f"recall@{k}", 0.0)
        print(f"{'Recall@'+str(k):<22}{rv:>18.4f}{iv:>18.4f}")
    print(f"{'MRR':<22}{raw['mrr']:>18.4f}{ix['mrr']:>18.4f}")
    print(f"{'Median latency (ms)':<22}{raw['median_latency_ms']:>18.1f}"
          f"{ix['median_latency_ms']:>18.1f}")
    print("-" * 78)
    h = summary["hardest_repo"]
    e = summary["easiest_repo"]
    print(f"Hardest repo:     {h['name']} (raw Recall@10: {h['raw_recall@10']:.4f})")
    print(f"Easiest repo:     {e['name']} (raw Recall@10: {e['raw_recall@10']:.4f})")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _build_retriever(name: str, **kwargs):
    if name == "voyage":
        from src.baselines.voyage import VoyageRetriever
        return VoyageRetriever()
    if name == "mamba_pooled":
        import torch
        from src.baselines.mamba_pooled import MambaPooledRetriever
        model_id = kwargs.get("model_id") or "state-spaces/mamba-130m-hf"
        dtype_str = kwargs.get("dtype") or "float32"
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[dtype_str]
        return MambaPooledRetriever(model_id=model_id, dtype=dtype)
    raise ValueError(f"unknown retriever: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True,
                    choices=["voyage", "mamba_pooled"])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--model-id", default=None,
                    help="HF model id (mamba_pooled only)")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--split", default=None, help="dev|test (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of instances (for debugging)")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate cost only; no API calls")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    instances = load_instances(split=args.split)
    if args.limit:
        instances = instances[: args.limit]
    log.info("Loaded %d instances", len(instances))

    if args.dry_run:
        if args.retriever == "voyage":
            from src.baselines.voyage import VoyageRetriever
            VoyageRetriever.dry_run(instances)
        return 0

    out_path = args.out or RESULTS_DIR / f"{args.retriever}_baseline.json"
    retriever = _build_retriever(args.retriever,
                                  model_id=args.model_id,
                                  dtype=args.dtype)
    report = run_eval(retriever, instances, top_k=args.top_k,
                      incremental_save=out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", out_path)
    print_summary(report["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
