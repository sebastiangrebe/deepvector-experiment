"""End-to-end smoke test.

Loads SWE-Bench Lite, picks a small repo, clones it at the issue's base commit,
encodes 10 source files via the Mamba encoder, runs cross-attention between a
fake query and the file latents.

Run from project root:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder, cross_attention_score
from src.utils import get_logger

log = get_logger("smoke_test")

DATASET_PATH = ROOT / "data" / "swe_bench_lite.json"
REPOS_DIR = ROOT / "data" / "repos"
PREFERRED_REPO = "pvlib/pvlib-python"


def pick_instance() -> dict:
    rows = json.loads(DATASET_PATH.read_text())
    log.info("Loaded %d instances", len(rows))
    candidates = [r for r in rows if r["repo"] == PREFERRED_REPO]
    if not candidates:
        # Fall back to smallest-population repo with any instance
        from collections import Counter
        counts = Counter(r["repo"] for r in rows)
        smallest = min(counts, key=counts.get)
        candidates = [r for r in rows if r["repo"] == smallest]
        log.info("Preferred repo not found, using %s", smallest)
    chosen = candidates[0]
    log.info("Chose instance %s (repo=%s, base_commit=%s)",
             chosen["instance_id"], chosen["repo"], chosen["base_commit"][:8])
    return chosen


def ensure_repo(instance: dict) -> Path:
    repo_slug = instance["repo"]
    base_commit = instance["base_commit"]
    target = REPOS_DIR / repo_slug.replace("/", "__")
    if target.exists():
        log.info("Repo already cloned at %s", target)
    else:
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo_slug}.git"
        log.info("Cloning %s -> %s", url, target)
        subprocess.check_call(["git", "clone", url, str(target)])
    log.info("Checking out base_commit %s", base_commit[:8])
    subprocess.check_call(["git", "-C", str(target), "fetch", "--all", "--quiet"])
    subprocess.check_call(["git", "-C", str(target), "checkout", "--quiet", base_commit])
    return target


def list_example_files(repo_path: Path, n: int = 10) -> list[Path]:
    py_files = sorted(repo_path.rglob("*.py"))
    py_files = [p for p in py_files if ".git" not in p.parts]
    py_files = [p for p in py_files if p.stat().st_size > 64]  # skip empty/tiny
    chosen = py_files[:n]
    log.info("Picked %d files (of %d non-trivial .py)", len(chosen), len(py_files))
    for p in chosen:
        log.info("  %s", p.relative_to(repo_path))
    return chosen


def main() -> int:
    print("=" * 70)
    print("DeepVector smoke test")
    print("=" * 70)

    instance = pick_instance()
    repo_path = ensure_repo(instance)
    files = list_example_files(repo_path, n=10)
    if not files:
        log.error("No .py files found in repo")
        return 1

    encoder = MambaEncoder()
    texts = [f.read_text(errors="replace")[:8000] for f in files]
    log.info("Encoding %d files...", len(texts))
    out = encoder.encode(texts)

    print()
    print(f"last_hidden shape : {tuple(out.last_hidden.shape)}  (B, T, D)")
    print(f"pooled shape      : {tuple(out.pooled.shape)}  (B, D)")
    print(f"device            : {out.last_hidden.device}")
    print(f"dtype             : {out.last_hidden.dtype}")
    print(f"on MPS            : {out.last_hidden.is_mps if hasattr(out.last_hidden, 'is_mps') else (out.last_hidden.device.type == 'mps')}")

    fake_query = "Fix bug in solar irradiance calculation when timezone is naive"
    log.info("Encoding fake query: %r", fake_query)
    q_out = encoder.encode([fake_query])
    q_latents = q_out.last_hidden[0]

    doc_latents = []
    masks = out.attention_mask
    for i in range(out.last_hidden.shape[0]):
        # Strip left-pad tokens
        m = masks[i].bool()
        doc_latents.append(out.last_hidden[i][m])

    scores = cross_attention_score(q_latents, doc_latents)
    print()
    print("Cross-attn scores (query -> 10 files):")
    for f, s in zip(files, scores.tolist()):
        print(f"  {s:7.4f}  {f.relative_to(repo_path)}")

    if not torch.isfinite(scores).all():
        log.error("Non-finite scores")
        return 1

    print()
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
