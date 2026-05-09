"""Codestral-Mamba-7B smoke test on M5 Max.

Verifies:
  - HF load with bf16 on MPS
  - Hidden-state shape (d_model expected = 4096)
  - No OOM on a representative file
  - Pairwise cos sim on 10 file vectors (collapse check)
  - Per-chunk encoding latency (for runtime extrapolation)
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.utils import list_eligible_files

MODEL = "mistralai/Mamba-Codestral-7B-v0.1"
N_FILES = 10
CHUNK_TOKENS = 1024


def main() -> int:
    print("=" * 70)
    print(f"Codestral smoke: {MODEL}")
    print("=" * 70)
    t_start = time.time()

    print("\n[1/4] loading model in bf16 on MPS...")
    enc = MambaEncoder(model_id=MODEL, dtype=torch.bfloat16, max_length=CHUNK_TOKENS)
    print(f"  d_model: {enc.d_model}")
    print(f"  device: {enc.device}")
    print(f"  load time: {time.time() - t_start:.1f}s")

    print(f"\n[2/4] picking {N_FILES} representative files from sqlfluff...")
    repo = ROOT / "data" / "repos" / "sqlfluff__sqlfluff"
    files = list_eligible_files(repo)[:N_FILES]
    for f in files:
        print(f"  {f.relative_to(repo).as_posix()}  ({f.stat().st_size:,}B)")

    print(f"\n[3/4] encoding {N_FILES} files (truncated at {CHUNK_TOKENS} tokens)...")
    file_vecs = np.zeros((N_FILES, enc.d_model), dtype=np.float32)
    per_file = []
    for i, f in enumerate(files):
        text = f.read_text(errors="replace")
        t0 = time.time()
        out = enc.encode([text])  # truncates internally to max_length
        dt = time.time() - t0
        per_file.append(dt)
        v = out.pooled.float().cpu().numpy()[0]
        file_vecs[i] = v
        seq_len = int(out.attention_mask.sum().item())
        print(f"  [{i+1}/{N_FILES}] seqlen={seq_len:>4}  pooled.shape={tuple(out.pooled.shape)}  "
              f"last_hidden.shape={tuple(out.last_hidden.shape)}  dt={dt:.2f}s")

    print(f"\n[4/4] vector quality + diagnostics")
    norms = np.linalg.norm(file_vecs, axis=1)
    print(f"  vector norms: min {norms.min():.2f}  max {norms.max():.2f}  mean {norms.mean():.2f}")
    file_vecs_n = file_vecs / np.maximum(norms[:, None], 1e-8)
    sim = file_vecs_n @ file_vecs_n.T
    np.fill_diagonal(sim, np.nan)
    print(f"  pairwise cos sim: mean {np.nanmean(sim):.4f}  std {np.nanstd(sim):.4f}  "
          f"min {np.nanmin(sim):.4f}  max {np.nanmax(sim):.4f}")
    mean_vec = file_vecs.mean(axis=0)
    mean_vec /= max(np.linalg.norm(mean_vec), 1e-8)
    align = file_vecs_n @ mean_vec
    print(f"  cos with corpus mean: mean {align.mean():.4f}  min {align.min():.4f}  max {align.max():.4f}")

    print()
    print("encoding latency (per file, truncated at 1024 tokens):")
    print(f"  mean {np.mean(per_file):.2f}s  median {np.median(per_file):.2f}s  "
          f"min {min(per_file):.2f}s  max {max(per_file):.2f}s")

    # Memory snapshot
    try:
        import resource
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"  peak RSS: {rss_gb:.2f} GB")
    except Exception:
        pass

    print()
    print(f"total smoke time: {time.time() - t_start:.1f}s")
    print("Smoke test passed" if np.nanstd(sim) > 0.005 else
          "WARNING: low vector spread (collapse risk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
