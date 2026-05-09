"""Compare fp32 vs fp16/bf16 cosine similarities on a sample of chunks.

Encodes the same N chunks under each dtype, L2-normalizes, computes pairwise
similarities, and reports whether reduced precision drifts the geometry.

Output appended to a JSON file (default data/results/dtype_sanity.json).
Used to decide whether Phase 3 latent-state storage can use fp16.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoder import MambaEncoder
from src.utils import list_eligible_files


def collect_chunks(repo_paths: list[Path], n: int, target_tokens: int,
                   tokenizer) -> list[str]:
    out: list[str] = []
    for repo in repo_paths:
        for f in list_eligible_files(repo):
            try:
                content = f.read_text(errors="replace")
            except OSError:
                continue
            if not content.strip():
                continue
            ids = tokenizer(content, return_tensors=None,
                            add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), target_tokens):
                sub = ids[i: i + target_tokens]
                if not sub:
                    continue
                out.append(tokenizer.decode(sub, skip_special_tokens=True))
                if len(out) >= n:
                    return out
    return out


def encode_all(model_id: str, dtype: torch.dtype, texts: list[str],
               batch: int = 4) -> tuple[np.ndarray, float]:
    enc = MambaEncoder(model_id=model_id, dtype=dtype, max_length=2048)
    out = np.zeros((len(texts), enc.d_model), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), batch):
        sub = texts[i: i + batch]
        o = enc.encode(sub)
        out[i: i + len(sub)] = o.pooled.float().cpu().numpy()
    return out, time.time() - t0


def cmp_matrices(a: np.ndarray, b: np.ndarray) -> dict:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    # Per-vector cos sim: a_i vs b_i
    per_vec_cos = (an * bn).sum(axis=1)
    sim_a = an @ an.T
    sim_b = bn @ bn.T
    triu = np.triu_indices_from(sim_a, k=1)
    diff = sim_a[triu] - sim_b[triu]
    return {
        "per_vector_cos": {
            "mean": float(per_vec_cos.mean()),
            "min": float(per_vec_cos.min()),
            "p5": float(np.percentile(per_vec_cos, 5)),
        },
        "pairwise_sim_diff": {
            "mean_abs": float(np.abs(diff).mean()),
            "max_abs": float(np.abs(diff).max()),
            "p95_abs": float(np.percentile(np.abs(diff), 95)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--n-chunks", type=int, default=100)
    ap.add_argument("--chunk-tokens", type=int, default=1500)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/results/dtype_sanity.json")
    ap.add_argument("--repos", nargs="*",
                    default=["sympy__sympy", "django__django",
                             "matplotlib__matplotlib"])
    args = ap.parse_args()

    print(f"sampling {args.n_chunks} chunks ({args.chunk_tokens} tok) "
          f"from {args.repos}")
    base_enc = MambaEncoder(model_id=args.model_id, dtype=torch.float32)
    repo_paths = [ROOT / "data" / "repos" / r for r in args.repos
                  if (ROOT / "data" / "repos" / r).exists()]
    if not repo_paths:
        print("ERROR: no eligible repos cloned; aborting")
        return 1

    texts = collect_chunks(repo_paths, args.n_chunks, args.chunk_tokens,
                           base_enc.tokenizer)
    print(f"collected {len(texts)} chunks")
    del base_enc

    print("encoding fp32...")
    e_fp32, t_fp32 = encode_all(args.model_id, torch.float32, texts)
    print(f"  fp32: {t_fp32:.1f}s")

    bf_dtype = torch.bfloat16
    print(f"encoding {bf_dtype}...")
    e_bf, t_bf = encode_all(args.model_id, bf_dtype, texts)
    print(f"  {bf_dtype}: {t_bf:.1f}s  ({t_fp32 / t_bf:.2f}x speedup)")

    print(f"encoding fp16...")
    e_fp16, t_fp16 = encode_all(args.model_id, torch.float16, texts)
    print(f"  fp16: {t_fp16:.1f}s  ({t_fp32 / t_fp16:.2f}x speedup)")

    report = {
        "model_id": args.model_id,
        "n_chunks": len(texts),
        "chunk_tokens": args.chunk_tokens,
        "encode_seconds": {"fp32": t_fp32, "bfloat16": t_bf, "fp16": t_fp16},
        "fp32_vs_bfloat16": cmp_matrices(e_fp32, e_bf),
        "fp32_vs_fp16": cmp_matrices(e_fp32, e_fp16),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
