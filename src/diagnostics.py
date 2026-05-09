"""Vector-quality diagnostics for retrieval indices.

Used by mamba_pooled (and eventually layer-2 latent-state) experiments to
detect collapse and quantify spread at scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def index_quality_stats(embeds: np.ndarray, sample_n: int = 1000,
                        rng_seed: int = 0) -> dict:
    """Compute spread diagnostics on a (N, D) embedding matrix.

    - Pairwise cosine on a random sub-sample (full pairwise is N^2; sample to bound cost)
    - Cosine with corpus centroid
    - Histogram of cosine similarities (10 buckets in [-1, 1])

    Returns a JSON-friendly dict.
    """
    if embeds.shape[0] == 0:
        return {"n": 0}

    e = embeds.astype(np.float32)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    e_n = e / np.maximum(norms, 1e-8)

    # Centroid alignment
    mean_vec = e_n.mean(axis=0)
    mean_vec /= max(np.linalg.norm(mean_vec), 1e-8)
    align = e_n @ mean_vec
    centroid = {
        "mean": float(align.mean()),
        "std": float(align.std()),
        "min": float(align.min()),
        "max": float(align.max()),
    }

    # Pairwise cos on sub-sample
    rng = np.random.default_rng(rng_seed)
    n = e_n.shape[0]
    if n <= sample_n:
        sample = e_n
    else:
        idx = rng.choice(n, size=sample_n, replace=False)
        sample = e_n[idx]

    sim = sample @ sample.T
    np.fill_diagonal(sim, np.nan)
    pairwise = {
        "sample_size": int(sample.shape[0]),
        "n_pairs": int(sample.shape[0] * (sample.shape[0] - 1) // 2),
        "mean": float(np.nanmean(sim)),
        "std": float(np.nanstd(sim)),
        "min": float(np.nanmin(sim)),
        "max": float(np.nanmax(sim)),
        "p50": float(np.nanpercentile(sim, 50)),
        "p95": float(np.nanpercentile(sim, 95)),
    }

    # Histogram (upper triangle only)
    flat = sim[np.triu_indices_from(sim, k=1)]
    flat = flat[~np.isnan(flat)]
    edges = np.linspace(-1.0, 1.0, 11)
    counts, _ = np.histogram(flat, bins=edges)
    histogram = {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
    }

    # Norm stats (sanity — should be uniform after L2-norm path; but pre-norm
    # tells us if the encoder is producing wildly different magnitudes)
    norm_stats = {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }

    # Collapse heuristic
    collapsed = pairwise["mean"] > 0.9 and pairwise["std"] < 0.05

    return {
        "n_vectors": int(n),
        "d_model": int(e.shape[1]),
        "pre_norm": norm_stats,
        "centroid_cos": centroid,
        "pairwise_cos_sample": pairwise,
        "pairwise_histogram": histogram,
        "collapse_flag": collapsed,
    }


def aggregate_per_repo(per_repo_embeds: dict[str, np.ndarray],
                      sample_n: int = 1000) -> dict:
    """Run index_quality_stats on each repo and aggregate."""
    out = {}
    for repo, emb in per_repo_embeds.items():
        out[repo] = index_quality_stats(emb, sample_n=sample_n)
    return out
