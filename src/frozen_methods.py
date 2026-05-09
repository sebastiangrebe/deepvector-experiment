"""Frozen-architecture matching methods over token-level Mamba latents.

All methods operate on already-encoded token-level latents (shape (T, D)).
No training. The same orthogonal projection matrix is used across all
queries and files (seeded) so scores are comparable.

Methods:
    pooled            — mean-pool both sides, cosine
    maxsim            — ColBERT-style late interaction, single-head
    multi_head_maxsim — random orthogonal projections, H heads, sum across
    late_interaction  — H=8 multi-head + extra pre-projection L2 norm
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# Stable orthogonal projection
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _orthogonal_projection(d: int, device_str: str, dtype_str: str,
                           seed: int = 42) -> torch.Tensor:
    """Generate a (d, d) orthogonal matrix. Cached so all calls reuse."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(d, d, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw)
    dtype = getattr(torch, dtype_str)
    return q.to(device=device_str, dtype=dtype).contiguous()


def get_projection(reference: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Return cached orthogonal projection matched to reference's device/dtype."""
    return _orthogonal_projection(
        d=int(reference.shape[-1]),
        device_str=str(reference.device),
        dtype_str=str(reference.dtype).replace("torch.", ""),
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────
# Method 0: pooled cosine
# ─────────────────────────────────────────────────────────────────────

def pooled_score(q_tokens: torch.Tensor,
                 candidate_tokens: list[torch.Tensor]) -> torch.Tensor:
    """Mean-pool both sides, cosine. Returns (N,)."""
    q_pool = F.normalize(q_tokens.mean(dim=0, keepdim=True), dim=-1)
    out = torch.empty(len(candidate_tokens), device=q_tokens.device,
                      dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        f_pool = F.normalize(f.mean(dim=0, keepdim=True), dim=-1)
        out[i] = (q_pool @ f_pool.T).squeeze()
    return out


# ─────────────────────────────────────────────────────────────────────
# Method 1: MaxSim (single-head ColBERT-style)
# ─────────────────────────────────────────────────────────────────────

def maxsim_score(q_tokens: torch.Tensor,
                 candidate_tokens: list[torch.Tensor]) -> torch.Tensor:
    qn = F.normalize(q_tokens, dim=-1)
    out = torch.empty(len(candidate_tokens), device=q_tokens.device,
                      dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        fn = F.normalize(f, dim=-1)
        sim = qn @ fn.T
        out[i] = sim.max(dim=-1).values.sum()
    return out


# ─────────────────────────────────────────────────────────────────────
# Method 2: multi-head MaxSim with random orthogonal projections
# ─────────────────────────────────────────────────────────────────────

def multi_head_maxsim_score(q_tokens: torch.Tensor,
                            candidate_tokens: list[torch.Tensor],
                            n_heads: int = 8,
                            seed: int = 42) -> torch.Tensor:
    """Sum of MaxSim scores over n_heads disjoint orthogonal subspaces."""
    d = q_tokens.shape[-1]
    if d % n_heads != 0:
        raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
    d_head = d // n_heads

    proj = get_projection(q_tokens, seed=seed)  # (d, d)
    q_proj = q_tokens @ proj                    # (n_q, d)
    cand_projs = [f @ proj for f in candidate_tokens]

    out = torch.zeros(len(candidate_tokens), device=q_tokens.device,
                      dtype=q_tokens.dtype)
    for h in range(n_heads):
        s, e = h * d_head, (h + 1) * d_head
        q_h = F.normalize(q_proj[:, s:e], dim=-1)
        for i, f_p in enumerate(cand_projs):
            f_h = F.normalize(f_p[:, s:e], dim=-1)
            sim = q_h @ f_h.T                    # (n_q, n_f)
            out[i] += sim.max(dim=-1).values.sum()
    return out


# ─────────────────────────────────────────────────────────────────────
# Method 3: late-interaction (H=8 + explicit pre-projection L2 norm)
# ─────────────────────────────────────────────────────────────────────

def late_interaction_score(q_tokens: torch.Tensor,
                           candidate_tokens: list[torch.Tensor],
                           n_heads: int = 8,
                           seed: int = 42) -> torch.Tensor:
    """L2-normalize each token before projection, then multi-head MaxSim."""
    qn = F.normalize(q_tokens, dim=-1)
    cand_n = [F.normalize(f, dim=-1) for f in candidate_tokens]
    return multi_head_maxsim_score(qn, cand_n, n_heads=n_heads, seed=seed)


# ─────────────────────────────────────────────────────────────────────
# Generic dedup-to-files (matches Voyage / mamba_pooled / maxsim)
# ─────────────────────────────────────────────────────────────────────

def dedup_to_files(scores: torch.Tensor, chunk_files: list[str],
                   top_k: int = 20) -> list[tuple[str, float]]:
    order = torch.argsort(scores, descending=True).tolist()
    seen: dict[str, float] = {}
    for row in order:
        f = chunk_files[row]
        if f in seen:
            continue
        seen[f] = float(scores[row].item())
        if len(seen) >= top_k:
            break
    return list(seen.items())


# ─────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MethodSpec:
    name: str
    fn: callable          # (q, cands, **kwargs) → (N,) scores
    kwargs: dict


def all_methods() -> list[MethodSpec]:
    return [
        MethodSpec("pooled",      pooled_score,            {}),
        MethodSpec("maxsim",      maxsim_score,            {}),
        MethodSpec("mh_4",        multi_head_maxsim_score, {"n_heads": 4}),
        MethodSpec("mh_8",        multi_head_maxsim_score, {"n_heads": 8}),
        MethodSpec("mh_16",       multi_head_maxsim_score, {"n_heads": 16}),
        MethodSpec("mh_32",       multi_head_maxsim_score, {"n_heads": 32}),
        MethodSpec("late_int_8",  late_interaction_score,  {"n_heads": 8}),
    ]
