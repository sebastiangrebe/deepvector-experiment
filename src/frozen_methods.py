"""Frozen-architecture matching methods over token-level Mamba latents.

All methods operate on already-encoded token-level latents (shape (T, D)).
No training. The same orthogonal projection matrix is used across all
queries and files (seeded) so scores are comparable.

Memory model: candidates are streamed from CPU one chunk at a time. Each
method moves a single chunk to the query's device, scores it, and drops the
GPU copy before moving to the next. Avoids OOM on large repos (matplotlib
pool ~35 GB; H100 80 GB minus encoder leaves no room for full pool).

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
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(d, d, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw)
    dtype = getattr(torch, dtype_str)
    return q.to(device=device_str, dtype=dtype).contiguous()


def get_projection(reference: torch.Tensor, seed: int = 42) -> torch.Tensor:
    return _orthogonal_projection(
        d=int(reference.shape[-1]),
        device_str=str(reference.device),
        dtype_str=str(reference.dtype).replace("torch.", ""),
        seed=seed,
    )


def _to_device(f: torch.Tensor, device: torch.device) -> torch.Tensor:
    return f if f.device == device else f.to(device, non_blocking=True)


# ─────────────────────────────────────────────────────────────────────
# Method 0: pooled cosine
# ─────────────────────────────────────────────────────────────────────

def pooled_score(q_tokens: torch.Tensor,
                 candidate_tokens: list[torch.Tensor]) -> torch.Tensor:
    device = q_tokens.device
    q_pool = F.normalize(q_tokens.mean(dim=0, keepdim=True), dim=-1)
    out = torch.empty(len(candidate_tokens), device=device, dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        f_d = _to_device(f, device)
        f_pool = F.normalize(f_d.mean(dim=0, keepdim=True), dim=-1)
        out[i] = (q_pool @ f_pool.T).squeeze()
        del f_pool
        if f_d is not f:
            del f_d
    return out


# ─────────────────────────────────────────────────────────────────────
# Method 1: MaxSim (single-head)
# ─────────────────────────────────────────────────────────────────────

def maxsim_score(q_tokens: torch.Tensor,
                 candidate_tokens: list[torch.Tensor]) -> torch.Tensor:
    device = q_tokens.device
    qn = F.normalize(q_tokens, dim=-1)
    out = torch.empty(len(candidate_tokens), device=device, dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        f_d = _to_device(f, device)
        fn = F.normalize(f_d, dim=-1)
        sim = qn @ fn.T
        out[i] = sim.max(dim=-1).values.sum()
        del sim, fn
        if f_d is not f:
            del f_d
    return out


# ─────────────────────────────────────────────────────────────────────
# Method 2: multi-head MaxSim (random orthogonal projections, sum)
# ─────────────────────────────────────────────────────────────────────

def multi_head_maxsim_score(q_tokens: torch.Tensor,
                            candidate_tokens: list[torch.Tensor],
                            n_heads: int = 8,
                            seed: int = 42) -> torch.Tensor:
    device = q_tokens.device
    d = q_tokens.shape[-1]
    if d % n_heads != 0:
        raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
    d_head = d // n_heads

    proj = get_projection(q_tokens, seed=seed)
    q_proj = q_tokens @ proj
    q_heads = [F.normalize(q_proj[:, h * d_head:(h + 1) * d_head], dim=-1)
               for h in range(n_heads)]

    out = torch.zeros(len(candidate_tokens), device=device, dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        f_d = _to_device(f, device)
        f_proj = f_d @ proj
        score = q_tokens.new_zeros(())
        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            f_h = F.normalize(f_proj[:, s:e], dim=-1)
            sim = q_heads[h] @ f_h.T
            score = score + sim.max(dim=-1).values.sum()
            del f_h, sim
        out[i] = score
        del f_proj
        if f_d is not f:
            del f_d
    return out


# ─────────────────────────────────────────────────────────────────────
# Method 3: late-interaction (H=8 + pre-projection L2 norm)
# ─────────────────────────────────────────────────────────────────────

def late_interaction_score(q_tokens: torch.Tensor,
                           candidate_tokens: list[torch.Tensor],
                           n_heads: int = 8,
                           seed: int = 42) -> torch.Tensor:
    """L2-normalize each token (pre-projection) → multi-head MaxSim.

    Streams candidates from CPU; never materializes a normalized list.
    """
    device = q_tokens.device
    d = q_tokens.shape[-1]
    if d % n_heads != 0:
        raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
    d_head = d // n_heads

    qn = F.normalize(q_tokens, dim=-1)
    proj = get_projection(qn, seed=seed)
    q_proj = qn @ proj
    q_heads = [F.normalize(q_proj[:, h * d_head:(h + 1) * d_head], dim=-1)
               for h in range(n_heads)]

    out = torch.zeros(len(candidate_tokens), device=device, dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        f_d = _to_device(f, device)
        fn = F.normalize(f_d, dim=-1)
        f_proj = fn @ proj
        del fn
        score = q_tokens.new_zeros(())
        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            f_h = F.normalize(f_proj[:, s:e], dim=-1)
            sim = q_heads[h] @ f_h.T
            score = score + sim.max(dim=-1).values.sum()
            del f_h, sim
        out[i] = score
        del f_proj
        if f_d is not f:
            del f_d
    return out


# ─────────────────────────────────────────────────────────────────────
# Generic dedup-to-files
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
    fn: callable
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
