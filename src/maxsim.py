"""MaxSim (ColBERT-style late interaction) scoring over token-level latents.

For a query Q (n_q, D) and a candidate F (n_f, D):

    MaxSim(Q, F) = sum_q max_f cos(q, f)
                 = sum_q max_f (q_norm @ f_norm)

We L2-normalize once per tensor and reduce via matmul + max.

Inputs are torch tensors. Caller is responsible for putting them on the desired
device (CUDA on cloud, MPS or CPU locally).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def maxsim_score(q_tokens: torch.Tensor, f_tokens: torch.Tensor) -> torch.Tensor:
    """Single (Q, F) score.

    q_tokens: (n_q, D)
    f_tokens: (n_f, D)

    Returns scalar tensor.
    """
    qn = _normalize(q_tokens)
    fn = _normalize(f_tokens)
    sim = qn @ fn.T                  # (n_q, n_f)
    per_q_max = sim.max(dim=-1).values  # (n_q,)
    return per_q_max.sum()


def maxsim_batch(q_tokens: torch.Tensor,
                 candidate_tokens: list[torch.Tensor]) -> torch.Tensor:
    """Score one query against a list of candidates with varying chunk lengths.

    Returns 1-D tensor of shape (len(candidate_tokens),).
    """
    qn = _normalize(q_tokens)
    out = torch.empty(len(candidate_tokens), device=q_tokens.device,
                      dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        fn = _normalize(f)
        sim = qn @ fn.T
        out[i] = sim.max(dim=-1).values.sum()
    return out


def pooled_score(q_tokens: torch.Tensor,
                 candidate_tokens: list[torch.Tensor]) -> torch.Tensor:
    """Method A: mean-pool both sides, cosine-score.

    Returns 1-D tensor (len(candidate_tokens),).
    """
    q_pool = _normalize(q_tokens.mean(dim=0, keepdim=True))   # (1, D)
    out = torch.empty(len(candidate_tokens), device=q_tokens.device,
                      dtype=q_tokens.dtype)
    for i, f in enumerate(candidate_tokens):
        f_pool = _normalize(f.mean(dim=0, keepdim=True))      # (1, D)
        out[i] = (q_pool @ f_pool.T).squeeze()
    return out


def dedup_to_files(scores: torch.Tensor, chunk_files: list[str],
                   top_k: int = 20) -> list[tuple[str, float]]:
    """Dedup chunk-level rankings to file-level, preserving best-rank order.

    Same logic as Voyage / mamba_pooled retrievers so methods are comparable.
    """
    n = scores.shape[0]
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
