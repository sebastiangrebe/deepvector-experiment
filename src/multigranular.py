"""Phase 2.8: multi-granularity matching over Mamba latent state.

Granularities, all derived from per-chunk token-level latents (already cached
by Phase 2.5/2.6/2.7 under data/maxsim_cache/):

  G0 — file-level pool: mean over all token vectors of every chunk in the file
       → 1 vector of shape (D,) per file.
  G1 — chunk-level pool: mean over each chunk's tokens
       → list[(D,)] per file, length = n_chunks.
  G2 — function-level pool. Two modes:
       - tree-sitter mode (when --g2-mode tree_sitter and tree_sitter_python
         is available + the file's source can be re-tokenized to recover
         chunk→file token mapping). Currently disabled by default because
         the existing chunk cache does not store the byte-offset metadata
         needed to map tree-sitter byte ranges back into cached tensors
         without re-encoding. See docstring of `_g2_via_treesitter`.
       - sliding-window mode (default): within each cached chunk tensor,
         take overlapping windows of `WIN_TOKENS` with stride `STRIDE_TOKENS`,
         mean-pool each window. Approximates "function-sized" segments
         without parser-level metadata.
  G3 — token-level: the cached chunk tensors verbatim.

Matching methods:

  pooled_chunk — Phase 1 baseline: query mean-pool vs G1 (max over chunks).
  pooled_file  — query mean-pool vs G0.
  func_pool    — query mean-pool vs G2 (max over function-sized segments).
  maxsim       — Phase 2.5 baseline: token-level MaxSim vs G3.
  mg_sum       — weighted sum of normalized G0/G1/G2/G3 scores.
  mg_max       — per-query max across normalized G0/G1/G2/G3 scores.
  mg_routed    — heuristic per-query route to one granularity.

Score normalization: per-granularity min-max from the candidate pool of
each query. We rescale each granularity's scores to [0, 1] *within the
query's candidate set*, so the sum/max combinations are commensurable.
This is local normalization (per-query), not global, which avoids the
calibration-set sampling bias the user flagged as a likely bug source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


WIN_TOKENS = 256
STRIDE_TOKENS = 128


# ─────────────────────────────────────────────────────────────────────
# Per-file representation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FileRepresentation:
    """All four granularities for one file. Tensors live on CPU; matching
    code moves them to the query's device per-chunk to bound VRAM."""
    file_rel: str
    g0_pool: torch.Tensor                         # (D,)
    g1_chunks: list[torch.Tensor]                 # K x (D,)
    g2_segments: list[torch.Tensor]               # F x (D,)
    g3_tokens: list[torch.Tensor]                 # K x (n_tok_k, D)
    g2_metadata: list[dict] = field(default_factory=list)


def build_file_representations(chunk_files: list[str],
                               chunk_tokens: list[torch.Tensor],
                               g2_mode: str = "sliding_window") -> list[FileRepresentation]:
    """Group cached chunks by file, derive G0/G1/G2/G3 from G3 tensors.

    Inputs are the cached-pool data structure (parallel lists of file ids
    and per-chunk token tensors) from `scripts/maxsim_test.py`.
    Returns one FileRepresentation per unique file.
    """
    by_file: dict[str, list[torch.Tensor]] = {}
    for f, t in zip(chunk_files, chunk_tokens):
        by_file.setdefault(f, []).append(t)

    out: list[FileRepresentation] = []
    for f, chunks in by_file.items():
        # G1: mean each chunk
        g1 = [c.mean(dim=0) for c in chunks]
        # G0: mean over all tokens of all chunks (weight by token count, not chunk count)
        all_tokens = torch.cat(chunks, dim=0)
        g0 = all_tokens.mean(dim=0)
        # G2: sliding window pool
        if g2_mode == "sliding_window":
            g2, g2_meta = _g2_sliding_window(chunks)
        elif g2_mode == "tree_sitter":
            g2, g2_meta = _g2_via_treesitter(f, chunks)
        else:
            raise ValueError(f"unknown g2_mode {g2_mode}")

        out.append(FileRepresentation(
            file_rel=f,
            g0_pool=g0,
            g1_chunks=g1,
            g2_segments=g2,
            g3_tokens=chunks,
            g2_metadata=g2_meta,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────
# G2 extractors
# ─────────────────────────────────────────────────────────────────────

def _g2_sliding_window(chunks: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[dict]]:
    """Per chunk, slide a (WIN_TOKENS, STRIDE_TOKENS) window and mean-pool."""
    segs: list[torch.Tensor] = []
    meta: list[dict] = []
    for ci, c in enumerate(chunks):
        n = c.shape[0]
        if n == 0:
            continue
        if n <= WIN_TOKENS:
            segs.append(c.mean(dim=0))
            meta.append({"chunk": ci, "start": 0, "end": n})
            continue
        for s in range(0, n - WIN_TOKENS + 1, STRIDE_TOKENS):
            e = s + WIN_TOKENS
            segs.append(c[s:e].mean(dim=0))
            meta.append({"chunk": ci, "start": s, "end": e})
        # Tail window if last stride didn't cover end of chunk.
        if (n - WIN_TOKENS) % STRIDE_TOKENS != 0:
            segs.append(c[-WIN_TOKENS:].mean(dim=0))
            meta.append({"chunk": ci, "start": n - WIN_TOKENS, "end": n})
    return segs, meta


def _g2_via_treesitter(file_rel: str,
                       chunks: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[dict]]:
    """Tree-sitter path. Currently NOT wired to cached tensors.

    Proper implementation requires byte-offset metadata per cached chunk so
    that tree-sitter byte ranges can be mapped back into the cached tensor
    indices. The current cache (data/maxsim_cache/*.pt) does not store this.
    Adding it requires re-encoding with offset_mapping enabled and saving the
    `(chunk_id, file_byte_start, file_byte_end, token_byte_offsets)` quadruple
    per chunk. Left as future work; sliding-window is used by default.
    """
    raise NotImplementedError(
        "tree_sitter G2 mapping requires cache schema extension; use sliding_window")


# ─────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────

def _minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a 1-D tensor to [0, 1]. Constant tensors → 0.5."""
    lo, hi = x.min(), x.max()
    rng = (hi - lo).clamp_min(1e-8)
    return (x - lo) / rng if (hi > lo) else torch.full_like(x, 0.5)


# ─────────────────────────────────────────────────────────────────────
# Per-method scoring against a list of FileRepresentations
# ─────────────────────────────────────────────────────────────────────

def _to_dev(t: torch.Tensor, device) -> torch.Tensor:
    return t if t.device == device else t.to(device, non_blocking=True)


def _q_pool(q_tokens: torch.Tensor) -> torch.Tensor:
    return F.normalize(q_tokens.mean(dim=0, keepdim=True), dim=-1).squeeze(0)


def _maxsim(qn: torch.Tensor, f_tokens: torch.Tensor) -> torch.Tensor:
    fn = F.normalize(f_tokens, dim=-1)
    sim = qn @ fn.T
    return sim.max(dim=-1).values.sum()


def score_pooled_chunk(q_tokens: torch.Tensor,
                       files: list[FileRepresentation]) -> torch.Tensor:
    """G1 only: query pool vs max over chunk pools."""
    device = q_tokens.device
    q = _q_pool(q_tokens)
    out = torch.empty(len(files), device=device, dtype=q_tokens.dtype)
    for i, fr in enumerate(files):
        best = q.new_tensor(-1e30)
        for c in fr.g1_chunks:
            cn = F.normalize(_to_dev(c, device).unsqueeze(0), dim=-1).squeeze(0)
            s = q @ cn
            if s > best:
                best = s
        out[i] = best
    return out


def score_pooled_file(q_tokens: torch.Tensor,
                      files: list[FileRepresentation]) -> torch.Tensor:
    """G0 only."""
    device = q_tokens.device
    q = _q_pool(q_tokens)
    out = torch.empty(len(files), device=device, dtype=q_tokens.dtype)
    for i, fr in enumerate(files):
        gn = F.normalize(_to_dev(fr.g0_pool, device).unsqueeze(0), dim=-1).squeeze(0)
        out[i] = q @ gn
    return out


def score_func_pool(q_tokens: torch.Tensor,
                    files: list[FileRepresentation]) -> torch.Tensor:
    """G2 only: query pool vs max over function-sized segments."""
    device = q_tokens.device
    q = _q_pool(q_tokens)
    out = torch.empty(len(files), device=device, dtype=q_tokens.dtype)
    for i, fr in enumerate(files):
        if not fr.g2_segments:
            out[i] = q.new_tensor(-1e30)
            continue
        best = q.new_tensor(-1e30)
        for seg in fr.g2_segments:
            sn = F.normalize(_to_dev(seg, device).unsqueeze(0), dim=-1).squeeze(0)
            s = q @ sn
            if s > best:
                best = s
        out[i] = best
    return out


def score_maxsim(q_tokens: torch.Tensor,
                 files: list[FileRepresentation]) -> torch.Tensor:
    """G3 only — Phase 2.5 baseline reproduced."""
    device = q_tokens.device
    qn = F.normalize(q_tokens, dim=-1)
    out = torch.empty(len(files), device=device, dtype=q_tokens.dtype)
    for i, fr in enumerate(files):
        best = qn.new_tensor(-1e30)
        for tok in fr.g3_tokens:
            s = _maxsim(qn, _to_dev(tok, device))
            if s > best:
                best = s
        out[i] = best
    return out


def _all_granularity_scores(q_tokens: torch.Tensor,
                            files: list[FileRepresentation]
                            ) -> dict[str, torch.Tensor]:
    return {
        "g0": score_pooled_file(q_tokens, files),
        "g1": score_pooled_chunk(q_tokens, files),
        "g2": score_func_pool(q_tokens, files),
        "g3": score_maxsim(q_tokens, files),
    }


def score_mg_sum(q_tokens: torch.Tensor,
                 files: list[FileRepresentation],
                 weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
                 ) -> torch.Tensor:
    """Weighted sum of per-query min-max-normalized scores from all granularities."""
    raw = _all_granularity_scores(q_tokens, files)
    norm = {k: _minmax(v.float()) for k, v in raw.items()}
    return (weights[0] * norm["g0"]
            + weights[1] * norm["g1"]
            + weights[2] * norm["g2"]
            + weights[3] * norm["g3"])


def score_mg_max(q_tokens: torch.Tensor,
                 files: list[FileRepresentation]) -> torch.Tensor:
    """Per-query, take per-file max across normalized G0/G1/G2/G3."""
    raw = _all_granularity_scores(q_tokens, files)
    norm = torch.stack([_minmax(raw[k].float()) for k in ("g0", "g1", "g2", "g3")])
    return norm.max(dim=0).values


# ─────────────────────────────────────────────────────────────────────
# mg_routed: per-query heuristic
# ─────────────────────────────────────────────────────────────────────

import re

_IDENT_PAT = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*[._][a-zA-Z_][a-zA-Z0-9_]*\b|\b[a-z]+[A-Z]\w*\b|\b[a-z]+_[a-z_]+\b")


def route_query(query_text: str, n_query_tokens: int) -> str:
    """Decide which granularity wins this query.

    Heuristic:
      - tokens < 30 AND has identifier-like pattern → "g3" (local token match)
      - tokens > 100 AND no identifier-like pattern → "g0" (file-level)
      - else                                        → "g2" (function-level)

    Keeping the rule simple and reproducible. Documented in the paper if
    Phase 2.8 results warrant it.
    """
    has_ident = bool(_IDENT_PAT.search(query_text))
    if n_query_tokens < 30 and has_ident:
        return "g3"
    if n_query_tokens > 100 and not has_ident:
        return "g0"
    return "g2"


def score_mg_routed(q_tokens: torch.Tensor,
                    query_text: str,
                    files: list[FileRepresentation]) -> tuple[torch.Tensor, str]:
    """Route the query to one granularity by heuristic, return scores + the
    chosen granularity (for routing-distribution stats)."""
    chosen = route_query(query_text, n_query_tokens=int(q_tokens.shape[0]))
    if chosen == "g0":
        return score_pooled_file(q_tokens, files), "g0"
    if chosen == "g1":
        return score_pooled_chunk(q_tokens, files), "g1"
    if chosen == "g2":
        return score_func_pool(q_tokens, files), "g2"
    return score_maxsim(q_tokens, files), "g3"


# ─────────────────────────────────────────────────────────────────────
# Dedup → top-K (file-level since FileRepresentation is per-file)
# ─────────────────────────────────────────────────────────────────────

def topk_files(scores: torch.Tensor, files: list[FileRepresentation],
               top_k: int = 20) -> list[tuple[str, float]]:
    n = min(scores.shape[0], top_k)
    order = torch.argsort(scores, descending=True).tolist()[:n]
    return [(files[i].file_rel, float(scores[i].item())) for i in order]
