"""Phase 3.1: nested Mamba encoding.

Three levels of encoding, all using the same frozen Codestral-Mamba-7B.

  L0 (token):    per-token last-layer states. Already cached from Phase 2.5.
  L1 (function): for each function/class in each file, run that function's
                 source as its own forward pass through Codestral. Take the
                 LAST-position last-layer state — Mamba's recurrent summary
                 of the sequence.  → 1 vec/function.
  L2 (file):     for each file, take the sequence of L1 vectors (in source
                 order), pass through Codestral via `inputs_embeds=...`,
                 take last-position last-layer state. → 1 vec/file.

Byte-range extraction uses tree-sitter directly (the existing tree-index
stores names but not byte ranges; we re-parse here to get them).

Caches: data/nested_cache/<repo_slug>_funcs.pt, _files.pt
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tree_sitter as ts
import tree_sitter_python as tsp
from tqdm import tqdm

from src.encoder import MambaEncoder
from src.tree_index import _unwrap_decorated
from src.utils import get_logger, list_eligible_files, project_root

log = get_logger("nested_encoding")
CACHE_DIR = project_root() / "data" / "nested_cache"

_LANGUAGE = ts.Language(tsp.language())
_PARSER = ts.Parser(_LANGUAGE)


# ─────────────────────────────────────────────────────────────────────
# Function byte-range extraction
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FuncSpan:
    file_rel: str
    name: str
    kind: str       # "function" | "class"
    start_byte: int
    end_byte: int

    def length(self) -> int:
        return self.end_byte - self.start_byte


def _id_text(node, src: bytes) -> str | None:
    for c in node.children:
        if c.type == "identifier":
            return src[c.start_byte:c.end_byte].decode("utf-8", errors="replace")
    return None


def extract_function_spans(file_rel: str, src: bytes,
                            include_nested: bool = False) -> list[FuncSpan]:
    """Top-level function/class definitions in source order. Class methods
    are not extracted by default (would explode count); set include_nested=True
    to include them.
    """
    tree = _PARSER.parse(src)
    out: list[FuncSpan] = []
    for top_raw in tree.root_node.children:
        top = _unwrap_decorated(top_raw)
        if top.type == "function_definition":
            n = _id_text(top, src) or "<anon>"
            out.append(FuncSpan(file_rel, n, "function",
                                 top_raw.start_byte, top_raw.end_byte))
        elif top.type == "class_definition":
            n = _id_text(top, src) or "<anon>"
            out.append(FuncSpan(file_rel, n, "class",
                                 top_raw.start_byte, top_raw.end_byte))
            if include_nested:
                body = next((c for c in top.children if c.type == "block"), None)
                if body:
                    for stmt in body.children:
                        inner = _unwrap_decorated(stmt)
                        if inner.type == "function_definition":
                            mn = _id_text(inner, src) or "<anon>"
                            out.append(FuncSpan(file_rel, f"{n}.{mn}", "function",
                                                 stmt.start_byte, stmt.end_byte))
    return out


# ─────────────────────────────────────────────────────────────────────
# L1: per-function encoding (forward pass over function source)
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_function_l1(encoder: MambaEncoder,
                       source_text: str) -> torch.Tensor:
    """Run function source through Codestral; return last-position last-layer
    state of shape (D,)."""
    out = encoder.encode([source_text])
    mask = out.attention_mask[0].bool()
    valid = out.last_hidden[0][mask]
    if valid.shape[0] == 0:
        return out.last_hidden[0][-1].detach().cpu()
    return valid[-1].detach().cpu()


# ─────────────────────────────────────────────────────────────────────
# L2: per-file encoding via inputs_embeds over function vectors
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_file_l2(encoder: MambaEncoder,
                   func_vectors: torch.Tensor) -> torch.Tensor:
    """Pass a (n_funcs, D) sequence through Codestral via inputs_embeds.
    Returns (D,) last-position last-layer state. If `inputs_embeds` is not
    supported by the model variant, raises RuntimeError so the caller can
    handle it explicitly (no silent fallback).

    func_vectors: (n_funcs, D) on CPU or device; will be moved to encoder.device.
    """
    if func_vectors.dim() != 2:
        raise ValueError(f"expected (n_funcs, D), got {tuple(func_vectors.shape)}")

    embeds = func_vectors.to(encoder.device, dtype=encoder.model.dtype).unsqueeze(0)
    # (1, n_funcs, D)
    n = embeds.shape[1]
    attn = torch.ones(1, n, dtype=torch.long, device=encoder.device)

    try:
        out = encoder.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    except TypeError as e:
        raise RuntimeError(
            f"This Mamba variant does not accept inputs_embeds: {e}") from e

    last_hidden = out.hidden_states[-1]   # (1, n, D)
    return last_hidden[0, -1].detach().cpu()


# ─────────────────────────────────────────────────────────────────────
# Repo-level batch driver with caches
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NestedRepoIndex:
    repo_slug: str
    func_spans: list[FuncSpan] = field(default_factory=list)
    func_vectors: torch.Tensor | None = None    # (n_total_funcs, D)
    file_to_func_idx: dict[str, list[int]] = field(default_factory=dict)
    file_vectors: dict[str, torch.Tensor] = field(default_factory=dict)


def _signature(model_id: str, files: list[Path], repo_root: Path) -> str:
    h = hashlib.sha256()
    h.update(b"nested_v1")
    h.update(model_id.encode())
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        h.update(f.relative_to(repo_root).as_posix().encode())
        h.update(f"{st.st_size}-{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def build_nested_index(encoder: MambaEncoder,
                        repo_path: Path,
                        max_func_chars: int = 12000,
                        max_funcs_per_file: int = 200,
                        ) -> NestedRepoIndex:
    """Encode L1 vectors for every top-level function/class and L2 vectors
    per file. Caches both on disk."""
    repo_path = repo_path.resolve()
    files = list_eligible_files(repo_path)
    sig = _signature(encoder.model_id, files, repo_path)
    cache = CACHE_DIR / f"{repo_path.name}_{Path(encoder.model_id).name}_{sig}.pt"

    if cache.exists():
        log.info("nested cache hit %s", cache.name)
        d = torch.load(cache, map_location="cpu", weights_only=False)
        idx = NestedRepoIndex(
            repo_slug=repo_path.name,
            func_spans=[FuncSpan(**fs) for fs in d["func_spans"]],
            func_vectors=d["func_vectors"],
            file_to_func_idx=d["file_to_func_idx"],
            file_vectors=d["file_vectors"],
        )
        return idx

    log.info("Building nested index for %s (%d files)", repo_path.name, len(files))
    spans: list[FuncSpan] = []
    file_to_funcs: dict[str, list[int]] = {}

    # Collect spans
    for p in files:
        try:
            src = p.read_bytes()
        except OSError:
            continue
        rel = p.relative_to(repo_path).as_posix()
        sps = extract_function_spans(rel, src)
        if not sps:
            continue
        if len(sps) > max_funcs_per_file:
            sps = sps[:max_funcs_per_file]
        start = len(spans)
        for s in sps:
            spans.append(s)
        file_to_funcs[rel] = list(range(start, len(spans)))

    log.info("[%s] %d total functions/classes across %d files",
             repo_path.name, len(spans), len(file_to_funcs))

    # Encode L1: per-function forward pass
    d_model = encoder.d_model
    func_vecs = torch.zeros(len(spans), d_model, dtype=torch.float32)
    file_to_src = {p.relative_to(repo_path).as_posix(): p.read_bytes()
                   for p in files}

    pbar = tqdm(total=len(spans), desc=f"L1 {repo_path.name}", leave=False)
    for i, sp in enumerate(spans):
        src = file_to_src.get(sp.file_rel)
        if src is None:
            pbar.update(1); continue
        text = src[sp.start_byte:sp.end_byte].decode("utf-8", errors="replace")
        if len(text) > max_func_chars:
            text = text[:max_func_chars]
        try:
            v = encode_function_l1(encoder, text)
        except Exception as e:
            log.warning("L1 failed on %s::%s: %s", sp.file_rel, sp.name, e)
            v = torch.zeros(d_model)
        func_vecs[i] = v.float()
        pbar.update(1)
    pbar.close()

    # Encode L2: per-file forward pass via inputs_embeds
    file_vectors: dict[str, torch.Tensor] = {}
    pbar = tqdm(total=len(file_to_funcs), desc=f"L2 {repo_path.name}", leave=False)
    for rel, idxs in file_to_funcs.items():
        if not idxs:
            pbar.update(1); continue
        seq = func_vecs[torch.tensor(idxs, dtype=torch.long)]
        try:
            v = encode_file_l2(encoder, seq)
        except RuntimeError as e:
            # No silent fallback — propagate to caller after first observation
            raise RuntimeError(
                f"L2 failed on {rel}: {e}. Codestral via transformers may "
                f"need direct backbone access; see nested_encoding for path."
            ) from e
        file_vectors[rel] = v.float()
        pbar.update(1)
    pbar.close()

    idx = NestedRepoIndex(
        repo_slug=repo_path.name,
        func_spans=spans,
        func_vectors=func_vecs,
        file_to_func_idx=file_to_funcs,
        file_vectors=file_vectors,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "func_spans": [fs.__dict__ for fs in spans],
        "func_vectors": func_vecs,
        "file_to_func_idx": file_to_funcs,
        "file_vectors": file_vectors,
    }, cache)
    log.info("Cached %s (%.2f MB)", cache.name, cache.stat().st_size / 1e6)

    return idx
