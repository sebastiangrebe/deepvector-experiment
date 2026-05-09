"""Mamba SSM mean-pooled vectors + FAISS baseline.

Same chunking strategy as Voyage for direct comparability:
  - 1500-token chunks per file (file path prepended as header)
  - Each chunk → encoder → mean-pool last_hidden over seqlen → 1 vec/chunk
  - All chunk vectors go into FAISS IndexFlatIP (after L2-normalize)
  - Search returns top-N chunks, dedup-to-file preserving rank → top-K files

Default model: state-spaces/mamba-130m-hf  (small, MPS-friendly, weak)
Recommended: mistralai/Mamba-Codestral-7B-v0.1  (Mamba-2, code-trained, needs cloud)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from tqdm import tqdm

from src.diagnostics import index_quality_stats
from src.encoder import MambaEncoder
from src.utils import get_logger, project_root

log = get_logger("mamba_pooled")

CHUNK_TOKENS = 1500   # match Voyage chunking exactly
ENCODE_BATCH = 4      # files-per-encoder-call; tune per device

CACHE_DIR = project_root() / "data" / "mamba_cache"


# ─────────────────────────────────────────────────────────────────────
# Chunk dataclass + chunking via encoder tokenizer
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    file_rel: str
    chunk_idx: int
    text: str

    def header_text(self) -> str:
        return f"# File: {self.file_rel}\n\n{self.text}"


def _split_chunks(tokenizer, file_rel: str, content: str,
                  target_tokens: int) -> list[Chunk]:
    """Tokenize once, split on token boundaries; rebuild text from token ids."""
    if not content.strip():
        return []
    enc = tokenizer(content, return_tensors=None, truncation=False,
                    padding=False, add_special_tokens=False)
    ids = enc["input_ids"]
    if not ids:
        return []
    chunks: list[Chunk] = []
    for ci, i in enumerate(range(0, len(ids), target_tokens)):
        sub = ids[i: i + target_tokens]
        text = tokenizer.decode(sub, skip_special_tokens=True)
        chunks.append(Chunk(file_rel, ci, text))
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Cache key
# ─────────────────────────────────────────────────────────────────────

def _scope_signature(model_id: str, repo_path: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(str(CHUNK_TOKENS).encode())
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        h.update(f.relative_to(repo_path).as_posix().encode())
        h.update(f"{st.st_size}-{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────

class MambaPooledRetriever:
    name = "mamba_pooled"

    def __init__(self,
                 model_id: str = "state-spaces/mamba-130m-hf",
                 dtype: torch.dtype = torch.float32) -> None:
        self.model_id = model_id
        self.encoder = MambaEncoder(model_id=model_id, dtype=dtype,
                                    max_length=CHUNK_TOKENS + 32)
        self.name = f"mamba_pooled_{Path(model_id).name}"
        self._reset()

    def _reset(self) -> None:
        self.repo_path: Path | None = None
        self.index_obj = None
        self.chunks: list[Chunk] = []
        self.chunk_files: list[str] = []  # parallel to FAISS rows
        self.last_quality: dict | None = None
        self.last_index_size_mb: float | None = None

    def index(self, repo_path: Path, files: list[Path]) -> None:
        self._reset()
        self.repo_path = repo_path
        sig = _scope_signature(self.model_id, repo_path, files)
        cache_file = CACHE_DIR / f"{repo_path.name}_{self.name}_{sig}.npz"
        meta_file = CACHE_DIR / f"{repo_path.name}_{self.name}_{sig}.json"

        if cache_file.exists() and meta_file.exists():
            log.info("Cache hit %s", cache_file.name)
            data = np.load(cache_file)
            embeds = data["embeds"]
            meta = json.loads(meta_file.read_text())
            self.chunks = [Chunk(**c) for c in meta["chunks"]]
            self.chunk_files = [c.file_rel for c in self.chunks]
            self._build_faiss(embeds)
            self.last_quality = index_quality_stats(embeds)
            self.last_index_size_mb = cache_file.stat().st_size / 1024 / 1024
            log.info("Loaded %d chunks from cache  pairwise_cos.mean=%.4f",
                     len(self.chunks),
                     self.last_quality["pairwise_cos_sample"]["mean"])
            return

        log.info("Indexing %s: %d files", repo_path.name, len(files))
        all_chunks: list[Chunk] = []
        for f in tqdm(files, desc=f"chunk {repo_path.name}", leave=False):
            try:
                content = f.read_text(errors="replace")
            except OSError:
                continue
            rel = f.relative_to(repo_path).as_posix()
            all_chunks.extend(_split_chunks(self.encoder.tokenizer, rel,
                                            content, CHUNK_TOKENS))

        log.info("Total chunks: %d", len(all_chunks))
        embeds = self._encode_chunks(all_chunks)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, embeds=embeds)
        meta_file.write_text(json.dumps({
            "chunks": [c.__dict__ for c in all_chunks],
            "model_id": self.model_id,
            "chunk_tokens": CHUNK_TOKENS,
        }))
        size_mb = cache_file.stat().st_size / 1024 / 1024
        log.info("Cached %d × %d vecs (%.2f MB) to %s",
                 embeds.shape[0], embeds.shape[1], size_mb, cache_file.name)

        self.chunks = all_chunks
        self.chunk_files = [c.file_rel for c in all_chunks]
        self._build_faiss(embeds)
        self.last_quality = index_quality_stats(embeds)
        self.last_index_size_mb = cache_file.stat().st_size / 1024 / 1024
        q = self.last_quality["pairwise_cos_sample"]
        log.info("Quality: pairwise_cos.mean=%.4f std=%.4f  collapse=%s",
                 q["mean"], q["std"], self.last_quality["collapse_flag"])

    def _encode_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.encoder.d_model), dtype=np.float32)
        d_model = self.encoder.d_model
        out = np.zeros((len(chunks), d_model), dtype=np.float32)
        BATCH = ENCODE_BATCH
        pbar = tqdm(total=len(chunks), desc="encode", leave=False)
        for i in range(0, len(chunks), BATCH):
            batch = [c.header_text() for c in chunks[i: i + BATCH]]
            o = self.encoder.encode(batch)
            out[i: i + len(batch)] = o.pooled.float().cpu().numpy()
            pbar.update(len(batch))
        pbar.close()
        return out

    def _build_faiss(self, embeds: np.ndarray) -> None:
        if embeds.shape[0] == 0:
            self.index_obj = None
            return
        embeds = embeds.astype(np.float32)
        faiss.normalize_L2(embeds)
        idx = faiss.IndexFlatIP(embeds.shape[1])
        idx.add(embeds)
        self.index_obj = idx

    def search(self, query: str, top_k: int = 20) -> list[tuple[Path, float]]:
        if self.index_obj is None or not self.chunks or self.repo_path is None:
            return []
        # Encode query as a single chunk (truncated by tokenizer if too long).
        out = self.encoder.encode([query])
        q = out.pooled.float().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(q)
        n_search = min(self.index_obj.ntotal, max(top_k * 10, 100))
        D, I = self.index_obj.search(q, n_search)
        seen: dict[str, float] = {}
        for score, row in zip(D[0].tolist(), I[0].tolist()):
            if row < 0:
                continue
            f = self.chunk_files[row]
            if f not in seen:
                seen[f] = float(score)
            if len(seen) >= top_k:
                break
        return [(self.repo_path / rel, score) for rel, score in seen.items()]
