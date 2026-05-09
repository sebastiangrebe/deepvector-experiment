"""Voyage embeddings + FAISS baseline.

- Model: voyage-code-3 (32K ctx, $0.18/M tokens, 2000 RPM, 3M TPM)
- Chunking: ~1500-token chunks; file path prepended as header
- Index: FAISS IndexFlatIP after L2 normalize (= cosine sim, exact search)
- Caching: on disk per (repo, base_commit) signature
- Dedup: chunk-level search, then collapse to file IDs preserving rank
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import get_logger, project_root

log = get_logger("voyage")

MODEL = "voyage-code-3"
CHUNK_TOKENS = 1500
PRICE_PER_M = 0.18           # USD
EMBED_BATCH_TEXTS = 128      # API max is 1000; stay conservative
EMBED_BATCH_TOKENS = 80_000   # API max is 120K — leave headroom
INTER_BATCH_SLEEP = 1.0      # seconds — keeps us well under 3M TPM
CACHE_DIR = project_root() / "data" / "voyage_cache"


# ─────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    file_rel: str
    chunk_idx: int
    text: str

    def header_text(self) -> str:
        return f"# File: {self.file_rel}\n\n{self.text}"


def _split_into_chunks(client, file_rel: str, content: str,
                      target_tokens: int = CHUNK_TOKENS) -> list[Chunk]:
    """Tokenize once, split on token boundaries by line groups."""
    if not content.strip():
        return []
    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    # Token budget per line via voyage tokenizer (one call per file)
    line_tok_counts = []
    BATCH = 256
    for i in range(0, len(lines), BATCH):
        batch = lines[i: i + BATCH]
        tokens_per_line = client.tokenize(batch, model=MODEL)
        line_tok_counts.extend(len(t) for t in tokens_per_line)

    chunks: list[Chunk] = []
    buf_lines: list[str] = []
    buf_tok = 0
    chunk_idx = 0
    for line, n in zip(lines, line_tok_counts):
        if buf_tok + n > target_tokens and buf_lines:
            chunks.append(Chunk(file_rel, chunk_idx, "".join(buf_lines)))
            chunk_idx += 1
            buf_lines, buf_tok = [], 0
        buf_lines.append(line)
        buf_tok += n
    if buf_lines:
        chunks.append(Chunk(file_rel, chunk_idx, "".join(buf_lines)))
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Cache key
# ─────────────────────────────────────────────────────────────────────

def _scope_signature(repo_path: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
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
# Voyage client + retries
# ─────────────────────────────────────────────────────────────────────

def _client(allow_no_key: bool = False):
    load_dotenv()
    import voyageai
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        if not allow_no_key:
            raise RuntimeError("VOYAGE_API_KEY not set")
        api_key = "dry-run-placeholder"
    return voyageai.Client(api_key=api_key, max_retries=0, timeout=120)


def _embed_with_backoff(client, texts: list[str], input_type: str) -> np.ndarray:
    delay = 1.0
    for attempt in range(8):
        try:
            r = client.embed(texts=texts, model=MODEL, input_type=input_type,
                             output_dtype="float", truncation=True)
            return np.asarray(r.embeddings, dtype=np.float32)
        except Exception as e:
            msg = str(e).lower()
            transient = any(s in msg for s in ("rate", "timeout", "429", "503", "502"))
            if not transient or attempt == 7:
                raise
            jitter = random.uniform(0, 0.5)
            log.warning("Voyage transient err (try %d): %s; sleep %.1fs",
                        attempt + 1, e, delay + jitter)
            time.sleep(delay + jitter)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("backoff exhausted")


# ─────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────

class VoyageRetriever:
    name = "voyage_code_3"

    def __init__(self) -> None:
        self.client = _client()
        self._reset()

    def _reset(self) -> None:
        self.index_obj = None
        self.repo_path: Path | None = None
        self.chunks: list[Chunk] = []
        self.chunk_files: list[str] = []  # parallel to FAISS rows

    def index(self, repo_path: Path, files: list[Path]) -> None:
        self._reset()
        self.repo_path = repo_path
        sig = _scope_signature(repo_path, files)
        cache_file = CACHE_DIR / f"{repo_path.name}_{sig}.npz"
        meta_file = CACHE_DIR / f"{repo_path.name}_{sig}.json"

        if cache_file.exists() and meta_file.exists():
            log.info("Cache hit %s", cache_file.name)
            data = np.load(cache_file)
            embeds = data["embeds"]
            meta = json.loads(meta_file.read_text())
            self.chunks = [Chunk(**c) for c in meta["chunks"]]
            self.chunk_files = [c.file_rel for c in self.chunks]
            self._build_faiss(embeds)
            log.info("Loaded %d chunks from cache", len(self.chunks))
            return

        log.info("Indexing %s: %d files", repo_path.name, len(files))
        all_chunks: list[Chunk] = []
        for f in tqdm(files, desc=f"chunk {repo_path.name}", leave=False):
            try:
                content = f.read_text(errors="replace")
            except OSError:
                continue
            rel = f.relative_to(repo_path).as_posix()
            all_chunks.extend(_split_into_chunks(self.client, rel, content))

        log.info("Total chunks: %d", len(all_chunks))
        embeds = self._embed_chunks(all_chunks)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, embeds=embeds)
        meta_file.write_text(json.dumps({
            "chunks": [c.__dict__ for c in all_chunks],
            "model": MODEL,
            "n_files": len(files),
        }))
        log.info("Cached to %s", cache_file.name)

        self.chunks = all_chunks
        self.chunk_files = [c.file_rel for c in all_chunks]
        self._build_faiss(embeds)

    def _embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, 1024), dtype=np.float32)

        # Pre-tokenize to get exact per-chunk token counts (single fast pass
        # via local HF tokenizer). Avoids batch overflow vs the 120K API cap.
        texts = [c.header_text() for c in chunks]
        TOKBATCH = 256
        tok_counts: list[int] = []
        for i in range(0, len(texts), TOKBATCH):
            tok_counts.extend(
                len(t) for t in self.client.tokenize(texts[i:i + TOKBATCH], model=MODEL)
            )

        out: list[np.ndarray] = []
        i = 0
        pbar = tqdm(total=len(chunks), desc="embed chunks", leave=False)
        while i < len(chunks):
            j = i
            tok_budget = 0
            while j < len(chunks) and (j - i) < EMBED_BATCH_TEXTS:
                if tok_budget + tok_counts[j] > EMBED_BATCH_TOKENS and j > i:
                    break
                tok_budget += tok_counts[j]
                j += 1
            batch_texts = texts[i:j]
            embs = _embed_with_backoff(self.client, batch_texts,
                                       input_type="document")
            out.append(embs)
            pbar.update(len(batch_texts))
            i = j
            if i < len(chunks):
                time.sleep(INTER_BATCH_SLEEP)
        pbar.close()
        return np.vstack(out).astype(np.float32)

    def _build_faiss(self, embeds: np.ndarray) -> None:
        if embeds.shape[0] == 0:
            self.index_obj = None
            return
        faiss.normalize_L2(embeds)
        idx = faiss.IndexFlatIP(embeds.shape[1])
        idx.add(embeds)
        self.index_obj = idx
        self._embed_dim = embeds.shape[1]

    def search(self, query: str, top_k: int = 20) -> list[tuple[Path, float]]:
        if self.index_obj is None or not self.chunks or self.repo_path is None:
            return []
        q = _embed_with_backoff(self.client, [query], input_type="query")
        faiss.normalize_L2(q)
        # Over-fetch chunk-level results so dedup-to-file yields enough files
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

    # ─────────────────────────────────────────────────────────
    # Dry-run (cost estimate)
    # ─────────────────────────────────────────────────────────

    @classmethod
    def dry_run(cls, instances) -> None:
        """Cost estimate. Avoids 320 git checkouts: estimates each repo once
        at its first observed base_commit, then multiplies by scope count.
        Real cost will scale with this since per-commit file deltas are small.
        """
        from collections import Counter
        from src.eval import ensure_repo_at
        from src.utils import list_eligible_files

        client = _client(allow_no_key=True)

        # Group: scopes per repo
        first_commit: dict[str, str] = {}
        scope_count: Counter = Counter()
        for inst in instances:
            scope_count[(inst.repo, inst.base_commit)] += 1
        scopes_per_repo: Counter = Counter()
        for (repo, commit), _ in scope_count.items():
            scopes_per_repo[repo] += 1
            first_commit.setdefault(repo, commit)

        log.info("Dry-run: %d repos, %d unique scopes, %d instances",
                 len(scopes_per_repo), sum(scopes_per_repo.values()),
                 len(instances))

        total_doc_tokens = 0
        per_repo_rows = []
        for repo_slug, n_scopes in tqdm(scopes_per_repo.items(),
                                        desc="repos", total=len(scopes_per_repo)):
            commit = first_commit[repo_slug]
            repo_path = ensure_repo_at(repo_slug, commit)
            files = list_eligible_files(repo_path)
            single_scope_tokens = 0
            n_chunks = 0
            for f in files:
                try:
                    content = f.read_text(errors="replace")
                except OSError:
                    continue
                if not content.strip():
                    continue
                try:
                    n = client.count_tokens([content], model=MODEL)
                except Exception as e:
                    log.warning("count_tokens failed on %s: %s", f.name, e)
                    n = max(1, len(content) // 3)
                est_chunks = max(1, (n + CHUNK_TOKENS - 1) // CHUNK_TOKENS)
                single_scope_tokens += n + est_chunks * 30
                n_chunks += est_chunks
            scope_total = single_scope_tokens * n_scopes
            per_repo_rows.append((repo_slug, n_scopes, len(files), n_chunks,
                                  single_scope_tokens, scope_total))
            total_doc_tokens += scope_total

        # Query embeddings (problem_statement per instance)
        query_tokens = 0
        for inst in instances:
            try:
                query_tokens += client.count_tokens([inst.problem_statement],
                                                   model=MODEL)
            except Exception:
                query_tokens += max(1, len(inst.problem_statement) // 3)

        # Per-repo indexing: each repo embedded once. Real doc cost = sum of
        # `tok/scope` (one indexing per repo, ignore scope count).
        doc_tokens_once = sum(r[4] for r in per_repo_rows)
        total_tokens = doc_tokens_once + query_tokens
        cost_usd = total_tokens / 1_000_000 * PRICE_PER_M

        print()
        print("=" * 90)
        print(f"Voyage dry-run ({MODEL}) — per-repo indexing")
        print("=" * 90)
        print(f"{'repo':<32} {'scopes':>7} {'files':>7} {'chunks':>9} {'tokens/repo':>14}")
        print("-" * 90)
        for repo, ns, nf, nc, tps, _tot in per_repo_rows:
            print(f"{repo:<32} {ns:>7} {nf:>7} {nc:>9} {tps:>14,}")
        print("-" * 90)
        print(f"Doc tokens (one index per repo): {doc_tokens_once:>15,}")
        print(f"Query tokens ({len(instances)}x):           {query_tokens:>15,}")
        print(f"Total tokens:                    {total_tokens:>15,}")
        print(f"Estimated cost:                  ${cost_usd:>14,.2f}")
        print("=" * 90)
