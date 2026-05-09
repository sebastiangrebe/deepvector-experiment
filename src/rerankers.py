"""Cross-encoder rerankers for the hybrid retrieval pipeline.

Each wrapper exposes:
    score_pairs(pairs: list[(query, passage)]) -> list[float]

Pairs are (issue_text, file_content[:max_doc_chars]). Truncation chars are
passed in by the caller (varies per reranker context length).

Three rerankers wired:
    bge_v2_m3            — BAAI/bge-reranker-v2-m3 (568M, multilingual, code-OK)
    jina_v2_multilingual — jinaai/jina-reranker-v2-base-multilingual (278M)
    minilm_l12           — cross-encoder/ms-marco-MiniLM-L-12-v2 (33M, weak baseline)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import get_device, get_logger

log = get_logger("rerankers")


# ─────────────────────────────────────────────────────────────────────
# Generic cross-encoder wrapper
# ─────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    name: str = "generic"
    max_length: int = 1024
    batch_size: int = 32

    def __init__(self, model_id: str, dtype: torch.dtype = torch.float32,
                 max_length: int | None = None,
                 batch_size: int | None = None) -> None:
        self.model_id = model_id
        self.device = get_device()
        log.info("Loading reranker %s on %s", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True)
        self.model.to(self.device).eval()
        if max_length is not None:
            self.max_length = max_length
        if batch_size is not None:
            self.batch_size = batch_size

    @torch.no_grad()
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i: i + self.batch_size]
            queries = [q for q, _ in batch]
            docs = [d for _, d in batch]
            enc = self.tokenizer(queries, docs, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # (B, 1) or (B,)
            scores = logits.squeeze(-1) if logits.ndim == 2 else logits
            out.extend(scores.float().cpu().tolist())
        return out


# ─────────────────────────────────────────────────────────────────────
# Concrete instances
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RerankerSpec:
    name: str
    model_id: str
    max_length: int
    batch_size: int


SPECS: list[RerankerSpec] = [
    # Ordered fastest → slowest. If budget cap hits mid-run, at least the
    # cheap rerankers finish and produce comparable results.
    RerankerSpec("minilm_l12",
                 "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 max_length=512, batch_size=64),
    RerankerSpec("jina_v2_multilingual",
                 "jinaai/jina-reranker-v2-base-multilingual",
                 max_length=1024, batch_size=32),
    RerankerSpec("bge_v2_m3",
                 "BAAI/bge-reranker-v2-m3",
                 max_length=1024, batch_size=16),
]


def build_reranker(spec: RerankerSpec, dtype: torch.dtype = torch.float32) -> CrossEncoderReranker:
    r = CrossEncoderReranker(spec.model_id, dtype=dtype,
                             max_length=spec.max_length,
                             batch_size=spec.batch_size)
    r.name = spec.name
    return r
