"""SSM encoder wrapper.

Loads a HuggingFace-compatible Mamba/Mamba-2 model and returns per-token hidden
states for downstream latent-state matching.

Defaults to state-spaces/mamba-130m-hf (Mamba-1, ~130M, fast) for smoke tests.
Recommended eval model: mistralai/Mamba-Codestral-7B-v0.1 (Mamba-2, code-trained).

NOTE: Original state-spaces/mamba2-* checkpoints are NOT loadable via transformers
(require mamba_ssm package, CUDA-only). See README for details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_device, get_logger

log = get_logger(__name__)


@dataclass
class EncoderOutput:
    last_hidden: torch.Tensor   # (batch, seqlen, d_model)
    pooled: torch.Tensor        # (batch, d_model) — mean over non-pad tokens
    attention_mask: torch.Tensor  # (batch, seqlen) — 1 for real tokens


class MambaEncoder:
    def __init__(
        self,
        model_id: str = "state-spaces/mamba-130m-hf",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        max_length: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device) if device else get_device()
        self.max_length = max_length

        log.info("Loading tokenizer %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"
        self.tokenizer.padding_side = "left"

        log.info("Loading model %s on %s (dtype=%s)", model_id, self.device, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

        self.d_model = int(self.model.config.hidden_size)
        log.info("Encoder ready: d_model=%d", self.d_model)

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> EncoderOutput:
        texts = list(texts)
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        out = self.model(
            **toks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]   # (B, T, D)
        mask = toks["attention_mask"].to(last_hidden.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (last_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        return EncoderOutput(
            last_hidden=last_hidden,
            pooled=pooled,
            attention_mask=toks["attention_mask"],
        )


def cross_attention_score(
    query_latents: torch.Tensor,    # (Tq, D) — single query
    doc_latents: List[torch.Tensor],  # list of (Td_i, D)
) -> torch.Tensor:
    """Compute a scalar relevance score per document via dot-product cross-attention.

    Score = mean over query tokens of max-pooled similarity against doc tokens.
    Late-interaction style (ColBERT-like). Inputs assumed already normalized OR
    of comparable magnitude; we L2-normalize internally.
    """
    qn = F.normalize(query_latents, dim=-1)  # (Tq, D)
    scores = []
    for d in doc_latents:
        dn = F.normalize(d, dim=-1)         # (Td, D)
        sim = qn @ dn.T                      # (Tq, Td)
        max_per_q = sim.max(dim=-1).values   # (Tq,)
        scores.append(max_per_q.mean())
    return torch.stack(scores)
