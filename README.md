# DeepVector Experiment

Research experiment: does SSM latent-state matching beat vector-search baselines for code retrieval?

## Goal

Test whether retaining full SSM per-token latents and matching them via cross-attention outperforms standard vector retrieval (mean-pooled embeddings + ANN) on code retrieval from SWE-Bench Lite.

## Encoder Decision (Step 4)

**Original plan**: Mamba-3 (ICLR 2026 paper).

**Pivot**: dropped Mamba-3 — no pretrained checkpoints exist (paper authors released kernels only, no weights), and `mamba_ssm.Mamba3` requires CUDA + Triton (won't run on Apple Silicon).

**Current backend**: HuggingFace-compatible Mamba (Mamba-1 / Mamba-2) via `transformers`.

- **Smoke test**: `state-spaces/mamba-130m-hf` (Mamba-1, ~130M, fast, MPS-compatible).
- **Real eval**: `mistralai/Mamba-Codestral-7B-v0.1` (Mamba-2, 7B, code-pretrained, HF-loadable). Note: original `state-spaces/mamba2-*` checkpoints are *not* HF-loadable — they need `mamba_ssm` (CUDA-only).
- **TODO**: swap to Mamba-3 once state-spaces releases pretrained weights.

`references/mamba3-minimal/` is kept for reference reading; consistency check (MIMO/SISO) passed on MPS at 1e-7.

## Success Criterion

**Layered Option A** (two-stage: cheap pooled retrieval → cross-attention re-rank over latents) must beat both baselines by **≥3 points Recall@10** at **≤2× latency** of the pooled baseline.

Baselines:
- Voyage embeddings + FAISS (industry SOTA dense retrieval).
- Mamba mean-pooled vectors + FAISS (controls for the encoder).

## Architecture (one paragraph)

Encode each repo file with a Mamba SSM and cache the per-token last-layer hidden states (a practical proxy for "SSM latent state"). At query time, encode the issue, run a fast Layer-1 pooled-vector ANN to fetch the top-K candidate files, then run Layer-2 late-interaction cross-attention (ColBERT-style: per-query-token max over doc tokens, then mean) between the query latents and candidate file latents to produce a fine-grained relevance score and re-rank. Hypothesis: per-token latents preserve positional/structural information lost in mean pooling, giving better recall on code where token-level matches matter more than topical similarity.

## Quickstart

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # fill in VOYAGE_API_KEY when needed
python scripts/smoke_test.py
```

## Layout

```
src/            # encoder, ingestion, matching, baselines, eval
data/           # repos, latent caches, dataset (gitignored)
references/    # mamba3-minimal clone (gitignored, reference only)
configs/        # experiment YAMLs
scripts/        # CLI entrypoints (smoke test, ingestion, eval)
notebooks/      # exploration
```
