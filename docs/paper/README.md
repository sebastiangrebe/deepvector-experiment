# `docs/paper/` — Research write-up

## What this paper covers

`deepvector_writeup.md` is the technical write-up of the DeepVector experiment: a four-phase study of how much retrieval signal lives in `mistralai/Mamba-Codestral-7B-v0.1`'s per-token last-layer activations on SWE-Bench Lite, and what frozen matching operations extract it.

Phases:

1. **Pooled baselines** — Voyage code-3 vs Codestral mean-pooled vectors. Establishes the gap (indexable Recall@10 of 0.95 vs 0.34) and rules out vector collapse as the cause.
2. **MaxSim discrimination** — On 80 instances where Voyage retrieves the gold and Codestral mean-pooling does not, ColBERT-style MaxSim over per-token Codestral latents recovers 35 (R@10 = 0.4375 vs 0.0125; 35× absolute-hit lift).
3. **Frozen ceiling** — Multi-head random-orthogonal-projection variants and a late-interaction normalization variant. None exceeds single-head MaxSim. Verdict: *CEILING* (Δ = −0.0125).
4. **Hybrid rerank** — `bge-reranker-v2-m3` and `cross-encoder/ms-marco-MiniLM-L-12-v2` on top of the MaxSim shortlist. Both underperformed the shortlist alone. Attributed narrowly to training-distribution mismatch.

## How to read

- **Quickest path to results**: read the abstract, then jump to §4 (Results). Each phase has one headline table and three labeled qualitative samples.
- **Methodology and reproducibility**: §3 describes the encoder, benchmark, indexing protocol, file filter, chunking, metrics, sanity checks, and seed/dtype choices.
- **Interpretation**: §5 (Discussion) is three points only — Codestral representation structure, why NL-trained rerankers fail on code, what this means for SSM-based retrieval research.
- **Caveats**: §6 (Limitations) is exhaustive (14 items). Every comparison in the paper sits under one or more of these caveats.
- **Next experiments**: §7 (Future Work) lists 5 items, each with the experiment, the gap it closes, the evidence required for closure, and a rough compute estimate.

## Data referenced

All numbers in the paper trace back to JSON files committed under `data/results/`:

- `voyage_baseline.json` — Voyage code-3, all 323 SWE-Bench Lite instances.
- `mamba_codestral_baseline.json` — Codestral mean-pooled, all 323 instances. Includes per-repo vector-quality stats.
- `frozen_ceiling_test.json` — Phase 2.6 (7 methods × 80 discriminating instances).
- `hybrid_rerank_test.json` — Phase 2.7 (MaxSim filter + 2 rerankers, 80 instances).
- `dtype_sanity_codestral.json` — fp32 vs bf16 vs fp16 cosine-drift check.

The Phase 2.5 standalone JSON was not preserved; the MaxSim numbers reported in §4.2 are reproduced from the Method 1 column of `frozen_ceiling_test.json` and the `maxsim_only` baseline of `hybrid_rerank_test.json`. See the §4.2 footnote.

## Reproducibility

All experiments run from scripts under `scripts/` with documented seeds:

- `scripts/maxsim_test.py` — Phase 2.5 (MaxSim discrimination test).
- `scripts/frozen_ceiling_test.py` — Phase 2.6 (frozen-ceiling comparison).
- `scripts/hybrid_rerank_test.py` — Phase 2.7 (hybrid MaxSim+reranker pipeline).
- `scripts/cloud_run.sh` — Lambda H100 provisioning + Codestral baseline run.
- `scripts/dtype_sanity.py` — fp32-vs-bf16/fp16 cosine-drift check.

Seeds: `random.seed(42)` for the discriminating-subset sample; `torch.Generator().manual_seed(42)` for the orthogonal-projection construction in `src/frozen_methods.py`. Encoder dtype: bf16 throughout (sanity check in §3.1). Hardware: single Lambda Cloud H100 PCIe (80 GB), `mamba_ssm` CUDA kernels + `causal-conv1d`. Cost per full run: ~$8 at $2.49/h.

## Status

Working draft / technical report. **Not peer-reviewed.** No claim to definitive treatment of the question. We treat the four findings as load-bearing for our own research planning and publish the write-up so that others can replicate, refute, or extend.

Contributions and corrections welcome via pull request. For substantive disagreements (e.g. an experiment that contradicts a claim), please open an issue with reproducible numbers attached.
