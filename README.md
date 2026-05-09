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

## Cloud Run (Codestral-Mamba-7B baseline on H100)

The full Codestral baseline does not fit M5 Max budgets (~17h on MPS) and the
pure-PyTorch Mamba-2 path OOMs at 1500-token chunks (it materializes a
`(B, T, T, ...)` intermediate). The fast-path CUDA kernels (`causal-conv1d`
+ `mamba-ssm`) are required.

### Provision

1. Lambda Cloud → spin up `gpu_1x_h100_pcie` ($2.49/h). Ubuntu 22.04 preferred,
   24.04 acceptable. CUDA 12.1/12.4/12.6/12.8 supported. CUDA 13+ may break
   `mamba-ssm` builds.
2. Add your `~/.ssh/id_ed25519.pub` to Lambda's "SSH Keys" page.
3. SSH in once the instance is up.

### Verify environment (dry-run)

```bash
git clone git@github.com:sebastiangrebe/deepvector-experiment.git
cd deepvector-experiment
bash scripts/cloud_run.sh --dry-run
```

Confirms CUDA version, GPU model, env vars. **Does not** install or run the
eval. Catches misconfigured boxes in <30 seconds.

### Full run

```bash
export HF_TOKEN=hf_...        # required (Codestral may be gated)
export GIT_PUSH=1             # optional — push results JSON via git
export GH_TOKEN=github_pat_... # required when GIT_PUSH=1 (HTTPS push token)
nohup bash scripts/cloud_run.sh > /tmp/cloud_run.log 2>&1 &
tail -f /tmp/cloud_run.log
```

Phases (each prints `=== Phase N/7 ===`): diagnostics → PyTorch → causal-conv1d
→ mamba-ssm → kernel sanity → dataset/repos → eval + dtype sanity.

Hard-fails on any unrecoverable issue. No silent fallbacks.

### Total time + cost

```
Phase 1 (diagnostics)       <30 s
Phase 2 (PyTorch install)   ~3 min
Phase 3 (causal-conv1d)     ~5-10 min     (skipped on rerun if importable)
Phase 4 (mamba-ssm)         ~10-20 min    (skipped on rerun if importable)
Phase 5 (kernel sanity)     <30 s
Phase 6 (dataset + repos)   ~5 min
Phase 7 (eval + dtype)      ~2-2.5 h

Total: ~2.5-3 h on a fresh box; $6-8 at $2.49/h
```

### Pull results + tear down

From local Mac:

```bash
bash scripts/cloud_teardown.sh ubuntu@<INSTANCE_IP>
```

Pulls `data/results/mamba_codestral_baseline.json`,
`data/results/dtype_sanity_codestral.json`,
`data/results/maxsim_discrimination_test.json` (if present),
and the `cloud_run.log` snapshot.
Then **terminate** the instance from the Lambda dashboard.

### MaxSim discrimination test (Phase 2.5)

After the Codestral baseline JSON is in `data/results/`, run the focused MaxSim
test on the same H100 (env already prepared by `cloud_run.sh`):

```bash
.venv/bin/python scripts/maxsim_test.py \
    --model-id mistralai/Mamba-Codestral-7B-v0.1 \
    --dtype bfloat16 \
    --max-test-instances 80 \
    --top-k 20 \
    --budget-hours 4
```

`--strict-sanity` is auto-enabled when the encoder matches the cached
baseline's encoder. Aborts (exit 2) if Method A pooled rankings diverge from
the cached Codestral baseline by >50% on the first 5 instances — that means
a harness bug, not a meaningful result.

Output: `data/results/maxsim_discrimination_test.json`. Token-level cache
(~117 GB) lives in `data/maxsim_cache/`, gitignored, **not transferred back**.

Estimated cost: ~3 h, ~$7.50 at $2.49/h.
