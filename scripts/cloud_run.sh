#!/usr/bin/env bash
# Cloud H100 runner for the Codestral-Mamba-7B mean-pooled baseline.
#
# Usage on a fresh Lambda/RunPod H100 instance with Ubuntu 22.04 + CUDA 12.x:
#
#   git clone git@github.com:sebastiangrebe/deepvector-experiment.git
#   cd deepvector-experiment
#   export HF_TOKEN=hf_...        # required if Codestral is gated
#   export GIT_PUSH=1             # set to push results JSON back via git
#   git config user.email "you@example.com"
#   git config user.name  "Your Name"
#   bash scripts/cloud_run.sh 2>&1 | tee /tmp/cloud_run.log
#
# Exits non-zero on hard failure. Tries CUDA-kernel install (mamba_ssm +
# causal_conv1d) with a 15-minute build budget; falls back to pure PyTorch
# if either step fails.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "=== Cloud run start: $(date -u) ==="
nvidia-smi || { echo "no nvidia-smi; aborting"; exit 1; }

# 1) Python env + base deps ----------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2) Optional CUDA kernels (15-min budget) ------------------------------------

CUDA_OK=0
echo "[setup] attempting mamba_ssm + causal_conv1d (15-min budget)..."
if timeout 900 uv pip install --no-build-isolation \
        causal-conv1d>=1.4.0; then
  if MAMBA_FORCE_BUILD=TRUE timeout 900 uv pip install --no-build-isolation \
        mamba-ssm; then
    CUDA_OK=1
    echo "[setup] CUDA kernels OK"
  else
    echo "[setup] mamba_ssm build failed; falling back to pure PyTorch"
  fi
else
  echo "[setup] causal-conv1d build failed; falling back to pure PyTorch"
fi

# 3) Pre-clone all SWE-Bench Lite repos at first base_commit ------------------

echo "[setup] pre-fetching SWE-Bench Lite + repos..."
.venv/bin/python -c "
import json
from datasets import load_dataset
from pathlib import Path
ds = load_dataset('princeton-nlp/SWE-bench_Lite')
all_rows = []
for split, d in ds.items():
    for row in d:
        row['_split'] = split
        all_rows.append(row)
Path('data').mkdir(exist_ok=True)
Path('data/swe_bench_lite.json').write_text(json.dumps(all_rows))
print('cached', len(all_rows), 'instances')
"

# 4) Run the Codestral baseline ------------------------------------------------

echo "[run] starting eval $(date -u)..."
.venv/bin/python -m src.eval --retriever mamba_pooled \
    --model-id mistralai/Mamba-Codestral-7B-v0.1 \
    --dtype bfloat16 --top-k 20 \
    --out data/results/mamba_codestral_baseline.json

# 5) fp32 vs fp16/bf16 sanity check --------------------------------------------

echo "[run] dtype sanity..."
.venv/bin/python scripts/dtype_sanity.py \
    --model-id mistralai/Mamba-Codestral-7B-v0.1 \
    --n-chunks 100 \
    --out data/results/dtype_sanity_codestral.json

# 6) Push results back ---------------------------------------------------------

if [[ "${GIT_PUSH:-0}" == "1" ]]; then
  echo "[git] committing results..."
  git add data/results/mamba_codestral_baseline.json \
          data/results/dtype_sanity_codestral.json
  git commit -m "Codestral-Mamba-7B baseline (cloud H100, $(uname -m))

CUDA_KERNELS=$CUDA_OK

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
  git push origin main
fi

echo "=== Cloud run done: $(date -u) ==="
