#!/usr/bin/env bash
# Cloud H100 runner for the Codestral-Mamba-7B mean-pooled baseline.
#
# Usage on a fresh Lambda/RunPod H100 instance with Ubuntu 22.04/24.04 + CUDA 12+:
#
#   git clone git@github.com:sebastiangrebe/deepvector-experiment.git
#   cd deepvector-experiment
#   export HF_TOKEN=hf_...        # required (Codestral may be gated)
#   export GIT_PUSH=1             # optional — push results JSON back via git
#   export GH_TOKEN=github_pat_... # required if GIT_PUSH=1 (HTTPS push token)
#   bash scripts/cloud_run.sh 2>&1 | tee /tmp/cloud_run.log
#
# Dry-run (no installs, no eval):
#   bash scripts/cloud_run.sh --dry-run
#
# Hard-fails fast on any unrecoverable issue. NO silent fallbacks: if the fast-path
# kernels can't be built, the run aborts (pure-PyTorch is OOM at 1500-tok chunks).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

phase() { printf "=== Phase %s ===\n" "$*"; }
ok()    { printf "    ✓ %s\n" "$*"; }
fail()  { printf "    ✗ %s\n" "$*" >&2; exit 1; }
note()  { printf "    • %s\n" "$*"; }

echo "=== Cloud run start: $(date -u) ==="
[ "$DRY_RUN" = "1" ] && echo "    (DRY RUN — no installs, no eval)"

# ─────────────────────────────────────────────────────────────────────
# Phase 1/7: Diagnostics — CUDA, GPU, env vars
# ─────────────────────────────────────────────────────────────────────
phase "1/7: Diagnostics"

if ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc not found. cloud_run.sh requires CUDA 12+. Aborting."
fi

CUDA_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
if [ -z "$CUDA_VER" ]; then
    fail "Could not parse CUDA version from nvcc."
fi

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
if [ "$CUDA_MAJOR" -lt 12 ]; then
    fail "CUDA $CUDA_VER detected; mamba-ssm requires CUDA 12+. Aborting."
fi

case "${CUDA_MAJOR}.${CUDA_MINOR}" in
    12.1) PYTORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    12.4) PYTORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    12.6) PYTORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    12.8) PYTORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    13.*) PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
          note "WARN: CUDA $CUDA_VER newer than pinned mamba-ssm targets. Build may fail." ;;
    *)
        # Map any 12.x to closest known wheel
        if [ "$CUDA_MINOR" -le 1 ]; then PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
        elif [ "$CUDA_MINOR" -le 4 ]; then PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
        elif [ "$CUDA_MINOR" -le 6 ]; then PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
        else PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
        fi
        note "CUDA $CUDA_VER unmapped; using nearest wheel $PYTORCH_INDEX"
        ;;
esac
ok "CUDA $CUDA_VER detected → wheel index $PYTORCH_INDEX"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "unknown")
    ok "GPU: $GPU_NAME"
else
    fail "nvidia-smi not found"
fi

if [ "$DRY_RUN" = "1" ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        note "WARN: HF_TOKEN not set (Codestral may require it)"
    else
        ok "HF_TOKEN set"
    fi
    if [ "${GIT_PUSH:-0}" = "1" ] && [ -z "${GH_TOKEN:-}" ]; then
        note "WARN: GIT_PUSH=1 but GH_TOKEN not set; push will be skipped"
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 2/7: Python venv + PyTorch matched to system CUDA
# ─────────────────────────────────────────────────────────────────────
phase "2/7: Python env + PyTorch (~3 min on fresh box)"

if [ "$DRY_RUN" = "1" ]; then
    note "would: install uv if missing, uv venv .venv, uv pip install torch from $PYTORCH_INDEX"
    note "would: verify torch.cuda.is_available()"
else
    if ! command -v uv >/dev/null 2>&1; then
        note "installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ ! -d .venv ]; then
        uv venv --python 3.11 .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate

    uv pip uninstall torch torchvision torchaudio 2>/dev/null || true
    uv pip install --index-url "$PYTORCH_INDEX" torch

    .venv/bin/python -c "
import torch, sys
print(f'    torch={torch.__version__}  cuda_built_for={torch.version.cuda}  cuda_avail={torch.cuda.is_available()}')
sys.exit(0 if torch.cuda.is_available() else 1)
" || fail "torch.cuda.is_available() == False after install"
    ok "PyTorch installed and sees CUDA"

    uv pip install -r requirements.txt
    ok "requirements.txt installed"
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 3/7: Build/verify causal-conv1d
# ─────────────────────────────────────────────────────────────────────
phase "3/7: causal-conv1d (~5-10 min on fresh box)"

KERNELS_OK=0
if [ "$DRY_RUN" = "1" ]; then
    note "would: check importability, else uv pip install --no-build-isolation causal-conv1d"
elif .venv/bin/python -c "from causal_conv1d import causal_conv1d_fn" 2>/dev/null; then
    ok "causal-conv1d already importable, skipping build"
else
    note "building causal-conv1d..."
    uv pip install --no-build-isolation causal-conv1d \
        || fail "causal-conv1d build failed. Pure-PyTorch fallback OOMs at 1500-tok chunks. Aborting."
    .venv/bin/python -c "from causal_conv1d import causal_conv1d_fn" \
        || fail "causal-conv1d installed but import failed"
    ok "causal-conv1d built"
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 4/7: Build/verify mamba-ssm
# ─────────────────────────────────────────────────────────────────────
phase "4/7: mamba-ssm (~10-20 min on fresh box)"

if [ "$DRY_RUN" = "1" ]; then
    note "would: check importability, else uv pip install --no-build-isolation mamba-ssm"
elif .venv/bin/python -c "from mamba_ssm.ops.triton.selective_state_update import selective_state_update" 2>/dev/null; then
    ok "mamba-ssm already importable, skipping build"
else
    note "building mamba-ssm..."
    uv pip install --no-build-isolation mamba-ssm \
        || fail "mamba-ssm build failed. Pure-PyTorch fallback OOMs at 1500-tok chunks. Aborting."
    .venv/bin/python -c "from mamba_ssm.ops.triton.selective_state_update import selective_state_update" \
        || fail "mamba-ssm installed but import failed"
    ok "mamba-ssm built"
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 5/7: Kernel import sanity
# ─────────────────────────────────────────────────────────────────────
phase "5/7: Kernel import sanity"

if [ "$DRY_RUN" = "1" ]; then
    note "would: import both kernels, abort if either fails"
else
    .venv/bin/python -c "
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
print('    fast-path kernels OK')
" || fail "Kernel import sanity check failed"
    KERNELS_OK=1
    ok "fast-path kernels verified"
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 6/7: Dataset + repos
# ─────────────────────────────────────────────────────────────────────
phase "6/7: SWE-Bench Lite + repos (~5 min)"

if [ "$DRY_RUN" = "1" ]; then
    note "would: load_dataset princeton-nlp/SWE-bench_Lite → data/swe_bench_lite.json"
    note "would: pre-clone all 18 unique repos at first base_commit"
else
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
print(f'    cached {len(all_rows)} instances')
"
    ok "SWE-Bench Lite cached"
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 7/7: Eval (Codestral) + dtype sanity
# ─────────────────────────────────────────────────────────────────────
phase "7/7: Codestral-Mamba-7B eval (~2-2.5h on H100)"

if [ "$DRY_RUN" = "1" ]; then
    note "would: python -m src.eval --retriever mamba_pooled \\"
    note "          --model-id mistralai/Mamba-Codestral-7B-v0.1 --dtype bfloat16 \\"
    note "          --top-k 20 --out data/results/mamba_codestral_baseline.json"
    note "would: grep eval log for 'Loading model ... on cuda' (abort if cpu/mps)"
    note "would: python scripts/dtype_sanity.py for fp32 vs bf16/fp16 drift"
    echo
    echo "=== DRY RUN COMPLETE — environment looks fit for cloud run ==="
    exit 0
fi

EVAL_LOG=/tmp/codestral_eval.log
.venv/bin/python -m src.eval --retriever mamba_pooled \
    --model-id mistralai/Mamba-Codestral-7B-v0.1 \
    --dtype bfloat16 --top-k 20 \
    --out data/results/mamba_codestral_baseline.json 2>&1 | tee "$EVAL_LOG" &
EVAL_PID=$!

# Quick device-check after model loads (poll for the encoder log line)
DEVICE_OK=0
for _ in $(seq 1 60); do
    if grep -q "Loading model.*on cuda" "$EVAL_LOG" 2>/dev/null; then
        DEVICE_OK=1; break
    fi
    if grep -qE "Loading model.*on (cpu|mps)" "$EVAL_LOG" 2>/dev/null; then
        kill -9 "$EVAL_PID" 2>/dev/null || true
        fail "Encoder loaded on CPU/MPS, not CUDA. get_device() regression?"
    fi
    sleep 30
done
[ "$DEVICE_OK" = "1" ] && ok "encoder confirmed on CUDA"

wait "$EVAL_PID"
EVAL_EXIT=$?
[ "$EVAL_EXIT" -eq 0 ] || fail "Eval exited with code $EVAL_EXIT"
ok "Codestral eval finished"

note "running dtype sanity (~5 min)..."
.venv/bin/python scripts/dtype_sanity.py \
    --model-id mistralai/Mamba-Codestral-7B-v0.1 \
    --n-chunks 100 \
    --out data/results/dtype_sanity_codestral.json
ok "dtype sanity finished"

# ─────────────────────────────────────────────────────────────────────
# Push results back (token-based HTTPS only)
# ─────────────────────────────────────────────────────────────────────
phase "Results upload"

git add data/results/mamba_codestral_baseline.json \
        data/results/dtype_sanity_codestral.json
git -c user.email="cloud-run@deepvector.local" \
    -c user.name="cloud-run" \
    commit -m "Codestral-Mamba-7B baseline (cloud H100, CUDA $CUDA_VER)" || true

if [ "${GIT_PUSH:-0}" = "1" ]; then
    if [ -z "${GH_TOKEN:-}" ]; then
        note "GIT_PUSH=1 but GH_TOKEN not set. Skipping push."
        note "scp from instance manually:"
        note "  scp ubuntu@<INSTANCE_IP>:~/deepvector-experiment/data/results/*.json data/results/"
    else
        git remote set-url origin \
            "https://${GH_TOKEN}@github.com/sebastiangrebe/deepvector-experiment.git"
        git push origin HEAD:main \
            && ok "pushed results to origin/main" \
            || note "push failed; commit is local. scp results manually."
    fi
else
    note "GIT_PUSH not set. Results committed locally. scp from instance manually."
fi

echo "=== Cloud run done: $(date -u) ==="
