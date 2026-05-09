#!/usr/bin/env bash
# Pull results + log back from a Lambda instance, then remind to terminate.
#
# Usage (from local Mac, NOT inside the cloud instance):
#   bash scripts/cloud_teardown.sh ubuntu@<INSTANCE_IP>
#
# Optional env:
#   REMOTE_PATH   — repo path on instance (default: ~/deepvector-experiment)
#   SSH_KEY       — path to identity file (default: ~/.ssh/id_ed25519)

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 user@host" >&2
    exit 2
fi

REMOTE="$1"
REMOTE_PATH="${REMOTE_PATH:-~/deepvector-experiment}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[teardown] pulling results from $REMOTE:$REMOTE_PATH/data/results/"
mkdir -p "$ROOT/data/results"
scp -i "$SSH_KEY" -p \
    "$REMOTE:$REMOTE_PATH/data/results/mamba_codestral_baseline.json" \
    "$REMOTE:$REMOTE_PATH/data/results/dtype_sanity_codestral.json" \
    "$ROOT/data/results/" || echo "[teardown] WARN: one or more result files missing"

echo "[teardown] pulling cloud_run log..."
mkdir -p "$ROOT/data/logs"
scp -i "$SSH_KEY" -p \
    "$REMOTE:/tmp/cloud_run.log" \
    "$ROOT/data/logs/cloud_run_$(date -u +%Y%m%d_%H%M%S).log" \
    || echo "[teardown] WARN: log file missing"

echo
echo "[teardown] DONE. Files now in:"
ls -la "$ROOT/data/results/" 2>/dev/null | tail -5
echo
echo "Next steps:"
echo "  1. git add data/results/*.json && git commit -m 'Codestral baseline results' && git push"
echo "  2. Terminate the Lambda instance from the dashboard ($REMOTE)"
