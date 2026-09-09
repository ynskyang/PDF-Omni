#!/usr/bin/env bash
# Evaluate a checkpoint on one dataset or on all five synthetic benchmarks.
# Usage: bash scripts/eval.sh [checkpoint] [dataset|all] [extra eval.py args...]
# Example: bash scripts/eval.sh checkpoints/pdfomni_finetune.pth all --save_result
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/pdfomni_finetune.pth}"
DB="${2:-all}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true
[ -s "$CKPT" ] || { echo "checkpoint not found: $CKPT (run scripts/download_checkpoints.sh)" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG="results/eval_$(basename "${CKPT%.pth}")_${DB}.log"
mkdir -p results
python eval.py --dbname "$DB" --restore_ckpt "$CKPT" "$@" 2>&1 | tee "$LOG"
echo "[eval] summary ($LOG):"
grep -oE "Result for [a-z]+:.*" "$LOG" || true
