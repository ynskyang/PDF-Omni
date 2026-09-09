#!/usr/bin/env bash
# Fine-tune a pre-trained model on OmniHouse and Sunny (16 epochs, lr 1e-4).
# Usage: bash scripts/finetune.sh [pretrain_ckpt] [extra train.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/pdfomni_pretrain.pth}"
shift || true
[ -s "$CKPT" ] || { echo "pre-trained checkpoint not found: $CKPT (run scripts/download_checkpoints.sh)" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python train.py --dbname omnihouse sunny \
    --base_channel 32 --mixed_precision --ot_iter 3 --lec_weight 0.1 \
    --use_ge --use_digru --ge_scale 1 --total_epochs 16 --lr 0.0001 \
    --pretrain_ckpt "$CKPT" "$@"
