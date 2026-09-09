#!/usr/bin/env bash
# Pre-train PDF-Omni on OmniThings (30 epochs, single GPU). Extra arguments are passed to train.py.
# Usage: bash scripts/train_omnithings.sh [--db_root ../omnidata] [--name my_run] ...
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python train.py --dbname omnithings \
    --base_channel 32 --mixed_precision --ot_iter 3 --lec_weight 0.1 \
    --use_ge --use_digru --ge_scale 1 --total_epochs 30 "$@"
