#!/usr/bin/env bash
# Download the paper checkpoints from the GitHub release into checkpoints/.
# Usage: bash scripts/download_checkpoints.sh [release_tag]
set -euo pipefail
TAG="${1:-v1.0.0}"
BASE_URL="https://github.com/ynskyang/PDF-Omni/releases/download/${TAG}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
mkdir -p "$DEST"

for f in pdfomni_pretrain.pth pdfomni_finetune.pth; do
    if [ -s "$DEST/$f" ]; then
        echo "[download] $f already present, skipping"
        continue
    fi
    echo "[download] $f"
    curl -L --fail --progress-bar -o "$DEST/$f.part" "$BASE_URL/$f"
    mv "$DEST/$f.part" "$DEST/$f"
done
ls -lh "$DEST"
