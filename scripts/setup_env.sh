#!/usr/bin/env bash
# Create the conda environment described in the README and install the dependencies.
# Usage: bash scripts/setup_env.sh [env_name] [cuda_tag]
#   env_name  conda environment name (default: pdf_omni)
#   cuda_tag  PyTorch wheel index tag (default: cu128)
set -euo pipefail
ENV_NAME="${1:-pdf_omni}"
CUDA_TAG="${2:-cu128}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] environment '$ENV_NAME' already exists, reusing it"
else
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

pip install torch==2.11.0 torchvision==0.26.0 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -r "$(dirname "$0")/../requirements.txt"

python - <<'PY'
import torch
print("[setup] torch", torch.__version__, "cuda available:", torch.cuda.is_available())
PY
echo "[setup] done. Activate with: conda activate $ENV_NAME"
