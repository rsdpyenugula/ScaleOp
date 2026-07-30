#!/usr/bin/env bash
# One-shot setup on any CUDA Linux box (AWS g5/g6/p4, GCP, Lambda, ...).
# No Spark required — run this ON the GPU instance after cloning the repo:
#   git clone <repo> && cd ScaleOp && bash tools/cloud_setup.sh [--bootstrap]
#
# --bootstrap additionally prepares the experiment prerequisites:
#   model downloads (~7 GB), frozen eval corpus, activation caches (~19 GB).
# Recommended instance: 24 GB+ GPU (g5.2xlarge / g6.2xlarge), 150 GB disk.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

UV_HTTP_TIMEOUT=900 uv sync --extra dev --extra eval
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'; \
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
uv run pytest -q tests/test_models.py tests/test_alignment.py tests/test_assemble.py tests/test_subclone.py

if [ "${1:-}" = "--bootstrap" ]; then
  uv run python tools/download_models.py
  uv run python tools/build_eval_tokens.py
  uv run python tools/capture_activations.py
fi
echo "setup complete — see README 'Run on AWS' for the experiment commands"
