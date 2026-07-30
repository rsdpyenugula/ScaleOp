#!/usr/bin/env bash
# ScaleOp <-> DGX Spark helper.
# The Mac is the git source-of-truth; the Spark (GB10, 128GB, CUDA) is the compute
# backend. This script mirrors code up, runs with uv on the Spark, and pulls
# generated reports/results back for committing.
#
# Usage:
#   ./spark.sh push                 # rsync code Mac -> Spark (no data/results/.venv)
#   ./spark.sh pull                 # rsync reports+results Spark -> Mac
#   ./spark.sh setup                # create uv venv on Spark, install torch(cu130)+deps
#   ./spark.sh run <cmd...>         # push, then run `uv run <cmd>` in the Spark project dir
#   ./spark.sh ssh <cmd...>         # raw ssh command in the Spark project dir
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/.spark.env"   # personal host settings (gitignored)
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH=(ssh -o BatchMode=yes -i "$SPARK_KEY" "$SPARK_HOST")
RSH="ssh -o BatchMode=yes -i \"$SPARK_KEY\""

# Non-interactive ssh doesn't source the login profile, so uv (~/.local/bin) is
# not on PATH. Prepend it for every remote command.
REMOTE_PATH='export PATH="$HOME/.local/bin:$PATH"'

_remote() { "${SSH[@]}" "$REMOTE_PATH; cd '$REMOTE_DIR' && $*"; }

push() {
  "${SSH[@]}" "mkdir -p '$REMOTE_DIR'"
  rsync -az --delete -e "ssh -o BatchMode=yes -i \"$SPARK_KEY\"" \
    --exclude '.git/' --exclude '.venv/' --exclude 'data/' \
    --exclude 'project_docs/results/' --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude '.DS_Store' --exclude '.pytest_cache/' \
    "$LOCAL_DIR"/ "$SPARK_HOST:$REMOTE_DIR"/
  echo "pushed -> $SPARK_HOST:$REMOTE_DIR"
}

pull() {
  # uv.lock is generated on the Spark by `uv sync`; bring it back so it lives on
  # the Mac source-of-truth (and survives the --delete on future pushes).
  rsync -az -e "ssh -o BatchMode=yes -i \"$SPARK_KEY\"" \
    "$SPARK_HOST:$REMOTE_DIR/uv.lock" "$LOCAL_DIR/uv.lock" 2>/dev/null || true
  # Bring back generated artifacts (reports are committed; results are gitignored
  # but useful to inspect locally).
  rsync -az -e "ssh -o BatchMode=yes -i \"$SPARK_KEY\"" \
    "$SPARK_HOST:$REMOTE_DIR/project_docs/reports"/ "$LOCAL_DIR/project_docs/reports"/
  rsync -az -e "ssh -o BatchMode=yes -i \"$SPARK_KEY\"" \
    "$SPARK_HOST:$REMOTE_DIR/project_docs/results"/ "$LOCAL_DIR/project_docs/results"/ 2>/dev/null || true
  echo "pulled reports+results <- $SPARK_HOST"
}

setup() {
  push
  # torch (cu130) is pinned in pyproject via [tool.uv.sources]; one sync does it all.
  _remote "UV_HTTP_TIMEOUT=900 uv sync --extra dev && \
    uv run python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
}

case "${1:-}" in
  push)  push ;;
  pull)  pull ;;
  setup) setup ;;
  run)   shift; push; _remote "uv run $*"; ;;
  ssh)   shift; _remote "$*"; ;;
  *) echo "usage: ./spark.sh {push|pull|setup|run <cmd>|ssh <cmd>}"; exit 1 ;;
esac
