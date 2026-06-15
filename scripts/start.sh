#!/usr/bin/env bash
# ============================================================
# FraudSentinel — launch script
#
#  Terminal 1:  bash scripts/start.sh vllm
#  Terminal 2:  bash scripts/start.sh app
#
# Access:
#   https://notebooks.amd.com/<pod>/proxy/7860/
#
# Pod name is in your Jupyter URL:
#   https://notebooks.amd.com/jupyter-hack-team-121-260610030428-51ed04de/lab
#   pod = jupyter-hack-team-121-260610030428-51ed04de
# ============================================================

MODE=${1:-app}
_HF_REPO="naazimsnh02/fraudsentinel-qwen3-14b-merged"
_HF_CACHE=~/.cache/huggingface/hub/models--naazimsnh02--fraudsentinel-qwen3-14b-merged/snapshots/7caf75818541f7fa95eabf5815d27dbd46dc21b3
# Use local cache if present (faster, no network); fall back to HF repo ID for first-time setup
if [ -n "$FRAUDSENTINEL_MODEL" ]; then
  MODEL="$FRAUDSENTINEL_MODEL"
elif [ -d "$_HF_CACHE" ]; then
  MODEL="$_HF_CACHE"
else
  MODEL="$_HF_REPO"
fi

case "$MODE" in

  vllm)
    echo "==> Starting vLLM on :8000"
    echo "    Model: $MODEL"
    echo "    Wait for 'Application startup complete.' (~2 min first run)"
    echo ""
    vllm serve "$MODEL" \
      --dtype bfloat16 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 \
      --host 0.0.0.0 \
      --port 8000 \
      --disable-log-requests
    ;;

  app)
    echo "==> Starting FraudSentinel app on :7860"
    echo ""
    echo "    URL: https://notebooks.amd.com/<your-pod>/proxy/7860/"
    echo ""
    cd "$(dirname "$0")/../backend"
    python -m uvicorn app:app \
      --host 0.0.0.0 \
      --port 7860 \
      --reload \
      --log-level info
    ;;

  *)
    echo "Usage: bash scripts/start.sh [vllm|app]"
    exit 1
    ;;

esac
