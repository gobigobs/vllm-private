#!/usr/bin/env bash
# Launch vLLM's OpenAI-compatible server — GPU host, private network only.
#   bash serve/vllm_serve.sh
#
# Requirements: NVIDIA GPU with >= 8 GB VRAM (12 GB+ recommended),
# CUDA driver, and:  pip install vllm
# Model weights (~6 GB bf16) download from Hugging Face on first run.
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
HOST="${VLLM_HOST:-127.0.0.1}"          # loopback only — never 0.0.0.0
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-}"            # set from env file, never in git
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"

if [ -z "$API_KEY" ]; then
  echo "ERROR: VLLM_API_KEY is not set. Generate one and export it, e.g.:" >&2
  echo "  export VLLM_API_KEY=\"\$(openssl rand -hex 24)\"" >&2
  exit 1
fi

# Optional: Hugging Face token for gated models (not needed for Qwen2.5).
# export HF_TOKEN=...

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-prefix-caching \
  --log-requests
