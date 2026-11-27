#!/usr/bin/env bash
# One-command demo (no GPU needed — Ollama fallback path).
#   ./run.sh
# Starts Ollama on 127.0.0.1:11434 if needed, waits for the model,
# runs the benchmark harness, prints the cost comparison.
set -euo pipefail
cd "$(dirname "$0")"

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/workspace/tools/ollama/dist/bin/ollama}"
MODEL="${OLLAMA_MODEL:-qwen2.5:3b-instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:11434/v1}"
PY="${PY:-python3}"

if ! curl -sf -m 2 "$BASE_URL/models" >/dev/null 2>&1; then
  echo "starting Ollama on 127.0.0.1:11434 (private, loopback only)..."
  OLLAMA_HOST=127.0.0.1:11434 nohup "$OLLAMA_BIN" serve \
    >/tmp/vllm-private-ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf -m 2 "$BASE_URL/models" >/dev/null 2>&1 && break
    sleep 1
  done
fi

if ! "$OLLAMA_BIN" list 2>/dev/null | grep -q "$MODEL"; then
  echo "pulling model $MODEL (~1.9 GB, one time only)..."
  "$OLLAMA_BIN" pull "$MODEL"
fi

# Warm-up: load the model once so the first benchmark request isn't the load.
curl -sf -m 120 "$BASE_URL/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"stream\":false}" \
  >/dev/null
echo "endpoint up: $BASE_URL  model: $MODEL"
echo

mkdir -p results
"$PY" bench/benchmark.py --base-url "$BASE_URL" --model "$MODEL" \
  --max-tokens 128 --concurrency 1 \
  --output results/benchmark-ollama-$(date +%F).json | tee results/demo-terminal-output.txt
echo
"$PY" cost/cost_model.py | tee -a results/demo-terminal-output.txt
