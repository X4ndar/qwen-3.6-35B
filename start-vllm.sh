#!/usr/bin/env bash
# (Re)start the Qwen3.6 vLLM server with the right flag profile for the
# pod's CUDA driver. Reuses the existing API key at /workspace/.qwen-api-key.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/.venv-vllm"
MODEL_DIR="$WORKSPACE/models/Qwen3.6-35B-A3B-FP8"
LOG_DIR="$WORKSPACE/logs"
KEY_FILE="$WORKSPACE/.qwen-api-key"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.6-35B-A3B-FP8}"

mkdir -p "$LOG_DIR"
[ -s "$KEY_FILE" ] || { echo "no API key at $KEY_FILE — run setup.sh first"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "model not found at $MODEL_DIR — run setup.sh first"; exit 1; }

DRIVER_MAJOR=$(nvidia-smi | awk '/CUDA Version/ { print $9 }' | head -1 \
    | awk -F. '{ printf "%d%02d\n", $1, $2 }')
[ "$DRIVER_MAJOR" -ge 1209 ] && PROFILE=modern || PROFILE=legacy
echo "profile=$PROFILE  (driver cuda=$DRIVER_MAJOR)"

COMMON=(
    --served-model-name "$SERVED_NAME"
    --host 0.0.0.0 --port 8000
    --tensor-parallel-size 1
    --max-model-len 131072
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --gpu-memory-utilization 0.9
    --kv-cache-dtype fp8
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
    --api-key "$(cat "$KEY_FILE")"
)
[ "$PROFILE" = "legacy" ] && \
    EXTRA=(--language-model-only --enforce-eager --attention-backend TRITON_ATTN) || \
    EXTRA=()

pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3
> "$LOG_DIR/vllm-serve.log"
setsid nohup "$VENV/bin/vllm" serve "$MODEL_DIR" "${COMMON[@]}" "${EXTRA[@]}" \
    > "$LOG_DIR/vllm-serve.log" 2>&1 < /dev/null &
disown

echo "started (PID $(pgrep -f 'vllm serve' | head -1)) — waiting for ready..."
until grep -qE "(Application startup complete|EngineCore failed|RuntimeError)" \
        "$LOG_DIR/vllm-serve.log" 2>/dev/null; do sleep 5; done
tail -3 "$LOG_DIR/vllm-serve.log"
