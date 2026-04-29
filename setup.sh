#!/usr/bin/env bash
# Qwen3.6-35B-A3B-FP8 — one-shot deploy script for a fresh RunPod GPU container.
#
# Usage (from inside the pod):
#   curl -fsSL https://raw.githubusercontent.com/X4ndar/qwen-3.6-35B/main/setup.sh | bash
# or:
#   git clone https://github.com/X4ndar/qwen-3.6-35B.git && cd qwen-3.6-35B && ./setup.sh
#
# Env overrides:
#   MODEL=Qwen/Qwen3.6-35B-A3B-FP8     model repo to fetch
#   WORKSPACE=/workspace               install root
#   API_KEY=<existing key>             reuse instead of generating a new one
#   SKIP_DOWNLOAD=1                    skip model download (already there)
#   SKIP_TUNNEL=1                      do not start cloudflared
#
# Requires: an NVIDIA GPU and an internet connection. Reads driver version
# automatically and falls back to the conservative "old-driver" flag set
# (Triton attn, eager mode, language-only) when CUDA driver < 12.9.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/.venv-vllm"
MODEL_DIR="$WORKSPACE/models/$(basename "$MODEL")"
LOG_DIR="$WORKSPACE/logs"
KEY_FILE="$WORKSPACE/.qwen-api-key"

REPO_RAW="${REPO_RAW:-https://raw.githubusercontent.com/X4ndar/qwen-3.6-35B/main}"

mkdir -p "$WORKSPACE" "$LOG_DIR" "$WORKSPACE/models"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

log "1/8 install uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

log "2/8 create venv at $VENV"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "3/8 detect CUDA driver version"
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
    | head -1 | awk -F. '{ printf "%d.%d\n", $1, $2 }')
DRIVER_MAJOR=$(nvidia-smi | awk '/CUDA Version/ { print $9 }' | head -1 \
    | awk -F. '{ printf "%d%02d\n", $1, $2 }')
echo "driver_version=$DRIVER_CUDA  cuda_runtime=$DRIVER_MAJOR"

# Decide the install + serve profile based on driver CUDA support.
# - driver supports CUDA >= 12.9  -> latest vLLM (cu13 wheels), full feature set
# - driver supports CUDA <= 12.8  -> vLLM 0.19.x cu128 wheels + workarounds
if [ "$DRIVER_MAJOR" -ge 1209 ]; then
    PROFILE="modern"
    VLLM_PIN=""
    TORCH_BACKEND="auto"
else
    PROFILE="legacy"
    VLLM_PIN=">=0.19.0,<0.20"
    TORCH_BACKEND="cu128"
fi
echo "profile=$PROFILE  vllm_pin='${VLLM_PIN}'  torch_backend=$TORCH_BACKEND"

log "4/8 install hf cli + (re)install vllm under the right CUDA"
uv pip install "huggingface_hub[cli,hf_transfer]" >/dev/null 2>&1 || \
    uv pip install "huggingface_hub[cli]"
uv pip install --reinstall "vllm${VLLM_PIN}" --torch-backend="$TORCH_BACKEND"
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| sm', torch.cuda.get_device_capability(0))"

log "5/8 download model weights to $MODEL_DIR"
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "SKIP_DOWNLOAD=1 — skipping"
elif [ -f "$MODEL_DIR/config.json" ]; then
    echo "already present — skipping"
else
    hf download "$MODEL" --local-dir "$MODEL_DIR"
fi

log "6/8 install cloudflared"
if ! command -v cloudflared >/dev/null 2>&1; then
    curl -fsSL -o /usr/local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /usr/local/bin/cloudflared
fi
cloudflared --version

log "7/8 install chat.py + qwen launcher"
mkdir -p "$WORKSPACE"
for f in chat.py QWEN_API.md; do
    if [ -f "./$f" ]; then
        cp "./$f" "$WORKSPACE/$f"
    else
        curl -fsSL "$REPO_RAW/$f" -o "$WORKSPACE/$f"
    fi
done
cat > /usr/local/bin/qwen <<EOF
#!/usr/bin/env bash
export OPENAI_API_KEY="\$(cat $KEY_FILE)"
exec $VENV/bin/python $WORKSPACE/chat.py "\$@"
EOF
chmod +x /usr/local/bin/qwen

log "8/8 generate API key + start vLLM (+ tunnel)"
if [ -n "${API_KEY:-}" ]; then
    echo "$API_KEY" > "$KEY_FILE"
elif [ ! -s "$KEY_FILE" ]; then
    echo "sk-qwen-$(openssl rand -hex 24)" > "$KEY_FILE"
fi
chmod 600 "$KEY_FILE"

# Build the vllm serve command per profile.
COMMON=(
    --served-model-name "$MODEL"
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
if [ "$PROFILE" = "legacy" ]; then
    EXTRA=(--language-model-only --enforce-eager --attention-backend TRITON_ATTN)
else
    EXTRA=()
fi

# Stop any previous instance (best effort).
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

> "$LOG_DIR/vllm-serve.log"
setsid nohup "$VENV/bin/vllm" serve "$MODEL_DIR" "${COMMON[@]}" "${EXTRA[@]}" \
    > "$LOG_DIR/vllm-serve.log" 2>&1 < /dev/null &
disown
echo "vLLM PID: $(pgrep -f 'vllm serve' | head -1)"

if [ "${SKIP_TUNNEL:-0}" != "1" ]; then
    pkill cloudflared 2>/dev/null || true
    sleep 2
    > "$LOG_DIR/cloudflared.log"
    setsid nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
        > "$LOG_DIR/cloudflared.log" 2>&1 < /dev/null &
    disown
fi

log "waiting for vLLM ready (~2-4 min)"
until grep -qE "(Application startup complete|EngineCore failed|RuntimeError)" \
        "$LOG_DIR/vllm-serve.log" 2>/dev/null; do sleep 5; done
tail -3 "$LOG_DIR/vllm-serve.log"

if [ "${SKIP_TUNNEL:-0}" != "1" ]; then
    until grep -qE "https://[a-z0-9-]+\.trycloudflare\.com" \
            "$LOG_DIR/cloudflared.log" 2>/dev/null; do sleep 2; done
    URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG_DIR/cloudflared.log" | head -1)
    echo
    echo "============================================================"
    echo "  Public URL : $URL/v1"
    echo "  API key    : $(cat "$KEY_FILE")"
    echo "  Model      : $MODEL"
    echo "  Docs       : $WORKSPACE/QWEN_API.md"
    echo "  CLI        : qwen [args...]"
    echo "============================================================"
fi
