#!/usr/bin/env bash
# (Re)start the Cloudflare quick tunnel for the local vLLM server on :8000.
# Prints the new public URL when it appears.

set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

pkill cloudflared 2>/dev/null || true
sleep 2
> "$LOG_DIR/cloudflared.log"
setsid nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
    > "$LOG_DIR/cloudflared.log" 2>&1 < /dev/null &
disown

until grep -qE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG_DIR/cloudflared.log" 2>/dev/null; do
    sleep 2
done
URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG_DIR/cloudflared.log" | head -1)
echo "Public URL: $URL/v1"
