# Qwen3.6-35B-A3B-FP8 — API Reference

OpenAI-compatible inference for **Qwen/Qwen3.6-35B-A3B-FP8** running on a RunPod
container, exposed publicly via a Cloudflare Tunnel.

---

## 1. Connection details

| Field | Value |
|---|---|
| Base URL | `https://<your-tunnel-id>.trycloudflare.com/v1` |
| API key  | stored at `/workspace/.qwen-api-key` on the pod |
| Model ID | `Qwen/Qwen3.6-35B-A3B-FP8` |
| Max context | 131,072 tokens (FP8 KV cache supports ~940K of headroom) |
| Auth header | `Authorization: Bearer <api-key>` |
| Tool calling | Enabled (`auto` parser: `qwen3_coder`) |
| MTP speculative decoding | Enabled (`mtp`, num_speculative_tokens=2) — faster decode |
| Vision input | **Disabled** on this pod (driver too old — text-only) |

> **⚠️ Quick-tunnel URL is ephemeral.** If `cloudflared` restarts, Cloudflare assigns a new random subdomain. See [§8 Operations](#8-operations) for recovery.

---

## 2. Quickstart — curl

```bash
export OPENAI_BASE_URL="https://<your-tunnel-id>.trycloudflare.com/v1"
export OPENAI_API_KEY="sk-qwen-..."   # from /workspace/.qwen-api-key

curl -s "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

---

## 3. Python — OpenAI SDK

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-tunnel-id>.trycloudflare.com/v1",
    api_key=os.environ["QWEN_API_KEY"],
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "Summarise the OSI model in 3 bullets."},
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

# When thinking is OFF, the answer arrives in `choices[0].message.reasoning`
# because vLLM's qwen3 reasoning parser routes pre-`</think>` text there.
# When thinking is ON, the answer arrives in `choices[0].message.content`
# and `reasoning` holds the chain-of-thought.
msg = resp.choices[0].message
print(msg.content or msg.reasoning)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Write a haiku about FP8."}],
    stream=True,
    max_tokens=200,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    delta = chunk.choices[0].delta
    piece = delta.content or getattr(delta, "reasoning", "") or ""
    print(piece, end="", flush=True)
```

---

## 4. Node / TypeScript — OpenAI SDK

```bash
npm install openai
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://<your-tunnel-id>.trycloudflare.com/v1",
  apiKey: process.env.QWEN_API_KEY!,
});

const stream = await client.chat.completions.create({
  model: "Qwen/Qwen3.6-35B-A3B-FP8",
  messages: [{ role: "user", content: "Pitch FP8 inference in two sentences." }],
  max_tokens: 256,
  stream: true,
  // @ts-expect-error vLLM-specific extension
  chat_template_kwargs: { enable_thinking: false },
} as any);

for await (const chunk of stream) {
  const d = chunk.choices[0].delta as any;
  process.stdout.write(d.content ?? d.reasoning ?? "");
}
```

---

## 5. Sampling presets

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Thinking — general | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| Thinking — coding  | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking       | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

Pass `top_k` and `min_p` via `extra_body`.

---

## 6. Tool calling

Tool calling is **enabled** with the `qwen3_coder` parser. Standard OpenAI shape works.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

msg = resp.choices[0].message
if msg.tool_calls:
    for call in msg.tool_calls:
        name = call.function.name
        args = call.function.arguments  # JSON string
        # ... execute the tool, then send result back as role="tool"
```

When the model picks a tool, `finish_reason == "tool_calls"`, `content == None`, and `tool_calls[]` is populated. See [tool-calling loop docs](https://platform.openai.com/docs/guides/function-calling) — the protocol is identical.

---

## 7. Thinking mode

The model thinks by default. Per-request control:

```json
{ "chat_template_kwargs": { "enable_thinking": false } }
```

Preserve thinking across turns (recommended for agent loops):

```json
{ "chat_template_kwargs": { "enable_thinking": true, "preserve_thinking": true } }
```

### Response field map (verified on this pod)

In **vLLM 0.19.1** with `--reasoning-parser qwen3`:

| Mode | `message.reasoning` | `message.content` |
|---|---|---|
| `enable_thinking: true`  | chain-of-thought | final answer |
| `enable_thinking: false` | empty string     | answer (clean) |

(The `reasoning` field is the OpenAI-extension field vLLM uses for parsed `<think>` blocks.)

### Preserve thinking across turns — VERIFIED working pattern

For agent loops, you **must** include the previous reasoning in the assistant message of the next turn. Two formats work; one does not:

| Assistant message format | Preserved? |
|---|---|
| `{"role":"assistant", "content":"...", "reasoning":"..."}`              | ✅ |
| `{"role":"assistant", "content":"<think>\n...\n</think>\n\n..."}`   | ✅ |
| `{"role":"assistant", "content":"...", "reasoning_content":"..."}`      | ❌ stripped by vLLM API |

**Recommended pattern (matches vLLM's response shape):**

```python
# Turn 1
r1 = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"chat_template_kwargs": {
        "enable_thinking": True, "preserve_thinking": True,
    }},
)
m1 = r1.choices[0].message

# Append assistant turn WITH reasoning so the next turn can leverage it
history = [
    {"role": "user", "content": "..."},
    {"role": "assistant",
     "content": m1.content or "",
     "reasoning": m1.reasoning or ""},   # <-- the field that survives
    {"role": "user", "content": "follow-up question"},
]

r2 = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=history,
    extra_body={"chat_template_kwargs": {
        "enable_thinking": True, "preserve_thinking": True,
    }},
)
```

**How to verify it's working:** turn-2 `usage.prompt_tokens` should grow by roughly *(turn-1 reasoning + turn-1 content)* tokens. If it only grows by *content* tokens, your reasoning isn't being preserved.

---

## 8. Operations

### Where things live on the pod

| Path | Purpose |
|---|---|
| `/workspace/.qwen-api-key` | The API key (mode 600) |
| `/workspace/models/Qwen3.6-35B-A3B-FP8/` | Model weights (~36 GB) |
| `/workspace/.venv-vllm/` | Python virtualenv with vLLM 0.19.1 + OpenAI SDK |
| `/workspace/chat.py` | Local terminal REPL client |
| `/workspace/QWEN_API.md` | This document |
| `/workspace/logs/vllm-serve.log` | vLLM server log |
| `/workspace/logs/cloudflared.log` | Cloudflare tunnel log (contains the public URL) |
| `/usr/local/bin/qwen` | `qwen` CLI launcher |
| `/usr/local/bin/cloudflared` | The tunnel binary |

### Find the current public URL

```bash
ssh root@<pod-ip> -p <pod-ssh-port> -i ~/.ssh/id_ed25519 \
  'grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /workspace/logs/cloudflared.log | head -1'
```

### Check both processes

```bash
ssh ... 'ps -ef | grep -E "(vllm serve|cloudflared)" | grep -v grep'
```

### Restart vLLM (this pod's working flags)

This pod's driver (570.x / CUDA 12.8) requires workarounds because the vLLM 0.19.1 wheel ships PTX compiled for CUDA 12.9+:

```bash
source /workspace/.venv-vllm/bin/activate
> /workspace/logs/vllm-serve.log
setsid nohup vllm serve /workspace/models/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.9 \
  --language-model-only \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --api-key "$(cat /workspace/.qwen-api-key)" \
  > /workspace/logs/vllm-serve.log 2>&1 < /dev/null &
disown
```

First request after restart: ~30-60s (Triton JIT-compiles attention kernels for sm_120). KV cache size: ~940K tokens thanks to `--kv-cache-dtype fp8`. MTP speculative decoding gives faster decode.

### Restart the tunnel (this **changes** the public URL)

```bash
pkill cloudflared
setsid nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
  > /workspace/logs/cloudflared.log 2>&1 < /dev/null &
disown
sleep 5
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /workspace/logs/cloudflared.log | head -1
```

---

## 9. Pod-specific caveats (driver 570.x)

This pod was provisioned with **driver 570.195.03 / CUDA 12.8**, which is older than the previous pod (driver 580.x / CUDA 13). Consequences:

| Aspect | Status | Impact |
|---|---|---|
| Text generation | ✅ Works | Slightly slower (eager mode, no CUDA graphs) |
| Tool calling | ✅ Works | Same as text |
| Vision input (image / video) | ❌ **Disabled** (`--language-model-only`) | Vision encoder PTX needs CUDA 12.9+ |
| FlashAttention 2/3 | ❌ Replaced by Triton attn | ~10-20% slower throughput |
| CUDA graphs | ❌ Disabled (`--enforce-eager`) | ~10-20% slower decode |

**To regain full performance + vision**, recreate the pod from a RunPod template with **driver 580+ / CUDA 13** — the same flags from the previous pod (no `--enforce-eager`, no `--language-model-only`, no `VLLM_ATTENTION_BACKEND` env) will work.

---

## 10. Quotas and cost

- **GPU**: Single NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM). 44 GB available for KV cache after model load = ~577K tokens of headroom.
- **Cost**: This is a paid hourly pod. For sporadic integration testing, a hosted Qwen API (OpenRouter, DashScope) is dramatically cheaper.
- **Cloudflare Quick Tunnels** are dev-only. No SLA, no published rate limits. For production traffic switch to a Named Tunnel or RunPod's port proxy.
