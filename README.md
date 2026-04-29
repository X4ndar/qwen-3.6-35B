# Qwen3.6-35B-A3B-FP8 — RunPod deploy

One-shot deploy of [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
on a single Blackwell-class GPU (e.g. RTX PRO 6000) via vLLM, exposed publicly
through a free Cloudflare quick tunnel.

The setup script auto-detects the host driver's CUDA support and picks the
right flag profile — works on both **driver ≥ 580 (CUDA 13)** *and* the older
**driver 570 / CUDA 12.8** pods that ship vLLM wheels with PTX too new for the
runtime.

---

## 1. One-line deploy on a fresh pod

```bash
git clone https://github.com/X4ndar/qwen-3.6-35B.git && cd qwen-3.6-35B && ./setup.sh
```

The script does:

1. Install `uv` + create venv at `/workspace/.venv-vllm`
2. Detect the host driver's CUDA version
3. Install the correct vLLM build (latest cu13 on modern drivers, `0.19.x` cu128 on legacy)
4. Download the ~36 GB FP8 weights to `/workspace/models/Qwen3.6-35B-A3B-FP8/`
5. Install `cloudflared`
6. Drop `chat.py` + the `qwen` CLI launcher in place
7. Generate an API key (or reuse `$API_KEY` if exported)
8. Start `vllm serve` with the right flag set + a Cloudflare quick tunnel

Total time on a clean pod: **~6–10 minutes** (most of it the model download).

When it's done you'll see something like:

```
============================================================
  Public URL : https://random-words.trycloudflare.com/v1
  API key    : sk-qwen-…
  Model      : Qwen/Qwen3.6-35B-A3B-FP8
  Docs       : /workspace/QWEN_API.md
  CLI        : qwen [args...]
============================================================
```

## 2. Use the API from your app

```bash
export OPENAI_BASE_URL="https://random-words.trycloudflare.com/v1"
export OPENAI_API_KEY="sk-qwen-..."
```

```python
from openai import OpenAI
client = OpenAI()  # picks up the env vars
r = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content or r.choices[0].message.reasoning)
```

Full reference (sampling presets, tool calling, preserve\_thinking pattern,
operations, troubleshooting) lives in [`QWEN_API.md`](./QWEN_API.md).

## 3. Use the CLI on the pod

```bash
qwen                                # interactive REPL with streaming
qwen "your prompt"                  # one-shot
qwen --no-think "fast reply"        # skip thinking
qwen --think "hard problem"         # show dimmed reasoning trace
qwen --preserve-thinking            # agent mode: keep reasoning across turns
qwen --system "you are..." "..."    # set a system prompt
```

In the REPL: `/reset`, `/system <text>`, `/think on|off`, `/quit`.

## 4. Helper scripts

| Script | What it does |
|---|---|
| `setup.sh`        | Full deploy from scratch (idempotent) |
| `start-vllm.sh`   | Just (re)start vLLM with the right flags |
| `start-tunnel.sh` | Just (re)start the Cloudflare tunnel (new URL each time) |
| `chat.py`         | The streaming REPL client |

## 5. Hardware / driver requirements

- Tested on **NVIDIA RTX PRO 6000 Blackwell** (96 GB VRAM, sm\_120)
- Works on driver ≥ 570 (CUDA 12.8). On driver < 12.9 the script auto-applies
  the legacy profile: `--language-model-only`, `--enforce-eager`,
  `--attention-backend TRITON_ATTN`. This means **no vision input** and ~10–20 %
  slower decode, but everything else (text + tool calling + thinking +
  preserve\_thinking + MTP) works.
- Model needs ~36 GB VRAM for weights; ~40 GB free for KV cache (with
  `--kv-cache-dtype fp8` you get ~940K tokens of cache headroom).

## 6. What's NOT in this repo

- `/workspace/.qwen-api-key` — generated per pod, kept off git
- `/workspace/models/Qwen3.6-35B-A3B-FP8/` — 36 GB of weights, downloaded on demand
- `/workspace/logs/` — runtime logs

## License

Setup scripts: MIT. Model weights: Apache-2.0 (Qwen team).
