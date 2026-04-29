#!/usr/bin/env python3
"""Streaming REPL for the Qwen3.6 vLLM server.

Usage:
    python chat.py                         # interactive REPL
    python chat.py "your question here"    # one-shot
    python chat.py --think "..."           # also keep thinking visible (dimmed)
    python chat.py --no-think "..."        # disable thinking entirely (faster)
    python chat.py --system "you are..."   # set system prompt

Env:
    OPENAI_BASE_URL  default http://localhost:8000/v1
    OPENAI_API_KEY   default EMPTY
    QWEN_MODEL       default Qwen/Qwen3.6-35B-A3B-FP8
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency. Run: pip install openai")

DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    enable_thinking: bool,
    show_thinking: bool,
    preserve_thinking: bool = False,
) -> str:
    if enable_thinking:
        params = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5)
    else:
        params = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=32768,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
                **({"preserve_thinking": True} if (enable_thinking and preserve_thinking) else {}),
            },
        },
        **params,
    )

    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    in_think_pane = False

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        rc = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        token = getattr(delta, "content", None) or ""

        if rc:
            reasoning_buf.append(rc)
            if not enable_thinking:
                sys.stdout.write(rc)
                sys.stdout.flush()
            elif show_thinking:
                if not in_think_pane:
                    sys.stdout.write(DIM + "[think] ")
                    in_think_pane = True
                sys.stdout.write(DIM + rc)
                sys.stdout.flush()

        if token:
            if in_think_pane:
                sys.stdout.write(RESET + "\n")
                in_think_pane = False
            content_buf.append(token)
            sys.stdout.write(token)
            sys.stdout.flush()

    if in_think_pane:
        sys.stdout.write(RESET)
    sys.stdout.write("\n")

    content = "".join(content_buf).strip()
    reasoning = "".join(reasoning_buf).strip()
    # Caller may want to preserve reasoning in agent history.
    return (content if content else reasoning), reasoning


def repl(client: OpenAI, model: str, system: str | None,
         show_thinking: bool, no_think: bool, preserve_thinking: bool) -> None:
    history: list[dict] = []
    if system:
        history.append({"role": "system", "content": system})

    print(f"{CYAN}Connected to {client.base_url} · model {model}{RESET}")
    print(f"{DIM}Commands: /reset  /system <text>  /think on|off  /quit{RESET}\n")

    enable_thinking = not no_think

    while True:
        try:
            line = input(f"{BOLD}{GREEN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        if line in ("/quit", "/exit"):
            return
        if line == "/reset":
            history = [m for m in history if m["role"] == "system"]
            print(f"{DIM}history cleared{RESET}")
            continue
        if line.startswith("/system "):
            sys_text = line[len("/system "):].strip()
            history = [m for m in history if m["role"] != "system"]
            history.insert(0, {"role": "system", "content": sys_text})
            print(f"{DIM}system prompt set{RESET}")
            continue
        if line == "/think on":
            enable_thinking = True
            print(f"{DIM}thinking ON{RESET}")
            continue
        if line == "/think off":
            enable_thinking = False
            print(f"{DIM}thinking OFF{RESET}")
            continue

        history.append({"role": "user", "content": line})
        print(f"{BOLD}{CYAN}qwen>{RESET} ", end="", flush=True)
        try:
            reply, reasoning_text = stream_completion(
                client, model, history,
                enable_thinking=enable_thinking,
                show_thinking=show_thinking,
                preserve_thinking=preserve_thinking,
            )
        except KeyboardInterrupt:
            print(f"\n{RED}[interrupted]{RESET}")
            history.pop()
            continue
        except Exception as e:
            print(f"\n{RED}error: {e}{RESET}")
            history.pop()
            continue
        msg = {"role": "assistant", "content": reply}
        if preserve_thinking and enable_thinking and reasoning_text:
            msg["reasoning"] = reasoning_text
        history.append(msg)


def one_shot(client: OpenAI, model: str, prompt: str, system: str | None,
             show_thinking: bool, no_think: bool, preserve_thinking: bool) -> None:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    _ = stream_completion(client, model, msgs,
                          enable_thinking=not no_think,
                          show_thinking=show_thinking,
                          preserve_thinking=preserve_thinking)


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3.6 chat client.")
    ap.add_argument("prompt", nargs="*", help="One-shot prompt. Omit for REPL.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--system", help="System prompt.")
    ap.add_argument("--think", action="store_true",
                    help="Show thinking blocks dimmed (only relevant when thinking is on).")
    ap.add_argument("--no-think", action="store_true",
                    help="Disable thinking mode (faster, less reasoning).")
    ap.add_argument("--preserve-thinking", action="store_true",
                    help="Preserve historical reasoning across turns (recommended for agents).")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.prompt:
        one_shot(client, args.model, " ".join(args.prompt),
                 system=args.system, show_thinking=args.think, no_think=args.no_think,
                 preserve_thinking=args.preserve_thinking)
    else:
        repl(client, args.model, args.system,
             show_thinking=args.think, no_think=args.no_think,
             preserve_thinking=args.preserve_thinking)


if __name__ == "__main__":
    main()
