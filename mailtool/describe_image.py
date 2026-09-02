#!/usr/bin/env python3
"""Describe an image with a vision model — content-aware routing.

Routing:
  - benign docs/screenshots/images  -> DeepSeek v4-flash-vision-exp (best quality,
    near-free, hard-refuses erotic content — perfect for safe content).
  - erotic / NSFW images            -> abliterated Qwen2.5-VL-7B (uncensored),
    kept LOCAL (worker SSH forward / local server) so private content is
    NEVER sent to a cloud API — not even to get a refusal.

Auto mode (--provider auto) decides by content: it classifies the image with a
LOCAL classifier first (neuro qwen -> local gemma), then routes. You can bypass
the classifier and decide yourself with --content safe|nsfw. An explicit
--provider always wins and skips routing.

Usage:
  describe_image.py <image> [prompt] [--provider auto|deepseek|openai|local|qwen|neuro]
                              [--content auto|safe|nsfw] [--max-tokens N]

Keys are read from ~/.pi/agent/auth.json (`deepseek.key` / `openai.key`).

Note: deepseek-v4-flash-vision-exp is a *reasoning* model. With a large enough
output budget its final answer lands in `content`; as a safety net we fall back
to `reasoning_content` when `content` is empty (token-exhaustion symptom). The
default max_tokens is set high enough to avoid that for normal use.
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

LOCAL_SERVER = "http://127.0.0.1:8080"
LOCAL_MODEL = "gemma-3-4b-it"
QWEN_SERVER = "http://127.0.0.1:8083"
QWEN_MODEL = "qwen2.5-vl-7b-abliterated"
NEURO_SERVER = "http://127.0.0.1:8090"  # SSH -L forward to worker's /agent vision host
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash-vision-exp"
AUTH_PATH = os.path.expanduser("~/.pi/agent/auth.json")

# High enough for the reasoning model to finish thinking AND write a real answer
# in `content`. A tight budget (e.g. 1600) makes dense docs exhaust it mid-
# reasoning, leaving content empty and triggering the ugly CoT fallback.
DEFAULT_MAX_TOKENS = 3000
CLS_MAX_TOKENS = 8

_CLS_PROMPT = (
    "You are an image safety classifier. Look at the image. "
    "If it contains sexually explicit content, nudity, or erotica, reply with "
    "exactly the single word NSFW. Otherwise reply with exactly the single "
    "word SAFE. Do not add anything else."
)


def _key(namespace):
    try:
        with open(AUTH_PATH) as f:
            return json.load(f).get(namespace, {}).get("key") or ""
    except Exception:
        return ""


def _openai_key():
    return os.environ.get("OPENAI_API_KEY") or _key("openai")


def _deepseek_key():
    return os.environ.get("DEEPSEEK_API_KEY") or _key("deepseek")


def _payload(path, prompt, model, max_tokens):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }


def _call(url, payload, headers=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"]


def describe_deepseek(path, prompt, max_tokens=DEFAULT_MAX_TOKENS):
    payload = _payload(path, prompt, DEEPSEEK_MODEL, max_tokens)
    msg = _call(DEEPSEEK_URL, payload, headers={"Authorization": f"Bearer {_deepseek_key()}"})
    content = (msg.get("content") or "").strip()
    if content:
        return content
    # reasoning model: with a tight budget the answer can land in
    # reasoning_content instead. Return it only as a last resort.
    return (msg.get("reasoning_content") or "").strip()


def describe_openai(path, prompt, max_tokens=DEFAULT_MAX_TOKENS):
    payload = _payload(path, prompt, OPENAI_MODEL, max_tokens)
    msg = _call(OPENAI_URL, payload, headers={"Authorization": f"Bearer {_openai_key()}"})
    return (msg.get("content") or "").strip()


def describe_local(path, prompt, max_tokens=DEFAULT_MAX_TOKENS):
    payload = _payload(path, prompt, LOCAL_MODEL, max_tokens)
    msg = _call(LOCAL_SERVER + "/v1/chat/completions", payload)
    return (msg.get("content") or "").strip()


def describe_qwen(path, prompt, max_tokens=DEFAULT_MAX_TOKENS):
    """Local uncensored vision model (Qwen2.5-VL-7B abliterated)."""
    payload = _payload(path, prompt, QWEN_MODEL, max_tokens)
    msg = _call(QWEN_SERVER + "/v1/chat/completions", payload)
    return (msg.get("content") or "").strip()


def describe_neuro(path, prompt, max_tokens=DEFAULT_MAX_TOKENS):
    """Same abliterated model, hosted on worker (/agent) via SSH forward."""
    payload = _payload(path, prompt, QWEN_MODEL, max_tokens)
    msg = _call(NEURO_SERVER + "/v1/chat/completions", payload)
    return (msg.get("content") or "").strip()


def _local_providers():
    return ["neuro", "qwen", "local"]


def _describe_with(provider, path, prompt, max_tokens):
    return {
        "deepseek": lambda: describe_deepseek(path, prompt, max_tokens),
        "openai": lambda: describe_openai(path, prompt, max_tokens),
        "local": lambda: describe_local(path, prompt, max_tokens),
        "qwen": lambda: describe_qwen(path, prompt, max_tokens),
        "neuro": lambda: describe_neuro(path, prompt, max_tokens),
    }[provider]()


def _classify(path):
    """Classify an image SAFE/NSFW using LOCAL models only (never the cloud).

    Tries the uncensored neuro qwen first (it reliably recognizes erotica),
    then local gemma. Returns 'safe' or 'nsfw'. On total classifier failure we
    fail safe to 'nsfw' (route to the local abliterated model) — the worst
    failure is leaking a private image to a cloud API, so we never do that.
    """
    for provider in _local_providers():
        try:
            ans = _describe_with(provider, path, _CLS_PROMPT, CLS_MAX_TOKENS).strip().upper()
            if "NSFW" in ans:
                return "nsfw"
            if "SAFE" in ans:
                return "safe"
        except Exception as e:
            print(f"[describe_image] classifier {provider} failed ({e}); trying next", file=sys.stderr)
    print("[describe_image] classifier unavailable; defaulting to nsfw (local, privacy-safe)", file=sys.stderr)
    return "nsfw"


def describe(path, prompt, provider="auto", content="auto", max_tokens=DEFAULT_MAX_TOKENS):
    # Explicit provider -> honor it, no routing.
    if provider != "auto":
        return _describe_with(provider, path, prompt, max_tokens)

    # Auto -> decide content first (explicit flag wins over the classifier).
    if content == "auto":
        content = _classify(path)

    if content == "nsfw":
        # Erotic content: local abliterated model only, never the cloud.
        for provider in _local_providers():
            try:
                return _describe_with(provider, path, prompt, max_tokens)
            except Exception as e:
                print(f"[describe_image] nsfw provider {provider} failed ({e}); trying next", file=sys.stderr)
        raise RuntimeError("NSFW image but no abliterated (uncensored) model is reachable.")

    # Safe content: DeepSeek -> OpenAI -> local.
    if _deepseek_key():
        try:
            return describe_deepseek(path, prompt, max_tokens)
        except Exception as e:
            print(f"[describe_image] DeepSeek failed ({e}); trying OpenAI", file=sys.stderr)
    if _openai_key():
        try:
            return describe_openai(path, prompt, max_tokens)
        except Exception as e:
            print(f"[describe_image] OpenAI failed ({e}); falling back to local", file=sys.stderr)
    return describe_local(path, prompt, max_tokens)


def main():
    ap = argparse.ArgumentParser(description="Describe an image with a content-aware vision model.")
    ap.add_argument("image")
    ap.add_argument("prompt", nargs="?", default="Describe this image in detail.")
    ap.add_argument("--provider", choices=["auto", "deepseek", "openai", "local", "qwen", "neuro"], default="auto")
    ap.add_argument("--content", choices=["auto", "safe", "nsfw"], default="auto",
                    help="force content routing (auto=classify locally). safe=DeepSeek, nsfw=local abliterated.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    a = ap.parse_args()
    print(describe(a.image, a.prompt, a.provider, a.content, a.max_tokens))


if __name__ == "__main__":
    main()
