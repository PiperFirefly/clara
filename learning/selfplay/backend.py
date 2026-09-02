"""
Model backends for self-play.

  local_ollama(host, model, messages, ...)  -> free, CPU, over SSH to worker/Local-box
  deepseek_chat(messages, ...)              -> paid (me), billed against the budget

No secrets here: the DeepSeek key is read from ~/.pi/agent/auth.json at call
time (the same file deepseek_balance.py and the webapp use), never echoed.
"""

import json
import os
import subprocess
import urllib.request

AUTH_PATH = os.path.expanduser("~/.pi/agent/auth.json")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-fable-5"  # strongest Claude (fable-5 > opus-5)

# Local hosts (from ~/.ssh/config). Mistral-7B is the workhorse opponent.
LOCAL_HOSTS = ["local-box", "worker"]
LOCAL_MODEL = "mistral"


def deepseek_key():
    with open(AUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["deepseek"]["key"]


def anthropic_key():
    with open(AUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["anthropic"]["key"]


def local_ollama(host, model, messages, max_tokens=256, temperature=0.7, timeout=300):
    """Run a chat completion on a remote box's ollama, streaming nothing.

    The JSON payload is piped through ssh stdin into `curl -d @-` on the remote,
    so there are no shell-quoting hazards and no temp files to clean up.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    )
    cmd = f"curl -s --max-time {timeout} http://localhost:11434/api/chat -d @-"
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, cmd],
        input=payload.encode(),
        capture_output=True,
        timeout=timeout + 40,
        check=False,
    )
    out = proc.stdout.decode("utf-8", "replace").strip()
    if proc.returncode != 0 or not out:
        raise RuntimeError(
            f"ollama@{host} failed (rc={proc.returncode}): {proc.stderr.decode()[:200] or 'no output'}"
        )
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"ollama@{host} returned non-JSON: {out[:200]}")
    return d.get("message", {}).get("content", "") or (d.get("response", "") or "")


def deepseek_chat(messages, max_tokens=1024, temperature=0.4):
    """One paid DeepSeek v4-pro call. Returns (content, prompt_tokens, completion_tokens)."""
    body = json.dumps(
        {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=body,
        headers={
            "Authorization": "Bearer " + deepseek_key(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    content = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = d.get("usage") or {}
    return (
        content,
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def anthropic_chat(system, user, max_tokens=1024):
    """One paid Claude (top of line) call. Returns (content, input_tokens, output_tokens).

    Note: claude-fable-5 rejects the `temperature` field (deprecated for reasoning
    models), so we omit it."""
    body = json.dumps(
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
    ).encode()
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=body,
        headers={
            "x-api-key": anthropic_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    content = "".join(b.get("text", "") for b in (d.get("content") or []))
    usage = d.get("usage") or {}
    return (
        content,
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )
