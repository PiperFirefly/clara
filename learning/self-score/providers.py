#!/usr/bin/env python3
"""
Provider registry — the seam that lets `onboard.py` test ANY model from ANY
vendor against the same frozen banks, so a new model drop (Claude 5.1, Gemini,
a new DeepSeek revision, an OpenAI-compatible host) is a one-command onboard.

Design rules:
  * Keys are fetched from the secret store at CALL TIME via get_secret()
    (M1: secrets never sit in model context, logs, or code). No auth.json reads.
  * Each provider has its own ask() (vendor wire formats differ) returning a
    normalized (text, reasoning_seen, prompt_tokens, completion_tokens) tuple.
  * Cost-per-1M-token is editable here. VERIFY against current vendor list
    prices before trusting a decision — these are seeded defaults to be refined.
  * Egress: only to the provider chat-completions endpoints (external https).
"""
import json
import os
import sys
import urllib.request

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "secrets"))
from secretstore import get_secret  # noqa: E402

ANSWER_ONLY = ("\n\nRespond with ONLY the final answer, nothing else — no "
               "explanation, no punctuation, no words around it.")

# max output tokens per thinking mode (off = answer-only; low/high = reasoning room)
MAX_TOKENS = {"off": 1024, "low": 4096, "high": 8192}


def _key(name):
    s = get_secret(name)
    if s and s.get("value"):
        return s["value"]
    raise RuntimeError(f"secret {name!r} not found in secret store")


def _post(url, headers, payload, timeout=180):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ----------------------------------------------------------------------------
# Per-vendor ask() — each returns (text, reasoning_seen, prompt_tok, comp_tok)
# ----------------------------------------------------------------------------

def ask_deepseek(model, instruction, thinking):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": instruction + ANSWER_ONLY}],
        "max_tokens": MAX_TOKENS[thinking],
        "temperature": 0,
    }
    if thinking == "off":
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = thinking  # 'low' | 'high' valid for flash
    d = _post("https://api.deepseek.com/chat/completions",
              {"Authorization": "Bearer " + _key("deepseek/api_key"),
               "Content-Type": "application/json"}, payload)
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    reas = bool((msg.get("reasoning_content") or "").strip())
    u = d.get("usage") or {}
    return text, reas, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def ask_anthropic(model, instruction, thinking):
    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS[thinking],
        "temperature": 0,
        "messages": [{"role": "user", "content": instruction + ANSWER_ONLY}],
    }
    if thinking != "off":
        payload["thinking"] = {"type": "enabled", "budget_tokens": MAX_TOKENS[thinking]}
    d = _post("https://api.anthropic.com/v1/messages",
              {"x-api-key": _key("anthropic/api_key"),
               "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}, payload)
    text = "".join(b.get("text", "") for b in d.get("content", [])
                   if b.get("type") == "text").strip()
    reas = any(b.get("type") == "thinking" for b in d.get("content", []))
    u = d.get("usage") or {}
    return text, reas, u.get("input_tokens", 0), u.get("output_tokens", 0)


def ask_openai(model, instruction, thinking, base_url, secret):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": instruction + ANSWER_ONLY}],
    }
    if thinking == "off":
        payload["max_tokens"] = MAX_TOKENS[thinking]
    else:
        # o-series / reasoning models take reasoning_effort + max_completion_tokens
        payload["reasoning_effort"] = thinking
        payload["max_completion_tokens"] = MAX_TOKENS[thinking]
    d = _post(base_url, {"Authorization": "Bearer " + _key(secret),
                         "Content-Type": "application/json"}, payload)
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    reas = bool((msg.get("reasoning_content") or "").strip())
    u = d.get("usage") or {}
    return text, reas, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


# ----------------------------------------------------------------------------
# Provider table — editable. prices are $ per 1M tokens (in / out).
# VERIFY against current vendor list prices before trusting a decision.
# ----------------------------------------------------------------------------

PROVIDERS = {
    "deepseek": {
        "ask": ask_deepseek,
        "models": {
            "deepseek-v4-flash": {"price_in": 0.20, "price_out": 0.60,
                                  "default_thinking": "low", "note": "Agent's current default"},
            "deepseek-v4-pro":   {"price_in": 0.60, "price_out": 2.00,
                                  "default_thinking": "high"},
        },
    },
    "anthropic": {
        "ask": ask_anthropic,
        "models": {
            "claude-sonnet-4-5": {"price_in": 3.00, "price_out": 15.00,
                                  "default_thinking": "low"},
            "claude-opus-4-5":   {"price_in": 15.00, "price_out": 75.00,
                                  "default_thinking": "high"},
        },
    },
    "openai": {
        "ask": lambda m, i, t: ask_openai(m, i, t,
              "https://api.openai.com/v1/chat/completions", "openai/api_key"),
        "models": {
            "gpt-4o-mini": {"price_in": 0.15, "price_out": 0.60,
                            "default_thinking": "off"},
            "gpt-4o":      {"price_in": 2.50, "price_out": 10.00,
                            "default_thinking": "low"},
        },
    },
}


def _model(provider, model):
    m = PROVIDERS[provider]["models"].get(model)
    if not m:
        raise KeyError(f"model {model!r} not in provider {provider!r}; "
                       f"known: {sorted(PROVIDERS[provider]['models'])}")
    return m


def resolve_model(model):
    """Find which provider owns `model`. Raises if unresolved/ambiguous."""
    hits = [p for p, c in PROVIDERS.items() if model in c["models"]]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"model {model!r} unknown. Add it to providers.py or pass "
                       f"--provider. Known: " +
                       ", ".join(f"{p}/{m}" for p, c in PROVIDERS.items()
                                 for m in c["models"]))
    raise KeyError(f"model {model!r} ambiguous across {hits}; pass --provider")


def ask(provider, model, instruction, thinking):
    """Normalized single call. thinking in ('off','low','high')."""
    return PROVIDERS[provider]["ask"](model, instruction, thinking)


def cost_usd(provider, model, prompt_tok, comp_tok):
    m = _model(provider, model)
    return (prompt_tok / 1e6) * m["price_in"] + (comp_tok / 1e6) * m["price_out"]


def default_thinking(provider, model):
    return _model(provider, model)["default_thinking"]
