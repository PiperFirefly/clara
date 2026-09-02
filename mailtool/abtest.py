#!/usr/bin/env python3
"""
abtest.py — run one prompt against two models side-by-side and compare.

The agent's A/B "second-opinion" tool. Defaults to DeepSeek V4 Flash vs V4 Pro
using the same key the rest of the stack reads from ~/.pi/agent/auth.json.

Usage:
  abtest.py "prompt"
  abtest.py "prompt" --system "you are terse" --models flash,pro
  abtest.py "prompt" --temp 1.0 --n 3          # multiple samples each
  abtest.py "prompt" --arbiter                 # ask Pro to diff the two answers
  abtest.py "prompt" --json                    # machine-readable output

Model aliases: flash|fast -> deepseek-v4-flash, pro -> deepseek-v4-pro.
You can also pass full model ids (e.g. deepseek-v4-flash,deepseek-v4-pro).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

AUTH_PATH = os.path.expanduser("~/.pi/agent/auth.json")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
ALIASES = {
    "flash": "deepseek-v4-flash",
    "fast": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}


def deepseek_key():
    with open(AUTH_PATH) as f:
        return json.load(f)["deepseek"]["key"]


def resolve_model(name):
    return ALIASES.get(name.strip().lower(), name.strip())


def call(model, messages, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {
        "Authorization": f"Bearer {deepseek_key()}",
        "Content-Type": "application/json",
    }
    t0 = time.time()
    r = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=300)
    r.raise_for_status()
    latency = time.time() - t0
    d = r.json()
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    return {
        "model": model,
        "content": content,
        "latency": round(latency, 2),
        "usage": d.get("usage") or {},
    }


def arbiter(question, answers):
    body = (
        "Two models were asked the same question. Here it is:\n\n"
        f"Q: {question}\n\n"
    )
    for m, a in answers.items():
        body += f"—— {m} ——\n{a}\n\n"
    body += (
        "In a few sentences, summarise the key points where they AGREE and "
        "where they DISAGREE, and note which answer you'd trust more and why. "
        "Be concise and specific."
    )
    return call("deepseek-v4-pro", [{"role": "user", "content": body}], 0.4, 1200)


def render_text(results, arbiter_text=None):
    out = []
    width = 62
    for i, res in enumerate(results):
        label = f" A · {res['model']} " if i == 0 else f" B · {res['model']} "
        toks = res["usage"].get("completion_tokens")
        meta = f"{res['latency']}s" + (f" · {toks}t" if toks else "")
        out.append("═" * width)
        out.append(f"{label:─<{width - len(meta)}}{meta}")
        out.append("─" * width)
        out.append(res["content"].rstrip())
        out.append("")
    if arbiter_text:
        out.append("╔" + "═" * (width - 2) + "╗")
        out.append("║ ARBITER — deepseek-v4-pro weighs in".ljust(width - 1) + "║")
        out.append("╚" + "═" * (width - 2) + "╝")
        out.append(arbiter_text.rstrip())
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="A/B test two DeepSeek models.")
    ap.add_argument("prompt", help="the question/prompt to send to both models")
    ap.add_argument("--system", help="optional system prompt", default=None)
    ap.add_argument("--models", help="comma-separated model ids or aliases",
                    default=",".join(DEFAULT_MODELS))
    ap.add_argument("--temp", type=float, default=0.7, help="temperature (0-2)")
    ap.add_argument("--max-tokens", type=int, default=0, help="cap output tokens")
    ap.add_argument("--n", type=int, default=1, help="samples per model")
    ap.add_argument("--arbiter", action="store_true",
                    help="have Pro diff the two (first) answers")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    models = [resolve_model(m) for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        sys.exit("need at least two models (--models a,b)")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    jobs = [(m, i) for m in models for i in range(args.n)]
    results = [None] * len(jobs)
    errors = {}

    def work(job):
        idx, (model, _i) = job
        try:
            return idx, call(model, messages, args.temp, args.max_tokens), None
        except Exception as e:
            return idx, None, f"{model}: {e}"

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for idx, res, err in ex.map(work, list(enumerate(jobs))):
            results[idx] = res
            if err:
                errors[jobs[idx][0]] = err

    ok = [r for r in results if r]
    if not ok:
        sys.exit("all models failed:\n" + "\n".join(errors.values()))

    arb = None
    if args.arbiter and args.n >= 1:
        first = {r["model"]: r["content"] for r in ok[:len(models)]}
        try:
            arb = arbiter(args.prompt, first)["content"]
        except Exception as e:
            arb = f"(arbiter failed: {e})"

    if args.json:
        print(json.dumps({
            "prompt": args.prompt,
            "results": ok,
            "arbiter": arb,
            "errors": errors or None,
        }, indent=2))
    else:
        print()
        print(f"Q: {args.prompt}")
        print()
        print(render_text(ok, arb))
        if errors:
            print("── errors ──")
            for m, e in errors.items():
                print(f"  {m}: {e}")


if __name__ == "__main__":
    main()
