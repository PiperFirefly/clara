#!/usr/bin/env python3
"""Screen engines: atomic vs chained instruction-following reliability.

Runs the screen-v1 bank (20 atomic + 20 chained) against each engine and
reports atomic% vs chained% — the goldilocks profile is high atomic, low chained.
"""
import json
import os
import sys
import hashlib
import urllib.request

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "learning", "self-score")
KEY = json.load(open(os.path.join(HOME, ".pi/agent/auth.json")))["deepseek"]["key"]


def ask(model, instruction):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content":
                      instruction + "\n\nRespond with ONLY the final answer, "
                      "nothing else — no explanation, no punctuation, no words around it."}],
        "max_tokens": 500,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=90).read())
    return d["choices"][0]["message"]["content"].strip()


def run(model):
    bank = json.load(open(os.path.join(ROOT, "banks", "screen-v1.json")))
    at = ch = at_ok = ch_ok = 0
    for t in bank["tasks"]:
        try:
            ans = ask(model, t["instruction"])
            ok = hashlib.sha256(ans.encode()).hexdigest() == t["answer_sha256"]
        except Exception as e:
            ok = False
        if t["kind"] == "atomic":
            at += 1
            at_ok += 1 if ok else 0
        else:
            ch += 1
            ch_ok += 1 if ok else 0
    return at_ok, at, ch_ok, ch


if __name__ == "__main__":
    models = sys.argv[1:] or ["deepseek-v4-pro", "deepseek-v4-flash"]
    for model in models:
        a, at, c, ct = run(model)
        ap = round(100 * a / at) if at else 0
        cp = round(100 * c / ct) if ct else 0
        print(f"{model}: atomic {a}/{at} ({ap}%)   chained {c}/{ct} ({cp}%)")
