#!/usr/bin/env python3
"""
Engine matrix — controlled comparison of DeepSeek V4 models x thinking modes.

Reuses the frozen screen-v1 bank (20 atomic + 20 chained, hash-pinned, minted by
claude-fable-5) and extends screen_runner.py with EXPLICIT thinking control.

Pi's wire format for deepseek (from pi-ai openai-completions.js):
    thinking ON  -> {"thinking": {"type": "enabled"}, "reasoning_effort": "<level>"}
    thinking OFF -> {"thinking": {"type": "disabled"}}

Matrix (default):
    deepseek-v4-flash  x  off | low | high
    deepseek-v4-pro    x  off | high          (pro only supports high/max)

Usage:
    python3 engine_matrix.py                  # run default matrix
    python3 engine_matrix.py flash off        # single cell
    python3 engine_matrix.py --report         # print stored results
"""
import json
import os
import sys
import time
import hashlib
import urllib.request

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "learning", "self-score")
KEY = json.load(open(os.path.join(HOME, ".pi/agent/auth.json")))["deepseek"]["key"]
URL = "https://api.deepseek.com/chat/completions"
RESULTS = os.path.join(ROOT, "work", "engine_matrix-results.jsonl")

# max_tokens per thinking mode: off is answer-only; on needs reasoning room.
MAX_TOKENS = {"off": 1024, "low": 4096, "high": 8192}


def ask(model, instruction, thinking):
    """One call. thinking = 'off' | 'low' | 'high'. Returns (content, reasoning_has, usage)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content":
                      instruction + "\n\nRespond with ONLY the final answer, "
                      "nothing else — no explanation, no punctuation, no words around it."}],
        "max_tokens": MAX_TOKENS[thinking],
        "temperature": 0,
    }
    if thinking == "off":
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = thinking  # 'low' | 'high' are valid for flash

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = bool((msg.get("reasoning_content") or "").strip())
    return content, reasoning, (d.get("usage") or {})


def run_cell(model, thinking):
    bank = json.load(open(os.path.join(ROOT, "banks", "screen-v1.json")))
    at = ch = at_ok = ch_ok = 0
    ptok = ctok = 0
    reasoning_hits = 0
    t0 = time.time()
    for t in bank["tasks"]:
        ok = False
        try:
            ans, reasoning, usage = ask(model, t["instruction"], thinking)
            ok = hashlib.sha256(ans.encode()).hexdigest() == t["answer_sha256"]
            ptok += int(usage.get("prompt_tokens") or 0)
            ctok += int(usage.get("completion_tokens") or 0)
            reasoning_hits += 1 if reasoning else 0
        except Exception as e:
            print(f"    [err] {t['id']}: {e}", file=sys.stderr)
        if t["kind"] == "atomic":
            at += 1
            at_ok += 1 if ok else 0
        else:
            ch += 1
            ch_ok += 1 if ok else 0
        time.sleep(0.3)  # human cadence, no rapid-fire
    ap = round(100 * at_ok / at) if at else 0
    cp = round(100 * ch_ok / ch) if ch else 0
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "thinking": thinking,
        "atomic_ok": at_ok, "atomic_total": at, "atomic_pct": ap,
        "chained_ok": ch_ok, "chained_total": ch, "chained_pct": cp,
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "reasoning_hits": reasoning_hits,
        "sec": round(time.time() - t0, 1),
    }
    print(f"  {model:24s} thinking={thinking:4s}  atomic {at_ok}/{at} ({ap}%)  "
          f"chained {ch_ok}/{ch} ({cp}%)  [reasoning_seen={reasoning_hits}/40]  "
          f"{rec['sec']}s")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    if "--report" in sys.argv:
        if not os.path.exists(RESULTS):
            print("no results yet")
            return
        for ln in open(RESULTS):
            r = json.loads(ln)
            print(f"{r['ts']}  {r['model']:24s} {r['thinking']:4s}  "
                  f"atomic {r['atomic_ok']}/{r['atomic_total']} ({r['atomic_pct']}%)  "
                  f"chained {r['chained_ok']}/{r['chained_total']} ({r['chained_pct']}%)  "
                  f"tok={r['prompt_tokens'] + r['completion_tokens']}  {r['sec']}s")
        return

    # single-cell mode: engine_matrix.py <model> <thinking>
    if len(sys.argv) >= 3:
        model, thinking = sys.argv[1], sys.argv[2]
        print(f"== {model} x {thinking} ==")
        run_cell(model, thinking)
        return

    matrix = [
        ("deepseek-v4-flash", "off"),
        ("deepseek-v4-flash", "low"),
        ("deepseek-v4-flash", "high"),
        ("deepseek-v4-pro", "off"),
        ("deepseek-v4-pro", "high"),
    ]
    print(f"Engine matrix ({len(matrix)} cells, 40 tasks each) -> {RESULTS}\n")
    for model, thinking in matrix:
        print(f"== {model} x {thinking} ==")
        run_cell(model, thinking)
        print()
    print("done. re-run with --report to see the full table.")


if __name__ == "__main__":
    main()
