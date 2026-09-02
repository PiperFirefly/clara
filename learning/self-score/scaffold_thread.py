#!/usr/bin/env python3
"""
Scaffold-threading test — does decomposition + state-threading let a weak
non-reasoning engine hold a chain it otherwise drops?

Baseline (single-shot, already measured): mistral 10% chained, qwen2.5:3b 0%.
This runs the SAME 20 chained screen tasks through a deterministic scaffold:
  1. parse the instruction into its numbered steps,
  2. feed each step to the engine as a single atomic call, threading the prior
     output as "Current value",
  3. hash-grade the final threaded result (identical to the single-shot grader).

The model is unchanged — only the protocol differs. If chained% jumps, the
lift is attributable to the scaffold (decomposition + state-threading), not the
engine. No LLM in the loop for the threading; it is plain parsing.

Usage:
  python3 scaffold_thread.py mistral
  python3 scaffold_thread.py qwen2.5:3b
  python3 scaffold_thread.py --smoke      # parse/thread c01..c20 with no calls
"""
import json
import os
import re
import sys
import time
import hashlib

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "self-score")
sys.path.insert(0, os.path.join(HOME, "selfplay"))
import backend as B  # noqa: E402  (local_ollama over ssh)

HOST = "worker"
RESULTS = os.path.join(ROOT, "work", "scaffold_thread-results.jsonl")


def parse_steps(instruction):
    """Return (initial_text_or_None, [(step_no, step_text), ...])."""
    m = re.search(r"Start with (.+?)\.\s*Step 1:", instruction)
    initial = m.group(1).strip() if m else None
    steps = re.findall(r"Step\s+(\d+)\s*:\s*(.*?)(?=\s*Step\s+\d+\s*:|$)",
                       instruction, re.DOTALL)
    out = []
    for i, (n, txt) in enumerate(steps):
        txt = txt.strip()
        # strip trailing "Give only ..." instruction from the final step
        txt = re.sub(r"\s*Give only.*$", "", txt, flags=re.DOTALL).strip()
        txt = txt.rstrip(".")
        out.append((int(n), txt))
    return initial, out


def step_prompt(initial, current, step_text, first):
    lines = ["You are doing ONE step of a multi-step computation. Do ONLY this step."]
    if initial and not first:
        lines.append(f"Original starting point: {initial}")
    if current is not None:
        lines.append(f"Current value: {current}")
    lines.append(f"This step: {step_text}")
    lines.append("Output only the result of this step (the new value). No explanation.")
    return "\n".join(lines)


def run_threaded(model, instruction):
    initial, steps = parse_steps(instruction)
    current = initial
    for idx, (n, txt) in enumerate(steps):
        prompt = step_prompt(initial, current, txt, first=(idx == 0))
        try:
            out = B.local_ollama(HOST, model, [{"role": "user", "content": prompt}],
                                 max_tokens=128, temperature=0)
            current = (out or "").strip()
        except Exception as e:
            current = f"<ERR {str(e)[:30]}>"
        time.sleep(0.1)
    return current


def run(model):
    bank = json.load(open(os.path.join(ROOT, "banks", "screen-v1.json")))
    chained = [t for t in bank["tasks"] if t["kind"] == "chained"]
    ok = 0
    fails = []
    t0 = time.time()
    for t in chained:
        final = run_threaded(model, t["instruction"])
        good = hashlib.sha256(final.encode()).hexdigest() == t["answer_sha256"]
        ok += 1 if good else 0
        if not good:
            fails.append((t["id"], final[:40]))
    pct = round(100 * ok / len(chained)) if chained else 0
    print(f"{model:14s} THREADED chained {ok}/{len(chained)} ({pct}%)  "
          f"[baseline single-shot: {'10%' if 'mistral' in model else '0%'}]")
    for tid, final in fails:
        print(f"    FAIL {tid}  final={final!r}")
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "host": HOST,
        "chained_ok": ok, "chained_total": len(chained), "chained_pct": pct,
        "sec": round(time.time() - t0, 1),
    }
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{model:14s} done in {rec['sec']}s -> {RESULTS}\n")


def smoke():
    bank = json.load(open(os.path.join(ROOT, "banks", "screen-v1.json")))
    for t in [x for x in bank["tasks"] if x["kind"] == "chained"]:
        initial, steps = parse_steps(t["instruction"])
        print(f"--- {t['id']}  initial={initial!r}  {len(steps)} steps")
        for n, txt in steps:
            print(f"      step {n}: {txt[:70]}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    elif len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        print(__doc__)
