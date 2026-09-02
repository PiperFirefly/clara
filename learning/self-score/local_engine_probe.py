#!/usr/bin/env python3
"""
Probe 1 — local non-reasoning engines through the frozen banks.

Runs a small local (CPU, free) engine through the same two frozen banks and
deterministic graders used for the DeepSeek matrix, so the numbers are directly
comparable:
  - screen-v1 (40 tasks: 20 atomic + 20 chained)  -> strict sha256 hash grading
  - selfplay games (30 deterministic games)       -> check_correct grading

Engines (already on the boxes, no downloads):
  mistral      (Mistral 7B, non-reasoning)  on worker
  qwen2.5:3b   (Qwen 2.5 3B, non-reasoning) on worker

Usage:
  python3 local_engine_probe.py mistral
  python3 local_engine_probe.py qwen2.5:3b
  python3 local_engine_probe.py --smoke      # 1 quick call per model, connectivity check
"""
import json
import os
import sys
import time
import random
import hashlib
import subprocess

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "self-score")
sys.path.insert(0, os.path.join(HOME, "selfplay"))
import questions as Q          # noqa: E402  (game bank + check_correct)
import backend as B            # noqa: E402  (local_ollama over ssh)

HOST = "worker"
RESULTS = os.path.join(ROOT, "work", "local_engine_probe-results.jsonl")

ANSWER_ONLY = ("\n\nRespond with ONLY the final answer, nothing else — no "
               "explanation, no punctuation, no words around it.")


def ask(model, instruction, max_tokens=512):
    out = B.local_ollama(HOST, model,
                         [{"role": "user", "content": instruction + ANSWER_ONLY}],
                         max_tokens=max_tokens, temperature=0)
    return (out or "").strip()


def run_screen(model):
    bank = json.load(open(os.path.join(ROOT, "banks", "screen-v1.json")))
    at = ch = at_ok = ch_ok = 0
    for t in bank["tasks"]:
        ok = False
        try:
            ans = ask(model, t["instruction"])
            ok = hashlib.sha256(ans.encode()).hexdigest() == t["answer_sha256"]
        except Exception as e:
            print(f"    [err] {t['id']}: {str(e)[:60]}", file=sys.stderr)
        if t["kind"] == "atomic":
            at += 1
            at_ok += 1 if ok else 0
        else:
            ch += 1
            ch_ok += 1 if ok else 0
        time.sleep(0.1)
    return at_ok, at, ch_ok, ch


def run_games(model):
    rng = random.Random(42)  # same instances as the DeepSeek game run
    kinds = [k for k, _ in Q.GAME_TYPES if k not in ("dilemma", "matrix_play")]
    games = [Q.gen_question(k, rng) for k in kinds]
    ok = 0
    fails = []
    for g in games:
        try:
            ans = ask(model, g["question"])
            c = Q.check_correct(ans, g["answer"], g["kind"])
            ok += 1 if c else 0
            if not c:
                fails.append((g["kind"], ans[:30], g["answer"][:30]))
        except Exception as e:
            fails.append((g["kind"], "ERR:" + str(e)[:40], g["answer"][:30]))
        time.sleep(0.1)
    return ok, len(games), fails


def run_model(model):
    t0 = time.time()
    a, at, c, ct = run_screen(model)
    ap = round(100 * a / at) if at else 0
    cp = round(100 * c / ct) if ct else 0
    print(f"{model:14s} SCREEN  atomic {a}/{at} ({ap}%)  chained {c}/{ct} ({cp}%)")
    g, gt, fails = run_games(model)
    gp = round(100 * g / gt) if gt else 0
    print(f"{model:14s} GAMES   {g}/{gt} ({gp}%)")
    for k, got, want in fails:
        print(f"    FAIL {k:18s} got={got!r:32s} want={want!r}")
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model, "host": HOST,
        "screen_atomic_ok": a, "screen_atomic_total": at, "screen_atomic_pct": ap,
        "screen_chained_ok": c, "screen_chained_total": ct, "screen_chained_pct": cp,
        "games_ok": g, "games_total": gt, "games_pct": gp,
        "sec": round(time.time() - t0, 1),
    }
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{model:14s} done in {rec['sec']}s -> {RESULTS}\n")


def smoke():
    for model in ["mistral", "qwen2.5:3b"]:
        t0 = time.time()
        try:
            ans = ask(model, "Compute 7 + 5. Answer with only the number.", 64)
            print(f"{model:14s} OK ({time.time()-t0:.1f}s) -> {ans!r}")
        except Exception as e:
            print(f"{model:14s} FAIL -> {str(e)[:120]}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    elif len(sys.argv) >= 2:
        run_model(sys.argv[1])
    else:
        print(__doc__)
