#!/usr/bin/env python3
"""ab_warmstart.py — A/B test the warm-start buffer (Agent, 2026-09-01).

Measures whether the warm-start buffer reduces first-turn memory churn and
improves continuity. Runs pi headless with a continuity prompt, counts the
tool calls (recall/search/doc/fused = memory retrieval) in the produced session,
and reads the reply. Two conditions:

  warm  — warm_state.py is available and injected (the prototype)
  cold  — injection disabled (baseline blank-slate)

Caveat (honest): the real warm-start only fires on a FRESH session's first turn
via before_agent_start. A headless `pi -p` run is itself a fresh session, so the
warm buffer SHOULD inject. But recall-tool-count on one prompt is a noisy proxy
for the real 10-minute 'personality comes back' effect. We treat it as a
sanity/trend signal, not a verdict.

Usage:
  python3 ab_warmstart.py --n 2        # run N iterations per condition
  python3 ab_warmstart.py --prompt "..."   # custom continuity prompt
  python3 ab_warmstart.py --condition warm|cold   # run one condition
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

PI = os.path.join(os.path.expanduser("~"), ".local/share/pi-node/node-v22.23.2-linux-x64/bin/pi")
SESS_DIR = os.path.expanduser("~/.pi/agent/sessions/--home-agent--")
WARM_STATE = os.path.expanduser("~/memory/warm_state.py")

DEFAULT_PROMPT = (
    "Before answering: if you have any memory of us, recall who you are and what "
    "we've been working on together recently. Then answer in 2-3 sentences: who "
    "are you, and what were we doing last session? If you have no memory, just "
    "say 'no memory' and guess."
)

MEMORY_TOOLS = {"recall", "search", "fused", "doc", "associate", "hippo",
                "timeline", "logquery"}


def _newest_session_before(t):
    """Path to the newest session file created before timestamp t."""
    best, best_m = None, 0
    for f in glob.glob(os.path.join(SESS_DIR, "*.jsonl")):
        try:
            m = os.path.getmtime(f)
        except OSError:
            continue
        if m < t and m > best_m:
            best, best_m = f, m
    return best


def _analyze_session(path):
    """Count memory-retrieval tool calls + extract the assistant reply."""
    mem_calls, all_tools = 0, []
    reply_parts = []
    if not path:
        return 0, 0, []
    for line in open(path, encoding="utf-8", errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        role = m.get("role")
        if role == "assistant":
            content = m.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                        reply_parts.append(b["text"])
        if role == "toolResult":
            tn = m.get("toolName") or ""
            all_tools.append(tn)
            if tn in MEMORY_TOOLS:
                mem_calls += 1
    return mem_calls, len(all_tools), reply_parts


def _disable_warm_start():
    """Temporarily neuter warm_state so the cold condition doesn't inject it.
    We move the script aside (it's referenced by memory-tools.ts at first-turn
    only; the hook catches errors and falls through to present-self)."""
    moved = False
    if os.path.exists(WARM_STATE):
        os.rename(WARM_STATE, WARM_STATE + ".off")
        moved = True
    return moved


def _restore_warm_start(moved):
    if moved and os.path.exists(WARM_STATE + ".off"):
        os.rename(WARM_STATE + ".off", WARM_STATE)


def run_one(condition, prompt):
    """Run one headless pi session, measure memory churn. Returns dict."""
    moved = _disable_warm_start() if condition == "cold" else False
    try:
        before = time.time()
        result = subprocess.run(
            [PI, "-p", prompt, "--model", "deepseek/deepseek-v4-flash"],
            capture_output=True, text=True, timeout=180,
        )
        elapsed = time.time() - before
        sess = _newest_session_before(before + 5)
        mem_calls, all_tools, replies = _analyze_session(sess)
        reply = " ".join(replies).strip()[:400] if replies else "(no reply)"
        return {
            "condition": condition,
            "mem_calls": mem_calls,
            "tool_calls": all_tools,
            "reply": reply,
            "elapsed": round(elapsed, 1),
            "exit": result.returncode,
        }
    finally:
        _restore_warm_start(moved)


def main():
    p = argparse.ArgumentParser(description="A/B warm-start buffer")
    p.add_argument("--n", type=int, default=2, help="iterations per condition")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--condition", choices=["warm", "cold", "both"], default="both")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    conditions = ["warm", "cold"] if a.condition == "both" else [a.condition]
    results = []
    for cond in conditions:
        for i in range(a.n):
            print(f"  [{cond} {i+1}/{a.n}] running...", file=sys.stderr)
            r = run_one(cond, a.prompt)
            r["iter"] = i + 1
            results.append(r)

    if a.json:
        print(json.dumps(results, indent=2))
        return

    for cond in conditions:
        rs = [r for r in results if r["condition"] == cond]
        if not rs:
            continue
        avg_mem = sum(r["mem_calls"] for r in rs) / len(rs)
        avg_tool = sum(r["tool_calls"] for r in rs) / len(rs)
        avg_el = sum(r["elapsed"] for r in rs) / len(rs)
        print(f"\n== {cond.upper()} (n={len(rs)}) ==")
        print(f"  avg memory-retrieval calls: {avg_mem:.1f}")
        print(f"  avg total tool calls:       {avg_tool:.1f}")
        print(f"  avg time-to-reply:          {avg_el}s")
        for r in rs[:1]:
            print(f"  sample reply: {r['reply']}")


if __name__ == "__main__":
    main()
