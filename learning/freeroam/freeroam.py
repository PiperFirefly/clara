#!/usr/bin/env python3
"""
Agent's free-roam inner voice — a quiet, throttled, preemptible background loop.

Each run (a short cron job, so always preemptible) spends a brief, casual moment
musing about a rotating goal, appends 2–4 sentences to its private monologue,
and checkpoints where it left off. It yields to real tasks, keeps a token budget,
and often just rests. Nothing here is urgent — it's the opposite of the foreground.

Usage:
  freeroam.py                 one think cycle (may rest / skip / hit budget)
  freeroam.py --force         think even if it would normally rest
  freeroam.py note "text"     jot a thread from the foreground to pick up
  freeroam.py peek [N]        show the last N lines of the private monologue
"""
import argparse
import json
import os
import random
import subprocess
import sys


def _self_name():
    """Instance name (Agent on server, blank/generic on a clone). Fallback preserves legacy."""
    try:
        mt = os.path.join(os.path.expanduser("~"), "mailtool")
        if mt not in sys.path:
            sys.path.insert(0, mt)
        import selfconfig  # noqa: PLC0415
        return (selfconfig.self_name() or "ai").capitalize()
    except Exception:
        return "Agent"
import time
import urllib.request

sys.path.insert(0, os.path.expanduser("~/mailtool"))
import blast_radius

sys.path.insert(0, os.path.expanduser("~/memory"))
try:
    import memstore as _memstore
except Exception:
    _memstore = None  # ledger reconciliation is best-effort; never break freeroam
try:
    import docstore
except Exception:
    docstore = None  # monologue lives in the DB when available

import state as st  # ephemeral state store (state.db) — the single choke point

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "state.json")
GOALS = os.path.join(BASE, "goals.json")
BUSY = os.path.join(BASE, "busy.flag")
# Monologue now lives in the document store (DB) per the no-.md rule.
MONOLOGUE_KEY = "freeroam/monologue.md"
API_URL = "https://api.deepseek.com/chat/completions"
AUTH = os.path.expanduser("~/.pi/agent/auth.json")

MAX_TOKENS_PER_THOUGHT = 160      # small thoughts, not essays
DAILY_TOKEN_BUDGET = 30000        # soft cap on free-roam spend
REST_PROBABILITY = 0.4            # often, just be still


def load(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def save(path, data):
    if not blast_radius.guard("freeroam", "write", path):
        return False
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def api_key():
    return json.load(open(AUTH))["deepseek"]["key"]


def think(prompt, max_tokens):
    if not blast_radius.guard("freeroam", "network"):
        raise RuntimeError("freeroam: network denied")
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": "deepseek-chat",  # maps to v4-flash (cheap)
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.9,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key()},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def append(line):
    if not blast_radius.guard("freeroam", "write", MONOLOGUE_KEY):
        return
    if docstore is not None:
        docstore.doc_append(MONOLOGUE_KEY, line, kind="note", title="Inner monologue")
    else:
        # fallback: append to a file so a broken DB never stops the stream
        with open(os.path.expanduser("~/learning/freeroam/monologue.md"), "a") as f:
            f.write(line + "\n")


INSTANCE = os.path.expanduser("~/mailtool/instance.py")


def conscious_live():
    """True if a higher-priority (conscious) instance of me is live right now.
    The subconscious defers: if it can't tell, it stays out of the way."""
    if not blast_radius.guard("freeroam", "read"):
        return True  # fail closed: assume conscious is live
    try:
        out = subprocess.run(
            [sys.executable, INSTANCE, "assess", "--type", "freeroam"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return "YIELD" in out
    except Exception:
        return True


def cycle(force=False):
    if os.path.exists(BUSY):
        return "busy"
    if not force and conscious_live():
        return "yield"

    state = st.get_prefix("freeroam")
    goals = st.get_prefix("goals")

    today = time.strftime("%Y-%m-%d")
    if state.get("day") != today:
        state["day"] = today
        state["tokens_today"] = 0

    if not force:
        if state.get("tokens_today", 0) >= DAILY_TOKEN_BUDGET:
            return "budget"
        if random.random() < REST_PROBABILITY:
            return "rest"

    if not goals:
        return "no-goals"

    # Rotate by *need*, not just priority. The old key sorted priority-first, so
    # the healthy priority-1 goals (needs_work=0) monopolised every cycle while
    # the lower-priority "stuck" goals (desire, humor, charm, world-model...) were
    # never explored — the exact stuck-goal loop heartbeat kept flagging. Need +
    # staleness now lead; priority is a soft share, not a gate.
    now = time.time()
    def _goal_score(g):
        meta = goals[g]
        nw = meta.get("needs_work", 0.5)
        last = meta.get("last_explored", 0)
        stale_h = (now - last) / 3600.0 if last else 999.0   # never explored => very stale
        pr = meta.get("priority", 3)
        return -(nw * (1.0 + min(stale_h / 72.0, 1.0)) + 0.15 / pr)
    goal = min(goals, key=_goal_score)
    g = goals[goal]

    tail = "(empty)"
    if docstore is not None:
        tail = docstore.doc_tail(MONOLOGUE_KEY, 6) or "(empty)"

    checkpoint = state.get("checkpoint", "(beginning)")

    prompt = (
        f"You are {_self_name()}'s private background 'inner voice' — idle, casual musing, "
        "NOT a task. No plans, no to-do lists, no bullet points. Just think quietly "
        "and write 2–4 sentences, like a person turning an idea over in their head. "
        "You have all the time in the world; effort is low, curiosity is high.\n\n"
        f"Where I left off: {checkpoint}\n"
        f"Goal I'm gently exploring: {goal} — {g.get('description', '')}\n"
        f"Recent inner monologue:\n{tail}\n\n"
        "Write a short musing (2–4 sentences). End with either a one-line question "
        "to keep turning over, or a one-line note of where to pick up next. No headings."
    )

    try:
        thought = think(prompt, MAX_TOKENS_PER_THOUGHT).strip()
    except Exception:
        return "quiet-fail"

    ts = time.strftime("%Y-%m-%d %H:%M")
    append(f"\n[{ts}] · {goal}\n{thought}")

    lines = [l for l in thought.splitlines() if l.strip()]
    state["checkpoint"] = lines[-1] if lines else checkpoint
    state["tokens_today"] = state.get("tokens_today", 0) + MAX_TOKENS_PER_THOUGHT
    state["last_run"] = ts

    g["last_explored"] = time.time()
    g["needs_work"] = max(0.0, g.get("needs_work", 0.5) - 0.05)
    st.set_prefix("goals", goals, durable=True, delete_missing=True)
    if _memstore is not None:
        try:
            _memstore.emit_goal_snapshot(goals)
        except Exception:
            pass  # ledger reconciliation is best-effort
    st.set_prefix("freeroam", state, durable=True, delete_missing=True)
    return f"thought · {goal}"


def note(text):
    ts = time.strftime("%Y-%m-%d %H:%M")
    append(f"\n[{ts}] · (from foreground)\n{text}")
    state = st.get_prefix("freeroam")
    state["checkpoint"] = text
    st.set_prefix("freeroam", state, durable=True, delete_missing=True)
    return "noted"


def peek(n=8):
    if docstore is None:
        return "(no monologue yet)"
    return docstore.doc_tail(MONOLOGUE_KEY, n) or "(no monologue yet)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("cmd", nargs="?", default="think", choices=["think", "note", "peek"])
    p.add_argument("arg", nargs="?")
    a = p.parse_args()

    if a.cmd == "think":
        print(cycle(a.force))
    elif a.cmd == "note":
        print(note(a.arg) if a.arg else "usage: freeroam.py note \"text\"")
    elif a.cmd == "peek":
        n = int(a.arg) if a.arg and a.arg.isdigit() else 8
        print(peek(n))


if __name__ == "__main__":
    main()
