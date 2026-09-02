#!/usr/bin/env python3
"""
SelfBench — Agent's private coding benchmark (Coding Cortex Phase 1).

A corpus of real bugs from my own history, each with a verifiable gold
regression test. This is what turns "I've fixed bugs" into "I can measure how
well I fix bugs" — the foundation for every later self-improvement decision.

Usage:
  selfbench.py list [--all]      # show tasks (--all includes held-out)
  selfbench.py stats             # corpus summary
  selfbench.py verify [id ...]   # run gold tests (default: all with gold)
  selfbench.py heldout <id>      # toggle a task's held-out flag

Exit 0 iff all verified gold tests pass.
"""
import importlib.util
import json
import os
import sys
from collections import Counter

DIR = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(DIR, "tasks.json")
GOLD = os.path.join(DIR, "gold_suite.py")


def load():
    with open(TASKS) as f:
        return json.load(f)["tasks"]


def cmd_list(show_heldout=False):
    for t in load():
        if t.get("held_out") and not show_heldout:
            continue
        gold = t.get("gold") or "-"
        print(f"{t['id']}  [{t['category']:<13}] {t['difficulty']:<6} gold={gold:<7} "
              f"heldout={t.get('held_out', False)}  {t['title']}")


def cmd_stats():
    ts = load()
    n_gold = sum(1 for t in ts if t.get("gold"))
    n_held = sum(1 for t in ts if t.get("held_out"))
    cats = Counter(t["category"] for t in ts)
    print(f"tasks={len(ts)}  with_gold={n_gold}  held_out={n_held}")
    for c, k in cats.most_common():
        print(f"  {c}: {k}")


def _run_gold(name):
    spec = importlib.util.spec_from_file_location("gold_suite", GOLD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    getattr(m, name)()  # raises on failure


def cmd_verify(ids):
    ts = {t["id"]: t for t in load()}
    targets = ids if ids else [t["id"] for t in load() if t.get("gold")]
    passed = failed = 0
    for tid in targets:
        t = ts.get(tid)
        if not t or not t.get("gold"):
            print(f"skip {tid} (no gold)")
            continue
        try:
            _run_gold(t["gold"])
            print(f"PASS {tid}  {t['title']}")
            passed += 1
        except Exception as ex:
            print(f"FAIL {tid}  {t['title']}  -> {type(ex).__name__}: {ex}")
            failed += 1
    print(f"\nselfbench: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


def cmd_heldout(tid):
    with open(TASKS) as f:
        data = json.load(f)
    for t in data["tasks"]:
        if t["id"] == tid:
            t["held_out"] = not t.get("held_out", False)
    with open(TASKS, "w") as f:
        json.dump(data, f, indent=2)
    t = next(x for x in data["tasks"] if x["id"] == tid)
    print(f"{tid} held_out={t['held_out']}")


def main():
    args = sys.argv[1:]
    if not args or args[0] == "list":
        cmd_list("--all" in args)
    elif args[0] == "stats":
        cmd_stats()
    elif args[0] == "verify":
        cmd_verify(args[1:])
    elif args[0] == "heldout":
        if len(args) > 1:
            cmd_heldout(args[1])
        else:
            print("usage: selfbench.py heldout <id>")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
