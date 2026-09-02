#!/usr/bin/env python3
"""
Hive orchestration (Stage 5+) — supervisor loop for the memory workers.

Replaces the naive "run every worker every hour" cron with an assess-then-
dispatch loop: it inspects the store for new work, queues only the workers
that are due, and runs them in priority order under a shared budget of
estimated LLM calls.

Workers (dispatch priority):
  decay    — prune stale low-value memories (no LLM; always runs)
  enhance  — rebuild the knowledge graph for un-graphed memories
  causal   — extract cause→effect links for un-causal-graphed memories
  evaluate — re-score importance of not-yet-scored memories
  dedup    — merge near-duplicate memories (soft-delete)
  verify   — scan for contradictions (log only)
  reason   — derive novel connections/insights

Each non-always worker carries a per-worker watermark (last max_id seen).
A worker is "due" when the store's max_id exceeds its watermark, so idle
hours cost nothing beyond the free `decay` pass. A worker's watermark is
advanced only after it completes successfully, so a failed run is retried.

Usage:
  python3 hive.py [run|status] [--budget N] [--dry-run]
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import memstore as M
import consolidate as C
import state as stmod  # ephemeral state store (state.db)

STATE = os.path.join(BASE, "hive-state.json")
DEFAULT_BUDGET = 40

# (name, priority, est LLM-call cost, always-run?, model)
# model: None = no LLM; "chat" = MODEL_WORKER (deepseek-chat, non-reasoning);
# the main agent runs MODEL_STRONG (deepseek-v4-pro, settings.json defaultModel).
WORKERS = [
    ("decay", 10, 0, True, None),
    ("enhance", 20, 1, False, "chat"),
    ("causal", 25, 1, False, "chat"),
    ("evaluate", 30, 1, False, "chat"),
    ("dedup", 40, 1, False, "chat"),
    ("verify", 50, 1, False, "chat"),
    ("reason", 60, 8, False, "chat"),
    ("belief_extract", 70, 2, False, "chat"),
    ("belief_propagate", 80, 0, False, None),
    ("forecast_extract", 72, 2, False, "chat"),
    ("forecast_resolve", 15, 0, True, None),
    ("person_model_extract", 74, 2, False, "chat"),
    ("affect_extract", 76, 2, False, "chat"),
]


def _state():
    return stmod.get("worker/hive", {})


def _save(st_):
    stmod.set("worker/hive", st_, durable=True)


def _max_id():
    with M.connect() as c:
        return c.execute("SELECT MAX(id) m FROM memories").fetchone()["m"] or 0


def assess(max_id=None):
    """Return (queue, max_id, state): workers that are due, in priority order."""
    max_id = max_id if max_id is not None else _max_id()
    st = _state()
    queue = []
    for name, pri, cost, always, model in WORKERS:
        if always or max_id > st.get(name, 0):
            queue.append({"name": name, "priority": pri, "cost": cost,
                          "always": always, "model": model})
    queue.sort(key=lambda w: w["priority"])
    return queue, max_id, st


def run(budget=None, dry_run=False):
    queue, max_id, st = assess()
    if budget is None:
        budget = int(os.environ.get("HIVE_BUDGET", str(DEFAULT_BUDGET)))
    budget = budget if budget > 0 else None  # <=0 means unlimited
    spent = 0
    dispatched = []
    for w in queue:
        if budget is not None and not w["always"] and spent + w["cost"] > budget:
            print(f"hive: budget hit ({spent}/{budget}); deferring {w['name']} (pri {w['priority']})")
            continue
        if dry_run:
            print(f"hive: [dry-run] would dispatch {w['name']} (pri {w['priority']}, ~{w['cost']} LLM)")
            dispatched.append(w["name"])
            continue
        print(f"hive: dispatch {w['name']} (pri {w['priority']}, ~{w['cost']} LLM, model {w['model'] or '-'})")
        getattr(C, w["name"])()
        spent += w["cost"]
        if w["name"] in ("reason", "belief_extract", "forecast_extract",
                           "person_model_extract", "affect_extract"):
            # self-watermarking workers: their budget may stop short of the store's
            # max_id, so read their watermark back (now from the state store).
            try:
                st[w["name"]] = stmod.get(f"worker/{w['name']}", {}).get("max_id", max_id)
            except Exception:
                st[w["name"]] = max_id
        else:
            st[w["name"]] = max_id
        _save(st)
        dispatched.append(w["name"])
    print(f"hive: done. dispatched={dispatched} spent~{spent} est LLM calls, watermark={max_id}")
    return {"dispatched": dispatched, "spent": spent, "max_id": max_id}


def status():
    queue, max_id, st = assess()
    print(f"store max_id = {max_id}")
    for name, pri, cost, always, model in WORKERS:
        last = st.get(name, 0)
        due = always or max_id > last
        mark = "DUE " if due else "idle"
        print(f"  {name:9s} pri={pri:2d} cost={cost} model={(model or '-'):6s} last={last:4d} -> {mark}")


def main():
    p = argparse.ArgumentParser(description="Hive supervisor for memory workers")
    p.add_argument("cmd", nargs="?", default="run", choices=["run", "status"])
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.cmd == "status":
        status()
    else:
        run(budget=a.budget, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
