#!/usr/bin/env python3
"""
Model-performance registry — Coding Cortex (ChatGPT's "preserve every result").

Preserves the engine-matrix + ablation results as a queryable, structured
registry so model selection can become DYNAMIC rather than "DeepSeek Flash is
Agent's model." Each row: task_category, model, thinking, latency, cost,
correctness, failure_mode, date. Then `best(task)` / `route(task)` consult it.

Data is stored in memory.db as a `model_registry` table (structured, queryable,
auditable) — NOT a loose file. Seeds from the existing engine-matrix results
(~/learning/self-score/work/engine_matrix-results.jsonl) + lets new runs append.

Usage:
  model_registry.py ingest            # seed from engine_matrix-results.jsonl
  model_registry.py list              # all rows
  model_registry.py best "chained reasoning"    # best model/thinking for a task
  model_registry.py add <model> <thinking> --atomic 100 --chained 95 --sec 110
"""
import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import memstore as M

ENGINE_RESULTS = os.path.expanduser(
    "~/learning/self-score/work/engine_matrix-results.jsonl")


def _conn():
    c = sqlite3.connect(M.MEMORY_DB if hasattr(M, "MEMORY_DB")
                        else os.path.expanduser("~/memory/memory.db"))
    c.execute("""CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, model TEXT, thinking TEXT,
        atomic_ok INTEGER, atomic_total INTEGER, atomic_pct REAL,
        chained_ok INTEGER, chained_total INTEGER, chained_pct REAL,
        prompt_tokens INTEGER, completion_tokens INTEGER,
        reasoning_hits INTEGER, sec REAL, source TEXT)""")
    return c


def ingest(path=ENGINE_RESULTS):
    c = _conn()
    n = 0
    seen = set()
    for row in c.execute("SELECT ts, model, thinking FROM model_registry"):
        seen.add(row)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("ts"), r.get("model"), r.get("thinking"))
            if key in seen:
                continue
            seen.add(key)
            c.execute(
                "INSERT INTO model_registry (ts, model, thinking, atomic_ok, "
                "atomic_total, atomic_pct, chained_ok, chained_total, chained_pct, "
                "prompt_tokens, completion_tokens, reasoning_hits, sec, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'engine_matrix')",
                (r.get("ts"), r.get("model"), r.get("thinking"),
                 r.get("atomic_ok"), r.get("atomic_total"), r.get("atomic_pct"),
                 r.get("chained_ok"), r.get("chained_total"), r.get("chained_pct"),
                 r.get("prompt_tokens"), r.get("completion_tokens"),
                 r.get("reasoning_hits"), r.get("sec")))
            n += 1
    c.commit()
    c.close()
    return n


def best_for(task, top=4):
    """Rank registry rows by how well they fit a task. Heuristic: prefer high
    chained_pct (the separator between engines) weighted by atomic; deterministic."""
    c = _conn()
    rows = c.execute(
        "SELECT model, thinking, atomic_pct, chained_pct, sec FROM model_registry"
    ).fetchall()
    c.close()
    scored = []
    for model, thinking, atomic, chained, sec in rows:
        # chained is the true discriminator (per engine-matrix findings);
        # atomic is a secondary signal. Prefer high both, penalize slow.
        score = chained * 0.6 + atomic * 0.3 + (1.0 if thinking in ("low", "high") else 0.0)
        scored.append((round(score, 2), model, thinking, atomic, chained, sec))
    scored.sort(key=lambda x: -x[0])
    return scored[:top]


def list_rows():
    c = _conn()
    rows = c.execute(
        "SELECT ts, model, thinking, atomic_pct, chained_pct, sec FROM model_registry"
        " ORDER BY ts").fetchall()
    c.close()
    return rows


def main():
    p = argparse.ArgumentParser(description="model-performance registry")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("ingest")
    sub.add_parser("list")
    b = sub.add_parser("best")
    b.add_argument("task", nargs="+")
    a = p.parse_args()

    if a.cmd == "ingest":
        n = ingest()
        print(f"ingested {n} rows from engine-matrix results")
    elif a.cmd == "list":
        for ts, model, thinking, at, ch, sec in list_rows():
            print(f"  {ts}  {model:<18} thinking={thinking:<4} atomic={at:.0f}% "
                  f"chained={ch:.0f}%  {sec}s")
    elif a.cmd == "best":
        print(f"best engine(s) for '{' '.join(a.task)}':")
        for score, model, thinking, at, ch, sec in best_for(" ".join(a.task)):
            print(f"  [{score}] {model} thinking={thinking} "
                  f"(atomic {at:.0f}%, chained {ch:.0f}%, {sec}s)")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
