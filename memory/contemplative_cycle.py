#!/usr/bin/env python3
"""contemplative_cycle.py — the logs→patterns→skills→rules knowledge cycle (idea #16).

From shimo4228/contemplative-agent: a CLI agent runs a knowledge cycle over its own
logs — logs → patterns → skills → rules — with EVERY promotion passing through a
human approval gate. Security by deliberate slowness: nothing self-modifies without
a person signing off.

This is the iterative self-improvement loop made *gated* and *auditable*. It reuses
my existing self-knowledge data (tool_uses, contradictions, self_thoughts) as the
"logs" input.

Pipeline:
  1. EXTRACT  — pull recent raw signals (tool failures, surprises, contradictions).
  2. PATTERN  — cluster them into recurring themes (deterministic grouping first).
  3. PROPOSE  — propose a distilled lesson/skill per pattern (a candidate reflex).
  4. QUEUE    — stage candidates for approval; nothing is applied until approved.
  5. APPROVE  — a human (operator) approves/rejects; approved ones become rules.

Usage:
  python3 contemplative_cycle.py extract [--n 200]   # pull raw signals
  python3 contemplative_cycle.py propose [--k 5]     # propose candidate lessons
  python3 contemplative_cycle.py queue               # show pending approvals
  python3 contemplative_cycle.py approve <id> [--reject]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_cycle(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT,                 -- pattern | skill | rule
    source TEXT,                -- what log/signal it came from
    content TEXT,               -- the pattern description / distilled lesson
    evidence TEXT,              -- supporting signal (json)
    status TEXT DEFAULT 'candidate',  -- candidate | approved | rejected
    created_at REAL, approved_at REAL)
"""


def _ensure(c):
    c.execute(_SCHEMA)


def _conn():
    c = M.connect()
    _ensure(c)
    return c


def extract(n=200):
    """Pull recent raw self-knowledge signals: tool failures + surprises."""
    signals = []
    with _conn() as c:
        fails = c.execute(
            "SELECT task, tool, outcome FROM tool_uses "
            "WHERE success=0 AND outcome != '' ORDER BY id DESC LIMIT ?", (int(n),)).fetchall()
        for r in fails:
            signals.append({"type": "tool-failure", "tool": r["tool"],
                            "task": r["task"], "outcome": r["outcome"]})
        su = c.execute("SELECT forecast_id, surprise FROM surprise_log "
                       "ORDER BY id DESC LIMIT 20").fetchall()
        for r in su:
            signals.append({"type": "surprise", "forecast_id": r["forecast_id"],
                            "surprise": r["surprise"]})
    # deterministic grouping: group tool-failures by tool
    from collections import defaultdict
    by_tool = defaultdict(list)
    for s in signals:
        if s["type"] == "tool-failure":
            by_tool[s["tool"]].append(s)
    print(f"# extracted {len(signals)} raw signals "
          f"({len(fails) if 'fails' in dir() else '?'} tool-failures, {len(su)} surprises)")
    for tool, items in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
        print(f"  {tool}: {len(items)} failures")
    return {"n_signals": len(signals), "by_tool": {k: len(v) for k, v in by_tool.items()}}


def _store(stage, source, content, evidence):
    with _conn() as c:
        cur = c.execute("INSERT INTO knowledge_cycle(stage, source, content, evidence, "
                        "status, created_at) VALUES(?,?,?,?, 'candidate', ?)",
                        (stage, source, content, json.dumps(evidence), time.time()))
        return cur.lastrowid


def propose(k=5):
    """Distill candidate lessons from recurring patterns and queue them for approval."""
    from collections import defaultdict
    with _conn() as c:
        fails = c.execute(
            "SELECT tool, COUNT(*) n FROM tool_uses WHERE success=0 "
            "GROUP BY tool HAVING n>=3 ORDER BY n DESC LIMIT ?", (int(k),)).fetchall()
    staged = 0
    for tool, n in fails:
        # a candidate skill: this tool fails often — verify/learn before trusting
        content = (f"Reflex: tool '{tool}' has {n} recorded failures; "
                   f"prefer verification/abstention before relying on it.")
        # skip if an identical candidate already exists
        exists = _conn().execute(
            "SELECT 1 FROM knowledge_cycle WHERE stage='skill' AND content=?", (content,)).fetchone()
        if exists:
            continue
        _store("skill", "tool_uses/success=0", content, {"tool": tool, "failures": n})
        staged += 1
        print(f"  queued skill candidate: {content}")
    print(f"# proposed {staged} new skill candidate(s) (waiting for approval)")
    return {"staged": staged}


def queue():
    with _conn() as c:
        rows = c.execute("SELECT id, stage, source, content, evidence, status "
                         "FROM knowledge_cycle WHERE status='candidate' "
                         "ORDER BY id").fetchall()
    if not rows:
        print("-- no candidates awaiting approval")
        return []
    for r in rows:
        print(f"#{r['id']} [{r['stage']}] ({r['source']})")
        print(f"   {r['content']}")
    print(f"-- {len(rows)} candidate(s) awaiting approval")
    return rows


def approve(fid, reject=False):
    with _conn() as c:
        row = c.execute("SELECT id, status FROM knowledge_cycle WHERE id=?", (int(fid),)).fetchone()
        if not row:
            print(f"no candidate #{fid}")
            return 1
        if row["status"] != "candidate":
            print(f"#{fid} already {row['status']}")
            return 1
        status = "rejected" if reject else "approved"
        c.execute("UPDATE knowledge_cycle SET status=?, approved_at=? WHERE id=?",
                  (status, time.time() if not reject else None, int(fid)))
    print(f"#{fid} -> {status}")
    return 0


def main():
    p = argparse.ArgumentParser(description="contemplative knowledge cycle (gated)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--n", type=int, default=200)
    pr = sub.add_parser("propose"); pr.add_argument("--k", type=int, default=5)
    sub.add_parser("queue")
    ap = sub.add_parser("approve"); ap.add_argument("id", type=int); ap.add_argument("--reject", action="store_true")
    a = p.parse_args()
    if a.cmd == "extract": extract(a.n)
    elif a.cmd == "propose": propose(a.k)
    elif a.cmd == "queue": queue()
    elif a.cmd == "approve": sys.exit(approve(a.id, a.reject))


if __name__ == "__main__":
    main()
