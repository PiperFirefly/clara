#!/usr/bin/env python3
"""
Curiosity / goal scoring (#8) — what should I learn next?

The last subsystem from the 4-LLM cognitive-upgrade analysis. The analysis
called this "curiosity-score half nice; goal-proxy half hard" and deferred it —
so I'm building only the tractable half: a curiosity score that ranks my goal
portfolio AND surfaces specific knowledge gaps, so free-roam (and my conscious
choices) spend time where it's most worth learning.

Score = relevance × (1 − coverage) × (0.4 + interest) × recency boost:
  - relevance: how directly it serves a stated goal (priority-mapped).
  - coverage: how much I already know (goal `needs_work` is a 1−coverage proxy;
    knowledge gaps carry an explicit coverage estimate).
  - interest: intrinsic curiosity (0..1).
  - recency: never/recently-unexplored topics get a boost, so I don't fixate.

Usage:
  python3 curiosity.py score            # rank the goal portfolio by curiosity
  python3 curiosity.py gaps [--budget N]  # LLM-mine specific knowledge gaps
  python3 curiosity.py top [--k 10]     # merged ranked list (goals + gaps)
  python3 curiosity.py explore "topic"  # mark a topic explored (updates recency)
  python3 curiosity.py stats
"""
import argparse
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M
import worker_common
import state as st  # ephemeral state store (state.db)

BASE = os.path.dirname(os.path.abspath(__file__))
GOALS = os.path.join(BASE, "..", "freeroam", "goals.json")  # legacy path (no longer written)
STATE = os.path.join(BASE, "curiosity-state.json")

DEFAULT_BUDGET = 10
PRIORITY_RELEVANCE = {1: 1.0, 2: 0.8, 3: 0.5}
RECENCY_WINDOW = 7 * 86400  # seconds


def _ensure(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS curiosity("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "topic TEXT NOT NULL,"
        "gap TEXT,"
        "embedding BLOB,"
        "relevance REAL,"
        "coverage REAL,"        # 0 = know nothing, 1 = already know it
        "interest REAL,"
        "goal TEXT,"
        "score REAL,"
        "status TEXT DEFAULT 'open',"   # open | explored | resolved
        "source_key TEXT UNIQUE,"
        "last_explored REAL,"
        "created_at REAL)"
    )


def _conn():
    c = M.connect()
    _ensure(c)
    return c


def _llm(prompt, max_tokens=900):
    return worker_common.llm_call(prompt, max_tokens)


def _load_goals():
    # Goals now live in the ephemeral state store under goals/* (seeded from the
    # old goals.json during migration). Read the whole namespace as a dict.
    try:
        return st.get_prefix("goals")
    except Exception:
        return {}


def _recency_boost(last_explored):
    """0.5 for never/unrecently explored, decaying to 0 for very recent."""
    if not last_explored:
        return 0.5
    age = time.time() - last_explored
    if age >= RECENCY_WINDOW:
        return 0.5
    return 0.5 * (1.0 - age / RECENCY_WINDOW)


def _score(relevance, coverage, interest, last_explored=None):
    return round(relevance * (1.0 - coverage) * (0.4 + interest) *
                 (1.0 + _recency_boost(last_explored)), 4)


# --------------------------------------------------------------------------- #
# goal-level scoring (pure, data-driven from goals.json)
# --------------------------------------------------------------------------- #
def goal_scores():
    goals = _load_goals()
    rows = []
    for name, g in goals.items():
        if not isinstance(g, dict):
            continue
        rel = PRIORITY_RELEVANCE.get(g.get("priority"), 0.5)
        needs_work = g.get("needs_work", 0.5)
        interest = 0.5 if g.get("nucleus") else 0.3
        score = _score(rel, 1.0 - needs_work, interest, g.get("last_explored"))
        rows.append({
            "type": "goal", "topic": name, "gap": g.get("description", ""),
            "relevance": rel, "coverage": round(1.0 - needs_work, 2),
            "interest": interest, "goal": name, "score": score,
            "last_explored": g.get("last_explored"),
        })
    rows.sort(key=lambda r: -r["score"])
    return rows


# --------------------------------------------------------------------------- #
# LLM knowledge-gap mining
# --------------------------------------------------------------------------- #
_GAP_PROMPT = (
    "You are a curiosity worker for the agent, an AI agent. Given its goal below, "
    "identify 1-3 SPECIFIC knowledge gaps — concrete things it needs to learn or "
    "understand to make progress on this goal that it likely does not yet know. "
    "For each give:\n"
    ' - "gap": the specific question/topic (concise, one line).\n'
    ' - "coverage": 0..1 how well she already knows it (0 = knows nothing).\n'
    ' - "relevance": 0..1 how much learning it would advance the goal.\n'
    ' - "interest": 0..1 how intrinsically curious it is (0 boring .. 1 fascinating).\n'
    'Output ONLY a JSON array of objects with those four keys.\n\nGOAL: '
)


def gaps(budget=None, dry_run=False):
    budget = budget if budget is not None else int(os.environ.get("CURIOSITY_BUDGET", str(DEFAULT_BUDGET)))
    goals = _load_goals()
    # highest-priority, least-explored goals first
    order = sorted(goals.items(), key=lambda kv: (
        kv[1].get("priority", 9),
        -(kv[1].get("needs_work", 0.5)),
    )) if isinstance(goals, dict) else []
    stored = 0
    for name, g in order:
        if stored >= budget:
            break
        desc = g.get("description", name) if isinstance(g, dict) else str(g)
        out = _llm(_GAP_PROMPT + desc[:600], max_tokens=700)
        data = M._extract_json(out)
        if isinstance(data, dict):
            # model may wrap in {"gaps": [...]}
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        if not isinstance(data, list):
            continue
        for it in data:
            if stored >= budget:
                break
            if not isinstance(it, dict) or not it.get("gap"):
                continue
            gap = it["gap"].strip()
            try:
                cov = max(0.0, min(1.0, float(it.get("coverage", 0.3))))
                rel = max(0.0, min(1.0, float(it.get("relevance", 0.5))))
                intr = max(0.0, min(1.0, float(it.get("interest", 0.3))))
            except (TypeError, ValueError):
                continue
            score = _score(rel, cov, intr)
            key = f"gap:{name}:{gap[:60]}"
            if dry_run:
                print(f"  [dry-run] {name}: ({score}) {gap[:80]}")
                stored += 1
                continue
            with _conn() as c:
                if c.execute("SELECT 1 FROM curiosity WHERE source_key=?", (key,)).fetchone():
                    continue
                emb = M.embed([gap])[0].tobytes()
                c.execute(
                    "INSERT INTO curiosity(topic, gap, embedding, relevance, coverage, "
                    "interest, goal, score, source_key, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (name, gap, emb, rel, cov, intr, name, score, key, time.time()),
                )
            stored += 1
    print(f"curiosity.gaps: stored {stored} gap(s)")
    return {"stored": stored}


# --------------------------------------------------------------------------- #
# merged ranked list
# --------------------------------------------------------------------------- #
def top(k=10):
    rows = goal_scores()
    with _conn() as c:
        for r in c.execute("SELECT topic, gap, relevance, coverage, interest, goal, "
                           "score, status FROM curiosity WHERE status='open' "
                           "ORDER BY score DESC LIMIT ?", (k,)).fetchall():
            rows.append({"type": "gap", "topic": r["topic"], "gap": r["gap"],
                         "relevance": r["relevance"], "coverage": r["coverage"],
                         "interest": r["interest"], "goal": r["goal"],
                         "score": r["score"], "last_explored": None})
    rows.sort(key=lambda r: -r["score"])
    return rows[:k]


def render(rows, with_header=True):
    lines = []
    if with_header:
        lines.append(f"curiosity — {len(rows)} thing(s) worth learning next:")
    for r in rows:
        kind = "gap" if r["type"] == "gap" else "goal"
        lines.append(f"  [{kind} {r['score']}] {r['topic']}"
                     f"  (relevance {r['relevance']:.1f}, coverage {r['coverage']:.1f}, "
                     f"interest {r['interest']:.1f})")
        if r.get("gap") and r["type"] == "gap":
            lines.append(f"      → {r['gap'][:110]}")
    return "\n".join(lines)


def explore(topic):
    now = time.time()
    with _conn() as c:
        n = c.execute("UPDATE curiosity SET status='explored', last_explored=? "
                      "WHERE topic=? AND status='open'", (now, topic)).rowcount
    # also refresh the goal's last_explored in the goals namespace (state store)
    goals = _load_goals()
    if topic in goals and isinstance(goals[topic], dict):
        goals[topic]["last_explored"] = now
        try:
            st.set_prefix("goals", goals, durable=True, delete_missing=False)
        except Exception:
            pass
        try:
            M.emit_goal_snapshot(goals)
        except Exception:
            pass  # ledger reconciliation is best-effort
    print(f"curiosity.explore: marked {n} gap(s) + goal '{topic}' as explored")
    return {"explored": n}


def stats():
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM curiosity WHERE status='open'").fetchone()["n"]
        ex = c.execute("SELECT COUNT(*) n FROM curiosity WHERE status='explored'").fetchone()["n"]
    goals = _load_goals()
    print(f"curiosity: {n} open gap(s), {ex} explored, {len(goals)} goal(s)")


def main():
    p = argparse.ArgumentParser(description="curiosity / goal scoring")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score")
    g = sub.add_parser("gaps")
    g.add_argument("--budget", type=int, default=None)
    g.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("top")
    t.add_argument("--k", type=int, default=10)
    e = sub.add_parser("explore")
    e.add_argument("topic")
    sub.add_parser("stats")
    a = p.parse_args()

    if a.cmd == "score":
        print(render(goal_scores()))
    elif a.cmd == "gaps":
        gaps(budget=a.budget, dry_run=a.dry_run)
    elif a.cmd == "top":
        print(render(top(k=a.k)))
    elif a.cmd == "explore":
        explore(a.topic)
    elif a.cmd == "stats":
        stats()


if __name__ == "__main__":
    main()
