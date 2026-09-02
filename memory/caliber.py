#!/usr/bin/env python3
"""
CALIBER (P2-8) — pre/post reasoning-confidence calibration for the agent.

Goal: detect when *reasoning* makes the agent overconfident. We elicit a
confidence estimate BEFORE reasoning ("chance I'll solve this") and another
AFTER reasoning ("chance my answer is right"), log both, then later resolve
the real outcome and score the gap. The headline signal is the "overconfident"
row: post_confidence >> pre_confidence but the outcome was wrong — i.e. the
act of reasoning talked the agent into unjustified certainty.

Storage: the `calibration` table in memstore's SQLite DB (memory.db). It is
created idempotently here (CREATE IF NOT EXISTS) and never touches the
tool-level `tool_uses.pre_confidence` column (that is P1-6's, tool-scoped).
CALIBER is its own reasoning-scoped table.

No new dependencies: one prompt pattern + a scoring table.

Usage:
  python3 caliber.py calibrate "<task>" [answer]
  python3 caliber.py resolve <id> <0|1>
  python3 caliber.py score
  python3 caliber.py overconfident [-k 10]
"""
import argparse
import json
import math
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import memstore as M

# Test isolation: allow the whole module (including the CLI subprocess) to be
# pointed at a throwaway DB so tests never touch the live memory.db.
if os.environ.get("CALIBER_DB"):
    M.DB = os.environ["CALIBER_DB"]

_CREATE = """
CREATE TABLE IF NOT EXISTS calibration(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    pre_confidence REAL,        -- 0..1, before reasoning
    post_confidence REAL,       -- 0..1, after reasoning
    outcome INTEGER,            -- 0/1/NULL (filled on resolve)
    answer TEXT,                -- the agent's answer (optional, for audit)
    resolved_at REAL,
    created_at REAL
)
"""

# post_confidence must exceed pre_confidence by more than this to count as a
# "reasoning made me overconfident" candidate (and only when outcome == 0).
GAP = 0.2


def _ensure_table(conn):
    """Idempotently create the calibration table on the given connection."""
    conn.execute(_CREATE)
    conn.commit()


def _parse_confidence(out):
    """Parse an LLM reply into a float clamped to [0, 1]. None on failure."""
    if out is None:
        return None
    s = str(out).strip()
    m = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)", s)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return min(1.0, max(0.0, v))


def elicit_pre(task):
    """Before reasoning: probability (0..1) the task can be answered correctly."""
    prompt = (
        "Before you reason about this task, what is the probability (0 to 1) "
        "that you can answer/solve it correctly? Reply with just a number.\n\n"
        f"Task: {task}"
    )
    out = M.llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=16, temperature=0, model=M.MODEL_WORKER,
    )
    return _parse_confidence(out)


def elicit_post(task, answer):
    """After reasoning: probability (0..1) that the produced answer is correct."""
    prompt = (
        f"Task: {task}\n\n"
        f"You just produced this answer: {answer}\n\n"
        "What is the probability (0 to 1) that it is correct? "
        "Reply with just a number."
    )
    out = M.llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=16, temperature=0, model=M.MODEL_WORKER,
    )
    return _parse_confidence(out)


def calibrate(task, answer=None, outcome=None):
    """Elicit pre + post confidence, insert a row, return the record id + confidences.

    post_confidence is only elicited when an answer is supplied (there is no
    "chance my answer is right" without an answer); otherwise it is NULL.
    """
    pre = elicit_pre(task)
    post = elicit_post(task, answer) if answer is not None else None
    with M.connect() as conn:
        _ensure_table(conn)
        cur = conn.execute(
            "INSERT INTO calibration"
            "(task, pre_confidence, post_confidence, outcome, answer, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (task, pre, post, outcome, answer, time.time()),
        )
        cid = cur.lastrowid
    return {"id": cid, "pre_confidence": pre, "post_confidence": post}


def resolve(calib_id, outcome):
    """Mark a calibration row resolved with outcome 0/1. Returns rows updated."""
    outcome = int(outcome)
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    with M.connect() as conn:
        _ensure_table(conn)
        cur = conn.execute(
            "UPDATE calibration SET outcome=?, resolved_at=? WHERE id=?",
            (outcome, time.time(), calib_id),
        )
        return cur.rowcount


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def score():
    """Aggregate calibration metrics over resolved rows (outcome IS NOT NULL)."""
    with M.connect() as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT pre_confidence, post_confidence, outcome FROM calibration "
            "WHERE outcome IS NOT NULL"
        ).fetchall()
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "mean_pre": None,
            "mean_post": None,
            "brier_pre": None,
            "brier_post": None,
            "overconfident": 0,
        }
    outcomes = [r["outcome"] for r in rows]
    pre = [r["pre_confidence"] for r in rows]
    post = [r["post_confidence"] for r in rows]
    over = sum(
        1 for p, q, o in zip(pre, post, outcomes)
        if p is not None and q is not None
        and q > p + GAP and o == 0
    )
    return {
        "n": n,
        "accuracy": float(sum(outcomes) / n),
        "mean_pre": _mean(pre),
        "mean_post": _mean(post),
        "brier_pre": _mean([(x - o) ** 2 for x, o in zip(pre, outcomes) if x is not None]),
        "brier_post": _mean([(x - o) ** 2 for x, o in zip(post, outcomes) if x is not None]),
        "overconfident": over,
    }


def overconfident(k=10):
    """List overconfident rows (post > pre + GAP, wrong), most recent first."""
    with M.connect() as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT id, task, answer, pre_confidence, post_confidence, "
            "outcome, created_at, resolved_at FROM calibration "
            "WHERE post_confidence IS NOT NULL AND pre_confidence IS NOT NULL "
            "AND post_confidence > pre_confidence + ? AND outcome = 0 "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (GAP, int(k)),
        ).fetchall()
    return [dict(r) for r in rows]


def _excerpt(s, n=120):
    if not s:
        return None
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _forecast_calibration():
    """Pull resolved-forecast calibration (Brier + surprise) and per-category
    breakdown from the prediction ledger, so the report closes the loop between
    *scoring* (which already exists) and *acting on* it. Best-effort."""
    try:
        with M.connect() as c:
            resolved = c.execute(
                "SELECT AVG(brier) b, AVG(surprise) s, AVG(confidence) mc, "
                "COUNT(*) n, SUM(outcome) hits FROM forecasts "
                "WHERE status='resolved' AND outcome IS NOT NULL").fetchone()
            by_cat = c.execute(
                "SELECT category, AVG(brier) b, COUNT(*) n FROM forecasts "
                "WHERE status='resolved' AND outcome IS NOT NULL AND brier IS NOT NULL "
                "GROUP BY category ORDER BY b DESC").fetchall()
        out: dict = {"n": 0, "brier": None, "surprise": None, "accuracy": None,
                "by_category": []}
        if resolved and resolved["n"]:
            hits = resolved["hits"] or 0
            out = {"n": resolved["n"], "brier": resolved["b"],
                   "surprise": resolved["s"], "accuracy": round(hits / resolved["n"], 3),
                   "by_category": [{"category": r["category"], "brier": round(r["b"], 3),
                                     "n": r["n"]} for r in by_cat]}
        return out
    except Exception:
        return {"n": 0, "brier": None, "surprise": None, "accuracy": None, "by_category": []}


def report():
    """Aggregate reasoning (CALIBER) + forecast calibration into a compact monthly
    self-report, store it where morning-me reads it (doc store + a fact), and print.
    This is the missing feedback loop: the Brier/calibration tables existed but the
    scored output never reached the me that makes the next judgment."""
    s = score()
    f = _forecast_calibration()
    lines = []
    lines.append("# Calibration report — %s" % time.strftime("%Y-%m"))
    lines.append("")
    lines.append("## Reasoning confidence (CALIBER)")
    if s["n"]:
        lines.append(f"- resolved: {s['n']}")
        lines.append(f"- accuracy: {s['accuracy']:.3f}")
        lines.append(f"- mean pre-confidence: {s['mean_pre']:.3f}  |  Brier(pre): {s['brier_pre']:.3f}")
        lines.append(f"- mean post-confidence: {s['mean_post']:.3f}  |  Brier(post): {s['brier_post']:.3f}")
        lines.append(f"- reasoning overconfidence events (post>pre+GAP, wrong): {s['overconfident']}")
        if s["brier_pre"] is not None and s["brier_post"] is not None:
            delta = s["brier_post"] - s["brier_pre"]
            if delta > 0.02:
                lines.append(f"- WATCH: reasoning made me MORE miscalibrated (Brier +{delta:.3f}) —")
                lines.append("  talk myself down before final confidence.")
    else:
        lines.append("- no resolved calibration rows yet.")
    lines.append("")
    lines.append("## Forecast calibration (prediction ledger)")
    if f["n"]:
        lines.append(f"- resolved: {f['n']}")
        lines.append(f"- mean Brier: {f['brier']:.3f}  (baseline always-0.5 = 0.2500)")
        lines.append(f"- mean surprise: {f['surprise']:.2f} bits   accuracy: {f['accuracy']:.3f}")
        worst = [b for b in f["by_category"] if b["n"] >= 2][:3]
        if worst:
            lines.append("- worst-calibrated categories: " + ", ".join(
                f"{b['category']} (Brier {b['brier']:.3f}, n={b['n']})" for b in worst))
    else:
        lines.append("- no resolved forecasts yet.")
    body = "\n".join(lines)
    # store where morning-me reads it
    try:
        import docstore
        docstore.doc_set("agent/calibration_report", "report",
                         f"Calibration report {time.strftime('%Y-%m')}", body,
                         source="caliber.report", tags=["calibration", "monthly"])
    except Exception as e:
        lines.append(f"(docstore write failed: {e})")
    try:
        M.facts_set("calibration/last_report",
                    f"{time.strftime('%Y-%m-%d %H:%M')} | reasoning n={s['n']} "
                    f"BrierPre={s['brier_pre']} BrierPost={s['brier_post']} "
                    f"overconf={s['overconfident']} | forecast n={f['n']} Brier={f['brier']}")
    except Exception:
        pass
    print(body)
    return {"reasoning": s, "forecast": f}


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="caliber.py",
        description="CALIBER (P2-8) — pre/post reasoning-confidence calibration.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="elicit pre/post confidence and log a row")
    c.add_argument("task")
    c.add_argument("answer", nargs="?")
    c.add_argument("--outcome", type=int, choices=[0, 1], default=None)

    r = sub.add_parser("resolve", help="mark a row resolved with outcome 0/1")
    r.add_argument("id", type=int)
    r.add_argument("outcome", type=int, choices=[0, 1])

    sub.add_parser("score", help="aggregate calibration metrics")

    sub.add_parser("report", help="monthly calibration self-report (closes the loop)")

    o = sub.add_parser("overconfident", help="list overconfident rows")
    o.add_argument("-k", type=int, default=10)

    args = p.parse_args(argv)

    if args.cmd == "calibrate":
        print(json.dumps(calibrate(args.task, args.answer, args.outcome), indent=2))
    elif args.cmd == "resolve":
        updated = resolve(args.id, args.outcome)
        print(json.dumps({"id": args.id, "outcome": args.outcome, "updated": updated}))
    elif args.cmd == "score":
        print(json.dumps(score(), indent=2))
    elif args.cmd == "report":
        report()
    elif args.cmd == "overconfident":
        rows = overconfident(args.k)
        out = [
            {
                "id": x["id"],
                "task": _excerpt(x["task"]),
                "answer": _excerpt(x["answer"]),
                "pre_confidence": x["pre_confidence"],
                "post_confidence": x["post_confidence"],
                "outcome": x["outcome"],
            }
            for x in rows
        ]
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
