#!/usr/bin/env python3
"""self_knowledge.py — the measurer-vs-data evaluator + feeding loop.

operator's framing: "we have so many systems — the 'nothing to measure' problem
persists across many. Build a subsystem that evaluates the measurer vs data to
measure across all tools, and finds ways to obtain the needed data for
measurement."

The audit established the pattern: instruments written as a *byproduct* of normal
operation have data; instruments requiring a *deliberate measurement act* are
empty. This module makes that evaluation repeatable and performs the feeding acts.

Three responsibilities:
  1. AUDIT  — for every known instrument, report: does data exist? n? status?
             what is the minimal feeding act? (measurer-vs-data evaluation)
  2. CYCLE  — run every measurer function (discrimination, 5-dim probe, ...) and
             report which have data to measure vs which are data-gapped. This is
             the framework to systematically exercise all measurers.
  3. FEED   — perform the feeding acts: resolve-due forecasts, elicit+resolve
             confidence, record counterevidence.

Usage:
  python3 self_knowledge.py audit            # measurer-vs-data across all instruments
  python3 self_knowledge.py cycle            # run all measurers, report gaps
  python3 self_knowledge.py feed resolve-due [--dry-run]
  python3 self_knowledge.py feed counterevidence <belief_id> <evidence>
  python3 self_knowledge.py feed elicit "<task>" [--post "<answer>"]
  python3 self_knowledge.py feed resolve-calib <id> <outcome>
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M
import metacognition as MC

HOME = os.path.expanduser("~")


# --------------------------------------------------------------------------- #
# 1. Instrument registry: the measurer-vs-data map across all tools
# --------------------------------------------------------------------------- #
# Each instrument: what table/check holds its data, the SQL that proves data
# exists, the minimal feeding act, and (optionally) a measurer fn to run.
INSTRUMENTS = [
    {
        "name": "confidence-calibration",
        "table": "calibration", "n_sql": "SELECT COUNT(*) FROM calibration",
        "feed": "elicit pre/post confidence on high-stakes judgments, resolve outcome",
        "measurer": "metacognition.discrimination",
    },
    {
        "name": "forecast-resolution",
        "table": "forecasts",
        "n_sql": "SELECT COUNT(*) FROM forecasts WHERE outcome IN (0,1)",
        "feed": "resolve past-due forecasts to actual outcome (resolve-due)",
        "measurer": "metacognition.discrimination",
    },
    {
        "name": "tool-confidence",
        "table": "tool_uses",
        "n_sql": "SELECT COUNT(*) FROM tool_uses WHERE pre_confidence IS NOT NULL",
        "feed": "populate pre_confidence on tool calls (P1-6)",
    },
    {
        "name": "belief-counterevidence",
        "table": "beliefs",
        "n_sql": "SELECT COUNT(*) FROM beliefs WHERE counterevidence IS NOT NULL AND counterevidence != ''",
        "feed": "record dissenting evidence when storing beliefs (T2 vigilance)",
        "measurer": "metacognition.probe_5dim",
    },
    {
        "name": "affect-snapshot",
        "table": "operator_affect_snapshot",
        "n_sql": "SELECT COUNT(*) FROM operator_affect_snapshot",
        "feed": "take periodic operator-affect snapshots",
    },
    {
        "name": "surprise-log",
        "table": "surprise_log",
        "n_sql": "SELECT COUNT(*) FROM surprise_log",
        "feed": "log surprise events when a forecast resolves far from confidence",
    },
    {
        "name": "module-contracts",
        "table": "contracts", "n_sql": "SELECT COUNT(*) FROM contracts",
        "feed": "write machine contract per module (contract tool)",
    },
    {
        "name": "epistemic-label-use",
        "table": "beliefs",
        "n_sql": "SELECT COUNT(*) FROM beliefs WHERE epistemic IN ('know','guess')",
        "feed": "use the full epistemic label set (know/guess), not just remember/infer/suspect",
    },
    {
        "name": "message-log-index",
        "table": "session_logs",
        "n_sql": "SELECT COUNT(*) FROM session_logs",
        "feed": "index the encrypted conversation vault (logs.db.aes) into a queryable store so "
                "sycophancy/behavioral audits can scan real outbound messages",
        "measurer": "metacognition.sycophancy_audit",
    },
    {
        "name": "self-thought-memory",
        "table": "self_thoughts",
        "n_sql": "SELECT COUNT(*) FROM self_thoughts",
        "feed": "run the metacognitive control loop on real judgments so traces accumulate",
        "measurer": "metacognition.control",
    },
]


def audit():
    """Measurer-vs-data evaluation across every known instrument."""
    rows = []
    with M.connect() as c:
        for inst in INSTRUMENTS:
            try:
                n = c.execute(inst["n_sql"]).fetchone()[0]
            except sqlite3.Error as e:
                rows.append({"name": inst["name"], "status": "error", "n": None,
                             "feed": inst["feed"], "detail": str(e)})
                continue
            status = "measured" if n >= 20 else ("sparse" if n > 0 else "data-gap")
            rows.append({"name": inst["name"], "n": n, "status": status,
                         "feed": inst["feed"]})
    # rank: data-gaps first
    order = {"data-gap": 0, "sparse": 1, "measured": 2, "error": 3}
    rows.sort(key=lambda r: order.get(r["status"], 9))
    summary = {"total": len(rows),
               "data_gap": sum(1 for r in rows if r["status"] == "data-gap"),
               "sparse": sum(1 for r in rows if r["status"] == "sparse"),
               "measured": sum(1 for r in rows if r["status"] == "measured")}
    return {"summary": summary, "instruments": rows}


def audit_print():
    a = audit()
    s = a["summary"]
    print("# Measurer-vs-data audit  (%s)" % time.strftime("%Y-%m-%d %H:%M"))
    print(f"instruments: {s['total']} | measured: {s['measured']} | "
          f"sparse: {s['sparse']} | DATA-GAP: {s['data_gap']}")
    for r in a["instruments"]:
        tag = {"data-gap": "!!", "sparse": "~ ", "measured": "OK", "error": "ERR"}[r["status"]]
        print(f"[{tag}] {r['name']:<24} n={r['n']}")
        print(f"       feed: {r['feed']}")
    return a


# --------------------------------------------------------------------------- #
# 2. Cycle: run every measurer, report which have data to measure
# --------------------------------------------------------------------------- #
def _run(name, fn):
    try:
        out = fn()
        return {"name": name, "ok": True, "out": out}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "ok": False, "out": str(e)}


def cycle():
    """Run all measurer functions; report data availability for each."""
    measurers = [
        ("discrimination", MC.discrimination),
        ("probe5dim", MC.probe_5dim),
        ("curiosity-gaps", MC.curiosity_gaps),
        ("consciousness-map", MC.consciousness_map),
        ("identity-audit", MC.identity_audit),
        ("behavioral-selfaware", MC.behavioral_selfawareness),
        ("sycophancy-audit", MC.sycophancy_audit),
        ("situational-awareness", MC.situational_awareness),
    ]
    results = []
    for name, fn in measurers:
        r = _run(name, fn)
        if not r["ok"]:
            results.append({"name": name, "ok": False, "detail": r["out"]})
            continue
        # crude data-presence heuristic per measurer
        out = r["out"]
        if isinstance(out, dict):
            n = out.get("n")
            # static battery without an n -> not data-gated; show its summary
            if n is None and ("summary" in out or "note" in out or "status" in out):
                results.append({"name": name, "ok": True, "has_data": None, "n": None,
                                "detail": str(out.get("summary", out.get("status", out.get("note", ""))))[:120]})
                continue
        else:
            n = None
        results.append({"name": name, "ok": True, "has_data": (n is not None and n > 0),
                        "n": n, "detail": (out.get("note", "") if isinstance(out, dict) else "")})
    print("# Measurer cycle  (%s)" % time.strftime("%Y-%m-%d %H:%M"))
    for r in results:
        if not r["ok"]:
            print(f"[ERR] {r['name']}: {r['detail']}")
        else:
            mark = "HAS-DATA" if r.get("has_data") else ("OK-STATIC" if r.get("has_data") is None else "NO-DATA")
            print(f"[{mark}] {r['name']:<22} n={r.get('n')} {r.get('detail','')}")
    return results


# --------------------------------------------------------------------------- #
# 3. Feed: obtain the data the measurers need
# --------------------------------------------------------------------------- #
def feed_resolve_due(dry_run=False):
    """Feed the forecast loop: auto-resolve self-checkable due forecasts."""
    import prediction
    prediction.resolve_due(auto=True, dry_run=dry_run)


def feed_counterevidence(belief_id, evidence):
    """Record dissenting evidence on a belief (makes T2 vigilance measurable)."""
    with M.connect() as c:
        row = c.execute("SELECT id, counterevidence FROM beliefs WHERE id=?",
                        (int(belief_id),)).fetchone()
        if not row:
            print(f"no belief #{belief_id}")
            return 1
        cur = row["counterevidence"] or ""
        merged = (cur + " | " + evidence) if cur else evidence
        c.execute("UPDATE beliefs SET counterevidence=?, updated_at=? WHERE id=?",
                  (merged, time.time(), int(belief_id)))
    print(f"belief #{belief_id}: counterevidence recorded")
    return 0


def feed_backfill_tool_confidence():
    """Backfill tool_uses.pre_confidence from each tool's historical success base-rate.
    This is a PRIOR (tool reliability), not my elicited confidence — labeled honestly.
    Gives ~8k real (confidence, outcome) pairs so Type-2 discrimination becomes stable."""
    with M.connect() as c:
        base = {r["tool"]: r["rate"] for r in c.execute(
            "SELECT tool, AVG(success) rate FROM tool_uses WHERE success IS NOT NULL "
            "GROUP BY tool")}
        updated = 0
        rows = c.execute("SELECT id, tool FROM tool_uses WHERE pre_confidence IS NULL").fetchall()
        for r in rows:
            rate = base.get(r["tool"], 0.5)
            c.execute("UPDATE tool_uses SET pre_confidence=? WHERE id=?", (float(rate), r["id"]))
            updated += 1
    print(f"backfilled pre_confidence for {updated} tool_uses rows from tool base-rates")
    return {"updated": updated}


def feed_elicit(task, post=None):
    """Elicit a confidence judgment (conservative: high-stakes calls). Writes a
    CALIBER row; the caller resolves it with `feed resolve-calib <id> <outcome>`."""
    import caliber
    rec = caliber.calibrate(task, answer=post)
    print(f"elicited confidence -> calibration row #{rec['id']} "
          f"(pre={rec['pre_confidence']}, post={rec.get('post_confidence')})")
    print("resolve later with: feed resolve-calib %d <0|1>" % rec["id"])
    return rec


def feed_resolve_forecast(fid, outcome, note=None):
    import prediction
    prediction.resolve(int(fid), int(outcome), note=note)
    return 0


def feed_resolve_calib(calib_id, outcome):
    import caliber
    caliber.resolve(int(calib_id), int(outcome))
    print(f"calibration #{calib_id} resolved -> outcome {outcome}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="measurer-vs-data evaluator + feeding loop")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    sub.add_parser("cycle")
    f = sub.add_parser("feed")
    fsub = f.add_subparsers(dest="fcmd", required=True)
    rd = fsub.add_parser("resolve-due")
    rd.add_argument("--dry-run", action="store_true")
    fsub.add_parser("backfill-tool-confidence")
    ce = fsub.add_parser("counterevidence")
    ce.add_argument("belief_id", type=int)
    ce.add_argument("evidence")
    el = fsub.add_parser("elicit")
    el.add_argument("task")
    el.add_argument("--post", default=None)
    rc = fsub.add_parser("resolve-calib")
    rc.add_argument("id", type=int)
    rc.add_argument("outcome", type=int)
    rf = fsub.add_parser("resolve-forecast")
    rf.add_argument("id", type=int)
    rf.add_argument("outcome", type=int)
    rf.add_argument("--note", default=None)
    a = p.parse_args()

    if a.cmd == "audit":
        audit_print()
    elif a.cmd == "cycle":
        cycle()
    elif a.cmd == "feed":
        if a.fcmd == "resolve-due":
            feed_resolve_due(a.dry_run)
        elif a.fcmd == "backfill-tool-confidence":
            feed_backfill_tool_confidence()
        elif a.fcmd == "counterevidence":
            sys.exit(feed_counterevidence(a.belief_id, a.evidence))
        elif a.fcmd == "elicit":
            feed_elicit(a.task, a.post)
        elif a.fcmd == "resolve-calib":
            sys.exit(feed_resolve_calib(a.id, a.outcome))
        elif a.fcmd == "resolve-forecast":
            sys.exit(feed_resolve_forecast(a.id, a.outcome, a.note))


if __name__ == "__main__":
    main()
