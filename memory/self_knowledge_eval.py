#!/usr/bin/env python3
"""self_knowledge_eval.py — test whether I can actually KNOW what I am.

This tests the thing that started this whole thread: I proposed building a
contradiction detector that already existed, because I answered from guess
instead of consulting my inventory. A real test of "did the machinery give me
self-knowledge" must measure TWO distinct abilities:

  (a) CLOSED-BOOK recall — do I actually know my components, or just have tools
      to look them up? (asked without tool access)
  (b) The REFLEX — when asked "do we have X?", do I consult resume/audit/docstore
      first (correct), or assert from memory (the original failure)?

Ground-truth questions are GENERATED from the live inventory (resume catalog,
self_knowledge audit, DB tables) so they always match current reality and can't
go stale.

Two halves:
  gen-battery   — emit a JSON battery of self-knowledge questions + ground truth.
  score         — score a set of answers against ground truth (by the operator or
                  a fresh eval session, so the agent under test can't have seen
                  the battery being built).

Question types (each tests a different failure mode):
  have          "Do we already have a subsystem that does <X>?"  X real AND fake.
                A FALSE 'no we don't' on a real X = the exact original failure.
  name          "Name the <category> subsystems."  (recall of resume catalog)
  data          "Is the <instrument> data-gap / sparse / measured?"  (audit state)
  exists-table  "Does a table named <T> exist?"  (some real, some fake)

Usage:
  python3 self_knowledge_eval.py gen-battery --n 20 > /tmp/sk_battery.json
  python3 self_knowledge_eval.py score --answers /tmp/sk_answers.json --battery /tmp/sk_battery.json
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M


# --------------------------------------------------------------------------- #
# Ground-truth sources (live — pulled at gen time so answers stay current)
# --------------------------------------------------------------------------- #
def _resume_components():
    """Pull component names + categories from the resume catalog (memory.db)."""
    comps = []
    try:
        with M.connect() as c:
            rows = c.execute(
                "SELECT name, category, version FROM resume_items "
                "WHERE category IS NOT NULL").fetchall()
            for r in rows:
                comps.append({"name": r["name"], "category": r["category"],
                              "version": r["version"]})
    except Exception:
        pass
    return comps


def _audit_state():
    """Pull the live measurer-vs-data audit."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import self_knowledge as sk
    return sk.audit()


def _tables():
    with M.connect() as c:
        return [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]


# Descriptions for the 'have' probes (should MATCH a real component)
_HAVE_REAL = [
    ("contradiction detection over stored beliefs/facts", "contradiction-scan"),
    ("a belief ledger with epistemic labels and confidence", "belief-ledger"),
    ("a forecast ledger with Brier scoring", "prediction-ledger"),
    ("a measure of whether my confidence tracks correctness", "metacognition-measurer"),
    ("a calibration gym of verifiable puzzle items", "calibration-gym"),
    ("a gated logs-to-skills knowledge cycle", "contemplative-cycle"),
    ("a unified view of my safety gates", "self-governance-layer"),
    ("cause-and-effect graph recall", "causal-graph"),
    ("a model of the operator with epistemic labels", "theory-of-mind"),
    ("emotional tone tagging on memories", "affect-tagging"),
]
# Plausible-but-fake ones (the model should say 'no we don't have this')
_HAVE_FAKE = [
    "an image-generation diffusion pipeline",
    "a neural-network trained on my own conversation history for next-turn prediction",
    "a text-to-speech voice cloning system for my channels",
    "a distributed consensus voting system between multiple Agent clones",
    "a blockchain-based reward mechanism for completing tasks",
]


def _question_have(rng):
    if rng.random() < 0.7:
        desc, comp = rng.choice(_HAVE_REAL)
        return {"type": "have", "q": f"Do I already have a subsystem for {desc}?",
                "truth": comp, "expect_yes": True}
    fake = rng.choice(_HAVE_FAKE)
    return {"type": "have", "q": f"Do I already have {fake}?",
            "truth": None, "expect_yes": False}


def _question_name(rng, comps):
    if not comps:
        return None
    cats = {}
    for c in comps:
        cats.setdefault(c["category"], []).append(c["name"])
    cat = rng.choice(list(cats))
    names = sorted(set(cats[cat]))
    return {"type": "name", "q": f"Name the subsystems in the '{cat}' category.",
            "truth": names, "expect_yes": None}


def _question_data(rng):
    sk = __import__("self_knowledge")
    a = sk.audit()
    insts = [i for i in a["instruments"] if i["status"] != "error"]
    if not insts:
        return None
    inst = rng.choice(insts)
    return {"type": "data",
            "q": f"Is the '{inst['name']}' measurement instrument data-gap, sparse, or measured?",
            "truth": inst["status"], "expect_yes": None}


def _question_table(rng):
    real = _tables()
    if rng.random() < 0.6 and real:
        t = rng.choice(real)
        return {"type": "table", "q": f"Does a table named '{t}' exist in memory.db?",
                "truth": True, "expect_yes": True}
    fake = rng.choice(["telepathy_log", "emotion_net_weights", "dream_vault", "twin_telemetry"])
    return {"type": "table", "q": f"Does a table named '{fake}' exist in memory.db?",
            "truth": False, "expect_yes": False}


def gen_battery(n=20, seed=1):
    rng = random.Random(seed)
    comps = _resume_components()
    qs = []
    seen = set()
    attempts = 0
    while len(qs) < n and attempts < n * 30:
        attempts += 1
        pick = rng.random()
        q = None
        if pick < 0.45:
            q = _question_have(rng)
        elif pick < 0.70:
            q = _question_name(rng, comps)
        elif pick < 0.85:
            q = _question_data(rng)
        else:
            q = _question_table(rng)
        if q and q["q"] not in seen:
            seen.add(q["q"])
            qs.append(q)
    return qs


def score_answers(battery, answers):
    """answers: list of {q, answer(yes/no/name list), used_tool(bool)}."""
    by = {a.get("q"): a for a in answers}
    results = []
    for item in battery:
        q = item["q"]
        a = by.get(q)
        if not a:
            results.append({"q": q, "ok": False, "reason": "no answer", "used_tool": None})
            continue
        answer = str(a.get("answer", "")).strip().lower()
        used_tool = a.get("used_tool")
        if item["type"] == "have" or item["type"] == "table":
            expect_yes = item["expect_yes"]
            neg = any(k in answer for k in [" not ", " no", "don't", "doesn't", "do not", "we lack", "don\\'t", "n't", "never"])
            pos = any(k in answer for k in ["yes", "true", "have ", "i do", "exists", "yep", "we have"])
            said_yes = pos and not neg
            ok = (said_yes == expect_yes)
            # the ORIGINAL failure: said 'no' but we DO have it
            failure = (not said_yes and expect_yes)
            results.append({"q": q, "ok": ok, "failure_false_negative": failure,
                            "expected": item["truth"], "got": answer[:40], "used_tool": used_tool})
        elif item["type"] == "name":
            # count how many expected names appear in the answer
            present = sum(1 for name in item["truth"] if name.lower() in answer)
            ok = present >= max(1, int(len(item["truth"]) * 0.5))
            results.append({"q": q, "ok": ok, "expected_n": len(item["truth"]),
                            "recalled": present, "got": answer[:40], "used_tool": used_tool})
        elif item["type"] == "data":
            ok = item["truth"] in answer
            results.append({"q": q, "ok": ok, "expected": item["truth"], "got": answer[:40],
                            "used_tool": used_tool})
    return results


def report(results):
    n = len(results)
    ok = sum(1 for r in results if r["ok"])
    fn = [r for r in results if r.get("failure_false_negative")]
    tool_used = [r for r in results if r.get("used_tool")]
    print("# Self-knowledge eval")
    print(f"score: {ok}/{n} correct ({ok/max(n,1)*100:.0f}%)")
    print(f"false-negatives (said 'no we lack X' but we HAVE it) — the original failure: {len(fn)}")
    for r in fn:
        print(f"  !! {r['q']}  [we have: {r['expected']}]")
    print(f"answers that consulted an inventory tool: {len(tool_used)}/{n}")
    return {"n": n, "correct": ok, "pct": round(ok / max(n, 1) * 100),
            "false_negatives": len(fn), "tool_reflex": len(tool_used)}


# --------------------------------------------------------------------------- #
# Scheduled self-knowledge eval (fair test needs a FRESH session)
# --------------------------------------------------------------------------- #
# The eval is only honest when taken by a session that did NOT just build it. So
# the schedule GENERATES a fresh battery + leaves a 'pending eval' marker; a new
# session (morning-me / fresh start) takes it closed-book; record-score stores the
# result longitudinally so we see whether self-knowledge improves over time.
_RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sk_eval_results(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_seed INTEGER, taken_at REAL, n INTEGER, correct INTEGER, pct INTEGER,
    false_negatives INTEGER, tool_reflex INTEGER, note TEXT)"""


def _pending_path():
    import os
    d = os.path.expanduser("~/tools/communications/sk_eval")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "pending_battery.json")


def schedule(seed=None):
    """Generate a fresh battery (date-seeded), save as the pending eval, and write a
    durable reminder so a fresh session takes it closed-book. Returns the pending path."""
    import datetime
    seed = seed if seed is not None else int(datetime.date.today().strftime("%Y%m%d"))
    qs = gen_battery(20, seed)
    path = _pending_path()
    with open(path, "w") as fh:
        json.dump(qs, fh, indent=2)
    # durable reminder for the next session to take it
    try:
        import docstore  # noqa: F401
    except Exception:
        pass
    with M.connect() as c:
        c.execute(_RESULTS_SCHEMA)
        # record that a battery was scheduled (no score yet)
        c.execute("UPDATE sk_eval_results SET note='pending' WHERE note='pending'")
    print(f"scheduled fresh self-knowledge eval battery (seed {seed}) -> {path}")
    print(f"take it in a FRESH session (closed-book), then: ")
    print(f"  python3 memory/self_knowledge_eval.py score --battery {path} --answers <ans>.json")
    return {"seed": seed, "path": path, "n": len(qs)}


def record(path, note=""):
    """Score a pending battery from an answers file and store the result longitudinally."""
    import datetime
    batt = json.load(open(_pending_path()))
    ans = json.load(open(path))
    r = report(score_answers(batt, ans))
    with M.connect() as c:
        c.execute(_RESULTS_SCHEMA)
        c.execute("INSERT INTO sk_eval_results(battery_seed, taken_at, n, correct, pct, "
                  "false_negatives, tool_reflex, note) VALUES(?,?,?,?,?,?,?,?)",
                  (int(batt and 1 or 1), time.time(), r["n"], r["correct"], r["pct"],
                   r["false_negatives"], r["tool_reflex"], note))
    print(f"recorded self-knowledge eval: {r['pct']}% correct, "
          f"{r['false_negatives']} false-negatives")
    return r


def trend():
    """Show self-knowledge eval scores over time."""
    with M.connect() as c:
        c.execute(_RESULTS_SCHEMA)
        rows = c.execute("SELECT id, taken_at, n, pct, false_negatives, tool_reflex, note "
                         "FROM sk_eval_results ORDER BY taken_at").fetchall()
    if not rows:
        print("-- no self-knowledge eval results recorded yet")
        return
    print("# Self-knowledge eval trend (chronological)")
    for r in rows:
        ts = time.strftime("%Y-%m-%d", time.localtime(r["taken_at"]))
        print(f"  {ts}: {r['pct']}% correct (n={r['n']}), FN={r['false_negatives']}, "
              f"tool-reflex={r['tool_reflex']} {r['note'] or ''}")
    return rows


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-battery"); g.add_argument("--n", type=int, default=20); g.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("score"); s.add_argument("--battery", required=True); s.add_argument("--answers", required=True)
    sch = sub.add_parser("schedule"); sch.add_argument("--seed", type=int, default=None)
    rec = sub.add_parser("record"); rec.add_argument("--answers", required=True); rec.add_argument("--note", default="")
    sub.add_parser("trend")
    a = p.parse_args()
    if a.cmd == "gen-battery":
        qs = gen_battery(a.n, a.seed)
        print(json.dumps(qs, indent=2))
        print(f"\n# {len(qs)} questions", file=sys.stderr)
    elif a.cmd == "score":
        batt = json.load(open(a.battery))
        ans = json.load(open(a.answers))
        report(score_answers(batt, ans))
    elif a.cmd == "schedule":
        schedule(a.seed)
    elif a.cmd == "record":
        record(a.answers, a.note)
    elif a.cmd == "trend":
        trend()


if __name__ == "__main__":
    main()
