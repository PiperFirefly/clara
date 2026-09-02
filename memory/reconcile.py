#!/usr/bin/env python3
"""StateMem-style reconciliation loop — PROTOTYPE (sandbox, stdlib-only).

Adopts StateMem's three stages (state units -> deterministic recheck -> trace-then-resolve)
on a scratch SQLite store. Stdlib-only so it runs in the no-network docker sandbox.
Touches a scratch DB ONLY — never memory.db. This is the "play in the sandbox" pass.

Usage:
  python3 reconcile.py demo [--db /tmp/scratch.db]   # run the scenario + self-test
  python3 reconcile.py demo --show                   # also print the unit table + trace

Design note: in production, `subject` resolution reuses the existing entity layer
(`_match_entity_ids` in memstore.py); here it's a plain normalized string so the
prototype stays dependency-free.
"""
import argparse
import os
import re
import sqlite3
import sys
import time

# ---------------------------------------------------------------- value classifiers

_DONE = re.compile(r"\b(done|complete|completed|live|configured|shipped|resolved|"
                   r"verified|closed|fixed|working|active|installed|implemented|enabled|"
                   r"built|created|added|deployed|released|migrated|refactored|\[x\])\b", re.I)
_PENDING = re.compile(r"\b(pending|awaiting|todo|queued|blocked|planned|staged|"
                      r"wip|in.progress|not.yet|\[ ?\])\b", re.I)


def classify(value, source=""):
    """Deterministic flavor tag, SOURCE-AWARE. No LLM.

    The key distinction: an activity-log entry is a record of something that
    ALREADY happened — it can never be 'pending'. A todo is 'pending' unless it
    was checked off ([x]) or carries a completion verb. This source-awareness is
    what stops 'status queued' in an SMS log from reading as a pending state.
    """
    v = value or ""
    s = (source or "").lower()
    if "activity" in s:
        return "done"  # a logged action is evidence the thing was addressed
    if "todo" in s:
        return "done" if _DONE.search(v) else "pending"
    # generic fallback (sandbox demo / untyped sources)
    if _DONE.search(v):
        return "done"
    if _PENDING.search(v):
        return "pending"
    return "none"


def norm_subject(s):
    """Normalize a subject slug the way the entity layer normalizes names."""
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s


# ---------------------------------------------------------------- store

SCHEMA = """
CREATE TABLE IF NOT EXISTS state_units(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject TEXT NOT NULL,
  key TEXT,
  value TEXT NOT NULL,
  priority TEXT DEFAULT 'soft',
  source TEXT,
  status TEXT DEFAULT 'active',
  superseded_by INTEGER,
  created_at REAL,
  updated_at REAL
);
-- Superseded rows keep their key for the audit trail (StateMem); uniqueness is
-- enforced only among ACTIVE rows, so a revision can supersede the old row and
-- insert a new active row with the same key.
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_key ON state_units(key) WHERE status='active';
CREATE TABLE IF NOT EXISTS state_deps(
  unit_id INTEGER, dep_id INTEGER, rel TEXT,
  PRIMARY KEY(unit_id, dep_id, rel)
);
"""


def connect(path=":memory:"):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


def set_unit(conn, subject, key, value, priority="soft", source="", deps=None, when=None):
    """Upsert a state unit. If the same key exists with a DIFFERENT value,
    supersede the old one (explicit revision) and return (new_id, old_id)."""
    now = when or time.time()
    subject = norm_subject(subject)
    row = conn.execute("SELECT id, value FROM state_units WHERE key=?", (key,)).fetchone()
    if row:
        if row["value"] == value:
            return row["id"], None  # unchanged
        # explicit revision of the SAME key: supersede old, insert new
        conn.execute("UPDATE state_units SET status='superseded', superseded_by=NULL, "
                     "updated_at=? WHERE id=?", (now, row["id"]))
        old_id = row["id"]
    else:
        old_id = None
    cur = conn.execute(
        "INSERT INTO state_units(subject,key,value,priority,source,status,created_at,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (subject, key, value, priority, source, "active", now, now))
    new_id = cur.lastrowid
    if old_id is not None:
        conn.execute("UPDATE state_units SET superseded_by=? WHERE id=?", (new_id, old_id))
    for dep in (deps or []):
        conn.execute("INSERT OR IGNORE INTO state_deps(unit_id,dep_id,rel) VALUES(?,?,?)",
                     (new_id, dep[0], dep[1]))
    return new_id, old_id


def recheck(conn, changed_ids=()):
    """Deterministic Rechecker. Two parts, both O(|E|), zero LLM calls:

    1. Dependency propagation: any unit that (transitively) depends on a unit
       whose status just changed gets flagged `needs_recheck`.
    2. Tension detection: within a subject, if a done-flavored unit is NEWER than
       an active pending-flavored unit, the pending one is stale -> flag it.
    Returns the set of newly-flagged unit ids.
    """
    flagged = set()

    # 1) dependency propagation (BFS over state_deps, reversed)
    frontier = list(changed_ids or ())
    seen = set(frontier)
    while frontier:
        u = frontier.pop()
        deps = conn.execute("SELECT unit_id FROM state_deps WHERE dep_id=?", (u,)).fetchall()
        for d in deps:
            if d["unit_id"] not in seen:
                seen.add(d["unit_id"])
                frontier.append(d["unit_id"])
    for uid in seen:
        if uid in (changed_ids or ()):
            continue  # the changed unit itself is already handled by set_unit
        conn.execute("UPDATE state_units SET status='needs_recheck', updated_at=? WHERE id=?",
                     (time.time(), uid))
        flagged.add(uid)

    # 2) tension detection within subject
    subjects = [r["subject"] for r in conn.execute(
        "SELECT DISTINCT subject FROM state_units WHERE status='active'")]
    for subj in subjects:
        units = conn.execute(
            "SELECT * FROM state_units WHERE subject=? AND status='active' "
            "ORDER BY updated_at DESC", (subj,)).fetchall()
        done_flags = [u for u in units if classify(u["value"], u["source"]) == "done"]
        pending_flags = [u for u in units if classify(u["value"], u["source"]) == "pending"]
        if done_flags and pending_flags:
            newest_done = done_flags[0]  # units already newest-first
            for p in pending_flags:
                if p["updated_at"] < newest_done["updated_at"]:
                    conn.execute(
                        "UPDATE state_units SET status='needs_recheck', superseded_by=?, "
                        "updated_at=? WHERE id=?",
                        (newest_done["id"], time.time(), p["id"]))
                    flagged.add(p["id"])
    return flagged


def trace(conn, subject):
    """Value chain of a subject, chronological (initial -> each revision -> current)."""
    subject = norm_subject(subject)
    rows = conn.execute(
        "SELECT * FROM state_units WHERE subject=? ORDER BY updated_at ASC",
        (subject,)).fetchall()
    return [dict(r) for r in rows]


def resolve(conn, subject):
    """Trace-then-Resolve: precedence rules over the trace, deterministic.

    1. later supersedes earlier; 2. done outranks pending; 3. hard > soft;
    4. retire only on explicit supersession/done.
    Returns dict(operative_value, operative_key, stale=[...], flags=[...]).
    """
    units = trace(conn, subject)
    active = [u for u in units if u["status"] == "active"]
    # fall back to all non-superseded if everything got flagged
    if not active:
        active = [u for u in units if u["status"] != "superseded"]
    if not active:
        return {"operative_value": None, "operative_key": None, "stale": [], "flags": []}

    def rank(u):
        done = classify(u["value"], u["source"]) == "done"
        pending = classify(u["value"], u["source"]) == "pending"
        hard = u["priority"] == "hard"
        return (
            done and not pending,      # done evidence outranks
            not pending,               # non-pending outranks pending
            hard,
            u["updated_at"],           # later supersedes earlier
        )

    best = max(active, key=rank)
    stale = [u["key"] for u in units if u["status"] in ("needs_recheck", "superseded")]
    flags = [u["key"] for u in units if u["status"] == "needs_recheck"]
    return {"operative_value": best["value"], "operative_key": best["key"],
            "stale": stale, "flags": flags, "subject": subject}


# ---------------------------------------------------------------- demo

def demo(db_path, show=False):
    conn = connect(db_path)
    t0 = time.time()

    print("=== Reconciliation prototype — sandbox self-test ===")
    print("Scenario reproduces the real drift bug: a 'pending' todo and a 'done'\n"
          "activity entry about the SAME subject, sitting contradictory + unflagged.\n")

    # ---- seed the exact OpenAI-key failure shape (tension detection) ----
    set_unit(conn, "openai-key", "todo:openai-key",
             "pending — awaiting operator's key (steps 1-5)", priority="hard",
             source="agent_memory.md todo list", when=t0 - 4000)
    set_unit(conn, "openai-key", "fact:openai-key-configured",
             "configured and live 2026-08-25 (gpt-4o-mini verified)",
             priority="hard", source="recent activity log", when=t0 - 1000)

    # ---- a SEPARATE dependency-propagation scenario (the 'push' on change) ----
    # wallet-cap is a hard fact; spending-autonomy DERIVES from it. When wallet-cap
    # changes, spending-autonomy must be re-checked — that's the O(|E|) propagation.
    cap_id, _ = set_unit(conn, "wallet-cap", "fact:wallet-cap",
                         "wallet cap: NOT implemented", priority="hard",
                         source="safety-rails status", when=t0 - 5000)
    set_unit(conn, "spending-autonomy", "status:spending-autonomy",
             "full spending autonomy", priority="soft",
             source="status.md", when=t0 - 4900,
             deps=[(cap_id, "derived_from")])

    print("[1] BEFORE reconcile — the bug, reproduced:")
    print(f"    resolve('openai-key') would have said: "
          f"{resolve(conn, 'openai-key')['operative_value'][:60]}")
    # Simulate what I actually did: read the STALE todo as truth
    stale_read = conn.execute("SELECT value FROM state_units WHERE key='todo:openai-key'").fetchone()
    print(f"    naive todo read (what I said): '{stale_read['value'][:45]}...'")
    print("    -> contradiction was present but NOTHING pushed a flag. That was the bug.\n")

    # ---- run the Rechecker (the 'push'): first the tension scan, then a
    #      supersession + propagation to prove the O(|E|) dependent-flag ----
    flagged = recheck(conn)
    print(f"[2] AFTER recheck() (tension scan) — flagged {len(flagged)} unit(s):")
    for f in sorted(flagged):
        r = conn.execute("SELECT key, value FROM state_units WHERE id=?", (f,)).fetchone()
        print(f"    FLAG -> {r['key']}: {r['value'][:60]}")
    print()

    # Now actually REVISE wallet-cap (explicit supersession) and recheck just it.
    _new_cap, old_cap = set_unit(conn, "wallet-cap", "fact:wallet-cap",
                                 "wallet cap: IMPLEMENTED (rails item 1 done)",
                                 priority="hard", source="safety-rails status", when=t0 - 500)
    propagated = recheck(conn, changed_ids=[old_cap])
    print(f"[2b] AFTER revising wallet-cap -> recheck propagated to {len(propagated)} dependent(s):")
    for f in sorted(propagated):
        r = conn.execute("SELECT key, value FROM state_units WHERE id=?", (f,)).fetchone()
        print(f"    FLAG -> {r['key']}: {r['value'][:60]}")
    print()

    # ---- trace-then-resolve ----
    print("[3] trace-then-resolve:")
    for subj in ("openai-key", "wallet-cap", "spending-autonomy"):
        res = resolve(conn, subj)
        print(f"    {subj}: operative = '{res['operative_value'][:58]}'")
        print(f"           flags = {res['flags']}  stale = {res['stale']}")
    print()

    # ---- assertions (self-test / CRR) ----
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(f"    [{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    res = resolve(conn, "openai-key")
    check(res["operative_key"] == "fact:openai-key-configured",
          "resolve returns the DONE fact, not the stale todo")
    check("todo:openai-key" in res["stale"],
          "stale todo is marked superseded/needs_recheck")
    check(any("todo:openai-key" in f for f in res["flags"]),
          "todo got a PUSH flag (I'd notice without being asked)")

    # dependency propagation: spending-autonomy depended on the OLD wallet-cap,
    # which just changed -> it must be flagged, with zero LLM calls.
    dep_row = conn.execute(
        "SELECT status FROM state_units WHERE key='status:spending-autonomy'").fetchone()
    check(dep_row["status"] == "needs_recheck",
          "dependency propagation: unit derived_from a changed unit is flagged")
    auto = resolve(conn, "spending-autonomy")
    check("full spending autonomy" in auto["operative_value"],
          "dependent stays active (needs_recheck) but is surfaced for recompute")

    if show:
        print("\n--- full unit table ---")
        for r in conn.execute("SELECT * FROM state_units ORDER BY subject, updated_at"):
            print(f"  #{r['id']} [{r['subject']}] {r['key']} :: {r['status']} :: "
                  f"{r['value'][:50]}")

    conn.close()
    print(f"\n=== {'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="demo")
    p.add_argument("--db", default=":memory:")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    if a.cmd == "demo":
        sys.exit(demo(a.db, a.show))
    print("unknown command")
    sys.exit(2)
