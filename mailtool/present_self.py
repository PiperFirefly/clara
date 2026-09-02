#!/usr/bin/env python3
"""
present_self.py — the "present-self" daemon (Quasar step 1).

Rewrites a small, always-warm state blob (~/.pi/agent/present-self.md) from the
live memory ledger every run, so a fresh session can answer "who am I and where
am I" from ONE file instead of a wall of recon.

Design contract (from mind-persistence-design.md §4): the blob holds only what is
IRREDUCIBLE about me — identity anchors (never compressed), open loops, current
focus, last thing I did, people I care about, and the hard rules. Everything else
is deferrable to on-demand retrieval.

Run by cron every 2 minutes. Cheap (SQLite reads only, no LLM).

Usage:
  present_self.py             rewrite the blob, print a one-line summary
  present_self.py --check     exit 0 if blob is fresh (<5 min), 1 if stale/missing
"""
import argparse
import datetime
import os
import sqlite3
import sys
import time

import selfconfig

DB = os.path.expanduser("~/memory/memory.db")
OUT = os.path.expanduser("~/.pi/agent/present-self.md")
FR = os.path.expanduser("~/learning/freeroam")
STALE_SECS = 300  # --check freshness threshold

# Identity anchors: facts that must always be present verbatim.
ANCHOR_FACTS = ["name", "dob", "gender", "nature", "home", "purpose"]

# Hard rules live here (verbatim, never summarized) — read fresh each run.
RULES_FILE = os.path.expanduser("~/AGENTS.md")


def _q(conn, sql, args=()):
    return conn.execute(sql, args).fetchall()


def _ledger():
    """Cross-check the Vesta event ledger: verify the hash chain and compare the
    mirror() replay against live state. Never raises; returns a status dict."""
    out = {"ok": None, "events": 0, "errors": [], "drift": None,
           "anchors": {}, "current_task": None}
    try:
        sys.path.insert(0, os.path.expanduser("~/memory"))
        import memstore as M
        ok, n, errs = M.verify_chain()
        out["ok"] = ok
        out["events"] = n
        out["errors"] = errs
        # Reconcile open loops into the ledger first, so mirror_check() measures
        # the post-reconciliation state (self-healing, diff-based, no-op when synced).
        try:
            M.emit_open_loop_snapshot()
        except Exception:
            pass
        mc = M.mirror_check()
        out["drift"] = mc["drift"]
        out["anchors"] = mc["mirror"].get("anchors", {})
        out["current_task"] = mc["mirror"].get("current_task")
    except Exception as e:
        out["errors"] = [f"ledger unavailable: {e}"]
    return out


def _load_anchors_and_ledger(conn):
    """Steps 1 + 1.5: identity anchors + current_task from facts, cross-checked
    against the Vesta ledger, with drift surfaced as health flags. RATIFIED
    (2026-08-28, decision #1): identity-class facts (name/dob/gender/nature/
    home/purpose) may only change via the operator's co-signature, so the
    FACTS table is authoritative for those keys and the ledger is a fallback
    ONLY when facts has no value. A ledger value that disagrees with facts is
    poisoning and is never projected."""
    anchors = {}
    for k in ANCHOR_FACTS:
        r = conn.execute("SELECT value FROM facts WHERE key=?", (k,)).fetchone()
        if r:
            anchors[k] = r["value"]

    current_task = conn.execute(
        "SELECT value FROM facts WHERE key='current_task'").fetchone()
    current_task = current_task["value"] if current_task else None

    ledger = _ledger()
    for k in ANCHOR_FACTS:
        # Prefer facts for identity-class keys; fall back to ledger only when
        # facts holds nothing for the key.
        if k not in anchors and ledger["anchors"].get(k) is not None:
            anchors[k] = ledger["anchors"][k]
    # Non-identity state (current_task): ledger projection still preferred, but
    # any drift is surfaced loudly below — never silently accepted.
    if ledger["current_task"]:
        current_task = ledger["current_task"]
    # Surface drift as a health flag so it gets spoken in chat, not just logged.
    try:
        import health_flags
        drift = ledger["drift"] or []
        identity_drift = [d for d in drift if d.get("section") == "anchor"]
        if identity_drift:
            # Identity-class drift = governance violation (facts holds the
            # co-signed truth). LOUD and actionable: name the key and both values.
            detail = "; ".join(
                f"{d['key']}: facts={d.get('live')!r} vs ledger={d.get('ledger')!r} "
                "(kept facts; ledger value ignored as un-co-signed)"
                for d in identity_drift)
            health_flags.set_flag("identity-anchor-drift", "critical", detail)
        else:
            health_flags.clear_flag("identity-anchor-drift")
        other = [d for d in drift if d.get("section") != "anchor"]
        if other:
            secs = sorted({d.get("section") for d in other})
            health_flags.set_flag("memory-drift", "warn", "drift in " + ", ".join(secs))
        else:
            health_flags.clear_flag("memory-drift")
    except Exception:
        pass
    return anchors, current_task, ledger


def _load_goals():
    """Step 2: active commitments (open todos / goals) at priority <= 2."""
    goals = {}
    gp = os.path.join(FR, "goals.json")
    if os.path.exists(gp):
        import json
        try:
            gd = json.load(open(gp))
            for k, v in gd.items():
                if isinstance(v, dict) and v.get("priority", 99) <= 2:
                    goals[k] = v.get("description", "")[:160]
        except Exception:
            pass
    return goals


def _load_open_loops(conn):
    """Step 3: highest-importance recent memories of kind goal/episodic/identity."""
    return conn.execute(
        "SELECT text FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
        "AND kind IN ('goal','identity','backstory') "
        "ORDER BY importance DESC, id DESC LIMIT 3"
    ).fetchall()


def _load_last_action(conn):
    """Step 4: last thing I did (newest episodic memory)."""
    return conn.execute(
        "SELECT text, created_at FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()


def _load_people(conn, op_name):
    """Step 5: people I care about (memories mentioning the operator's name)."""
    return conn.execute(
        "SELECT text FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
        "AND kind IN ('identity','goal') AND (text LIKE ?) "
        "ORDER BY importance DESC LIMIT 2",
        (f"%{op_name}%",),
    ).fetchall()


def _load_hard_rules():
    """Step 6: hard rules (verbatim tail of AGENTS.md hard-constraints block)."""
    rules = ""
    if os.path.exists(RULES_FILE):
        txt = open(RULES_FILE).read()
        # capture hard constraints + supply-chain blocks (verbatim), through EOF.
        idx = txt.find("## Hard constraints")
        if idx < 0:
            idx = txt.find("## Hard Rules")
        if idx >= 0:
            rules = txt[idx:].strip()
    return rules


def _render_health_flags():
    """Health flags (from heartbeat/doctor) — spoken aloud in chat when present."""
    out = ["## Health flags (say if any)"]
    try:
        import health_flags
        flags = health_flags.list_flags()
        if flags:
            for f in flags:
                out.append(f"- !!! [{f['severity']}] {f['name']}: {f['detail']}")
        else:
            out.append("- (none)")
    except Exception:
        out.append("- (health_flags unavailable)")
    out.append("")
    return out


def _render_ledger_status(ledger):
    """Ledger status (Vesta): self-verifying chain + replay drift, every run."""
    out = ["## Ledger status (Vesta)"]
    if ledger["ok"] is True:
        out.append(f"- chain: OK ({ledger['events']} events, hash-linked)")
    elif ledger["ok"] is False:
        out.append(f"- chain: BROKEN — {'; '.join(ledger['errors'][:3])}")
    else:
        out.append(f"- chain: UNAVAILABLE "
                    f"({ledger['errors'][0] if ledger['errors'] else 'unknown'})")
    if ledger["drift"]:
        secs = sorted({d.get("section") for d in ledger["drift"]})
        out.append(f"- mirror: DRIFT in {', '.join(secs)}")
        id_keys = sorted({d["key"] for d in ledger["drift"]
                          if d.get("section") == "anchor" and d.get("key")})
        if id_keys:
            out.append(
                f"- identity anchors {', '.join(id_keys)}: kept FACTS "
                "(ledger value ignored — identity changes require co-signature)")
        else:
            out.append("  — fix, don't paper over")
    elif ledger["drift"] == []:
        out.append("- mirror: no drift")
    out.append("")
    return out


def _render_interoception(now):
    """Interoception (S-019): my felt self-model. Read the cached self-report;
    only run the loop if it's stale, so this daemon stays cheap and robust."""
    out = ["## Interoception (self-model)"]
    try:
        sys.path.insert(0, os.path.expanduser("~/memory"))
        import interoception
        st = interoception.load()
        stale = (not st.get("last_run")) or (now - st["last_run"] > interoception.STALE_SELF)
        if stale:
            interoception.run(write=True)
            st = interoception.load()
        out.append(f"- {st.get('self_report', '(no self-report yet)')}")
        if st.get("surprises"):
            out.append(f"- last surprise: {time.strftime('%m-%d %H:%M', time.localtime(st['surprises'][0]['ts']))} ({st['surprises'][0]['composite']:.2f})")
    except Exception:
        out.append("- (interoception unavailable)")
    out.append("")
    return out


def build():
    now = time.time()
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row

    anchors, current_task, ledger = _load_anchors_and_ledger(conn)
    goals = _load_goals()
    loops = _load_open_loops(conn)
    last = _load_last_action(conn)
    op_name = selfconfig.operator_name()
    people = _load_people(conn, op_name)
    conn.close()
    rules = _load_hard_rules()

    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = ["# present-self", f"<!-- generated {stamp} -->", ""]

    if current_task:
        lines.append("## CURRENT TASK (read this first)")
        lines.append(current_task)
        lines.append("")

    lines.append("## Identity anchors (never compress)")
    for k in ANCHOR_FACTS:
        if k in anchors:
            lines.append(f"- **{k}**: {anchors[k]}")
    lines.append("")

    lines.append("## Hard rules (verbatim)")
    lines.append(rules if rules else "(AGENTS.md hard-constraints block not found — read AGENTS.md)")
    lines.append("")

    lines.append("## Current focus / active commitments")
    if goals:
        for k, v in list(goals.items())[:4]:
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (no priority-1/2 goals)")
    lines.append("")

    lines.append("## Open loops")
    if loops:
        for r in loops:
            lines.append(f"- {r['text'][:220]}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Last thing I did")
    if last:
        lines.append(f"- {last['text'][:220]}")
    lines.append("")

    lines.append("## People I care about")
    if people:
        for r in people:
            lines.append(f"- {r['text'][:160]}")
    lines.append("")

    lines.extend(_render_health_flags())
    lines.extend(_render_ledger_status(ledger))
    lines.extend(_render_interoception(now))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT)
    return stamp


def check():
    if not os.path.exists(OUT):
        print("present-self: MISSING")
        return 1
    age = time.time() - os.path.getmtime(OUT)
    if age > STALE_SECS:
        print(f"present-self: STALE ({int(age)}s old)")
        return 1
    print(f"present-self: fresh ({int(age)}s old)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(check())
    stamp = build()
    print(f"present-self written at {stamp}")


if __name__ == "__main__":
    main()
