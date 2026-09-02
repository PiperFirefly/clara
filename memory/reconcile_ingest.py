#!/usr/bin/env python3
"""Ingest agent_memory.md -> state units (the TurnEncoder step), then reconcile.

This is the LLM-costly part of the StateMem adoption: deepseek-chat parses my
prose todos + activity log into structured state units. The downstream recheck
+ resolve is the deterministic, LLM-free prototype already sandboxed.

SAFETY: writes to a SCRATCH db (default /tmp/reconcile_scratch.db) — never
memory.db. agent_memory.md is read-only here.

Usage:
  python3 reconcile_ingest.py [--db /tmp/reconcile_scratch.db] [--dry-run]
  python3 reconcile_ingest.py --report [--db ...]   # list surfaced drift candidates
"""
import argparse
import calendar
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reconcile as R
from memstore import llm_chat, _extract_json

MD = os.path.expanduser("~/agent_memory.md")

_DATE = re.compile(r"\[(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}:\d{2}))?\]")


def extract_section(text, name):
    m = re.search(rf"^##\s+{re.escape(name)}\s*$(.*?)(?=^##\s|\Z)",
                  text, re.M | re.S)
    return m.group(1).strip() if m else ""


def split_entries(section):
    """Split a bullet list ('- [x] ...', '- [ ] ...', '- [date] ...') into items."""
    items, cur = [], None
    for line in section.splitlines():
        if re.match(r"^\s*-\s+\[", line) or re.match(r"^\s*-\s+\S", line):
            if cur:
                items.append(cur)
            cur = line
        elif cur:
            cur += " " + line.strip()
    if cur:
        items.append(cur)
    return items


def parse_date(entry):
    m = _DATE.search(entry)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = mm = 0
    if m.group(4):
        hh, mm = (int(x) for x in m.group(4).split(":"))
    return calendar.timegm((y, mo, d, hh, mm, 0))


TODO_PROMPT = (
    "You are parsing an agent's personal markdown journal into structured STATE "
    "UNITS for a memory-reconciliation system.\n\n"
    "Below is the agent's TODO list. Each item starts with [x] (already done) or "
    "[ ] (still pending). For each item output ONE JSON unit:\n"
    '  "subject": a narrow, SPECIFIC lowercase-hyphenated slug for the exact topic '
    "(a tool, person, decision, or system — NOT a broad project bucket). Two items "
    "share a slug ONLY if they are genuinely about the same thing.\n"
    '  "key": "todo:<subject>" (dedupe with -2, -3 if needed).\n'
    '  "value": "done — " + summary if the item is [x]; "pending — " + summary if [ ].\n'
    '  "priority": "hard" if a hard rule / safety / security constraint, else "soft".\n'
    '  "source": "agent_memory.md todo".\n\n'
    "Output ONLY a JSON array, no commentary.\n\n"
    "TODOS:\n{text}\n"
)

ACTIVITY_PROMPT = (
    "You are parsing an agent's personal markdown journal into structured STATE "
    "UNITS for a memory-reconciliation system.\n\n"
    "Below are dated ACTIVITY-LOG entries (things the agent already did). For each "
    "entry output ONE JSON unit:\n"
    '  "subject": a narrow, SPECIFIC lowercase-hyphenated slug for the exact topic. '
    "CRITICAL: if an entry is about one of these ALREADY-KNOWN TODO topics, REUSE "
    "that EXACT slug (so related items align):\n"
    "{todo_subjects}\n"
    "Otherwise create a new narrow slug (a tool, person, decision, or system). Do "
    "NOT lump unrelated entries together.\n"
    '  "key": "fact:<subject>-<nn>" (unique).\n'
    '  "value": a factual statement of what happened, INCLUDING the date, with a '
    "completion verb when the entry describes finishing something (built/configured/"
    "fixed/installed/added/done). Keep under ~25 words.\n"
    '  "priority": "hard" if safety/security/critical, else "soft".\n'
    '  "source": "agent_memory.md activity".\n\n'
    "Rules: do NOT invent facts; preserve dates; one unit per entry.\n"
    "Output ONLY a JSON array, no commentary.\n\n"
    "ACTIVITY ENTRIES:\n{text}\n"
)


def _normalize_unit(u):
    if not isinstance(u, dict) or not all(k in u for k in ("subject", "value")):
        return None
    u.setdefault("key", f"fact:{R.norm_subject(u['subject'])}-{int(time.time()*1000)}")
    u.setdefault("priority", "soft")
    u.setdefault("source", "agent_memory.md activity")
    return u


def ingest(db_path, dry_run=False, verbose=True):
    text = open(MD).read()
    todos = extract_section(text, "Plans & Todos")
    activity = extract_section(text, "Recent Activity")
    todo_items = split_entries(todos)
    act_items = split_entries(activity)

    units = []
    # 1) todos — one call
    t = llm_chat([{"role": "user", "content": TODO_PROMPT.format(text=todos)}],
                 max_tokens=4000, temperature=0.1, model="deepseek-chat")
    t_units = [u for u in (_extract_json(t) or []) if _normalize_unit(u)]
    for u in t_units:
        u["source"] = "agent_memory.md todo"
    units += t_units
    todo_subjects = sorted({u["subject"] for u in t_units})
    if verbose:
        print(f"todos -> {len(t_units)} units (from {len(todo_items)} items)")

    # 2) activity — chunked; real dates become the units' updated_at; seed the
    #    prompt with the todo subjects so related entries REUSE the same slug.
    CHUNK = 10
    calls = 0
    for i in range(0, len(act_items), CHUNK):
        chunk = act_items[i:i + CHUNK]
        dates = [parse_date(e) for e in chunk]
        a = llm_chat([{"role": "user", "content": ACTIVITY_PROMPT.format(
            todo_subjects=", ".join(todo_subjects), text="\n".join(chunk))}],
            max_tokens=4000, temperature=0.1, model="deepseek-chat")
        a_units = [u for u in (_extract_json(a) or []) if _normalize_unit(u)]
        # attach real dates where the LLM kept a 1:1 mapping with the chunk
        if len(a_units) == len(chunk):
            for u, d in zip(a_units, dates):
                if d:
                    u["when"] = d
        else:
            for u in a_units:
                u["when"] = None
        for u in a_units:
            u["source"] = "agent_memory.md activity"
        units += a_units
        calls += 1
        if not dry_run:
            time.sleep(0.6)
    if verbose:
        print(f"activity -> {len(units) - len(t_units)} units across {calls} calls "
              f"({len(act_items)} entries)")

    if dry_run:
        print(f"\n[dry-run] would ingest {len(units)} units into {db_path}")
        return units

    # 3) write to scratch DB
    conn = R.connect(db_path)
    seen = set()
    for u in units:
        key = u["key"]
        base, n = key, 2
        while key in seen:
            key = f"{base}-{n}"
            n += 1
        seen.add(key)
        R.set_unit(conn, u["subject"], key, u["value"],
                   priority=u.get("priority", "soft"), source=u.get("source", ""),
                   when=u.get("when"))
    flagged = R.recheck(conn)
    conn.commit()
    conn.close()
    if verbose:
        print(f"\nwrote {len(units)} units -> {db_path}")
        print(f"recheck flagged {len(flagged)} unit(s) as needs_recheck")
    return units


def report(db_path):
    conn = R.connect(db_path)
    print("=== Drift candidates (a [ ] todo whose topic has a done activity entry) ===\n")
    rows = conn.execute(
        "SELECT * FROM state_units WHERE status IN ('active','needs_recheck') "
        "ORDER BY subject, updated_at").fetchall()
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)

    found = 0
    for subj, us in sorted(by_subj.items()):
        pending = [u for u in us if R.classify(u["value"], u["source"]) == "pending"]
        done = [u for u in us if R.classify(u["value"], u["source"]) == "done"]
        if pending and done:
            for p in pending:
                # activity entries are 'done' by source; show the newest one
                newest_done = max(done, key=lambda u: u["updated_at"] or 0)
                found += 1
                d_ts = newest_done["updated_at"]
                when = time.strftime("%Y-%m-%d", time.gmtime(d_ts)) if d_ts else "?"
                print(f"[{subj}]")
                print(f"    still-pending todo: {p['value'][:105]}")
                print(f"    done activity ({when}): {newest_done['value'][:105]}")
                print()
    if not found:
        print("(none)")
    print(f"TOTAL drift candidates: {found}  (needs my eyeball — continuous todos are false positives)")
    conn.close()
    return found


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="/tmp/reconcile_scratch.db")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        sys.exit(0 if report(a.db) >= 0 else 1)
    if os.path.exists(a.db):
        os.unlink(a.db)  # fresh scratch each ingest run (not before a report)
    ingest(a.db, dry_run=a.dry_run)
