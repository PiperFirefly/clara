#!/usr/bin/env python3
"""First-run check — print 'fresh' if the agent has no name and no operator yet.

Lightweight (sqlite3 only, no numpy) so it can run on every turn cheaply.
Used by memory-tools.ts `before_agent_start` to decide whether to inject the
first-run onboarding prompt.
"""
import os
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "memory.db")

if not os.path.exists(DB):
    print("fresh")
    raise SystemExit

try:
    c = sqlite3.connect(DB)
    name = c.execute(
        "SELECT value FROM facts WHERE key='name' AND invalidated_at IS NULL"
    ).fetchone()
    op = c.execute(
        "SELECT 1 FROM operators WHERE role='primary' AND active=1 LIMIT 1"
    ).fetchone()
    c.close()
except sqlite3.OperationalError:
    print("fresh")
    raise SystemExit

print("ready" if (name and name[0] and op) else "fresh")
