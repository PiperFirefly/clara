#!/usr/bin/env python3
"""state.py — a single, coherent ephemeral state store for Agent.

Consolidates the ~20 scattered *_state.json files into one small WAL SQLite store
(~/memory/state.db). Design rule: ephemeral state is a nervous system, not
autobiographical memory — anything that must survive a reboot is marked `durable`
and is also mirrored into `facts` (`state/<key>`).

Schema:
    kv(key TEXT PRIMARY KEY, value TEXT, durable INTEGER, updated_at REAL)

Keys are namespaced: "<area>/<name>" (e.g. "freeroam/checkpoint", "hive/decay",
"cron/warden", "goals/coding", "heartbeat/escalated").

CLI:
    state.py get <key>
    state.py set <key> <json> [--durable]
    state.py keys [--durable]
    state.py del <key>
    state.py last <key>
"""
import argparse
import json
import os
import sqlite3
import sys
import time

DB = os.path.join(os.path.expanduser("~"), "memory", "state.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  durable    INTEGER DEFAULT 0,
  updated_at REAL NOT NULL
);
"""


def _connect():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure():
    with _connect() as c:
        c.executescript(_SCHEMA)


def _mirror_facts(key, value_json):
    """If durable, mirror the value into the facts store under `state/<key>`."""
    try:
        sys.path.insert(0, os.path.expanduser("~/memory"))
        import memstore  # noqa: PLC0415
        memstore.facts_set("state/" + key, value_json)
    except Exception:
        pass


def get(key, default=None):
    """Return the parsed value for key, or default if absent. Never raises."""
    _ensure()
    try:
        with _connect() as c:
            r = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if r is None:
            return default
        return json.loads(r["value"])
    except Exception:
        return default


def set(key, value, durable=False):
    """Set key to value (JSON-serialized). If durable, also mirror to facts."""
    _ensure()
    now = time.time()
    vj = json.dumps(value)
    with _connect() as c:
        c.execute(
            "INSERT INTO kv(key, value, durable, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "durable=excluded.durable, updated_at=excluded.updated_at",
            (key, vj, 1 if durable else 0, now),
        )
    if durable:
        _mirror_facts(key, vj)
    return value


def update(key, fn, default=None, durable=False):
    """Read-modify-write under a write lock. fn(old) -> new value."""
    _ensure()
    conn = _connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        r = conn.execute("SELECT value, durable FROM kv WHERE key=?", (key,)).fetchone()
        old = json.loads(r["value"]) if r is not None else default
        if r is not None:
            durable = durable or bool(r["durable"])
        new = fn(old)
        now = time.time()
        conn.execute(
            "INSERT INTO kv(key, value, durable, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "durable=excluded.durable, updated_at=excluded.updated_at",
            (key, json.dumps(new), 1 if durable else 0, now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    if durable:
        _mirror_facts(key, json.dumps(new))
    return new


def delete(key):
    _ensure()
    with _connect() as c:
        c.execute("DELETE FROM kv WHERE key=?", (key,))


def last_updated(key):
    _ensure()
    try:
        with _connect() as c:
            r = c.execute("SELECT updated_at FROM kv WHERE key=?", (key,)).fetchone()
        return r["updated_at"] if r else None
    except Exception:
        return None


def keys(durable_only=False):
    _ensure()
    q = "SELECT key FROM kv"
    if durable_only:
        q += " WHERE durable=1"
    with _connect() as c:
        return [r["key"] for r in c.execute(q + " ORDER BY key")]


def as_dict():
    """Return {key: parsed_value} for the whole store (useful for diagnostics)."""
    _ensure()
    with _connect() as c:
        rows = c.execute("SELECT key, value, durable, updated_at FROM kv").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def get_prefix(prefix):
    """Return {name: value} for all keys under `prefix/`. Namespaces map dicts."""
    _ensure()
    out = {}
    p = prefix.rstrip("/") + "/"
    with _connect() as c:
        rows = c.execute("SELECT key, value FROM kv WHERE key LIKE ?", (p + "%",)).fetchall()
    for r in rows:
        name = r["key"][len(p):]
        out[name] = json.loads(r["value"])
    return out


def set_prefix(prefix, d, durable=False, delete_missing=False):
    """Set each {name: value} under `prefix/`. If delete_missing, remove keys in the
    store's namespace not present in d (mirrors a dict-write of a whole file)."""
    _ensure()
    p = prefix.rstrip("/") + "/"
    want = {p + str(k): json.dumps(v) for k, v in (d or {}).items()}
    if delete_missing:
        with _connect() as c:
            c.execute("SELECT key FROM kv WHERE key LIKE ?", (p + "%",)).fetchall()
            # delete keys not in `want`
            cur = c.execute("SELECT key FROM kv WHERE key LIKE ?", (p + "%",)).fetchall()
            for r in cur:
                if r["key"] not in want:
                    c.execute("DELETE FROM kv WHERE key=?", (r["key"],))
    now = time.time()
    with _connect() as c:
        for k, vj in want.items():
            c.execute(
                "INSERT INTO kv(key, value, durable, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "durable=excluded.durable, updated_at=excluded.updated_at",
                (k, vj, 1 if durable else 0, now),
            )
    if durable:
        for k, vj in want.items():
            _mirror_facts(k, vj)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keys").add_argument("--durable", action="store_true")
    g = sub.add_parser("get"); g.add_argument("key")
    s = sub.add_parser("set"); s.add_argument("key"); s.add_argument("value")
    s.add_argument("--durable", action="store_true")
    d = sub.add_parser("del"); d.add_argument("key")
    l = sub.add_parser("last"); l.add_argument("key")
    a = ap.parse_args()

    if a.cmd == "keys":
        for k in keys(a.durable):
            print(k)
    elif a.cmd == "get":
        v = get(a.key, "<absent>")
        print(json.dumps(v) if v != "<absent>" else "(absent)")
    elif a.cmd == "set":
        set(a.key, json.loads(a.value), durable=a.durable)
        print("set", a.key)
    elif a.cmd == "del":
        delete(a.key)
    elif a.cmd == "last":
        print(last_updated(a.key))


if __name__ == "__main__":
    main()
