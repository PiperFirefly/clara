#!/usr/bin/env python3
"""
Operator config — the single source of truth for who this agent may interact
with and obey.

Cadence step 2 (cadence-decustomize-plan.md §4). This module is *machinery*:
it defines the schema, the one-time backfill from legacy scattered config, and
the read/write/lookup primitives. The actual operator values (names, channel
ids) live in the instance database — they are never hardcoded here.

Schema (two normalized tables in memory.db):

  operators          one row per operator (role = primary | associate)
  operator_channels  one row per (operator, channel-type, handle)

A "channel" is any address the agent can be contacted on or reach out to:
telegram id, email address, sms number, discord handle. The `handle` field is
the type-appropriate identifier (id / addr / num). Normalized on write so
lookups are case/digit-insensitive.

Migration: `migrate_from_legacy()` runs once (when no primary exists) and reads
the OLD scattered sources — the trusted-sender json files and the legacy
"collaborator" fact — into the new tables. The old files are left untouched
(fallback until step 7/8 wires the consumers and deletes them). Fails closed:
an empty/absent config means no primary operator, no trusted senders, no
outbound.

Usage:
  operator_config.py show                      dump the full config as JSON
  operator_config.py migrate [--name N]        backfill from legacy sources
  operator_config.py set-primary --name N --telegram ID --email ADDR --sms NUM
  operator_config.py add-associate --name N --telegram ID [--perm P] [--restr R]
  operator_config.py rm <name>                 remove an operator (any role)
  operator_config.py lookup <type> <handle>    who owns this channel, if anyone
"""
import argparse
import json
import os
import sqlite3
import time

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "memory.db")

# ---- Legacy scattered sources (read ONCE during backfill; never mutated) ----
# These are instance-local files that step 7/8 will retire. Until then they
# remain the authority and this module merely mirrors them into the table.
LEGACY_TELEGRAM_CFG = os.path.expanduser("~/.pi/agent/telegram.json")
LEGACY_EMAIL_CFG = os.path.expanduser("~/mailtool/agent_loop.json")
LEGACY_SMS_CFG = os.path.expanduser("~/mailtool/sms_config.json")
LEGACY_PRIMARY_FACT = "collaborator"            # legacy fact key holding the name

# Channel types this config understands.
CHANNEL_TYPES = ("telegram", "email", "sms", "discord")

# Conservative permission model (forward path for step 7's consumers).
# `authorized()` uses this until blast-radius / trusted-sender filters are
# wired to read the operator table directly.
ASSOCIATE_DEFAULT_PERMISSIONS = ["read", "notify"]
ASSOCIATE_DEFAULT_RESTRICTIONS = [
    "no-irreversible",
    "no-delete",
    "no-system",
    "no-git",
    "no-outbound-to-third-parties",
]


# --------------------------------------------------------------------------
# connection + schema
# --------------------------------------------------------------------------

def _connect():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def ensure_schema(conn):
    """Idempotently create the operator tables + lookup index."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS operators("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "role TEXT NOT NULL, "
        "name TEXT NOT NULL, "
        "trust_tier TEXT NOT NULL DEFAULT 'full', "
        "permissions TEXT, "
        "restrictions TEXT, "
        "active INTEGER NOT NULL DEFAULT 1, "
        "created_at REAL NOT NULL, "
        "updated_at REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS operator_channels("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "operator_id INTEGER NOT NULL REFERENCES operators(id) ON DELETE CASCADE, "
        "type TEXT NOT NULL, "
        "handle TEXT NOT NULL, "
        "created_at REAL NOT NULL, "
        "UNIQUE(operator_id, type, handle))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_operator_channels_lookup "
        "ON operator_channels(type, handle)"
    )


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

def normalize_handle(channel_type, handle):
    t = (channel_type or "").lower()
    h = str(handle or "").strip()
    if t == "email":
        return h.lower()
    if t == "sms":
        # digits only, preserving an optional leading '+' for readability.
        digits = "".join(ch for ch in h if ch.isdigit())
        return ("+" + digits) if h.lstrip().startswith("+") and digits else digits
    if t == "telegram":
        return "".join(ch for ch in h if ch.isdigit())
    return h


def _json_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(list(value))


def _json_array(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


# --------------------------------------------------------------------------
# write API
# --------------------------------------------------------------------------

def _upsert_channels(conn, operator_id, channels):
    for ch in channels or []:
        t = (ch.get("type") or "").lower()
        h = normalize_handle(t, ch.get("handle") or ch.get("id") or ch.get("addr") or ch.get("num"))
        if not t or not h or t not in CHANNEL_TYPES:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO operator_channels(operator_id, type, handle, created_at) "
            "VALUES(?,?,?,?)",
            (operator_id, t, h, time.time()),
        )


def set_primary(name, channels=None, trust_tier="full",
                permissions=None, restrictions=None):
    """Create or replace the single primary operator. Returns the operator id."""
    now = time.time()
    with _connect() as conn:
        ensure_schema(conn)
        cur = conn.execute("SELECT id FROM operators WHERE role='primary'")
        row = cur.fetchone()
        if row:
            oid = row["id"]
            conn.execute(
                "UPDATE operators SET name=?, trust_tier=?, permissions=?, "
                "restrictions=?, active=1, updated_at=? WHERE id=?",
                (name, trust_tier, _json_list(permissions),
                 _json_list(restrictions), now, oid),
            )
        else:
            cur = conn.execute(
                "INSERT INTO operators(role, name, trust_tier, permissions, "
                "restrictions, active, created_at, updated_at) "
                "VALUES('primary',?,?,?,?,1,?,?)",
                (name, trust_tier, _json_list(permissions),
                 _json_list(restrictions), now, now),
            )
            oid = cur.lastrowid
        _upsert_channels(conn, oid, channels)
        conn.commit()
    return oid


def add_associate(name, channels=None, permissions=None, restrictions=None):
    """Add an associate operator. Returns the operator id."""
    now = time.time()
    perms = list(permissions) if permissions else list(ASSOCIATE_DEFAULT_PERMISSIONS)
    restr = list(restrictions) if restrictions else list(ASSOCIATE_DEFAULT_RESTRICTIONS)
    with _connect() as conn:
        ensure_schema(conn)
        cur = conn.execute(
            "INSERT INTO operators(role, name, trust_tier, permissions, "
            "restrictions, active, created_at, updated_at) "
            "VALUES('associate',?,'limited',?,?,1,?,?)",
            (name, _json_list(perms), _json_list(restr), now, now),
        )
        oid = cur.lastrowid
        _upsert_channels(conn, oid, channels)
        conn.commit()
    return oid


def remove_operator(name):
    """Remove an operator (any role) by name. Returns True if one was removed."""
    with _connect() as conn:
        ensure_schema(conn)
        cur = conn.execute("DELETE FROM operators WHERE name=?", (name,))
        conn.commit()
        return cur.rowcount > 0


# --------------------------------------------------------------------------
# read API
# --------------------------------------------------------------------------

def _row_to_dict(conn, row):
    channels = [
        {"type": r["type"], "handle": r["handle"]}
        for r in conn.execute(
            "SELECT type, handle FROM operator_channels WHERE operator_id=? "
            "ORDER BY type, handle",
            (row["id"],),
        )
    ]
    d = {
        "name": row["name"],
        "role": row["role"],
        "trust_tier": row["trust_tier"],
        "channels": channels,
    }
    perms = _json_array(row["permissions"])
    restr = _json_array(row["restrictions"])
    if perms is not None:
        d["permissions"] = perms
    if restr is not None:
        d["restrictions"] = restr
    return d


def get_primary():
    with _connect() as conn:
        ensure_schema(conn)
        row = conn.execute(
            "SELECT * FROM operators WHERE role='primary' AND active=1 "
            "ORDER BY id LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return _row_to_dict(conn, row)


def get_associates():
    with _connect() as conn:
        ensure_schema(conn)
        rows = conn.execute(
            "SELECT * FROM operators WHERE role='associate' AND active=1 ORDER BY id"
        ).fetchall()
        return [_row_to_dict(conn, r) for r in rows]


def primary_channel(channel_type):
    """The primary operator's handle for a given channel type (e.g. 'email',
    'telegram', 'sms'), or None if unconfigured. Generic — no ids hardcoded."""
    p = get_primary()
    if not p:
        return None
    for ch in p.get("channels", []):
        if ch.get("type") == channel_type:
            return ch.get("handle")
    return None


def get_config():
    """The full config in the plan's §4 shape."""
    primary = get_primary()
    out = {"primary": primary, "associates": get_associates()}
    return out


def _operator_by_id(conn, oid):
    row = conn.execute("SELECT * FROM operators WHERE id=? AND active=1", (oid,)).fetchone()
    return _row_to_dict(conn, row) if row else None


def channel_owner(channel_type, handle):
    """Which active operator (dict) owns this channel, or None."""
    h = normalize_handle(channel_type, handle)
    with _connect() as conn:
        ensure_schema(conn)
        row = conn.execute(
            "SELECT oc.operator_id AS oid FROM operator_channels oc "
            "JOIN operators o ON o.id = oc.operator_id "
            "WHERE oc.type=? AND oc.handle=? AND o.active=1 LIMIT 1",
            (channel_type.lower(), h),
        ).fetchone()
        if not row:
            return None
        return _operator_by_id(conn, row["oid"])


def is_trusted_channel(channel_type, handle):
    """True if any active operator owns this channel (trusted-sender primitive)."""
    return channel_owner(channel_type, handle) is not None


# --------------------------------------------------------------------------
# authority (conservative; consumers wired in step 7)
# --------------------------------------------------------------------------

def authorized(requester_type, requester_handle, action):
    """Does this requester hold authority for `action`? Fails closed.

    Conservative stand-in for the blast-radius/outbound consumers (step 7).
    Primary operator: full authority. Associate: read/notify only, and only
    actions they are not restricted from. Unknown/absent requester: denied.
    """
    owner = channel_owner(requester_type, requester_handle)
    if owner is None:
        return False
    if owner["role"] == "primary" and owner.get("trust_tier") == "full":
        return True
    perms = owner.get("permissions") or ASSOCIATE_DEFAULT_PERMISSIONS
    restr = owner.get("restrictions") or ASSOCIATE_DEFAULT_RESTRICTIONS
    if action in restr:
        return False
    return action in perms


# --------------------------------------------------------------------------
# migration (one-time backfill from legacy scattered config)
# --------------------------------------------------------------------------

def _read_json(path, default):
    try:
        with open(os.path.expanduser(path)) as f:
            return json.load(f)
    except Exception:
        return default


def migrate_from_legacy(name=None):
    """Backfill the primary operator from legacy scattered config.

    Reads (never mutates) the old trusted-sender files + the legacy fact key.
    No-ops if a primary operator already exists, so operator edits are never
    clobbered. Returns a summary dict.
    """
    with _connect() as conn:
        ensure_schema(conn)
        existing = conn.execute(
            "SELECT 1 FROM operators WHERE role='primary' LIMIT 1"
        ).fetchone()
        if existing:
            return {"changed": False, "reason": "primary already exists"}

        # Primary name: explicit arg, else the legacy fact, else None.
        if not name:
            try:
                r = conn.execute(
                    "SELECT value FROM facts WHERE key=?", (LEGACY_PRIMARY_FACT,)
                ).fetchone()
                name = r["value"] if r else None
            except sqlite3.OperationalError:
                name = None
        if not name:
            return {"changed": False, "reason": "no primary name (legacy fact missing)"}

        channels = []
        # telegram id (from bridge config; no hardcoded fallback)
        tg = _read_json(LEGACY_TELEGRAM_CFG, {})
        tg_id = ((tg.get("profiles") or {}).get("default") or {}).get("allowedUserId")
        if tg_id:
            channels.append({"type": "telegram", "handle": str(tg_id)})
        # email trusted senders (first address is the primary contact)
        email_cfg = _read_json(LEGACY_EMAIL_CFG, {})
        emails = email_cfg.get("trusted_senders") or []
        if emails:
            channels.append({"type": "email", "handle": str(emails[0])})
        # sms trusted senders (first number)
        sms_cfg = _read_json(LEGACY_SMS_CFG, {})
        sms_nums = sms_cfg.get("trusted_senders") or []
        if sms_nums:
            channels.append({"type": "sms", "handle": str(sms_nums[0])})

        set_primary(name, channels=channels)
        return {"changed": True, "name": name, "channels": channels}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_config():
    print(json.dumps(get_config(), indent=2, ensure_ascii=False))


def _channels_from_args(args):
    ch = []
    if getattr(args, "telegram", None):
        ch.append({"type": "telegram", "handle": args.telegram})
    if getattr(args, "email", None):
        ch.append({"type": "email", "handle": args.email})
    if getattr(args, "sms", None):
        ch.append({"type": "sms", "handle": args.sms})
    if getattr(args, "discord", None):
        ch.append({"type": "discord", "handle": args.discord})
    return ch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("show")

    m = sub.add_parser("migrate")
    m.add_argument("--name")

    sp = sub.add_parser("set-primary")
    sp.add_argument("--name", required=True)
    sp.add_argument("--telegram"); sp.add_argument("--email")
    sp.add_argument("--sms"); sp.add_argument("--discord")

    aa = sub.add_parser("add-associate")
    aa.add_argument("--name", required=True)
    aa.add_argument("--telegram"); aa.add_argument("--email")
    aa.add_argument("--sms"); aa.add_argument("--discord")
    aa.add_argument("--perm", action="append")
    aa.add_argument("--restr", action="append")

    rm = sub.add_parser("rm")
    rm.add_argument("name")

    lk = sub.add_parser("lookup")
    lk.add_argument("type"); lk.add_argument("handle")

    args = p.parse_args()

    if args.cmd == "show":
        _print_config()
    elif args.cmd == "migrate":
        print(json.dumps(migrate_from_legacy(args.name), indent=2))
    elif args.cmd == "set-primary":
        set_primary(args.name, channels=_channels_from_args(args))
        _print_config()
    elif args.cmd == "add-associate":
        add_associate(args.name, channels=_channels_from_args(args),
                      permissions=args.perm, restrictions=args.restr)
        _print_config()
    elif args.cmd == "rm":
        print("removed" if remove_operator(args.name) else "not found")
    elif args.cmd == "lookup":
        owner = channel_owner(args.type, args.handle)
        print(json.dumps(owner, indent=2, ensure_ascii=False) if owner else "no owner")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
