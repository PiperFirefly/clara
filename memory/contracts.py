#!/usr/bin/env python3
"""Explicit invariants and contracts (coding-stack item #12).

Every nontrivial module should accumulate machine-readable assertions:
    input guarantees, output guarantees, side effects, error semantics,
    performance assumptions, thread-safety.

Agent edits AGAINST contracts instead of reconstructing intent every
session. This is a per-module contract registry stored in memory.db
(the same DB as code_nodes / concept_index), so it composes with
code_graph / concept / structural_edit: when opening a module, read its
contract; when changing it, update the contract so intent persists
across sessions instead of being re-derived each time.

Design: contracts keyed by (module_path, symbol). Each has the six
named fields plus an optional free-form note. machine-readable = JSON
field values + a machine-checkable 'check' expression (see properties.py,
which verifies contracts hold). No LLM in the loop.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegraph import connect  # noqa: E402

# the six canonical contract dimensions (plus 'note')
FIELDS = [
    "input_guarantees",
    "output_guarantees",
    "side_effects",
    "error_semantics",
    "performance_assumptions",
    "thread_safety",
]


def _ensure_contract_schema(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS contracts("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "module TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '', "
        "input_guarantees TEXT DEFAULT '', "
        "output_guarantees TEXT DEFAULT '', "
        "side_effects TEXT DEFAULT '', "
        "error_semantics TEXT DEFAULT '', "
        "performance_assumptions TEXT DEFAULT '', "
        "thread_safety TEXT DEFAULT '', "
        "note TEXT DEFAULT '', "
        "updated_at REAL)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_contracts_key "
                 "ON contracts(module, symbol)")
    conn.commit()


def _canonical(module, symbol):
    return module.strip(), (symbol or "").strip()


def upsert_contract(module, symbol, **fields):
    """Create or update a contract. fields keys are FIELDS + 'note'."""
    conn = connect()
    _ensure_contract_schema(conn)
    module, symbol = _canonical(module, symbol)
    now = time.time()
    existing = conn.execute(
        "SELECT * FROM contracts WHERE module=? AND symbol=?",
        (module, symbol)).fetchone()
    row = dict(existing) if existing else {
        "module": module, "symbol": symbol, "input_guarantees": "",
        "output_guarantees": "", "side_effects": "", "error_semantics": "",
        "performance_assumptions": "", "thread_safety": "", "note": "",
    }
    for k, v in fields.items():
        if k in FIELDS or k == "note":
            row[k] = v
    conn.execute(
        "INSERT OR REPLACE INTO contracts(module,symbol,input_guarantees,"
        "output_guarantees,side_effects,error_semantics,"
        "performance_assumptions,thread_safety,note,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (row["module"], row["symbol"], row["input_guarantees"],
         row["output_guarantees"], row["side_effects"], row["error_semantics"],
         row["performance_assumptions"], row["thread_safety"], row["note"], now))
    conn.commit()
    conn.close()
    return {"ok": True, "module": module, "symbol": symbol}


def get_contract(module, symbol=""):
    conn = connect()
    _ensure_contract_schema(conn)
    module, symbol = _canonical(module, symbol)
    if symbol:
        r = conn.execute(
            "SELECT * FROM contracts WHERE module=? AND symbol=?",
            (module, symbol)).fetchone()
    else:
        # prefer the module-level row; fall back to any symbol row
        r = conn.execute(
            "SELECT * FROM contracts WHERE module=? AND symbol=''",
            (module,)).fetchone() or conn.execute(
                "SELECT * FROM contracts WHERE module=? ORDER BY updated_at DESC "
                "LIMIT 1", (module,)).fetchone()
    conn.close()
    return dict(r) if r else None


def list_contracts(module=None, missing=False):
    """List contracts. If missing=True, report modules in code_nodes that
    have NO contract yet (coverage gap for the registry)."""
    conn = connect()
    _ensure_contract_schema(conn)
    if module:
        rows = conn.execute(
            "SELECT module, symbol, updated_at FROM contracts WHERE module=? "
            "ORDER BY symbol", (module,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    if missing:
        # modules = file node paths under the known repos
        have = {r["module"] for r in conn.execute(
            "SELECT DISTINCT module FROM contracts")}
        files = conn.execute(
            "SELECT DISTINCT path FROM code_nodes WHERE kind='file' "
            "ORDER BY path").fetchall()
        gap = []
        for f in files:
            p = f["path"]
            if p not in have:
                gap.append(p)
        conn.close()
        return gap
    rows = conn.execute(
        "SELECT module, symbol, updated_at FROM contracts ORDER BY module, symbol")
    conn.close()
    return [dict(r) for r in rows]


def stats():
    conn = connect()
    _ensure_contract_schema(conn)
    total = conn.execute("SELECT count(*) c FROM contracts").fetchone()["c"]
    symbols = conn.execute(
        "SELECT count(*) c FROM contracts WHERE symbol!=''").fetchone()["c"]
    conn.close()
    return {"contracts": total, "symbol_scoped": symbols}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=3, metavar=("MODULE", "SYMBOL", "JSON"),
                    help="upsert contract; JSON has field keys as keys")
    ap.add_argument("--get", nargs="+", metavar="MODULE",
                    help="get contract(s) for a module [and symbol]")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--missing", action="store_true",
                    help="list modules in code_nodes lacking a contract")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    if a.set:
        module, symbol = a.set[0], a.set[1]
        payload = json.loads(a.set[2])
        print(json.dumps(upsert_contract(module, symbol, **payload), indent=2))
    if a.get:
        module = a.get[0]
        symbol = a.get[1] if len(a.get) > 1 else ""
        c = get_contract(module, symbol)
        if not c:
            print(json.dumps({"module": module, "contract": None}))
        else:
            print(json.dumps(c, indent=2, default=str))
    if a.list:
        rows = list_contracts(missing=a.missing)
        for r in rows:
            if isinstance(r, str):
                print(f"  [no contract] {r}")
            else:
                tag = "" if r["symbol"] else "  (module)"
                print(f"  {r['module']}{(':'+r['symbol']) if r['symbol'] else ''}{tag}")
    if a.stats:
        print(json.dumps(stats(), indent=2))
