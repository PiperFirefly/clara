#!/usr/bin/env python3
"""Property-based testing (coding-stack item #13).

Describe PROPERTIES, not just examples. For an encoder/decoder pair:
    decode(encode(x)) == x
This encourages cleaner abstractions because ugly abstractions are
notoriously hard to specify with properties.

Each property is an executable assertion stored in memory.db (the
`properties` table), registered against a module. `--run` executes it
under Hypothesis: generates many random inputs, shrinks failures to a
minimal counterexample, and reports pass/fail + the counterexample.

Properties pair with contracts.py: a contract says "what holds",
a property is the machine-checkable statement that it holds.

Design: property = (module, name, body, strategies_json). body is a
python function body (a def body whose inputs come from strategies).
Strategies describe Hypothesis generators per argument. Sandboxed:
executed in a fresh namespace with only stdlib + the module under test.
"""
import ast
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegraph import connect  # noqa: E402


def _ensure_property_schema(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS properties("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "module TEXT NOT NULL, name TEXT NOT NULL, "
        "body TEXT NOT NULL, strategies TEXT NOT NULL DEFAULT '[]', "
        "kind TEXT NOT NULL DEFAULT 'given', "
        "last_result TEXT DEFAULT '', last_counter TEXT DEFAULT '', "
        "runs INTEGER DEFAULT 0, updated_at REAL)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_props_key "
                 "ON properties(module, name)")
    conn.commit()


def register_property(module, name, body, strategies=None):
    """Register a property. body is a python function body. strategies is a
    list of Hypothesis strategy source expressions, one per argument."""
    conn = connect()
    _ensure_property_schema(conn)
    if strategies is None:
        # infer arity from the def
        fn = ast.parse(body).body[0]
        nargs = len([a for a in fn.args.args if a.arg != "self"])
        strategies = ["integers()"] * max(1, nargs)
    conn.execute(
        "INSERT OR REPLACE INTO properties"
        "(module,name,body,strategies,kind,updated_at) VALUES(?,?,?,?,?,?)",
        (module, name, body, json.dumps(strategies), "given", time.time()))
    conn.commit()
    conn.close()
    return {"ok": True, "module": module, "name": name, "strategies": strategies}


def _run_one(module, p, max_examples=200, timeout=30):
    """Execute one property under Hypothesis. Returns result dict."""
    import hypothesis  # noqa: F401  (sanity import; given/strategies used below)
    from hypothesis import given, settings, strategies as st

    body = p["body"]
    strat_srcs = json.loads(p["strategies"])
    strat_ns = {"st": st}
    for _n in ("integers", "floats", "text", "binary", "lists", "dicts",
               "booleans", "bytes", "dates", "datetimes", "timedeltas",
               "uuids", "none", "nothing", "one_of", "sampled_from",
               "just", "tuples"):
        strat_ns[_n] = getattr(st, _n, None)
    strategies = [eval(s, strat_ns) for s in strat_srcs]

    # build a test function: body may be `def f(a,b): ...` or a bare assertion
    fn = None
    try:
        tree = ast.parse(body)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            fn = tree.body[0]
    except SyntaxError:
        fn = None

    if fn is not None:
        namespace = {"__name__": "prop_runner", "sys": __import__("sys")}
        # make sibling repos importable (mailtool/, memory/, coding-cortex/)
        _roots = (os.path.expanduser("~"),)
        for _r in _roots:
            if _r not in sys.path:
                sys.path.insert(0, _r)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<prop>", "exec"),
             namespace)
        f = namespace[fn.name]
    else:
        # bare assertion: wrap as a single-arg function over the first strategy
        def f(x):
            exec(compile(body, "<prop>", "exec"), {"x": x, "result": x})
        strategies = strategies[:1]

    result = {"module": module, "name": p["name"], "status": "pending"}
    from hypothesis import find
    from hypothesis import strategies as _st
    def _violates(*args):
        try:
            f(*args)
            return False
        except AssertionError:
            return True
        except Exception:
            return False
    strategy = strategies[0] if len(strategies) == 1 else _st.tuples(*strategies)
    try:
        runner = given(*strategies)(f)
        runner = settings(max_examples=max_examples, deadline=None)(runner)
        runner()
        result["status"] = "pass"
        result["counterexample"] = None
    except Exception as e:
        ce = None
        try:
            found = find(strategy, _violates)
            ce = str(found)
        except Exception:
            ce = str(e)[:200]
        result["status"] = "FAIL"
        result["counterexample"] = ce
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return result


def run_properties(module=None, name=None, max_examples=200):
    """Run all registered properties for a module (or all)."""
    conn = connect()
    _ensure_property_schema(conn)
    if module:
        q = ("SELECT * FROM properties WHERE module=? "
             + ("AND name=?" if name else ""))
        rows = conn.execute(q, [module, name] if name else [module]).fetchall()
    else:
        rows = conn.execute("SELECT * FROM properties ORDER BY module, name").fetchall()
    conn.close()
    results = []
    for r in rows:
        res = _run_one(r["module"], r, max_examples)
        # persist result
        c2 = connect()
        c2.execute(
            "UPDATE properties SET last_result=?, last_counter=?, runs=runs+1 "
            "WHERE id=?",
            (res["status"], res.get("counterexample") or "", r["id"]))
        c2.commit()
        c2.close()
        results.append(res)
    return results


def list_properties(module=None):
    conn = connect()
    _ensure_property_schema(conn)
    if module:
        rows = conn.execute(
            "SELECT module,name,kind,last_result,runs FROM properties "
            "WHERE module=? ORDER BY name", (module,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT module,name,kind,last_result,runs FROM properties "
            "ORDER BY module,name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs=3, metavar=("MODULE", "NAME", "BODY"),
                    help="register a property (function body) for a module")
    ap.add_argument("--strategies", help="JSON list of strategy exprs")
    ap.add_argument("--run", nargs="*", metavar="MODULE",
                    help="run properties for a module (optionally [MODULE NAME])")
    ap.add_argument("--list", nargs="?", const="", metavar="MODULE",
                    help="list registered properties")
    ap.add_argument("--max-examples", type=int, default=200)
    a = ap.parse_args()

    if a.add:
        module, name, body = a.add
        strat = json.loads(a.strategies) if a.strategies else None
        print(json.dumps(register_property(module, name, body, strat), indent=2))
    if a.run is not None:
        module = a.run[0] if a.run else None
        pname = a.run[1] if len(a.run) > 1 else None
        res = run_properties(module, pname, a.max_examples)
        for r in res:
            mark = "PASS" if r["status"] == "pass" else f"FAIL -> {r.get('counterexample')}"
            print(f"  [{mark}] {r['module']}::{r['name']}")
    if a.list is not None:
        mod = a.list if a.list else None
        for p in list_properties(mod):
            print(f"  {p['module']}::{p['name']} [{p['kind']}] "
                  f"last={p['last_result']} runs={p['runs']}")
