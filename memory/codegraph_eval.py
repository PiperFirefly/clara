#!/usr/bin/env python3
"""
CodeGraph eval — deterministic assertions that the tree-sitter code structure
graph is (a) populated, (b) resolves symbols correctly, and (c) answers the
callers/callees/imports questions it exists to answer.

Checks:
  A. Graph sanity — node/edge counts above floor; calls + imports + inherits all present.
  B. Definition resolution — a known symbol resolves to its real file:line.
  C. Callers — memstore.remember is called by session_distiller.run (alias case,
     `import memstore as M`) and by at least one in-module caller.
  D. Callees — hive.run calls multiple known functions.
  E. Idempotency — re-ingesting one file does not duplicate its nodes.
  F. Cross-module imports — a file's import edges exist.

Usage: python3 codegraph_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codegraph as C
import memstore as M

ROOTS = [os.path.expanduser(p) for p in
         ("~/memory", "~/mailtool", "~/coding-cortex", "~/cognitive-upgrades")]

_fail = 0


def check(name, ok, detail=""):
    global _fail
    mark = "PASS" if ok else "FAIL"
    if not ok:
        _fail += 1
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    print("code graph eval\n")

    # A. graph sanity
    s = C.stats()
    check("nodes >= 1000", s["nodes"] >= 1000, f"{s['nodes']} nodes")
    check("edges >= 10000", s["edges"] >= 10000, f"{s['edges']} edges")
    check("calls edges present", s["by_rel"].get("calls", 0) > 0,
          f"{s['by_rel'].get('calls', 0)} calls")
    check("imports edges present", s["by_rel"].get("imports", 0) > 0,
          f"{s['by_rel'].get('imports', 0)} imports")
    check("inherits edges present", s["by_rel"].get("inherits", 0) > 0,
          f"{s['by_rel'].get('inherits', 0)} inherits")
    check("func+method nodes", s["by_kind"].get("func", 0) + s["by_kind"].get("method", 0) > 0,
          f"{s['by_kind']}")

    # B. definition resolution
    d = C.code_graph("memstore.remember", "defs", k=1)
    top = d["result"][0] if d["result"] else None
    ok = (top is not None and top["kind"] == "func"
          and "memstore" in top["path"] and top["name"].endswith(".remember"))
    check("resolve memstore.remember -> correct def", ok,
          f"{top['name'] if top else 'NONE'} @ {top['path'] if top else ''}:{top['line'] if top else ''}")

    # C. callers (alias case is the real test)
    callers = C.code_graph("memstore.remember", "callers", depth=1, k=50)
    caller_names = [x["name"] for x in callers["result"]]
    has_alias = any("session_distiller" in n for n in caller_names)
    has_internal = any(n == "memory.memstore.seed" or n == "memory.memstore.main"
                       for n in caller_names)
    check("alias caller: session_distiller -> memstore.remember", has_alias,
          f"{len(caller_names)} callers")
    check("in-module caller present", has_internal)

    # D. callees
    callees = C.code_graph("hive.run", "callees", depth=1, k=50)
    callee_names = [x["name"] for x in callees["result"]]
    has_real = any("hive" in n and "assess" in n for n in callee_names)
    check("hive.run calls hive.assess", has_real,
          f"{len(callee_names)} callees")

    # E. idempotency — re-ingest one file, node count must not grow
    conn = C.connect()
    before = conn.execute("SELECT count(*) c FROM code_nodes WHERE path=?",
                          (os.path.join(os.path.expanduser("~/memory"), "codegraph.py"),)).fetchone()["c"]
    C.ingest_file(conn, os.path.expanduser("~/memory/codegraph.py"), "memory", os.path.expanduser("~/memory"))
    after = conn.execute("SELECT count(*) c FROM code_nodes WHERE path=?",
                         (os.path.join(os.path.expanduser("~/memory"), "codegraph.py"),)).fetchone()["c"]
    conn.close()
    check("re-ingest is idempotent (no node duplication)", before == after,
          f"{before} -> {after}")

    # F. imports
    imp = C.code_graph("session_distiller", "imports", depth=1, k=30)
    imp_names = [x["name"] for x in imp["result"]]
    check("session_distiller has import edges", any("memstore" in n for n in imp_names),
          f"{len(imp_names)} imports")

    print(f"\n{'ALL PASS' if _fail == 0 else f'{_fail} FAILURE(S)'}")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
