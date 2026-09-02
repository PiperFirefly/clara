#!/usr/bin/env python3
"""
Multi-repository transplantation tooling — Coding Cortex item #18.

Understands the DEPENDENCY ENVELOPE around a component — not just the function,
but everything it needs to live in a new repo: its callee closure, the modules
it imports, external/third-party dependencies, config keys, DB tables/schemas,
tests, and assumptions. Then a subsystem can be TRANSPLANTED, not copy-pasted.

Walks the existing code_graph (callers/callees/imports) transitively to find the
closure, then annotates each member's envelope.

Usage:
  transplant.py <symbol> [--depth N] [--repo X]
    e.g. transplant.py memstore.remember
    e.g. transplant.py consolidate.dedup --depth 3

  transplant.py plan <symbol>    # emit a transplant checklist (files, deps, tests)
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import codegraph as CG

# heuristic: config/schema/test/dep markers in a path
_CONFIG = re.compile(r"(config|settings|\.env|secrets|schema|migration|migrate|"
                    r"model|table|ddl)", re.IGNORECASE)
_TEST = re.compile(r"(test|eval|fixture|spec)", re.IGNORECASE)


def closure(symbol, depth=2, k=50):
    """Walk callees + imports transitively; return dedup set of qualified names."""
    seen = set()
    queue = [symbol]
    out = []
    for _ in range(depth):
        nxt = []
        for q in queue:
            if q in seen:
                continue
            seen.add(q)
            try:
                r = CG.code_graph(q, direction="callees", depth=1, k=k)
            except Exception as e:  # noqa: BLE001 - skip unresolvable
                out.append({"query": q, "error": str(e)})
                continue
            resolved = r.get("resolved", []) if isinstance(r, dict) else []
            res = r.get("result", []) if isinstance(r, dict) else []
            info = {"query": q}
            if resolved:
                info["resolved"] = resolved[0]
            info["callees"] = [x.get("name") for x in res]
            out.append(info)
            for x in res:
                nm = x.get("name", "")
                if nm and x.get("kind") == "func":
                    nxt.append(nm)
            for x in resolved:
                nm = x.get("name", "")
                if nm and x.get("kind") == "func":
                    nxt.append(nm)
        queue = [q for q in nxt if q not in seen][:k]
    return out


def classify(path):
    """Categorize a source file by its likely envelope role."""
    tags = []
    base = os.path.basename(path or "")
    if _TEST.search(path or ""):
        tags.append("test")
    if _CONFIG.search(path or ""):
        tags.append("config/schema")
    if base.endswith(".py"):
        tags.append("python-source")
    elif base.endswith((".js", ".ts")):
        tags.append("js-source")
    return tags


_BUILTINS = {"any", "astype", "c", "conn", "dict", "e", "float", "len",
             "list", "mat", "math", "max", "min", "os", "round", "sqlite3",
             "time", "vec", "str", "int", "bool", "set", "tuple",
             "json", "open", "range", "print", "enumerate", "sum", "abs",
             "sorted", "isinstance", "getattr", "setattr", "hasattr",
             "Exception", "BaseException", "None", "True", "False", "self",
             "return", "raise", "except", "if", "for", "while", "lambda",
             "import", "from"}


def _is_external(name):
    """True if a callee name looks like a real external/library dependency (a
    module or dotted package), not a builtin, single-letter loop var, or
    control-flow fragment."""
    base = name.split("(")[0].split(".")[0]
    return not (base in _BUILTINS or len(base) <= 1 or not base[0].isalpha())


def envelope(symbol, depth=2, k=50):
    """Return structured envelope: closure, files, externals, config, tests."""
    cl = closure(symbol, depth=depth, k=k)
    files = {}
    externals = set()
    configs = set()
    tests = set()
    for info in cl:
        r = info.get("resolved")
        if isinstance(r, dict) and r.get("path"):
            p = r["path"]
            files.setdefault(p, set()).add(info["query"])
            for t in classify(p):
                if t == "test":
                    tests.add(p)
                elif t == "config/schema":
                    configs.add(p)
        for c in info.get("callees", []):
            if _is_external(c):
                externals.add(c.split("(")[0].split(".")[0])
    return {
        "symbol": symbol,
        "closure_size": len(cl),
        "source_files": sorted(files.keys()),
        "external_deps": sorted(e for e in externals if e),
        "config_schema_files": sorted(configs),
        "test_files": sorted(tests),
        "members": cl,
    }


def plan(symbol, depth=2, k=50):
    e = envelope(symbol, depth=depth, k=k)
    lines = [f"TRANSPLANT PLAN: {symbol}", ""]
    lines.append(f"  Closure size: {e['closure_size']} nodes (depth {depth})")
    lines.append("\n  Files to transplant (source):")
    for f in e["source_files"]:
        lines.append(f"    - {f}")
    lines.append("\n  Config / schema / migration files (envelope):")
    for f in e["config_schema_files"] or ["(none detected)"]:
        lines.append(f"    - {f}")
    lines.append("\n  Tests to carry over:")
    for f in e["test_files"] or ["(none detected)"]:
        lines.append(f"    - {f}")
    lines.append("\n  External dependencies to re-resolve in the new repo:")
    for d in e["external_deps"] or ["(none detected — pure stdlib/builtin?)"]:
        lines.append(f"    - {d}")
    lines.append("\n  Assumptions to verify in the new repo:")
    for a in ["DB path / connection lifecycle",
              "config keys and env vars",
              "error-handling contract",
              "logging conventions",
              "migration ordering"]:
        lines.append(f"    ? {a}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="transplant a subsystem w/ its envelope")
    p.add_argument("symbol", help="qualified symbol, e.g. memstore.remember")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--plan", action="store_true",
                   help="emit a transplant checklist (default: JSON envelope)")
    a = p.parse_args()
    if a.plan:
        print(plan(a.symbol, depth=a.depth, k=a.k))
    else:
        print(json.dumps(envelope(a.symbol, depth=a.depth, k=a.k), indent=1))


if __name__ == "__main__":
    main()
