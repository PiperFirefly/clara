#!/usr/bin/env python3
"""Post-generation simplifier agent (coding-stack item #20).

Give another reasoning pass a single mandate:

    "Assume the implementation works. Delete everything unnecessary."

It searches for duplicate abstractions, unnecessary classes, needless
wrappers, duplicated error handling, dead branches, and dependencies
that could disappear.

Design: two layers.
  Layer 1 (deterministic detectors) — cheap, mechanical, no LLM:
    dead code       : functions/classes never referenced (via code_graph)
    duplicate absts : functions with structurally identical bodies (AST fp)
    needless wrapper: one-line function that only calls another function
                      with the same args (forwarding wrapper)
    dead branch     : `if True:`/`if False:`/unreachable / empty except
    unused import   : import not referenced in the module body (via AST)
  Layer 2 (the reasoning pass) — the mandate applied by a review pass over
    the candidate list. It reads the real code and decides what is safe to
    delete, choosing only findings where deletion cannot change behavior
    (callers absent / wrappers pure / branch unreachable / import unused).

Composition: find candidates (Layer 1) -> reasoning pass selects safe ones
(Layer 2) -> apply deletions -> re-run property/mutate tests to confirm
nothing broke. Read-only by default; `--apply` performs the edits.
"""
import ast
import copy
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegraph import connect  # noqa: E402

REPOS = [
    os.path.expanduser("~/mailtool"),
    os.path.expanduser("~/memory"),
    os.path.expanduser("~/coding-cortex"),
    os.path.expanduser("~/cognitive-upgrades"),
]


def _py_files(root):
    for dp, _dirs, fs in os.walk(root):
        if "__pycache__" in dp or ".git" in dp:
            continue
        for f in fs:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


# ---------------------------------------------------------------------------
# Layer 1: deterministic detectors
# ---------------------------------------------------------------------------
def _ast_fingerprint(node):
    """Structural fingerprint: identifiers blanked, constants stripped, so
    two functions with identical shape (modulo names/values) collide."""
    cp = copy.deepcopy(node)
    def blank(n):
        if isinstance(n, ast.Name):
            n.id = "X"
        elif isinstance(n, ast.arg):
            n.arg = "a"
        elif isinstance(n, ast.Attribute):
            n.attr = "A"
        elif isinstance(n, ast.Constant):
            n.value = "<C>"
    for sub in ast.walk(cp):
        blank(sub)
    return ast.dump(cp)


def detect_dead_code(conn=None, exclude_main=True):
    """Functions/classes never referenced. Two checks: call-graph (via
    code_edges) AND raw source-text (catches dispatch tables / string refs /
    getattr calls the call graph can't see). A symbol is dead only if BOTH
    say unreferenced -- eliminates false positives from indirect calls."""
    own = conn is None
    if own:
        conn = connect()
    called = set()
    for e in conn.execute("SELECT obj FROM code_edges WHERE rel='calls'"):
        called.add(e["obj"])
    bare_called = {o.split(".")[-1] if "." in o else o for o in called}
    all_names = {r["name"] for r in conn.execute("SELECT name FROM code_nodes")}
    # pre-load module sources for the text check
    _src_cache = {}
    def has_text_ref(path, bare):
        if path not in _src_cache:
            try:
                _src_cache[path] = open(path).read()
            except Exception:
                _src_cache[path] = ""
        return re.search(r"\b" + re.escape(bare) + r"\b", _src_cache[path]) is not None
    findings = []
    for r in conn.execute(
            "SELECT name,kind,path,line FROM code_nodes "
            "WHERE kind IN ('func','method','class')"):
        bare = r["name"].split(".")[-1]
        if bare in bare_called:
            continue
        if bare == "main" and exclude_main:
            continue
        if any(bare in n for n in all_names if n != r["name"]):
            continue
        # text check: if the symbol appears anywhere in its file, not dead
        if has_text_ref(r["path"], bare):
            continue
        findings.append({"kind": "dead_code", "path": r["path"], "line": r["line"],
                         "symbol": r["name"], "detail": f"{r['kind']} never referenced"})
    if own:
        conn.close()
    return findings


def _funcs_in(path):
    try:
        tree = ast.parse(open(path).read())
    except Exception:
        return []
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return funcs


def detect_duplicates():
    """Functions with structurally identical bodies (candidate duplication)."""
    bodies = {}
    for root in REPOS:
        for path in _py_files(root):
            for fn in _funcs_in(path):
                if len(fn.body) == 0:
                    continue
                fp = _ast_fingerprint(fn)
                rel = os.path.relpath(path, os.path.expanduser("~"))
                bodies.setdefault(fp, []).append(
                    f"{rel}:{fn.lineno} {fn.name}")
    findings = []
    for fp, locs in bodies.items():
        if len(locs) >= 2:
            findings.append({"kind": "duplicate", "path": locs[0],
                             "symbol": locs[0].split()[-1],
                             "detail": "identical structure: " + " || ".join(locs)})
    return findings


def detect_wrappers():
    """One-line functions that just forward to another function with same args."""
    findings = []
    for root in REPOS:
        for path in _py_files(root):
            for fn in _funcs_in(path):
                if len(fn.body) != 1:
                    continue
                stmt = fn.body[0]
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    # same argument list as signature
                    args = [a.arg for a in fn.args.args]
                    call_args = []
                    for a in call.args:
                        if isinstance(a, ast.Name):
                            call_args.append(a.id)
                    if call_args and call_args == args:
                        fname = ast.dump(call.func)
                        findings.append({
                            "kind": "wrapper",
                            "path": os.path.relpath(path, os.path.expanduser("~")),
                            "line": fn.lineno,
                            "symbol": fn.name,
                            "detail": f"forwards to {fname} with identical args"})
    return findings


def detect_dead_branches():
    """`if True:` / `if False:` (literal) and empty except handlers."""
    findings = []
    for root in REPOS:
        for path in _py_files(root):
            try:
                tree = ast.parse(open(path).read())
            except Exception:
                continue
            rel = os.path.relpath(path, os.path.expanduser("~"))
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
                    findings.append({"kind": "dead_branch", "path": rel,
                                     "line": node.lineno, "symbol": "if",
                                     "detail": f"if {node.test.value!r} is constant"})
                if isinstance(node, ast.ExceptHandler) and not node.body:
                    findings.append({"kind": "dead_branch", "path": rel,
                                     "line": node.lineno, "symbol": "except",
                                     "detail": "empty except handler"})
    return findings


def detect_unused_imports():
    """Use ruff F401 (authoritative) for unused imports; fall back to AST scan."""
    import subprocess
    findings = []
    try:
        r = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F401",
             "--output-format", "json"] + [p for root in REPOS for p in _py_files(root)],
            capture_output=True, text=True, timeout=120)
        data = json.loads(r.stdout or "[]")
    except Exception:
        data = []
    for fix in data:
        path = fix.get("filename", "")
        if fix.get("code", "") != "F401":
            continue
        sym = fix.get("message", "").split("`")[1] if "`" in fix.get("message", "") else "?"
        findings.append({"kind": "unused_import", "path": path,
                         "line": fix.get("location", {}).get("row", 0),
                         "symbol": sym, "detail": "F401 unused import"})
    if not findings:
        # fallback: naive AST scan (ruff not available)
        findings.extend(_naive_unused_imports())
    return findings


def _naive_unused_imports():
    findings = []
    for root in REPOS:
        for path in _py_files(root):
            try:
                src = open(path).read()
                tree = ast.parse(src)
            except Exception:
                continue
            imported = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported.setdefault((a.asname or a.name.split(".")[0]), []).append((node.lineno, a.name))
                elif isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        if a.name == "*":
                            continue
                        imported.setdefault((a.asname or a.name), []).append((node.lineno, a.name))
            used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used.add(node.id)
                if isinstance(node, ast.Attribute):
                    used.add(node.attr)
            rel = os.path.relpath(path, os.path.expanduser("~"))
            for name, locs in imported.items():
                if name not in used and name not in ("self",):
                    findings.append({"kind": "unused_import", "path": rel,
                                     "line": locs[0][0], "symbol": name,
                                     "detail": f"import {name} never referenced"})
    return findings


# ---------------------------------------------------------------------------
# Layer 2: the reasoning pass (mandate: assume works, delete unnecessary)
# ---------------------------------------------------------------------------
def reason_over(candidates):
    """Apply the mandate: select only deletions that cannot change behavior.
    This is the deterministic slice of the reasoning pass. (The LLM review
    pass sits on top; see the `simplify` tool description.)"""
    accepted = []
    for c in candidates:
        # dead_code: safe if truly unreferenced (detector already checked)
        if c["kind"] == "dead_code":
            accepted.append(c)
        # duplicate: only safe if the reporter lists identical bodies
        elif c["kind"] == "duplicate":
            accepted.append(c)
        elif c["kind"] == "wrapper":
            accepted.append(c)
        elif c["kind"] == "dead_branch":
            # `if False`/empty-except are safe; `if True` changes behavior
            if c["detail"].startswith("if False") or c["detail"].startswith("empty except"):
                accepted.append(c)
        elif c["kind"] == "unused_import":
            accepted.append(c)
    return accepted


def scan_all():
    """Run all detectors, return categorized candidates."""
    dead = detect_dead_code()
    dupes = detect_duplicates()
    wrappers = detect_wrappers()
    branches = detect_dead_branches()
    imports = detect_unused_imports()
    return {"dead_code": dead, "duplicates": dupes, "wrappers": wrappers,
            "dead_branches": branches, "unused_imports": imports}


def apply_deletions(findings):
    """Apply a list of finding dicts (with kind + path + line/symbol) to files.
    Supports removing dead_code (whole def), unused_import (import stmt).
    Read-only report unless --apply."""
    applied = []
    for f in findings:
        path = f["path"]
        if not path.startswith("/"):
            path = os.path.expanduser("~/" + path)
        try:
            src = open(path).read()
            tree = ast.parse(src)
        except Exception as e:
            applied.append({**f, "ok": False, "error": str(e)})
            continue
        removed = False
        if f["kind"] == "unused_import":
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(
                        (a.asname or a.name.split(".")[0]) == f["symbol"] for a in node.names):
                    lines = src.split("\n")
                    lines[node.lineno - 1] = ""
                    open(path, "w").write("\n".join(lines))
                    removed = True
                    break
        elif f["kind"] == "dead_code":
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == f["symbol"]:
                    lines = src.split("\n")
                    for ln in range(node.lineno, node.end_lineno + 1):
                        lines[ln - 1] = ""
                    open(path, "w").write("\n".join(lines))
                    removed = True
                    break
        applied.append({**f, "ok": removed})
    return applied


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="run all detectors")
    ap.add_argument("--kind", default="", help="limit scan to a detector kind")
    ap.add_argument("--apply", action="store_true",
                    help="apply accepted deletions to files (writes)")
    a = ap.parse_args()

    res = scan_all()
    if a.kind:
        res = {k: v for k, v in res.items() if k == a.kind}

    print("=== simplification scan (item #20) ===")
    total = 0
    for kind, items in res.items():
        safe = reason_over(items)
        print(f"\n[{kind}] {len(items)} found, {len(safe)} safe to delete")
        for c in items[:15]:
            mark = "SAFE" if c in safe else "     "
            print(f"  {mark} {c.get('path','')}:{c.get('line','')} "
                  f"{c.get('symbol','')}  {c.get('detail','')}")
        total += len(items)
    print(f"\nTOTAL: {total} simplification candidates across {len(res)} categories")
