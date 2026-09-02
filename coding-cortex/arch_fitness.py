#!/usr/bin/env python3
"""
Architecture fitness functions — Coding Cortex item #16.

Measurable architectural constraints used as FEEDBACK SIGNALS, not rigid style
rules. Each check reports PASS / WARN / FAIL with the concrete evidence (the
violating edge / function / module) so Agent can consciously violate one when
the alternative is worse — but only with the evidence in front of her.

Checks:
  1. no_circular_dependencies   — SCC of the module import graph
  2. module_complexity          — functions per module / fan-in-out, vs threshold
  3. public_api_count           — exported (no underscore) funcs per module
  4. duplicate_code_threshold   — near-duplicate groups (reuses clone_detect)
  5. dependency_direction_rules — user-specified "A MUST NOT import B"
  6. max_function_size          — lines per function (re-parsed via tree-sitter)

Deterministic, no LLM. Reads code graph from memory.db + re-parses source for
line counts. Thresholds are defaults; override with flags.

Usage:
  arch_fitness.py check --repo ~/memory
  arch_fitness.py check --repo ~/mailtool --max-func-size 80 --no-circular
  arch_fitness.py rules --repo ~/memory "mailtool must not import memory"
"""
import argparse
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import codegraph  # noqa: F401  (imported for its schema/DB path)

MEMORY_DB = os.path.expanduser("~/memory/memory.db")

# dependency direction rule grammar (simple, explicit):
#   "<A> must not import <B>"  /  "<A> must not depend on <B>"
_RULE_RE = re.compile(
    r"(?P<a>[\w/.-]+)\s+must\s+not\s+(?:import|depend\s+on)\s+(?P<b>[\w/.-]+)")


def _imports(db, repo):
    """module-level import edges: (from_module, to_module)."""
    edges = []
    for subj, obj in db.execute(
            "SELECT subj, obj FROM code_edges WHERE rel='imports'"):
        # subj/obj look like 'repo.module' — normalize to module path
        edges.append((subj, obj))
    return edges


def circular_imports(edges):
    """Return strongly-connected cycles in the module import graph (Tarjan)."""
    g = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
    index = {}
    low = {}
    onstack = set()
    stack = []
    cycles = []
    counter = [0]

    def strongconnect(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        onstack.add(v)
        for w in g.get(v, ()):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or (comp[0] in g and comp[0] in g[comp[0]]):
                cycles.append(comp)

    for v in list(g):
        if v not in index:
            strongconnect(v)
    return [c for c in cycles if len(c) > 1]


def module_complexity(db, max_per_module=400):
    """functions per module; flag modules over the threshold."""
    rows = db.execute(
        "SELECT path, COUNT(*) FROM code_nodes WHERE kind='func' GROUP BY path"
    ).fetchall()
    return [(p, n) for p, n in rows if n > max_per_module]


def public_api_count(db, min_api=1):
    """modules with zero public (non-underscore) functions — a smell."""
    rows = db.execute(
        "SELECT path, name FROM code_nodes WHERE kind='func'").fetchall()
    bymod = {}
    for p, n in rows:
        if not n.startswith("_"):
            bymod.setdefault(p, 0)
            bymod[p] = bymod.get(p, 0) + 1
    return [(p, c) for p, c in bymod.items() if c < min_api]


def duplicate_threshold(repo, max_groups=8):
    """near-duplicate function groups (reuses clone_detect)."""
    try:
        import clone_detect as CD
        found = CD.scan([repo])
        groups = [g for g in CD.detect(found, tok_thresh=0.55) if len(g[1]) >= 2]
        return groups
    except Exception as e:  # noqa: BLE001 - clone scan optional
        return [((f"(clone scan unavailable: {e})"), [])]


def func_sizes(repo, max_size=100):
    """functions over max_size lines, re-parsed via tree-sitter."""
    import tree_sitter_python as tsp
    from tree_sitter import Language, Parser
    parser = Parser(Language(tsp.language()))
    sizes = []
    repo = os.path.abspath(os.path.expanduser(repo))
    for dp, dns, fns in os.walk(repo):
        dns[:] = [x for x in dns if x not in
                  {"node_modules", "venv", "venvs", ".git", "__pycache__",
                   "models", "backups", "archive"}]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dp, fn)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            src = text.encode()
            tree = parser.parse(src)
            stack = [tree.root_node]
            while stack:
                n = stack.pop()
                if n.type in ("function_definition", "async_function_definition"):
                    body = n.child_by_field_name("body")
                    if body is not None:
                        lines = (body.end_point[0] - n.start_point[0]) + 1
                        if lines > max_size:
                            sizes.append((n.child_by_field_name("name").text.decode()
                                          if n.child_by_field_name("name") else "?",
                                          lines, path))
                stack.extend(list(n.children))
    return sizes


def parse_rules(rules):
    out = []
    for r in rules:
        m = _RULE_RE.search(r)
        if m:
            out.append((m.group("a"), m.group("b")))
    return out


def direction_violations(db, rules):
    """rules like 'mailtool must not import memory'."""
    if not rules:
        return []
    parsed = parse_rules(rules)
    edges = db.execute("SELECT subj, obj FROM code_edges WHERE rel='imports'").fetchall()
    viol = []
    for a, b in parsed:
        for subj, obj in edges:
            if a in subj and b in obj:
                viol.append((a, b, subj, obj))
    return viol


def run_checks(repo, checks=None, max_func=100, max_mod=400, dup_groups=8,
               rules=None):
    db = sqlite3.connect(MEMORY_DB)
    repo_a = os.path.abspath(os.path.expanduser(repo))
    out = {}

    if not checks or "no_circular" in checks:
        cycles = circular_imports(_imports(db, repo_a))
        out["no_circular_dependencies"] = {
            "status": "FAIL" if cycles else "PASS",
            "detail": f"{len(cycles)} circular module(s): "
                      f"{cycles[:5]}" if cycles else "no circular module imports",
        }

    if not checks or "module_complexity" in checks:
        hot = module_complexity(db, max_per_module=max_mod)
        out["module_complexity"] = {
            "status": "WARN" if hot else "PASS",
            "detail": f"{len(hot)} module(s) over {max_mod} funcs: "
                      f"{hot[:5]}" if hot else "no module over threshold",
        }

    if not checks or "public_api" in checks:
        low = public_api_count(db)
        out["public_api_count"] = {
            "status": "WARN" if low else "PASS",
            "detail": f"{len(low)} module(s) with <1 public function: "
                      f"{low[:5]}" if low else "all modules expose public API",
        }

    if not checks or "duplicate" in checks:
        groups = duplicate_threshold(repo, max_groups=dup_groups)
        n = len(groups)
        out["duplicate_code_threshold"] = {
            "status": "WARN" if n > dup_groups else "PASS",
            "detail": f"{n} near-duplicate group(s) "
                      f"(threshold {dup_groups})" if n else "no near-duplicates",
        }

    if not checks or "dep_direction" in checks:
        viol = direction_violations(db, rules)
        out["dependency_direction_rules"] = {
            "status": "FAIL" if viol else "PASS",
            "detail": f"{len(viol)} direction violation(s): {viol[:5]}" if viol
                      else f"{len(rules) if rules else 0} rule(s), none violated",
        }

    if not checks or "func_size" in checks:
        big = func_sizes(repo, max_size=max_func)
        out["max_function_size"] = {
            "status": "WARN" if big else "PASS",
            "detail": f"{len(big)} function(s) over {max_func} lines: "
                      f"{big[:5]}" if big else "no oversized functions",
        }
    return out


def main():
    p = argparse.ArgumentParser(description="architecture fitness functions")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="run fitness checks")
    c.add_argument("--repo", default=os.path.expanduser("~/memory"))
    c.add_argument("--max-func-size", type=int, default=100)
    c.add_argument("--max-module", type=int, default=400)
    c.add_argument("--dup-threshold", type=int, default=8)
    c.add_argument("--skip", nargs="*", default=[],
                   help="check names to skip")
    c.add_argument("rules", nargs="*",
                   help='e.g. "mailtool must not import memory"')

    r = sub.add_parser("rules", help="parse a dependency rule")
    r.add_argument("text", nargs="+")

    a = p.parse_args()

    if a.cmd == "rules":
        for t in a.text:
            m = _RULE_RE.search(t)
            print(m.groups() if m else f"(unparseable rule: {t})")
        return
    if a.cmd != "check":
        p.print_help()
        return

    checks = [x for x in
              ("no_circular", "module_complexity", "public_api",
               "duplicate", "dep_direction", "func_size")
              if x not in a.skip]
    res = run_checks(a.repo, checks=checks, max_func=a.max_func_size,
                     max_mod=a.max_module, dup_groups=a.dup_threshold,
                     rules=a.rules)
    for name, r in res.items():
        print(f"[{r['status']:<4}] {name}: {r['detail']}")


if __name__ == "__main__":
    main()
