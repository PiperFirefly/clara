#!/usr/bin/env python3
"""behavior_gate.py — enforce behavior-preservation on refactors (fail-closed).

Bridges two ideas into a HARD, dependency-free gate (stdlib `ast` only):
  * classify every function MOVED / REWRITTEN / ADDED / REMOVED / UNCHANGED
    by comparing normalized AST bodies. MOVED/UNCHANGED = same body, safe.
  * for every REWRITTEN function, differential-test old vs new (differ) across
    generated inputs. If the new body isn't equivalent, the refactor silently
    changed behavior.

Gate rule (deterministic checkable version of "don't ship vibes"):
  a refactor is NOT "done" until every REWRITTEN function is proven equivalent to
  its reference, OR the rewrite is explicitly acknowledged (`--allow`) as an
  intentional behavior change. FAIL-CLOSED: any rewritten function we can't prove
  -> exit 1.

To differential-test a rewritten function you need BOTH bodies importable at once,
so the gate compares two FILE VERSIONS and imports each function from its own file.
For a genuine refactor keep the old file around, or pass pre/post paths from git.

Usage:
  behavior_gate.py check old.py new.py
      classify + prove equivalence; exit 1 on unproven rewrite.
  behavior_gate.py check old.py new.py --allow <name>[,<name>]
      accept these rewrites without proof (intentional behavior change).
  behavior_gate.py check old.py new.py --dry-run
      classify + report, but never fail (audit mode).
  behavior_gate.py git-check [--ref <rev>] [--path <subdir>] [--dry-run]
      gate every .py file changed between `ref` (default HEAD) and the working
      tree, pulling the old body from git. Run this after a refactor to prove
      nothing silently changed across the whole change set. Fail-closed.
"""

import argparse
import ast
import importlib.util
import inspect
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "coding-cortex"))
import differ


# Edge-case seed values to try before random inputs (mirrors differ's generator).
_EDGES = [0, 1, -1, 2, 10, 100, "", "a", "hello world", None, 3.14, -3.14, 0.0,
          float("inf"), float("-inf")]


def _gen_args(arity, n, seed):
    """Yield tuples of `arity` args — edges first, then random. Deterministic."""
    rng = random.Random(seed)
    for i in range(n):
        args = []
        for _ in range(arity):
            if rng.random() < 0.2:
                args.append(rng.choice(_EDGES))
            elif rng.random() < 0.5:
                args.append(rng.randrange(-1000, 1000))      # signed ints (catches sign bugs)
            else:
                args.append(rng.random() * 200 - 100)        # signed floats
        yield tuple(args)


def _differential(fn_a, fn_b, n=1000, seed=12345):
    """Arity-aware differential test. Returns (ok, mismatches)."""
    try:
        arity = len(inspect.signature(fn_a).parameters)
    except (TypeError, ValueError):
        arity = 1
    mismatches = []
    for args in _gen_args(arity, n, seed):
        try:
            try:
                ra = ("ret", differ._canonical(fn_a(*args)))
            except BaseException as e:  # noqa: BLE001
                ra = ("exc", differ._exc_family(e))
            try:
                rb = ("ret", differ._canonical(fn_b(*args)))
            except BaseException as e:  # noqa: BLE001
                rb = ("exc", differ._exc_family(e))
        except TypeError:
            continue  # fn rejects this combo — not a divergence
        if ra != rb:
            mismatches.append((args, ra, rb))
            if len(mismatches) >= 10:
                break
    return len(mismatches) == 0, mismatches


def extract_funcs(src):
    """Return {name: (normalized_body_digest, lineno)} for module-level defs."""
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = (ast.dump(node), node.lineno)
        elif isinstance(node, (ast.AsyncFunctionDef,)):
            out[node.name] = (ast.dump(node), node.lineno)
    return out


def semantic_diff(old_src, new_src):
    of = extract_funcs(old_src)
    nf = extract_funcs(new_src)
    report = {"added": [], "removed": [], "rewritten": [], "unchanged": [], "moved": []}
    for name, (nh, nline) in nf.items():
        if name in of:
            oh, oline = of[name]
            if nh == oh:
                if oline != nline:
                    report["moved"].append((name, oline, nline))
                else:
                    report["unchanged"].append(name)
            else:
                report["rewritten"].append(name)
        else:
            report["added"].append(name)
    for name in set(of) - set(nf):
        report["removed"].append(name)
    return report


def _load_module(path):
    spec = importlib.util.spec_from_file_location("_gate_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_fn(mod, name):
    cur = mod
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def gate(old_path, new_path, allow=None, dry_run=False):
    allow = set(allow or [])
    if not os.path.exists(old_path) or not os.path.exists(new_path):
        return 2, f"missing path: need both {old_path} and {new_path}"
    with open(old_path, encoding="utf-8", errors="ignore") as fh:
        old_src = fh.read()
    with open(new_path, encoding="utf-8", errors="ignore") as fh:
        new_src = fh.read()

    report = semantic_diff(old_src, new_src)
    lines = []
    for key, label in [("moved", "MOVED (same body, new location)"),
                       ("rewritten", "REWRITTEN (same name, body changed)"),
                       ("added", "ADDED (new)"),
                       ("removed", "REMOVED (gone)")]:
        if report[key]:
            lines.append(f"{label} ({len(report[key])}): {report[key]}")
    lines.append(f"UNCHANGED ({len(report['unchanged'])}): {report['unchanged'][:8]}"
                 f"{'...' if len(report['unchanged']) > 8 else ''}")

    rewritten = report["rewritten"]
    if not rewritten:
        lines.append("GATE: nothing rewritten -> no proof needed. PASS")
        return 0, "\n".join(lines)

    old_mod = _load_module(old_path)
    new_mod = _load_module(new_path)

    unproven, proven = [], []
    for name in rewritten:
        if name in allow:
            lines.append(f"  REWRITTEN {name}: explicitly ALLOWED (intentional change)")
            continue
        try:
            fa, fb = _find_fn(old_mod, name), _find_fn(new_mod, name)
        except Exception as e:  # noqa: BLE001
            unproven.append((name, f"cannot import: {e}"))
            continue
        if not callable(fa) or not callable(fb):
            unproven.append((name, "not both callable"))
            continue
        try:
            ok, mismatches = _differential(fa, fb, n=1000)
        except Exception as e:  # noqa: BLE001
            unproven.append((name, f"differ crashed: {e}"))
            continue
        if ok:
            proven.append(name)
        else:
            unproven.append((name, f"NOT equivalent ({len(mismatches)}+ diverging inputs)"))

    if proven:
        lines.append(f"GATE: proven equivalent: {proven}")
    if unproven:
        for name, why in unproven:
            lines.append(f"  FAIL {name}: {why}")
        if not dry_run:
            lines.append("GATE: FAIL — behavior change unproven (fail-closed).")
            return 1, "\n".join(lines)
    lines.append("GATE: PASS")
    return 0, "\n".join(lines)


def git_check(ref="HEAD", path=None, allow=None, dry_run=False):
    """Gate every .py file modified between `ref` and the working tree."""
    import subprocess
    allow = set(allow or [])
    if not os.path.isdir(".git"):
        return 2, "not a git repo (run from the repo root)"
    files = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", ref],
        capture_output=True, text=True).stdout.splitlines()
    files = [f for f in files if f.endswith(".py")]
    if path:
        files = [f for f in files if f.startswith(path)]
    if not files:
        return 0, "no changed .py files vs " + ref

    results = []
    any_fail = False
    for f in files:
        if not os.path.exists(f):
            continue  # deleted — nothing to compare
        old = subprocess.run(["git", "show", f"{ref}:{f}"], capture_output=True, text=True)
        if old.returncode != 0:
            continue  # new file (added) — no reference to compare
        tmp_old = os.path.join("/tmp", "_bgate_old_" + str(abs(hash(f))) + ".py")
        with open(tmp_old, "w", encoding="utf-8") as fh:
            fh.write(old.stdout)
        try:
            rc, out = gate(tmp_old, f, allow=allow, dry_run=dry_run)
        finally:
            os.remove(tmp_old)
        status = "PASS" if rc == 0 else "FAIL"
        results.append(f"[{status}] {f}")
        results.append("\n".join("    " + ln for ln in out.splitlines()))
        if rc != 0:
            any_fail = True
    body = "\n".join(results)
    if any_fail and not dry_run:
        return 1, "GATE FAIL (git-check): a refactor changed behavior unproven.\n" + body
    return 0, body


def main():
    p = argparse.ArgumentParser(description="behavior-preservation gate for refactors")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("old")
    c.add_argument("new")
    c.add_argument("--allow", help="comma-separated names to accept without proof")
    c.add_argument("--dry-run", action="store_true", help="report but never fail")
    g = sub.add_parser("git-check")
    g.add_argument("--ref", default="HEAD")
    g.add_argument("--path", help="restrict to a subdir (e.g. mailtool/)")
    g.add_argument("--allow", help="comma-separated names to accept without proof")
    g.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.cmd == "check":
        rc, out = gate(a.old, a.new, allow=(a.allow.split(",") if a.allow else None),
                       dry_run=a.dry_run)
    else:
        rc, out = git_check(a.ref, a.path,
                            allow=(a.allow.split(",") if a.allow else None),
                            dry_run=a.dry_run)
    print(out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
