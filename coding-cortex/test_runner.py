#!/usr/bin/env python3
"""
Agent's test-runner harness — Coding Cortex Phase 0.3.

One command that runs every acceptance test in the codebase and returns a
machine-readable pass/fail/perf summary. This is the foundation every later
phase stands on (SelfBench, patch dossiers, property tests): "does it pass"
must be a *number* before it can gate anything.

Layers:
  1. syntax gate  — py_compile every .py, node --check every .js
  2. unit/eval    — run every *_eval.py / test_*.py / *_test.py

Output: human-readable, plus a final JSON summary line. Exit 0 iff nothing failed.
No LLM calls of its own; evals run as-is (the current eval suite is deterministic).
"""
import json
import os
import subprocess
import sys
import time

VENV_PY = os.path.expanduser("~/venvs/memory/bin/python")
# Syntax gate covers everything (tools + tests); test DISCOVERY is narrower so the
# harness never treats its own tool files (test_runner.py, show_test.py, code_check.py)
# as tests and recurses on itself.
SYNTAX_ROOTS = [os.path.expanduser(p) for p in
    ("~/memory", "~/mailtool", "~/cognitive-upgrades", "~/coding-cortex")]
TEST_ROOTS = [os.path.expanduser(p) for p in ("~/memory", "~/cognitive-upgrades")]
TIMEOUT = 180
SKIP_DIRS = {"node_modules", "venv", "venvs", ".git", "__pycache__", "backups", "archive"}


def discover(roots, test_match, ext_match):
    tests, sources = [], []
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                p = os.path.join(dp, f)
                if test_match(f):
                    tests.append(p)
                if ext_match(f):
                    sources.append(p)
    return sorted(set(tests)), sorted(set(sources))


def is_test(fn):
    return fn.endswith("_eval.py") or fn.startswith("test_") or fn.endswith("_test.py")


def run(cmd, timeout=TIMEOUT):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr,
                "secs": round(time.time() - t0, 2)}
    except subprocess.TimeoutExpired:
        return {"rc": -1, "out": "", "err": "TIMEOUT after %ds" % timeout,
                "secs": timeout}


def tail(s, n=6):
    lines = [l for l in s.strip().splitlines() if l.strip()]
    return lines[-n:]


def main():
    _, py_sources = discover(SYNTAX_ROOTS, lambda f: False, lambda f: f.endswith(".py"))
    js_sources, _ = discover(SYNTAX_ROOTS, lambda f: False, lambda f: f.endswith(".js"))
    tests, _ = discover(TEST_ROOTS, is_test, lambda f: False)

    out = {"syntax": [], "tests": [], "summary": {}}
    failed = 0

    # --- layer 1: syntax gate ---
    for p in py_sources:
        r = run([VENV_PY, "-m", "py_compile", p], timeout=60)
        ok = r["rc"] == 0
        if not ok:
            failed += 1
        out["syntax"].append({"file": p, "ok": ok, "secs": r["secs"],
                              "err": tail(r["err"], 3) if not ok else []})
    for p in js_sources:
        r = run(["node", "--check", p], timeout=60)
        ok = r["rc"] == 0
        if not ok:
            failed += 1
        out["syntax"].append({"file": p, "ok": ok, "secs": r["secs"],
                              "err": tail(r["err"], 3) if not ok else []})

    # --- layer 2: unit/eval ---
    for p in tests:
        r = run([VENV_PY, p])
        ok = r["rc"] == 0
        if not ok:
            failed += 1
        out["tests"].append({"file": p, "ok": ok, "rc": r["rc"], "secs": r["secs"],
                             "tail": tail(r["out"], 6) if not ok else tail(r["out"], 3)})

    n_syntax_bad = sum(1 for s in out["syntax"] if not s["ok"])
    n_tests_bad = sum(1 for t in out["tests"] if not t["ok"])
    out["summary"] = {
        "syntax_files": len(out["syntax"]),
        "syntax_failed": n_syntax_bad,
        "test_files": len(out["tests"]),
        "test_failed": n_tests_bad,
        "passed": (len(out["syntax"]) - n_syntax_bad) + (len(out["tests"]) - n_tests_bad),
        "failed": failed,
    }

    # human-readable
    print(f"syntax gate: {len(out['syntax'])} files, {n_syntax_bad} failed")
    for t in out["tests"]:
        mark = "PASS" if t["ok"] else "FAIL"
        print(f"  [{mark}] {t['file']}  ({t['secs']}s)")
        if not t["ok"]:
            for ln in t["tail"]:
                print(f"          {ln}")
    print("JSON_SUMMARY " + json.dumps(out["summary"]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
