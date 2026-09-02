#!/usr/bin/env python3
"""
show_test — run one test/eval and, on failure, show the traceback WITH source
context (the lines around the failing frame). ACI hardening (Phase 0.2): the
"show me the failing test plus ~15 lines around the traceback" tool.

This turns "an exception happened" into "here is the exact code, with the
failing line marked" — the difference between reading a crash and seeing it.

Usage:
  show_test.py <testfile.py> [--context N]

Exits 0 always (it is a viewer, not a gate — the harness/runner gates on pass/fail).
"""
import os
import re
import subprocess
import sys

VENV_PY = os.path.expanduser("~/venvs/memory/bin/python")
DEFAULT_CONTEXT = 15


def run_test(path):
    return subprocess.run([VENV_PY, path], capture_output=True, text=True)


def frame_context(err_text, ctx):
    frames = re.findall(r'File "([^"]+)", line (\d+)', err_text)
    if not frames:
        return None
    fpath, lineno = frames[-1]
    lineno = int(lineno)
    try:
        with open(fpath) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    lo = max(0, lineno - ctx - 1)
    hi = min(len(lines), lineno + ctx)
    out = [f"--- {fpath}  (lines {lo + 1}-{hi}, FAILING LINE {lineno}) ---"]
    for i in range(lo, hi):
        mark = ">>>" if i == lineno - 1 else "   "
        out.append(f"{mark} {i + 1:4d}  {lines[i]}")
    return "\n".join(out)


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: show_test.py <testfile.py> [--context N]")
        sys.exit(2)
    path = argv[0]
    ctx = DEFAULT_CONTEXT
    if "--context" in argv:
        try:
            ctx = int(argv[argv.index("--context") + 1])
        except (IndexError, ValueError):
            ctx = DEFAULT_CONTEXT

    r = run_test(path)
    print(f"=== {path}  (exit {r.returncode}) ===")
    if r.returncode == 0:
        body = (r.stdout or "").strip()
        print(body[-2000:] if len(body) > 2000 else body)
        print("\n[PASS]")
    else:
        err = (r.stderr or "") or (r.stdout or "")
        print(err.strip()[-3000:])
        ctx_text = frame_context(err, ctx)
        if ctx_text:
            print("\n" + ctx_text)
        else:
            print("\n(no source context resolvable — traceback may reference a frame "
                  "outside this checkout)")


if __name__ == "__main__":
    main()
