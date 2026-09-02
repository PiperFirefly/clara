#!/usr/bin/env python3
"""
code_check — Agent's ACI-hardening tool (Coding Cortex Phase 0.2).
Syntax + lint gate on a file or tree, structured output.

Why: the SWE-agent finding is that *interface quality* moves coding-agent
results more than raw reasoning. This is the "auto lint/typecheck on every edit,
fed back cleanly" piece — deterministic, no LLM, runnable after any change.

Usage:
  code_check.py [path ...] [--full] [--json]
  - path defaults to ~/memory, ~/mailtool, ~/coding-cortex
  - default lint = high-signal rules (E9, F); --full = entire ruff ruleset
  - --json = machine-readable

Exit 0 iff clean (so the harness can gate on it).
"""
import json
import os
import subprocess
import sys
import time

VENV_PY = os.path.expanduser("~/venvs/memory/bin/python")
RUFF = os.path.expanduser("~/venvs/memory/bin/ruff")
ROOTS = [os.path.expanduser(p) for p in ("~/memory", "~/mailtool", "~/coding-cortex")]
SKIP_DIRS = {"node_modules", "venv", "venvs", ".git", "__pycache__",
             "backups", "archive", "ingested", "library", "models"}


def collect(paths):
    files = []
    for p in paths:
        if os.path.isfile(p):
            if p.endswith(".py"):
                files.append(p)
        elif os.path.isdir(p):
            for dp, dns, fns in os.walk(p):
                dns[:] = [d for d in dns if d not in SKIP_DIRS]
                for f in fns:
                    if f.endswith(".py"):
                        files.append(os.path.join(dp, f))
    return sorted(set(files))


def pycompile(files):
    diags = []
    for f in files:
        r = subprocess.run([VENV_PY, "-m", "py_compile", f],
                           capture_output=True, text=True)
        if r.returncode != 0:
            diags.append({"file": f, "kind": "syntax", "line": None,
                          "code": None, "msg": r.stderr.strip()[:300]})
    return diags


def ruff_lint(files, full):
    if not os.path.exists(RUFF):
        return None, "ruff not installed"
    sel = ["--select", "ALL"] if full else ["--select", "E9,F"]
    r = subprocess.run([RUFF, "check", *sel, "--output-format", "json",
                        "--quiet", "--no-cache", *files],
                       capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
    except Exception:
        data = []
    diags = []
    for d in data:
        diags.append({"file": d.get("filename"), "kind": "lint",
                      "line": (d.get("location") or {}).get("row"),
                      "code": d.get("code"), "msg": d.get("message", "")})
    return diags, None


def main():
    argv = sys.argv[1:]
    full = "--full" in argv
    as_json = "--json" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        paths = ROOTS

    files = collect(paths)
    t0 = time.time()
    diags = pycompile(files)
    lint, lint_err = ruff_lint(files, full)
    if lint is not None:
        diags.extend(lint)

    summary = {
        "files": len(files),
        "diagnostics": len(diags),
        "syntax": sum(1 for d in diags if d["kind"] == "syntax"),
        "lint": sum(1 for d in diags if d["kind"] == "lint"),
        "secs": round(time.time() - t0, 2),
        "ruff": "ok" if lint is not None else lint_err,
    }
    if as_json:
        print(json.dumps({"summary": summary, "diagnostics": diags[:80]}, indent=2))
    else:
        print(f"code_check: {summary['files']} files, {summary['diagnostics']} issues "
              f"({summary['syntax']} syntax, {summary['lint']} lint) in {summary['secs']}s "
              f"[ruff: {summary['ruff']}]")
        for d in diags[:40]:
            loc = f":{d['line']}" if d.get("line") else ""
            print(f"  {d['kind'].upper()} {os.path.basename(d['file'])}{loc} "
                  f"({d.get('code') or '-'}) {d['msg'][:130]}")
    sys.exit(1 if diags else 0)


if __name__ == "__main__":
    main()
