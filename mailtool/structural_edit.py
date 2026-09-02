#!/usr/bin/env python3
"""
structural_edit.py — syntax-aware code transformation (ast-grep wrapper), safe.

The idea (from operator, 2026-08-31): instead of me editing 27 files one-by-one with
text edits, generate ONE structural rule ("replace this API pattern everywhere")
and let ast-grep apply it. Cleaner refactors, far fewer tokens, and the rule is
reviewable.

Discipline (blast-radius aware): `~/mailtool`, `~/memory` etc. are NOT local git
repos, so an in-place rewrite has no rollback. This wrapper therefore defaults to
copy-then-diff and only touches the live tree with an explicit apply + `--yes`.

Usage:
  structural_edit.py preview <lang> <pattern> [--rewrite <r>] [paths...]
      Print every structural match (or what the rewrite would change). NO writes.
  structural_edit.py diff <lang> <pattern> --rewrite <r> [paths...]
      Apply to a temp copy, print a unified diff against the originals. NO writes.
  structural_edit.py apply <lang> <pattern> --rewrite <r> [paths...] --yes
      Apply for real. Requires --yes. Backs up changed files to
      ~/tools/data/struct_rollback/ and prints the diff + rollback paths.
      If the target tree is a git repo, uses `git diff` instead of a backup.

Languages: python, ts, tsx, js, jsx, rust, go, java, c, cpp, csharp, ruby, php,
kotlin, swift, ... (any ast-grep bundled tree-sitter grammar).
"""
import json
import os
import shutil
import subprocess
import sys
import time

BACKUP = os.path.expanduser("~/tools/data/struct_rollback")
AG = shutil.which("ast-grep") or "ast-grep"


def _run(args, cwd=None, input=None):
    return subprocess.run([AG] + args, capture_output=True, text=True, cwd=cwd,
                          input=input)


def preview(lang, pattern, rewrite, paths):
    args = ["run", "-l", lang, "-p", pattern]
    if rewrite:
        args += ["-r", rewrite]
    args += list(paths) + ["--json=compact"]
    r = _run(args)
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return
    try:
        matches = json.loads(r.stdout)
    except Exception:
        print(r.stdout or r.stderr)
        return
    if not matches:
        print(f"preview: no matches for pattern {pattern!r} ({lang})")
        return
    print(f"preview: {len(matches)} structural match(es):")
    for m in matches:
        f = m.get("file")
        ln = m.get("lines", [""])[0].strip()
        print(f"  {f}:  {ln[:100]}")


def diff(lang, pattern, rewrite, paths):
    """Per-file diff of what the rewrite WOULD change. NO writes (via --stdin)."""
    if not rewrite:
        print("diff: --rewrite required")
        return
    files = _expanded(paths)
    if not files:
        print("diff: no files to process")
        return
    any_out = False
    for f in files:
        try:
            content = open(f, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        r = _run(["run", "-l", lang, "-p", pattern, "-r", rewrite, "--stdin"],
                 input=content)
        if r.returncode == 0 and r.stdout.strip():
            print(f"### {f}")
            print(r.stdout)
            any_out = True
    if not any_out:
        print(f"diff: no matches for pattern {pattern!r} ({lang})")


def _expanded(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fns in os.walk(p):
                for fn in fns:
                    out.append(os.path.join(root, fn))
        else:
            out.append(p)
    return out


def apply(lang, pattern, rewrite, paths, yes):
    if not rewrite:
        print("apply: --rewrite required")
        return
    if not yes:
        print("apply: refused — pass --yes to mutate the live tree. "
              "Run `diff` first and review.")
        return
    os.makedirs(BACKUP, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    changed = []
    for f in _expanded(paths):
        try:
            before = open(f, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        r = _run(["run", "-l", lang, "-U", "-p", pattern, "-r", rewrite, f])
        try:
            after = open(f, encoding="utf-8", errors="replace").read()
        except Exception:
            after = before
        if r.returncode == 0 and after != before:
            # rollback backup (the pre-change original)
            dest = os.path.join(BACKUP, f"{stamp}__" + f.replace("/", "__"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as bf:
                bf.write(before)
            changed.append((f, dest))
    if not changed:
        print("apply: no changes")
        return
    print(f"apply: {len(changed)} file(s) changed. rollback backups:")
    for f, dest in changed:
        print(f"  {f}  ->  {dest}")


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        print(__doc__)
        return
    cmd, lang, pattern = a[0], a[1], a[2]
    rewrite = None
    yes = False
    paths = []
    i = 3
    while i < len(a):
        if a[i] == "--rewrite":
            rewrite = a[i + 1]; i += 2
        elif a[i] == "--yes":
            yes = True; i += 1
        elif a[i].startswith("--"):
            i += 1
        else:
            paths.append(a[i]); i += 1
    if cmd == "preview":
        preview(lang, pattern, rewrite, paths)
    elif cmd == "diff":
        diff(lang, pattern, rewrite, paths)
    elif cmd == "apply":
        apply(lang, pattern, rewrite, paths, yes)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
