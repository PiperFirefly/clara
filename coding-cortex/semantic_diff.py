#!/usr/bin/env python3
"""
Semantic diff — Coding Cortex item #19.

Review your own work as changed SYNTAX/BEHAVIOR, not changed lines. A syntax-aware
diff distinguishes "function moved" from "function rewritten" — enormously more
informative for large refactors than a line diff.

Approach (reuses clone_detect's normalized-shape machinery):
  * parse old and new file via tree-sitter
  * extract each function's normalized AST shape (identifiers/literals stripped)
    + call signature
  * match functions across versions by name first, then by shape
  * classify each function:
      added      — new name, no shape match
      removed    — old name gone
      moved      — same shape, different line (pure relocation)
      renamed    — different name, same shape (the body didn't change)
      rewritten  — same name, shape changed (behavior may have changed)
      unchanged  — same name AND same shape

Usage:
  semantic_diff.py <old_file> <new_file>
  semantic_diff.py --old old.py --new new.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser("~/coding-cortex"))
import clone_detect as CD


def extract_funcs(src):
    """Return {name: (norm_hash, call_signature, line)} for a source buffer."""
    import tree_sitter_python as tsp
    from tree_sitter import Language, Parser
    parser = Parser(Language(tsp.language()))
    tree = parser.parse(src)
    funcs = {}
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in ("function_definition", "async_function_definition"):
            nm = n.child_by_field_name("name")
            body = n.child_by_field_name("body")
            if nm is not None and body is not None:
                name = nm.text.decode()
                b0, b1 = body.start_byte, body.end_byte
                types, _toks = CD.normalized_shape(src[b0:b1], src.decode(errors="ignore"))
                calls = CD.call_signature(src[b0:b1], src.decode(errors="ignore"))
                funcs[name] = (CD.norm_hash(types), calls, n.start_point[0])
        stack.extend(list(n.children))
    return funcs


def semantic_diff(old_path, new_path):
    with open(old_path, encoding="utf-8", errors="ignore") as fh:
        old = fh.read().encode()
    with open(new_path, encoding="utf-8", errors="ignore") as fh:
        new = fh.read().encode()
    of = extract_funcs(old)
    nf = extract_funcs(new)
    o_names = set(of)
    n_names = set(nf)

    report = {"added": [], "removed": [], "moved": [], "renamed": [],
              "rewritten": [], "unchanged": []}

    # unchanged / rewritten / renamed / moved
    for name, (nh, nc, nline) in nf.items():
        if name in of:
            oh, _oc, oline = of[name]
            if nh == oh:
                if oline != nline:
                    report["moved"].append((name, oline, nline))
                else:
                    report["unchanged"].append(name)
            else:
                report["rewritten"].append(name)
    # renamed: old name gone, but its shape appears under a new name
    old_by_shape = {of[n][0]: n for n in o_names}
    for name, (nh, nc, nline) in nf.items():
        if name in o_names:
            continue  # already handled
        if nh in old_by_shape and old_by_shape[nh] not in n_names:
            report["renamed"].append((old_by_shape[nh], name))
        else:
            report["added"].append(name)
    for name in o_names - n_names:
        # removed unless it was renamed (renamed entries already consumed)
        if not any(name == old for old, _n in report["renamed"]):
            report["removed"].append(name)
    return report, of, nf


def render(report, of, nf):
    L = []
    for key, label in [("moved", "MOVED (same body, new location)"),
                       ("renamed", "RENAMED (same body, new name)"),
                       ("rewritten", "REWRITTEN (same name, body changed)"),
                       ("added", "ADDED (new)"),
                       ("removed", "REMOVED (gone)")]:
        items = report[key]
        if not items:
            continue
        L.append(f"{label} ({len(items)}):")
        for it in items:
            if key == "moved":
                L.append(f"    {it[0]}  line {it[1]} -> {it[2]}")
            elif key == "renamed":
                L.append(f"    {it[0]}  ->  {it[1]}")
            else:
                L.append(f"    {it}")
    L.append(f"UNCHANGED ({len(report['unchanged'])}): {report['unchanged'][:8]}{'...' if len(report['unchanged'])>8 else ''}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description="semantic diff (behavior, not lines)")
    p.add_argument("old", nargs="?", help="old file")
    p.add_argument("new", nargs="?", help="new file")
    p.add_argument("--old-file", dest="oldf")
    p.add_argument("--new-file", dest="newf")
    a = p.parse_args()
    old = a.old or a.oldf
    new = a.new or a.newf
    if not (old and new):
        p.print_help()
        return
    report, of, nf = semantic_diff(old, new)
    print(render(report, of, nf))


if __name__ == "__main__":
    main()
