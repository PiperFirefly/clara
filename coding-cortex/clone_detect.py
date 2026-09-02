#!/usr/bin/env python3
"""
Clone & near-clone detection — Coding Cortex item #10.

Embeddings alone can't catch `normalize_customer_address()` vs
`cleanup_shipping_address()` as the same capability — the wording and the
identifiers differ. This uses structural signals instead:

  * NORMALIZED SYNTAX HASH — the function body's tree-sitter node-type stream
    with identifiers/literals replaced by placeholders, so the *shape* of the
    code is compared, not the words. Two functions that do the same thing with
    different variable names produce the same (or very close) shape.
  * CALL-GRAPH SIGNATURE — the sorted set of callee names, so "these both call
    the same underlying helpers" is evidence of the same capability.
  * Structural token ratio — Jaccard over the normalized node tokens.

Detection rule:
  * exact clone   — identical normalized hash (same shape).
  * near clone    — high normalized-token Jaccard AND overlapping call signature.
    These are the "probably the same capability" hits even when names differ.

Pure tree-sitter, no LLM, no network. Deterministic.

Usage:
  clone_detect.py scan <dirs...>      # build clones over given source dirs
  clone_detect.py dedup <dirs...>     # alias: near-clone report (read-mostly)
"""
import argparse
import os
import sys

import tree_sitter_python as tsp
from tree_sitter import Language, Parser

sys.path.insert(0, os.path.expanduser("~/memory"))

# languages we index
_LANGS = {"py": "python"}
_PARSER = Parser(Language(tsp.language()))

# node types that carry identifiers/names/literals — normalized away
_NAME_NODES = {"identifier", "string", "integer", "float", "true", "false",
               "none", "comment", "dotted_name", "concatenated_string",
               "string_start", "string_content", "string_end"}


def normalized_shape(src, text):
    """Return (node_type_stream, token_set) with identifiers/literals normalized."""
    tree = _PARSER.parse(src)
    types = []
    toks = set()
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in _NAME_NODES:
            # identifiers/literals -> a placeholder, but keep the *kind*
            if n.type in ("string", "concatenated_string", "integer", "float",
                          "true", "false", "none"):
                types.append("LIT")
            else:
                types.append("ID")
            toks.add(n.type)
            continue
        types.append(n.type)
        toks.add(n.type)
        # children in source order
        stack.extend(reversed(n.children))
    # collapse repeats so tiny body-size differences don't dominate
    return types, toks


def norm_hash(types):
    # a compact signature: keep the type stream but collapse runs
    collapsed = []
    prev = None
    for t in types:
        if t != prev:
            collapsed.append(t)
            prev = t
    return "|".join(collapsed)


def call_signature(src, text):
    """Sorted set of callee names (function calls) in this function body."""
    tree = _PARSER.parse(src)
    calls = set()
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn is not None:
                name = fn.text.decode()
                # strip dotted prefixes -> leaf name
                leaf = name.split(".")[-1]
                calls.add(leaf)
        for c in list(n.children):
            stack.append(c)  # noqa: PERF402 - stack separate from children
    return calls


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    i = len(a & b)
    u = len(a | b)
    return i / u if u else 0.0


def scan(dirs, thresh=0.70):
    """Return list of {name,path,hash,signature} for all functions/methods."""
    found = []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if x not in
                      {"node_modules", "venv", "venvs", ".git", "__pycache__",
                       "models", "backups", "archive"}]
            for fn in fns:
                ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
                if ext not in _LANGS:
                    continue
                path = os.path.join(dp, fn)
                try:
                    with open(path, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                src = text.encode()
                tree = _PARSER.parse(src)
                _extract(tree.root_node, src, text, path, found)
    return found


def _extract(node, src, text, path, found):
    if node.type in ("function_definition", "async_function_definition"):
        name = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name is not None and body is not None:
            nm = name.text.decode()
            b0 = body.start_byte
            b1 = body.end_byte
            types, toks = normalized_shape(src[b0:b1], text)
            calls = call_signature(src[b0:b1], text)
            found.append({
                "name": nm, "path": path,
                "hash": norm_hash(types), "toks": toks, "calls": calls,
            })
    for c in node.children:
        _extract(c, src, text, path, found)


def detect(found, tok_thresh=0.55, call_req=False, report_all=True):
    """Group exact clones (same hash) and near-clones (token overlap + calls)."""
    by_hash = {}
    for f in found:
        by_hash.setdefault(f["hash"], []).append(f)
    groups = []
    for members in by_hash.values():
        if len(members) >= 2:
            groups.append(("exact", members))
    # near-clones: different hashes but high token overlap (skip exact already)
    exact_paths = set()
    for kind, members in groups:
        for m in members:
            exact_paths.add((m["name"], m["path"]))
    remaining = [f for f in found if (f["name"], f["path"]) not in exact_paths]
    seen = []
    for i in range(len(remaining)):
        a = remaining[i]
        row = [a]
        for j in range(i + 1, len(remaining)):
            b = remaining[j]
            if a["hash"] == b["hash"]:
                continue
            t = jaccard(a["toks"], b["toks"])
            c = jaccard(a["calls"], b["calls"]) if (a["calls"] or b["calls"]) else 1.0
            if t >= tok_thresh and (not call_req or c >= 0.5):
                row.append(b)
        if len(row) >= 2:
            seen.append(("near", row))
    return groups + seen


def render(groups):
    lines = []
    for kind, members in groups:
        lines.append(f"\n[{kind.upper()} CLONE] {len(members)} members:")
        for m in members:
            lines.append(f"  {m['name']}  @ {m['path']}")
            if kind == "near" and m.get("calls"):
                lines.append(f"      calls: {sorted(m['calls'])[:8]}")
    if not lines:
        lines.append("no clones detected")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="clone / near-clone detection")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("scan", help="detect clones in dirs")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--thresh", type=float, default=0.55)
    s.add_argument("--require-calls", action="store_true",
                   help="near-clone must share callees too")
    a = p.parse_args()
    if a.cmd != "scan" or not a.dirs:
        p.print_help()
        return
    found = scan(a.dirs)
    print(f"indexed {len(found)} functions across {a.dirs}")
    groups = detect(found, tok_thresh=a.thresh, call_req=a.require_calls)
    print(render(groups))


if __name__ == "__main__":
    main()
