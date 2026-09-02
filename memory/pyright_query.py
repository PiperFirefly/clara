#!/usr/bin/env python3
"""
PyrightLSP — on-demand language-server query bridge (the "meaning" layer under
the tree-sitter cortex).

Tree-sitter codegraph.py gives STRUCTURE (parse tree -> symbol graph). pyright
gives MEANING: it resolves names across modules, knows types, tracks imports and
re-exports, and answers "who implements this class". This module spawns pyright's
language server ON DEMAND over stdio (boot, ask one question, kill) — no resident
daemon, so swap pressure stays flat. It is the Python-first slice of operator's
"language servers underneath tree-sitter" idea; Go/Rust/C++ servers are deferred
to a future-upgrade list (the box has no toolchains and is swap-bound).

Protocol: LSP 3.17 over stdio (Content-Length framed JSON-RPC). We use the Python
`pyright-langserver` binary installed locally at ~/memory/pyright_lsp/. No LLM,
no network — pure deterministic language-server intelligence.

Queries:
  hover       — "what type comes out of this symbol/expression?"  (textDocument/hover)
  definition  — "what does this symbol resolve to?"               (textDocument/definition)
  implementors— "who implements/extends this class?"              (workspace/symbol + refs)
  references  — "everywhere this symbol is used"                  (textDocument/references)
"""
import json
import os
import subprocess
import time

# --- locate the pyright langserver binary (installed locally via npm) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_LS = os.path.join(_HERE, "pyright_lsp", "node_modules", ".bin", "pyright-langserver")
_DEFAULT_TIMEOUT = 30  # seconds for the whole one-shot exchange

SKIP_DIRS = {"node_modules", "venv", "venvs", ".git", "__pycache__", "backups",
             "archive", "models", ".pytest_cache", "dist", "build", "site-packages"}


def _ensure_ls():
    if not os.path.exists(_LS):
        raise RuntimeError(
            f"pyright-langserver not found at {_LS}. "
            "Run: cd ~/memory/pyright_lsp && npm install pyright")


def _frame(obj):
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return b"Content-Length: %d\r\n\r\n%s" % (len(body), body)


def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_message(fp):
    headers = {}
    while True:
        line = fp.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("ascii").partition(":")
        headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", 0))
    if length == 0:
        return None
    return json.loads(fp.read(length).decode("utf-8"))


def _query(file_path, position, req, timeout=_DEFAULT_TIMEOUT, settle=0.0):
    """Run one LSP query against `file_path` at `position` (line,col 0-based)."""
    _ensure_ls()
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        return {"ok": False, "error": f"file not found: {file_path}"}
    uri = "file://" + file_path

    proc = subprocess.Popen(
        [_LS, "--stdio"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    try:
        # opening a workspace root lets pyright index the whole project, which
        # is needed for references / implementors; hover+definition work from a
        # single didOpen alone.
        root = os.path.dirname(os.path.abspath(file_path)) or "."
        root_uri = "file://" + root
        # Write initialize (with workspaceFolders so pyright discovers the real
        # project root, not its own install dir), initialized, and didOpen all
        # together up front with NO reads in between — reading between init
        # messages desyncs pyright's workspace-root resolution. The query-read
        # loop below skips notifications and the id:0 initialize response.
        for msg in [
            {"jsonrpc": "2.0", "id": 0, "method": "initialize",
             "params": {"processId": None, "rootUri": root_uri,
                         "workspaceFolders": [{"uri": root_uri, "name": root}],
                         "capabilities": {}}},
            {"jsonrpc": "2.0", "method": "initialized", "params": {}},
            {"jsonrpc": "2.0", "method": "textDocument/didOpen",
             "params": {"textDocument": {"uri": uri, "languageId": "python",
                                         "version": 1,
                                         "text": _read_text(file_path)}}},
        ]:
            proc.stdin.write(_frame(msg))
            proc.stdin.flush()

        if settle:
            time.sleep(settle)  # let async indexing finish before the query
        mid = 1
        params = {"textDocument": {"uri": uri}, "position": position}
        # req arrives as the LSP method name ("textDocument/references"); pyright
        # REQUIRES the references context param or it errors out.
        if req == "textDocument/references":
            params["context"] = {"includeDeclaration": True}
        proc.stdin.write(_frame({"jsonrpc": "2.0", "id": mid,
                                 "method": req, "params": params}))
        proc.stdin.flush()
        while True:
            resp = _read_message(proc.stdout)
            if resp is None:
                break
            if resp.get("id") == mid:
                return {"ok": True, "result": resp.get("result")}
    finally:
        proc.kill()
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
    return {"ok": False, "error": "no response from language server"}


def _position_for(file_path, line, col):
    """line/col are 1-based human; LSP wants 0-based."""
    return {"line": max(0, int(line) - 1), "character": max(0, int(col) - 1)}


def _fmt_hover(res):
    if not res or not res.get("result"):
        return "no hover info (unresolved symbol?)"
    h = res["result"]
    parts = []
    contents = h.get("contents")
    if isinstance(contents, str):
        parts.append(contents)
    elif isinstance(contents, dict) and contents.get("value"):
        parts.append(contents["value"])
    elif isinstance(contents, list):
        for c in contents:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and c.get("value"):
                parts.append(c["value"])
    rng = h.get("range")
    loc = f" @lines {rng['start']['line']+1}-{rng['end']['line']+1}" if rng else ""
    return "".join(parts).strip() + loc


def _fmt_definition(res):
    if not res or not res.get("result"):
        return "no definition found"
    locs = res["result"]
    if isinstance(locs, dict):
        locs = [locs]
    out = []
    for l in locs:
        rng = l.get("range", {})
        out.append(f"{l.get('uri','?')} lines {rng.get('start',{}).get('line',0)+1}-{rng.get('end',{}).get('line',0)+1}")
    return "\n".join(out)


def _fmt_references(res):
    if not res or not res.get("result"):
        return "no references"
    seen = set()
    out = []
    for r in res["result"]:
        uri = r.get("uri", "")
        line = r.get("range", {}).get("start", {}).get("line", 0) + 1
        if (uri, line) in seen:
            continue
        seen.add((uri, line))
        out.append(f"{uri} line {line}")
    return "\n".join(out) if out else "no references"


def _find_implementors(root_dir, class_name):
    """Find concrete subclasses/implementors of a class by scanning workspace
    source for inheritance. Pure textual scan over .py files (no LLM):
      class Sub(Base)        -> python subclass
      class Sub(Base1, Base) -> base listed
      class Sub(Base): pass  -> direct
    We match the class name as a whole word in the base-clause. This is the
    deterministic "who implements this interface" answer."""
    import re
    root = os.path.abspath(root_dir)
    pat = re.compile(r"class\s+(\w+)\s*\(([^)]*)\)")
    base_pat = re.compile(rf"(?:^|[\s,\.])({re.escape(class_name)})(?:$|[\s,\.:\)])")
    hits = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            try:
                lines = _read_text(p).splitlines()
            except OSError:
                continue
            for i, ln in enumerate(lines, 1):
                m = pat.search(ln)
                if not m:
                    continue
                cls, bases = m.group(1), m.group(2)
                if bases and base_pat.search(bases):
                    hits.append(f"{p} line {i}  class {cls}({bases.strip()})")
    if hits:
        return {"ok": True, "result": "\n".join(sorted(set(hits)))}
    return {"ok": True, "result": f"no subclass of '{class_name}' found in {root}"}


def query(file_path, req, line=None, col=1, root_dir=None, symbol=None):
    """Public entry. req: hover|definition|references|implementors.
    implementors uses `symbol` (or the file's basename) as the class to search."""
    req = (req or "hover").lower()
    if req == "implementors":
        if not root_dir:
            root_dir = os.path.dirname(os.path.abspath(file_path)) or "."
        cls = symbol or os.path.splitext(os.path.basename(file_path))[0]
        return _find_implementors(root_dir, cls)
    if line is None:
        line = 1
    lsp = {"hover": "textDocument/hover",
           "definition": "textDocument/definition",
           "references": "textDocument/references"}.get(req)
    if not lsp:
        return {"ok": False, "error": f"unknown query '{req}' (hover|definition|references|implementors)"}
    pos = _position_for(file_path, line, col)
    # references/indexing queries need a settle for async indexing; hover/def don't
    settle = 3.0 if req == "references" else 0.0
    res = _query(file_path, pos, lsp, settle=settle)
    if not res["ok"]:
        return res
    if req == "hover":
        return {"ok": True, "result": _fmt_hover(res)}
    if req == "definition":
        return {"ok": True, "result": _fmt_definition(res)}
    if req == "references":
        return {"ok": True, "result": _fmt_references(res)}
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Python file to query")
    ap.add_argument("req", nargs="?", default="hover",
                    choices=["hover", "definition", "references", "implementors"])
    ap.add_argument("--line", type=int, default=1)
    ap.add_argument("--col", type=int, default=1)
    ap.add_argument("--root", help="project root (for implementors)")
    ap.add_argument("--symbol", help="class/interface name for implementors")
    a = ap.parse_args()
    print(json.dumps(query(a.file, a.req, a.line, a.col, a.root, a.symbol), indent=2))
