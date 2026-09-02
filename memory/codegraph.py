#!/usr/bin/env python3
"""
CodeGraph — tree-sitter code structure graph, ingested into the SAME SQLite
store as the agent's memory (memory.db), but in dedicated tables so the semantic
memory graph (entities/edges → Personalized PageRank) is never polluted with
function names or slowed by them.

Thesis (comp_sci_plan.md Phase 4.1, pulled forward): the single largest measured
win in the 2026 agentic-coding literature is replacing file-scanning/grep with a
structured code graph (~94% fewer tool calls, ~77% faster exploration). This
module builds that graph deterministically — pure tree-sitter parsing, NO LLM
calls, NO network — and exposes a `code_graph()` query for callers/callees/
imports/definitions.

Languages: python, javascript/typescript, go, rust, php (lazy-loaded grammars;
add more by extending _lang_for/_parser/_extract_* and _files_under). Mixed-language
repos share one graph, so cross-language references (Go→fmt, Rust→Point, PHP→use)
are plain edges.

Storage (same memory.db, separate tables):
  code_nodes(id, kind, name, path, line, parent, signature, repo, lang, exported)
  code_edges(id, subj, obj, rel, path, repo)

Node `name` is globally-unique: `{repo}.{module}.{qualname}` (e.g.
"memory.memstore.MemStore.remember"). Edge rels: calls, imports, inherits.
Python import aliases (`import memstore as M`) are resolved so `M.remember`
refers to `memstore.remember`.

Pure and idempotent: re-ingesting a file deletes its prior nodes+edges first.
Parse errors are skipped, never fatal.
"""
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memstore import DB  # reuse the exact memory.db the hive already owns

# --- languages (lazy-loaded; only parsed when their files are seen) ---
_LANG = {}


def _lang_for(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return "python"
    if ext in (".js", ".mjs", ".cjs"):
        return "javascript"
    if ext in (".ts", ".tsx"):
        return "typescript"
    if ext == ".go":
        return "go"
    if ext in (".rs",):
        return "rust"
    if ext in (".php",):
        return "php"
    return None


def _parser(lang):
    if lang not in _LANG:
        from tree_sitter import Language, Parser
        if lang == "python":
            import tree_sitter_python
            L = Language(tree_sitter_python.language())
        elif lang == "javascript":
            import tree_sitter_javascript
            L = Language(tree_sitter_javascript.language())
        elif lang == "typescript":
            import tree_sitter_typescript
            L = Language(tree_sitter_typescript.language_typescript())
        elif lang == "go":
            import tree_sitter_go
            L = Language(tree_sitter_go.language())
        elif lang == "rust":
            import tree_sitter_rust
            L = Language(tree_sitter_rust.language())
        elif lang == "php":
            import tree_sitter_php
            L = Language(tree_sitter_php.language_php())
        else:
            raise ValueError(f"unsupported lang {lang}")
        _LANG[lang] = (Parser(L), L)
    return _LANG[lang][0]


SKIP_DIRS = {"node_modules", "venv", "venvs", ".git", "__pycache__", "backups",
             "archive", "models", ".pytest_cache", "dist", "build", "site-packages"}


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def _ensure_schema(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS code_nodes("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "kind TEXT NOT NULL, name TEXT NOT NULL, path TEXT NOT NULL, "
        "line INTEGER, parent TEXT, signature TEXT, "
        "repo TEXT NOT NULL, lang TEXT NOT NULL, exported INTEGER DEFAULT 0, "
        "mtime REAL, created_at REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cn_name ON code_nodes(name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cn_kind ON code_nodes(kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cn_path ON code_nodes(path)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS code_edges("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "subj TEXT NOT NULL, obj TEXT NOT NULL, rel TEXT NOT NULL, "
        "path TEXT, repo TEXT, created_at REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ce_subj ON code_edges(subj, rel)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ce_obj ON code_edges(obj, rel)")
    # migration: code_nodes gained an mtime column for incremental re-ingest
    try:
        cn_cols = {r["name"] for r in conn.execute("PRAGMA table_info(code_nodes)")}
    except sqlite3.OperationalError:
        cn_cols = set()
    if cn_cols and "mtime" not in cn_cols:
        conn.execute("ALTER TABLE code_nodes ADD COLUMN mtime REAL")
    conn.commit()


def connect():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _ensure_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# tree-sitter helpers
# ---------------------------------------------------------------------------
def _text(node, src):
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _field(node, name):
    return node.child_by_field_name(name)


def _dotted_name(node, src):
    """Best-effort dotted name from an identifier/attribute/call chain."""
    if node is None:
        return ""
    t = node.type
    if t == "identifier" or t == "type_identifier" or t == "field_identifier" or t == "name":
        return _text(node, src)
    if t in ("attribute", "member_expression", "property_identifier"):
        obj = _field(node, "object")
        attr = _field(node, "property") or _field(node, "attribute")
        obj_n = _dotted_name(obj, src) if obj else ""
        attr_n = _text(attr, src) if attr else ""
        if obj_n and attr_n:
            return f"{obj_n}.{attr_n}"
        return attr_n or obj_n
    if t == "call" or t == "call_expression":
        f = _field(node, "function")
        return _dotted_name(f, src) if f else ""
    # go selector_expression: operand.field
    if t == "selector_expression":
        op = _field(node, "operand")
        fld = _field(node, "field")
        op_n = _dotted_name(op, src) if op else ""
        fld_n = _text(fld, src) if fld else ""
        if op_n and fld_n:
            return f"{op_n}.{fld_n}"
        return fld_n or op_n
    # rust scoped_identifier: path::name ; field_expression: value.field
    if t == "scoped_identifier":
        p = _field(node, "path")
        nm = _field(node, "name")
        p_n = _dotted_name(p, src) if p else ""
        nm_n = _text(nm, src) if nm else ""
        if p_n and nm_n:
            return f"{p_n}.{nm_n}"
        return nm_n or p_n
    if t == "field_expression":
        v = _field(node, "value")
        fld = _field(node, "field")
        v_n = _dotted_name(v, src) if v else ""
        fld_n = _text(fld, src) if fld else ""
        if v_n and fld_n:
            return f"{v_n}.{fld_n}"
        return fld_n or v_n
    # php member/scoped access
    if t == "member_access_expression" or t == "member_call_expression":
        obj = _field(node, "object")
        nm = _field(node, "name")
        obj_n = _dotted_name(obj, src) if obj else ""
        nm_n = _text(nm, src) if nm else ""
        if obj_n and nm_n:
            return f"{obj_n}.{nm_n}"
        return nm_n or obj_n
    if t == "scoped_call_expression" or t == "scoped_property_access_expression":
        sc = _field(node, "scope")
        nm = _field(node, "name")
        sc_n = _dotted_name(sc, src) if sc else ""
        nm_n = _text(nm, src) if nm else ""
        if sc_n and nm_n:
            return f"{sc_n}::{nm_n}"
        return nm_n or sc_n
    return ""


# ---------------------------------------------------------------------------
# extraction: python (alias-aware)
# ---------------------------------------------------------------------------
def _extract_python(root, src, module, path):
    """Yield items: (kind, ...) where kind in
    {func, method, class, call, import, inherit}."""
    out = []
    aliases = {}  # local name -> real dotted name

    def expand(name):
        if not name:
            return name
        first, _, rest = name.partition(".")
        if first in aliases:
            return aliases[first] + (f".{rest}" if rest else "")
        return name

    def collect_imports(node):
        for c in node.named_children:
            if c.type == "import_statement":
                for d in c.named_children:
                    if d.type == "dotted_name":
                        full = _text(d, src)
                        aliases[full] = full
                        aliases.setdefault(full.split(".")[0], full.split(".")[0])
                        out.append(("import", module, full))
                    elif d.type == "aliased_import":
                        nm, al = _field(d, "name"), _field(d, "alias")
                        if nm is not None and al is not None:
                            aliases[_text(al, src)] = _text(nm, src)
                            out.append(("import", module, _text(nm, src)))
            elif c.type == "import_from_statement":
                m = _field(c, "module_name")
                mod = _text(m, src) if m is not None else ""
                if mod:
                    out.append(("import", module, mod))
                for d in c.named_children:
                    if d.type == "dotted_name":
                        nm = _text(d, src)
                        aliases[nm] = f"{mod}.{nm}" if mod else nm
                    elif d.type == "aliased_import":
                        nm, al = _field(d, "name"), _field(d, "alias")
                        if nm is not None and al is not None:
                            aliases[_text(al, src)] = (
                                f"{mod}.{_text(nm, src)}" if mod else _text(nm, src))
            else:
                collect_imports(c)

    def walk_calls(node, owner):
        """Collect call edges for owner, NOT descending into nested defs."""
        if node.type in ("function_definition", "class_definition", "lambda"):
            return
        if node.type == "call":
            fn = _field(node, "function")
            callee = _dotted_name(fn, src)
            if callee:
                out.append(("call", owner, expand(callee)))
        for ch in node.named_children:
            walk_calls(ch, owner)

    def walk_def(node, cls=None):
        if node is None:
            return
        if node.type == "class_definition":
            name_node = _field(node, "name")
            name = _text(name_node, src) if name_node else "<anon>"
            qn = f"{cls}.{name}" if cls else f"{module}.{name}"
            out.append(("class", qn, node.start_point[0] + 1, cls, name))
            args = _field(node, "superclasses")
            if args is not None:
                for c in args.named_children:
                    if c.type in ("identifier", "attribute"):
                        out.append(("inherit", qn, expand(_text(c, src))))
            body = _field(node, "body")
            if body is not None:
                for c in body.named_children:
                    walk_def(c, cls=qn)
        elif node.type == "function_definition":
            name_node = _field(node, "name")
            name = _text(name_node, src) if name_node else "<anon>"
            qn = f"{cls}.{name}" if cls else f"{module}.{name}"
            kind = "method" if cls else "func"
            params = _field(node, "parameters")
            psig = _text(params, src)[:80] if params else ""
            sig = f"{name}{psig}" if psig else name
            out.append((kind, qn, node.start_point[0] + 1, cls, sig))
            body = _field(node, "body")
            if body is not None:
                walk_calls(body, qn)
                for c in body.named_children:
                    if c.type in ("function_definition", "class_definition"):
                        walk_def(c, cls=qn)
        elif node.type == "decorated_definition":
            d = _field(node, "definition")
            if d is not None:
                walk_def(d, cls)

    collect_imports(root)
    for child in root.named_children:
        walk_def(child, None)
    return out


def _iter_descendants(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.named_children)


def _extract_javascript(root, src, module, path):
    """JS/TS: functions, methods, classes, calls, imports."""
    out = []

    def walk(node, cls):
        if node.type in ("function_declaration", "method_definition"):
            name_node = _field(node, "name")
            name = _text(name_node, src) if name_node else "<anon>"
            qn = f"{cls}.{name}" if cls else f"{module}.{name}"
            kind = "method" if cls else "func"
            out.append((kind, qn, node.start_point[0] + 1, cls, name))
            for c in _iter_descendants(node):
                if c.type == "call_expression":
                    fn = _field(c, "function")
                    callee = _dotted_name(fn, src)
                    if callee:
                        out.append(("call", qn, callee))
        elif node.type == "class_declaration":
            name_node = _field(node, "name")
            name = _text(name_node, src) if name_node else "<anon>"
            qn = f"{cls}.{name}" if cls else f"{module}.{name}"
            out.append(("class", qn, node.start_point[0] + 1, cls, name))
            body = _field(node, "body")
            if body is not None:
                for c in body.named_children:
                    walk(c, qn)
        elif node.type == "import_statement":
            src_node = _field(node, "source")
            if src_node is not None:
                out.append(("import", module, _text(src_node, src).strip("'\"")))
        else:
            for c in node.named_children:
                walk(c, cls)

    for child in root.named_children:
        walk(child, None)
    return out


def _extract_go(root, src, module, path):
    """Go: funcs, methods, struct/interface types, calls, imports."""
    out = []

    def walk_calls(node, owner):
        if node.type in ("function_declaration", "method_declaration", "func_literal"):
            return
        if node.type == "call_expression":
            fn = _field(node, "function")
            callee = _dotted_name(fn, src)
            if callee:
                out.append(("call", owner, callee))
        for c in node.named_children:
            walk_calls(c, owner)

    for child in root.named_children:
        t = child.type
        if t == "function_declaration":
            nm = _field(child, "name")
            name = _text(nm, src) if nm else "<anon>"
            qn = f"{module}.{name}"
            out.append(("func", qn, child.start_point[0] + 1, None, name))
            body = _field(child, "body")
            if body is not None:
                walk_calls(body, qn)
        elif t == "method_declaration":
            nm = _field(child, "name")
            recv = _field(child, "receiver")
            recv_t = ""
            if recv is not None:
                for pd in recv.named_children:
                    ty = _field(pd, "type")
                    if ty is not None:
                        recv_t = _text(ty, src).lstrip("*")
                        break
            name = _text(nm, src) if nm else "<anon>"
            qn = f"{module}.{recv_t}.{name}" if recv_t else f"{module}.{name}"
            out.append(("method", qn, child.start_point[0] + 1, recv_t or None, name))
            body = _field(child, "body")
            if body is not None:
                walk_calls(body, qn)
        elif t == "type_declaration":
            for spec in child.named_children:
                if spec.type != "type_spec":
                    continue
                nm = _field(spec, "name")
                ty = _field(spec, "type")
                name = _text(nm, src) if nm else "<anon>"
                qn = f"{module}.{name}"
                kind = "class"
                if ty is not None:
                    if ty.type == "interface_type":
                        kind = "interface"
                    elif ty.type == "struct_type":
                        kind = "struct"
                out.append((kind, qn, spec.start_point[0] + 1, None, name))
        elif t == "import_declaration":
            for spec in child.named_children:
                if spec.type == "import_spec":
                    p = _field(spec, "path")
                    if p is not None:
                        out.append(("import", module, _text(p, src).strip("'\"")))
    return out


def _extract_rust(root, src, module, path):
    """Rust: functions, structs/enums/traits, impl methods, calls, uses."""
    out = []

    def walk_calls(node, owner):
        if node.type in ("function_item", "closure_expression"):
            return
        if node.type == "call_expression":
            fn = _field(node, "function")
            callee = _dotted_name(fn, src)
            if callee:
                out.append(("call", owner, callee))
        for c in node.named_children:
            walk_calls(c, owner)

    def walk_item(node, cls=None):
        """Process one node; if it is a definition, index it, else recurse."""
        t = node.type
        if t == "function_item":
            nm = _field(node, "name")
            name = _text(nm, src) if nm else "<anon>"
            qn = f"{cls}.{name}" if cls else f"{module}.{name}"
            kind = "method" if cls else "func"
            out.append((kind, qn, node.start_point[0] + 1, cls, name))
            body = _field(node, "body")
            if body is not None:
                walk_calls(body, qn)
        elif t in ("struct_item", "enum_item", "trait_item", "union_item", "type_item"):
            nm = _field(node, "name")
            name = _text(nm, src) if nm else "<anon>"
            qn = f"{cls}.{name}" if cls else f"{module}.{name}"
            out.append(("class", qn, node.start_point[0] + 1, cls, name))
        elif t == "impl_item":
            ity = _field(node, "type")
            itn = _text(ity, src) if ity else ""
            body = _field(node, "body")
            base = f"{module}.{itn}" if itn else None
            if body is not None:
                for m in body.named_children:
                    if m.type == "function_item":
                        nm = _field(m, "name")
                        name = _text(nm, src) if nm else "<anon>"
                        qn = f"{base}.{name}" if base else f"{module}.{name}"
                        out.append(("method", qn, m.start_point[0] + 1, base, name))
                        mb = _field(m, "body")
                        if mb is not None:
                            walk_calls(mb, qn)
        else:
            for c in node.named_children:
                walk_item(c, cls)

    # collect uses, then walk everything else
    for child in root.named_children:
        if child.type == "use_declaration":
            arg = _field(child, "argument")
            if arg is not None:
                if arg.type == "scoped_identifier":
                    out.append(("import", module, _dotted_name(arg, src)))
                elif arg.type == "identifier":
                    out.append(("import", module, _text(arg, src)))
        else:
            walk_item(child, None)
    return out


def _extract_php(root, src, module, path):
    """PHP: classes/interfaces/traits/enums, methods, functions, calls, uses."""
    out = []

    def walk_calls(node, owner):
        if node.type in ("function_definition", "method_declaration", "anonymous_function"):
            return
        if node.type in ("function_call_expression", "member_call_expression",
                         "scoped_call_expression", "nullsafe_member_call_expression"):
            fn = _field(node, "function") or _field(node, "name") or _field(node, "scope")
            callee = _dotted_name(node, src)
            if not callee and fn is not None:
                callee = _dotted_name(fn, src)
            if callee:
                out.append(("call", owner, callee))
        elif node.type == "object_creation_expression":
            ty = _field(node, "type")
            if ty is not None:
                out.append(("call", owner, _text(ty, src)))
        for c in node.named_children:
            walk_calls(c, owner)

    def walk_members(node, cls):
        body = _field(node, "body") or _field(node, "declaration_list")
        if body is None:
            return
        for m in body.named_children:
            if m.type == "method_declaration":
                nm = _field(m, "name")
                name = _text(nm, src) if nm else "<anon>"
                qn = f"{cls}.{name}"
                out.append(("method", qn, m.start_point[0] + 1, cls, name))
                mb = _field(m, "body")
                if mb is not None:
                    walk_calls(mb, qn)

    for child in root.named_children:
        t = child.type
        if t in ("class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"):
            nm = _field(child, "name")
            name = _text(nm, src) if nm else "<anon>"
            qn = f"{module}.{name}"
            kind = t.replace("_declaration", "")
            out.append((kind, qn, child.start_point[0] + 1, None, name))
            # inheritance
            if t == "class_declaration":
                ic = _field(child, "class_interface_clause")
                if ic is not None:
                    for b in ic.named_children:
                        if b.type == "name":
                            out.append(("inherit", qn, _text(b, src)))
            walk_members(child, qn)
        elif t == "function_definition":
            nm = _field(child, "name")
            name = _text(nm, src) if nm else "<anon>"
            qn = f"{module}.{name}"
            out.append(("func", qn, child.start_point[0] + 1, None, name))
            body = _field(child, "body")
            if body is not None:
                walk_calls(body, qn)
        elif t == "namespace_use_declaration":
            for clause in child.named_children:
                if clause.type in ("namespace_use_clause", "namespace_use_group"):
                    qn = _field(clause, "qualified_name") or _field(clause, "name")
                    if qn is not None:
                        out.append(("import", module, _dotted_name(qn, src)))
    return out


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def _files_under(root):
    exts = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs", ".php"}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for f in fns:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(dp, f)


def _module_of(repo_root, path):
    rel = os.path.relpath(path, repo_root)
    rel = os.path.splitext(rel)[0]
    return rel.replace(os.sep, ".")


def parse_file(path, repo, repo_root):
    """Parse one file → (nodes, edges)."""
    lang = _lang_for(path)
    if lang is None:
        return [], []
    try:
        with open(path, "rb") as f:
            src = f.read()
    except (OSError, UnicodeDecodeError):
        return [], []
    try:
        tree = _parser(lang).parse(src)
    except Exception:
        return [], []
    module = _module_of(repo_root, path)
    root = tree.root_node
    if root.type == "ERROR" or not root.named_child_count:
        return [], []

    items = {
        "python": _extract_python,
        "javascript": _extract_javascript,
        "typescript": _extract_javascript,
        "go": _extract_go,
        "rust": _extract_rust,
        "php": _extract_php,
    }[lang](root, src, module, path)

    nodes = {}
    edges = []
    # the file itself is a node, so module-level imports can attach to it
    file_name = f"{repo}.{module}"
    nodes[file_name] = {
        "kind": "file", "name": file_name, "path": path, "line": 1,
        "parent": None, "signature": module, "repo": repo, "lang": lang,
    }
    for item in items:
        kind = item[0]
        if kind == "call":
            _, owner, callee = item
            edges.append((f"{repo}.{owner}", callee, "calls", path, repo))
        elif kind == "import":
            _, owner, mod = item
            edges.append((f"{repo}.{owner}", mod, "imports", path, repo))
        elif kind == "inherit":
            _, owner, base = item
            edges.append((f"{repo}.{owner}", base, "inherits", path, repo))
        elif kind in ("func", "method", "class"):
            _, qn, line, parent, sig = item
            name = f"{repo}.{qn}"
            nodes[name] = {
                "kind": kind, "name": name, "path": path, "line": line,
                "parent": parent, "signature": sig, "repo": repo, "lang": lang,
            }
    return list(nodes.values()), edges


def ingest_file(conn, path, repo, repo_root):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return 0, 0
    row = conn.execute(
        "SELECT mtime FROM code_nodes WHERE path=? LIMIT 1", (path,)).fetchone()
    if row is not None and row["mtime"] is not None and abs(row["mtime"] - mtime) < 0.001:
        return 0, 0  # unchanged since last ingest
    nodes, edges = parse_file(path, repo, repo_root)
    if not nodes and not edges:
        return 0, 0
    conn.execute("DELETE FROM code_nodes WHERE path=?", (path,))
    conn.execute("DELETE FROM code_edges WHERE path=?", (path,))
    now = time.time()
    for n in nodes:
        conn.execute(
            "INSERT INTO code_nodes(kind,name,path,line,parent,signature,repo,lang,mtime,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (n["kind"], n["name"], n["path"], n["line"], n["parent"],
             n["signature"], n["repo"], n["lang"], mtime, now))
    for e in edges:
        conn.execute(
            "INSERT INTO code_edges(subj,obj,rel,path,repo,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (e[0], e[1], e[2], e[3], e[4], now))
    conn.commit()
    return len(nodes), len(edges)


def ingest(roots, repo=None, clear=False):
    """Ingest all source files under each root. Returns (nodes, edges, files)."""
    conn = connect()
    if clear:
        conn.execute("DELETE FROM code_edges")
        conn.execute("DELETE FROM code_nodes")
        conn.commit()
    total_n = total_e = total_f = 0
    for root in roots:
        root = os.path.abspath(root)
        r = repo or os.path.basename(root.rstrip(os.sep)) or "root"
        for path in _files_under(root):
            n, e = ingest_file(conn, path, r, root)
            total_n += n
            total_e += e
            total_f += 1
    conn.close()
    return total_n, total_e, total_f


def update_from_diff(repo_root, repo=None, base="HEAD"):
    """Reparse only files changed since git base (incremental)."""
    import subprocess
    root = os.path.abspath(repo_root)
    r = repo or os.path.basename(root.rstrip(os.sep)) or "root"
    try:
        out = subprocess.check_output(
            ["git", "-C", root, "diff", "--name-only", base],
            stderr=subprocess.DEVNULL).decode()
    except Exception:
        return 0, 0, 0
    changed = [os.path.join(root, l) for l in out.splitlines()
               if _lang_for(os.path.join(root, l)) is not None]
    conn = connect()
    n = e = 0
    for path in changed:
        if os.path.exists(path):
            nn, ee = ingest_file(conn, path, r, root)
            n += nn
            e += ee
    conn.close()
    return n, e, len(changed)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
def _resolve(query, conn, limit=30):
    """Suffix/substring resolve a query string to node rows."""
    q = query.strip().lower()
    rows = conn.execute(
        "SELECT * FROM code_nodes WHERE lower(name) LIKE ? OR lower(name)=? "
        "ORDER BY (name=?) DESC, length(name) ASC LIMIT ?",
        (f"%{q}%", q, query, limit)).fetchall()
    return rows


def code_graph(query, direction="all", depth=1, k=25):
    """Resolve a symbol and return its definition + neighbors.

    direction: defs, callers, callees, imports, all.
    """
    conn = connect()
    nodes = _resolve(query, conn)
    if not nodes:
        conn.close()
        return {"query": query, "resolved": [], "result": []}

    resolved = [{"kind": r["kind"], "name": r["name"], "path": r["path"],
                 "line": r["line"], "signature": r["signature"], "repo": r["repo"]}
                for r in nodes]

    # Reference index: source-writable form → full node name. Call edges store
    # obj as-written (possibly bare or aliased); node names are repo-qualified.
    all_names = [r["name"] for r in conn.execute("SELECT name FROM code_nodes")]
    stripped_map = {}
    bare_map = {}
    full_lower = {}
    for n in all_names:
        full_lower[n.lower()] = n
        stripped = n.split(".", 1)[1] if "." in n else n
        bare = n.split(".")[-1]
        stripped_map.setdefault(stripped.lower(), []).append(n)
        bare_map.setdefault(bare.lower(), []).append(n)

    def resolve_ref(obj):
        o = obj.lower()
        hits = []
        if o in full_lower:
            hits.append(full_lower[o])
        hits.extend(stripped_map.get(o, []))
        hits.extend(bare_map.get(o, []))
        if not hits:
            hits = [n for n in all_names if n.lower().endswith("." + o)]
        seen = set()
        return [h for h in hits if not (h in seen or seen.add(h))]

    call_edges = conn.execute(
        "SELECT subj, obj FROM code_edges WHERE rel='calls'").fetchall()
    import_edges = conn.execute(
        "SELECT subj, obj FROM code_edges WHERE rel='imports'").fetchall()

    # name -> path, and path -> file-node name, so an imports query can attach
    # module-level imports to whatever symbol the user named.
    path_by_name = {r["name"]: r["path"]
                    for r in conn.execute("SELECT name, path FROM code_nodes")}
    file_by_path = {r["path"]: r["name"]
                    for r in conn.execute(
                        "SELECT name, path FROM code_nodes WHERE kind='file'")}

    out = {}

    def defn(name):
        if name in out:
            return out[name]
        r = conn.execute("SELECT * FROM code_nodes WHERE name=?", (name,)).fetchone()
        if r:
            out[name] = {"kind": r["kind"], "name": r["name"], "path": r["path"],
                         "line": r["line"], "signature": r["signature"], "repo": r["repo"]}
        else:
            out[name] = {"kind": "unknown", "name": name, "path": "", "line": None,
                         "signature": "", "repo": ""}
        return out[name]

    for r in nodes:
        defn(r["name"])

    frontier = [r["name"] for r in nodes]
    seen = set(frontier)
    for _ in range(max(1, depth)):
        nxt = []
        for name in frontier:
            if direction in ("callers", "all"):
                for e in call_edges:
                    if e["subj"] != name and name in resolve_ref(e["obj"]) and e["subj"] not in seen:
                        seen.add(e["subj"])
                        nxt.append(e["subj"])
            if direction in ("callees", "all"):
                for e in call_edges:
                    if e["subj"] == name:
                        refs = resolve_ref(e["obj"])
                        for m in refs:
                            if m not in seen:
                                seen.add(m)
                                nxt.append(m)
                        if not refs and e["obj"] not in seen:
                            seen.add(e["obj"])
                            nxt.append(e["obj"])
            if direction in ("imports", "all"):
                # imports attach to the file node; map any symbol to its file
                fname = file_by_path.get(path_by_name.get(name, ""), name)
                for e in import_edges:
                    if e["subj"] in (name, fname) and e["obj"] not in seen:
                        seen.add(e["obj"])
                        nxt.append(e["obj"])
        frontier = nxt

    for n in list(seen):
        defn(n)

    conn.close()
    return {"query": query, "resolved": resolved,
            "result": [out[n] for n in seen if n in out][:k]}


def stats():
    conn = connect()
    nn = conn.execute("SELECT count(*) c FROM code_nodes").fetchone()["c"]
    ne = conn.execute("SELECT count(*) c FROM code_edges").fetchone()["c"]
    by_kind = {r["kind"]: r["c"] for r in conn.execute(
        "SELECT kind, count(*) c FROM code_nodes GROUP BY kind")}
    by_rel = {r["rel"]: r["c"] for r in conn.execute(
        "SELECT rel, count(*) c FROM code_edges GROUP BY rel")}
    conn.close()
    return {"nodes": nn, "edges": ne, "by_kind": by_kind, "by_rel": by_rel}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--roots", nargs="+", default=[
        os.path.expanduser("~/memory"), os.path.expanduser("~/mailtool"),
        os.path.expanduser("~/coding-cortex"), os.path.expanduser("~/cognitive-upgrades")])
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("--direction", default="all")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    if a.ingest:
        n, e, f = ingest(a.roots, clear=a.clear)
        print(f"ingested {n} nodes, {e} edges from {f} files")
    if a.stats:
        print(json.dumps(stats(), indent=2))
    if a.query:
        print(json.dumps(code_graph(a.query, a.direction), indent=2))
