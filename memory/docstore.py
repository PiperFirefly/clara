#!/usr/bin/env python3
"""docstore.py — the document store: the database home for long-form content.

The migration target for `.md` files that are NOT instruction surface, release
docs, READMEs, or the reading library. Holds plans, designs, research notes,
identity/canon, stories, and ephemera as structured rows (with a stable `key`),
with a full-text index for search.

Schema (in memory.db, alongside facts/memories/beliefs):

  documents(id PK, key UNIQUE, kind, title, content, source, tags,
            created_at, updated_at, invalidated_at)

`key` is the stable identifier (e.g. the original relative file path, so a
migration is idempotent). `kind` is one of: plan, design, research, identity,
canon, story, note, ephemera, report, spec. `invalidated_at` is a soft delete.

Every accessor fails closed (returns None/[] on a missing table), so nothing
depends on the store existing.

Usage:
  docstore.py list [--kind K] [--all]         list keys/titles
  docstore.py get KEY                          print one document's body
  docstore.py search QUERY [--limit N]         full-text search
  docstore.py set KEY KIND TITLE [--content F] [--source S] [--tags "a,b"]
  docstore.py rm KEY                           soft-delete
  docstore.py migrate                          (see migrate_md.py for bulk import)
"""
import argparse
import json
import os
import sys
import time

DB = os.path.join(os.path.expanduser("~"), "memory", "memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  source TEXT,
  tags TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  invalidated_at REAL
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
  key, title, content,
  content='documents', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
  INSERT INTO documents_fts(rowid, key, title, content)
  VALUES (new.id, new.key, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
  INSERT INTO documents_fts(documents_fts, rowid, key, title, content)
  VALUES ('delete', old.id, old.key, old.title, old.content);
END;
CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
  INSERT INTO documents_fts(documents_fts, rowid, key, title, content)
  VALUES ('delete', old.id, old.key, old.title, old.content);
  INSERT INTO documents_fts(rowid, key, title, content)
  VALUES (new.id, new.key, new.title, new.content);
END;
"""

VALID_KINDS = {"plan", "design", "research", "identity", "canon", "story",
               "note", "ephemera", "report", "spec", "memory"}


def _connect():
    import sqlite3
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def ensure_schema():
    with _connect() as c:
        c.executescript(_SCHEMA)


def _now():
    return time.time()


def doc_set(key, kind, title, content, source=None, tags=None):
    """Insert or update a document by key. Returns the key. Idempotent."""
    ensure_schema()
    kind = kind if kind in VALID_KINDS else "note"
    tags = ",".join(tags) if isinstance(tags, (list, tuple, set)) else (tags or "")
    now = _now()
    with _connect() as c:
        row = c.execute("SELECT id FROM documents WHERE key=?", (key,)).fetchone()
        if row:
            c.execute(
                "UPDATE documents SET kind=?, title=?, content=?, source=?, "
                "tags=?, updated_at=? WHERE key=? AND invalidated_at IS NULL",
                (kind, title, content, source, tags, now, key),
            )
        else:
            c.execute(
                "INSERT INTO documents(key, kind, title, content, source, tags, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (key, kind, title, content, source, tags, now, now),
            )
    return key


def doc_get(key):
    """Return the active document row for key, or None. Never raises."""
    try:
        with _connect() as c:
            return c.execute(
                "SELECT * FROM documents WHERE key=? AND invalidated_at IS NULL",
                (key,),
            ).fetchone()
    except Exception:
        return None


def doc_list(kind=None, include_invalidated=False):
    try:
        with _connect() as c:
            q = "SELECT key, kind, title, source, updated_at FROM documents"
            args = []
            if not include_invalidated:
                q += " WHERE invalidated_at IS NULL"
            if kind:
                q += " AND kind=?" if "WHERE" in q else " WHERE kind=?"
                args.append(kind)
            q += " ORDER BY kind, key"
            return c.execute(q, args).fetchall()
    except Exception:
        return []


def doc_list_by_prefix(prefix, kind=None):
    """List active documents whose key starts with prefix (e.g. 'vixen/curriculum/').
    Returns rows or [] on failure."""
    try:
        with _connect() as c:
            q = "SELECT key, kind, title, source FROM documents " \
                "WHERE invalidated_at IS NULL AND key LIKE ?"
            args = [prefix + "%"]
            if kind:
                q += " AND kind=?"
                args.append(kind)
            q += " ORDER BY key"
            return c.execute(q, args).fetchall()
    except Exception:
        return []


def doc_search(query, limit=10):
    """Full-text search over key/title/content. Returns rows or [] on failure."""
    try:
        with _connect() as c:
            return c.execute(
                "SELECT d.key, d.kind, d.title, d.source, "
                "snippet(documents_fts, 2, '[', ']', '...', 8) AS snip "
                "FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid "
                "WHERE documents_fts MATCH ? AND d.invalidated_at IS NULL "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
    except Exception:
        return []


def doc_invalidate(key):
    with _connect() as c:
        c.execute(
            "UPDATE documents SET invalidated_at=? WHERE key=? AND invalidated_at IS NULL",
            (_now(), key),
        )


def doc_append(key, block, kind="note", title=None):
    """Append a block of text to a document (create it if absent). Preserves prior
    content; timestamp each block. Returns the key. Idempotent by key."""
    ensure_schema()
    now = _now()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    block = block.rstrip("\n")
    if not block:
        return key
    with _connect() as c:
        row = c.execute(
            "SELECT id, content FROM documents WHERE key=? AND invalidated_at IS NULL",
            (key,),
        ).fetchone()
        added = f"\n\n--- {stamp} ---\n{block}"
        if row:
            c.execute(
                "UPDATE documents SET content=content || ?, updated_at=? WHERE key=?",
                (added, now, key),
            )
        else:
            c.execute(
                "INSERT INTO documents(key, kind, title, content, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (key, kind, title or key, block, now, now),
            )
    return key


def doc_tail(key, n=10):
    """Return the last n lines of a document, or an empty string. Never raises."""
    try:
        r = doc_get(key)
        if r is None:
            return ""
        lines = r["content"].strip().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _row_dict(r):
    return {k: r[k] for k in r.keys()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list"); sp.add_argument("--kind"); sp.add_argument("--all", action="store_true")
    sp = sub.add_parser("get"); sp.add_argument("key")
    sp = sub.add_parser("search"); sp.add_argument("query"); sp.add_argument("--limit", type=int, default=10)
    sp = sub.add_parser("set"); sp.add_argument("key"); sp.add_argument("kind"); sp.add_argument("title")
    sp.add_argument("--content", help="file to read body from (else stdin)"); sp.add_argument("--source"); sp.add_argument("--tags")
    sp = sub.add_parser("rm"); sp.add_argument("key")
    sp = sub.add_parser("append"); sp.add_argument("key"); sp.add_argument("--kind", default="note"); sp.add_argument("--title")
    sp.add_argument("--text", help="text to append (else stdin)")
    sp = sub.add_parser("tail"); sp.add_argument("key"); sp.add_argument("-n", type=int, default=10)
    sp = sub.add_parser("count")

    a = p.parse_args()
    ensure_schema()
    if a.cmd == "list":
        rows = doc_list(a.kind, a.all)
        for r in rows:
            print(f"  [{r['kind']}] {r['key']}  — {r['title']}")
        print(f"({len(rows)} document(s))")
    elif a.cmd == "get":
        r = doc_get(a.key)
        if r is None:
            print(f"(no document '{a.key}')", file=sys.stderr); sys.exit(1)
        print(f"# {r['title']}  [{r['kind']}]")
        if r["source"]:
            print(f"# source: {r['source']}")
        print("")
        print(r["content"])
    elif a.cmd == "search":
        for r in doc_search(a.query, a.limit):
            print(f"[{r['kind']}] {r['key']}: {r['snip']}")
    elif a.cmd == "set":
        content = open(a.content, encoding="utf-8").read() if a.content else sys.stdin.read()
        doc_set(a.key, a.kind, a.title, content, source=a.source, tags=a.tags)
        print(f"stored {a.key}")
    elif a.cmd == "rm":
        doc_invalidate(a.key); print(f"removed {a.key}")
    elif a.cmd == "append":
        text = a.text if a.text else sys.stdin.read()
        doc_append(a.key, text, kind=a.kind, title=a.title); print(f"appended {a.key}")
    elif a.cmd == "tail":
        print(doc_tail(a.key, a.n))
    elif a.cmd == "count":
        print(len(doc_list()))


if __name__ == "__main__":
    main()
