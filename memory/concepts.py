#!/usr/bin/env python3
"""Concept layer for the coding stack.

A software ontology ABOVE the languages. Rather than "is there a function
called parse_foo()?", answer "what implementations of *parser* already
exist?" -- cross-language, cross-repo, ranked. The concept is the
first-class key; names are just one realization of it.

The concept index is a table in the same memory.db the codegraph owns,
so it composes with code_graph / structural_edit / code_meaning: every hit
is hot-linkable back to its node for callers/callees.

Populated three ways:
  1. deterministic auto-tag of existing code_nodes by name/type patterns
  2. manual tagging whenever a new implementation is written (so we stop
     re-inventing things we already own)
  3. vocabulary grows as new concepts are needed

No LLM in the loop -- this is a deterministic registry, not an inference
layer over our own code.
"""
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegraph import connect  # noqa: E402


# ---------------------------------------------------------------------------
# concept vocabulary
# ---------------------------------------------------------------------------
# Each concept has:
#   strong  -> high-confidence tokens. Fire anywhere (incl. methods). quality 9.
#   weak    -> generic tokens. Only fire on CLASSES and standalone FUNCS (the
#              implementation units), never on individual methods. quality 5.
#   desc    -> one-line meaning.
VOCAB = {
    "queue": {
        "strong": [r"\bqueue\b", r"\bfifo\b", r"\blifo\b", "deque", "worker_queue"],
        "weak": [r"\bbuffer\b", r"\bstack\b"],
        "desc": "ordered collection of items to process (FIFO/LIFO/buffer)",
    },
    "serializer": {
        "strong": ["serializ", "deserializ", "to_bytes", "from_bytes", "marshal", "unmarshal"],
        "weak": ["encode", "decode", "pickle", "json_", r"\bjson\b", r"\bpack\b", r"\bunpack\b", r"\bdump\b", r"\bload\b"],
        "desc": "convert objects to/from bytes/text/formats",
    },
    "cache": {
        "strong": [r"\bcache\b", "memoiz", r"\bmemo\b", r"\blru\b", "ttl_cache", "_cached"],
        "weak": [],
        "desc": "store computed results to avoid recomputation",
    },
    "parser": {
        "strong": ["parser", "parse_", "lexer", "tokeniz", "tokenize", r"\bast\b"],
        "weak": [r"\btoken\b", "syntax", "read_", "extract"],
        "desc": "turn raw input into structured data / AST",
    },
    "state machine": {
        "strong": ["state.?machine", r"\bfsm\b", "finite_state", "state_machine"],
        "weak": ["transition", r"\bstate\b", "_machine"],
        "desc": "model with discrete states and transitions",
    },
    "retry policy": {
        "strong": [r"\bretry\b", "retries", "backoff", "exponential", r"\bjitter\b"],
        "weak": [r"\battempt\b", "_do_with_retry"],
        "desc": "policy for retrying failed operations with backoff",
    },
    "transaction": {
        "strong": [r"\btransaction\b", r"\bcommit\b", r"\brollback\b", r"\batomic\b", r"begin\b"],
        "weak": [r"\bcommit_"],
        "desc": "atomic unit of work (all-or-nothing)",
    },
    "adapter": {
        "strong": [r"\badapter\b", r"\badapt\b"],
        "weak": ["wrapper", r"\bbridge\b", "connector", "driver", "gateway"],
        "desc": "translate one interface to another",
    },
    "repository": {
        "strong": [r"\brepository\b", r"\bdao\b", r"\brepo\b"],
        "weak": [r"\bstore\b", "persistence", r"\bstorage\b"],
        "desc": "data-access layer abstracting persistence",
    },
    "worker pool": {
        "strong": [r"\bworker\b", "thread.?pool", r"\bpool\b", "multiprocess", "worker_pool"],
        "weak": ["concurren", "parallel"],
        "desc": "bounded set of workers consuming tasks",
    },
    "rate limiter": {
        "strong": ["rate.?limit", "rate_limit", r"\bthrottle\b", r"\blimiter\b", "token.?bucket", "leaky"],
        "weak": ["cooldown", r"\bsleep\b"],
        "desc": "control the rate of operations",
    },
    "event bus": {
        "strong": ["event.?bus", "eventbus", "pub.?sub", "publish", "subscribe", "dispatcher"],
        "weak": [r"\bbus\b", r"\bemit\b", r"\bsignal\b", r"\bevent\b"],
        "desc": "publish/subscribe message passing",
    },
    "scheduler": {
        "strong": ["scheduler", r"\bschedule\b", r"\bcron\b", "periodic"],
        "weak": [r"\btimer\b", r"\binterval\b", r"\bevery\b"],
        "desc": "run tasks at defined times/intervals",
    },
    "notifier": {
        "strong": [r"\bnotif\b", r"\bnotify\b", r"\balert\b", "send_alert", r"\bwebhook\b"],
        "weak": [],
        "desc": "push alerts/notifications out",
    },
    "logger": {
        "strong": [r"\blogger\b", r"\blogging\b", r"\baudit\b"],
        "weak": [r"\blog\b", r"\blog_", "log_rec"],
        "desc": "structured logging / audit trail",
    },
    "config": {
        "strong": [r"\bconfig\b", r"\bsettings\b", "dotenv", "env_vars"],
        "weak": [r"\boptions\b", r"\benv\b", r"\bargs\b"],
        "desc": "configuration loading / validation",
    },
}


def _concepts():
    return list(VOCAB.keys())


_STRONG = "strong"
_WEAK = "weak"


def _w(frag):
    """Turn \b...\b into an underscore-aware word boundary.

    \b treats `_` as a word char, so it fails on snake_case (`_load_queue`).
    Use explicit lookarounds that also treat `_` as a boundary.
    """
    if not frag:
        return frag
    out = frag.replace(r"\b", "")
    if frag.startswith(r"\b"):
        out = "(?<![a-z0-9])" + out
    if frag.endswith(r"\b"):
        out = out + "(?![a-z0-9])"
    return out


def _alias_res(concept):
    """Return (strong_re, weak_re) for a concept. weak may be None if empty."""
    v = VOCAB[concept]
    strong = re.compile("|".join(f"(?:{_w(f)})" for f in v.get("strong", [])), re.I)
    wk = v.get("weak", [])
    weak = re.compile("|".join(f"(?:{_w(f)})" for f in wk), re.I) if wk else None
    return strong, weak


def _ensure_concept_schema(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS concept_index("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "concept TEXT NOT NULL, node_name TEXT NOT NULL, "
        "kind TEXT, path TEXT, repo TEXT, lang TEXT, "
        "source TEXT NOT NULL DEFAULT 'auto', quality INTEGER DEFAULT 5, "
        "added_at REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_concept ON concept_index(concept)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ci_node "
                 "ON concept_index(concept, node_name)")
    # ---- software-memory / capability columns (item: capability library) ----
    # is_best: this impl is the preferred one; reason: WHY (engineering note);
    # used_by: callers/projects depending on it; test_count: test coverage;
    # known_limitation: what it does NOT handle.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(concept_index)")}
    for col, ddl in [
        ("is_best", "INTEGER DEFAULT 0"),
        ("reason", "TEXT DEFAULT ''"),
        ("used_by", "TEXT DEFAULT ''"),
        ("test_count", "INTEGER DEFAULT 0"),
        ("known_limitation", "TEXT DEFAULT ''"),
        ("last_verified", "REAL"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE concept_index ADD COLUMN {col} {ddl}")
    conn.commit()


# ---------------------------------------------------------------------------
# auto-tag
# ---------------------------------------------------------------------------
def auto_tag(conn, dry_run=False):
    """Tag existing code_nodes.

    strong tokens match any kind (incl. methods); weak tokens match only
    classes and standalone functions (the implementation units). quality:
    strong=9 on class/func, 8 on method; weak=5 on class/func.
    """
    ensure = _ensure_concept_schema
    if not dry_run:
        ensure(conn)
    added, skipped = 0, 0
    nodes = conn.execute(
        "SELECT name, kind, path, repo, lang FROM code_nodes "
        "WHERE kind IN ('func','method','class')").fetchall()
    for concept in _concepts():
        srx, wrx = _alias_res(concept)
        for n in nodes:
            hay = n["name"]
            bare = hay.split(".")[-1]
            impl_unit = n["kind"] in ("class", "func")
            qual = None
            if srx.search(bare) or srx.search(hay):
                qual = 9 if impl_unit else 8
            # weak tokens only match the bare symbol, never the module path
            elif wrx is not None and impl_unit and wrx.search(bare):
                qual = 5
            if qual is None:
                continue
            if dry_run:
                added += 1
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO concept_index"
                    "(concept,node_name,kind,path,repo,lang,source,quality,added_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (concept, n["name"], n["kind"], n["path"], n["repo"],
                     n["lang"], "auto", qual, __import__("time").time()))
                if conn.execute("SELECT changes()").fetchone()[0]:
                    added += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError:
                skipped += 1
    if not dry_run:
        conn.commit()
    return added, skipped


# ---------------------------------------------------------------------------
# manual tag
# ---------------------------------------------------------------------------
def manual_tag(conn, concept, node_name, quality=5, source="manual"):
    """Attach a concept to a node by hand (e.g. right after writing it)."""
    if concept not in VOCAB:
        # allow unknown concepts; vocabulary grows on demand
        VOCAB[concept] = {"alias": [concept], "desc": "user-defined"}
    _ensure_concept_schema(conn)
    r = conn.execute(
        "SELECT name,kind,path,repo,lang FROM code_nodes WHERE name=? OR name LIKE ? LIMIT 1",
        (node_name, f"%{node_name}")).fetchone()
    if not r:
        return {"ok": False, "error": f"no node found for {node_name!r}"}
    conn.execute(
        "INSERT OR REPLACE INTO concept_index"
        "(concept,node_name,kind,path,repo,lang,source,quality,added_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (concept, r["name"], r["kind"], r["path"], r["repo"], r["lang"],
         source, quality, __import__("time").time()))
    conn.commit()
    return {"ok": True, "node": r["name"], "path": r["path"]}


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
def concept_lookup(concept, conn=None, k=25):
    """Return implementations of a concept, ranked by quality. Hot-linkable
    back into code_graph via node_name."""
    own = conn is None
    if own:
        conn = connect()
        _ensure_concept_schema(conn)
    concept_l = concept.lower()
    results = conn.execute(
        "SELECT * FROM concept_index WHERE concept=? ORDER BY quality DESC, added_at DESC LIMIT ?",
        (concept_l, k)).fetchall()
    if not results and concept_l not in VOCAB:
        # unknown concept: broad name match as a best-effort fallback
        results = conn.execute(
            "SELECT * FROM concept_index WHERE concept LIKE ? ORDER BY quality DESC LIMIT ?",
            (f"%{concept_l}%", k)).fetchall()
    if own:
        conn.close()
    return [dict(r) for r in results]


def vocab_list():
    out = []
    for c, v in VOCAB.items():
        out.append({"concept": c, "desc": v["desc"],
                    "strong": v.get("strong", []), "weak": v.get("weak", [])})
    return out


# ---------------------------------------------------------------------------
# capability / software-memory lookup (item: capability library)
# ---------------------------------------------------------------------------
def set_capability_meta(concept, node_name, *, is_best=None, reason=None,
                        used_by=None, test_count=None, known_limitation=None,
                        last_verified=None):
    """Attach distilled engineering knowledge to a concept implementation.
    This is the 'software memory' layer: best-impl, reason, used-by, tests,
    limitation -- so Agent recalls 'we solved this before and here's why this
    one is the good one' systematically, not accidentally."""
    import time as _t
    conn = connect()
    _ensure_concept_schema(conn)
    r = conn.execute(
        "SELECT id FROM concept_index WHERE concept=? AND node_name=?",
        (concept, node_name)).fetchone()
    if not r:
        # attach to the most recent matching node for that concept if none exact
        r = conn.execute(
            "SELECT id FROM concept_index WHERE concept=? AND node_name LIKE ? "
            "ORDER BY quality DESC LIMIT 1", (concept, f"%{node_name}%")).fetchone()
    if not r:
        conn.close()
        return {"ok": False, "error": f"no concept_index row for {concept}::{node_name}"}
    sets, vals = [], []
    for col, val in [("is_best", is_best), ("reason", reason),
                     ("used_by", used_by), ("test_count", test_count),
                     ("known_limitation", known_limitation)]:
        if val is not None:
            sets.append(f"{col}=?")
            vals.append(val)
    if last_verified is not None or not sets:
        sets.append("last_verified=?")
        vals.append(last_verified if last_verified is not None else _t.time())
    vals.append(r["id"])
    conn.execute(f"UPDATE concept_index SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
    conn.close()
    return {"ok": True, "concept": concept, "node": node_name, "id": r["id"]}


def capability_lookup(concept, conn=None, k=25):
    """Software-memory query: return implementations WITH distilled metadata
    (best/reason/used_by/tests/limitation). Best impl sorts first."""
    own = conn is None
    if own:
        conn = connect()
        _ensure_concept_schema(conn)
    concept_l = concept.lower()
    results = conn.execute(
        "SELECT * FROM concept_index WHERE concept=? "
        "ORDER BY is_best DESC, quality DESC, added_at DESC LIMIT ?",
        (concept_l, k)).fetchall()
    if not results and concept_l not in VOCAB:
        results = conn.execute(
            "SELECT * FROM concept_index WHERE concept LIKE ? "
            "ORDER BY is_best DESC, quality DESC LIMIT ?",
            (f"%{concept_l}%", k)).fetchall()
    if own:
        conn.close()
    return [dict(r) for r in results]


def stats(conn=None):
    own = conn is None
    if own:
        conn = connect()
        _ensure_concept_schema(conn)
    total = conn.execute("SELECT count(*) c FROM concept_index").fetchone()["c"]
    by_concept = {r["concept"]: r["c"] for r in conn.execute(
        "SELECT concept, count(*) c FROM concept_index GROUP BY concept "
        "ORDER BY c DESC")}
    by_source = {r["source"]: r["c"] for r in conn.execute(
        "SELECT source, count(*) c FROM concept_index GROUP BY source")}
    best = conn.execute(
        "SELECT count(*) c FROM concept_index WHERE is_best=1").fetchone()["c"]
    if own:
        conn.close()
    return {"indexed": total, "concepts_covered": len(by_concept),
            "by_concept": by_concept, "by_source": by_source,
            "best_impls_marked": best}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="store_true", help="auto-tag existing nodes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--add", nargs=2, metavar=("CONCEPT", "NODE_NAME"),
                    help="manually tag a node with a concept")
    ap.add_argument("--query", help="concept to look up")
    ap.add_argument("--capability", help="software-memory lookup for a concept")
    ap.add_argument("--set-meta", nargs=3, metavar=("CONCEPT", "NODE", "JSON"),
                    help="attach distilled metadata (is_best/reason/used_by/"
                         "test_count/known_limitation) to a concept impl")
    ap.add_argument("--vocab", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()

    if a.capability:
        print(json.dumps(capability_lookup(a.capability, k=a.k), indent=2))

    if a.set_meta:
        concept, node, payload_s = a.set_meta
        payload = json.loads(payload_s)
        r = set_capability_meta(concept, node, **payload)
        print(json.dumps(r))

    if a.vocab:
        print(json.dumps(vocab_list(), indent=2))
    if a.tag:
        c = connect()
        _ensure_concept_schema(c)
        if a.dry_run:
            n, s = auto_tag(c, dry_run=True)
            print(f"dry-run: {n} matches")
        else:
            n, s = auto_tag(c)
            print(f"auto-tagged: {n} added, {s} skipped")
    if a.add:
        c = connect()
        r = manual_tag(c, a.add[0], a.add[1])
        print(json.dumps(r))
    if a.query:
        print(json.dumps(concept_lookup(a.query, k=a.k), indent=2))
    if a.stats:
        print(json.dumps(stats(), indent=2))
