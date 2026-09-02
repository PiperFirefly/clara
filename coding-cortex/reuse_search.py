#!/usr/bin/env python3
"""
Reuse search over the code graph — Coding Cortex.

Answers: "I need UUID normalization; there are already three implementations
of that idea in these repositories." Embeds code symbol names + signatures
with fastembed (already in the stack), clusters near-duplicates by cosine
similarity, so a new task can find existing implementations instead of
re-inventing one.

Reuses the existing code_nodes table in memory.db (tree-sitter graph) —
no second store. Deterministic embedding, pure read of the graph.

Usage:
  reuse_search.py uuid normalize      # find existing implementations of this idea
  reuse_search.py --dup               # list near-duplicate symbol clusters
  reuse_search.py --stats
"""
import argparse
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.expanduser("~/memory"))

import numpy as np

try:
    import memstore as M  # reuse its model config + cache (no second fetch)
    MODEL_CACHE = M.MODEL_CACHE
    MODEL_NAME = M.MODEL_NAME
except Exception:  # noqa: BLE001 - env setup, fall back to defaults
    MODEL_CACHE = os.path.expanduser("~/memory/models")
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

try:
    from fastembed import TextEmbedding
except Exception:  # noqa: BLE001 - env setup
    TextEmbedding = None

MEMORY_DB = os.path.expanduser("~/memory/memory.db")
INDEX_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reuse_index.npz"
)
# The code graph changes rarely; embedding all symbols every query is wasteful.
# Persist the vectors once, reload on subsequent calls.
CACHE_STALE_SECS = 86400  # rebuild the index at most daily


def load_index(rows):
    """Return (vecs, rows). Embed once and cache to disk; rebuild only on expiry.

    The code graph re-ingests frequently (cron ~10min) so the row count churns
    slightly; we do NOT invalidate on count drift (that would rebuild the whole
    index every run). New symbols join on the next daily rebuild. Stale-by-a-few-
    symbols is an acceptable cost for a reuse *search* (approximate by design)."""
    if os.path.exists(INDEX_CACHE) and \
            time.time() - os.path.getmtime(INDEX_CACHE) < CACHE_STALE_SECS:
        try:
            d = np.load(INDEX_CACHE, allow_pickle=True)
            return d["vecs"], d["rows"]
        except (OSError, ValueError):
            pass
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE)
    docs = [" ".join(tokens(n)) + " | " + (s or "") for (n, s, _p, _r) in rows]
    vecs = np.array(list(model.embed(docs)))
    np.savez(INDEX_CACHE, vecs=vecs, rows=np.array(rows, dtype=object))
    return vecs, np.array(rows, dtype=object)

# Tokenize a symbol name like "normalize_uuid" / "uuidNorm" / "get_user_id"
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokens(name):
    return [w.lower() for w in _WORD.findall(name)]


def load_symbols(db_path=MEMORY_DB, kinds=("func", "method")):
    db = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(kinds))
    rows = db.execute(
        "SELECT name, signature, path, repo FROM code_nodes "
        f"WHERE kind IN ({placeholders})",
        kinds,
    ).fetchall()
    return rows


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def find_similar(vecs, rows, qvec, top=10, thresh=0.55):
    q = qvec / (np.linalg.norm(qvec) + 1e-12)
    sims = [cosine(q, v) for v in vecs]
    order = sorted(range(len(sims)), key=lambda i: -sims[i])
    out = []
    for i in order[:top]:
        if sims[i] >= thresh:
            out.append((sims[i], rows[i]))
    return out


def clusters(vecs, rows, thresh=0.82, min_size=2):
    """Group near-duplicate symbols (same idea implemented in several places)."""
    used = set()
    groups = []
    for i in range(len(rows)):
        if i in used:
            continue
        grp = [i]
        for j in range(i + 1, len(rows)):
            if j in used:
                continue
            if cosine(vecs[i], vecs[j]) >= thresh:
                grp.append(j)
        if len(grp) >= min_size:
            used.update(grp)
            groups.append([rows[k] for k in grp])
    return groups


def main():
    p = argparse.ArgumentParser(description="reuse search over the code graph")
    p.add_argument("query", nargs="*", help="space-separated idea, e.g. uuid normalize")
    p.add_argument("--dup", action="store_true", help="list near-duplicate symbol clusters")
    p.add_argument("--stats", action="store_true", help="index size")
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args()

    if TextEmbedding is None:
        sys.exit("fastembed not importable — is ~/venvs/memory active?")

    rows = load_symbols()
    vecs, rows = load_index(rows)

    if a.stats:
        print(f"indexed {len(rows)} symbols ({len(rows)} vecs)")
        return
    if a.dup:
        gs = clusters(vecs, rows)
        if not gs:
            print("no near-duplicate clusters above threshold")
        for g in gs:
            print(f"\ncluster ({len(g)}):")
            for s in g:
                print(f"  {s[0]}  @ {s[2]}")
        return

    q = " ".join(a.query)
    if not q:
        p.print_help()
        return
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE)
    qvec = np.array(next(iter(model.embed([" ".join(tokens(q))]))))
    print(f"existing implementations of '{q}':\n")
    for sim, (name, sig, path, repo) in find_similar(vecs, rows, qvec, top=a.top):
        print(f"  [{sim:.2f}] {name} {sig or ''}")
        print(f"          {path}")


if __name__ == "__main__":
    main()
