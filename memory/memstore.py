#!/usr/bin/env python3
"""
The agent's memory store — Stage 1 (CoALA split: facts + episodic/semantic + vectors).

Storage: SQLite (single file). Embeddings: fastembed (ONNX, no torch) ->
BAAI/bge-small-en-v1.5 (384-dim). Retrieval: brute-force cosine via numpy —
plenty for tens of thousands of memories; swap in an ANN index (sqlite-vec /
lancedb) only when scale demands it.

Usage:
  python3 memstore.py init
  python3 memstore.py remember "text" [--kind episodic] [--importance 0.5]
  python3 memstore.py recall "query" [--k 5]
  python3 memstore.py facts set <key> <value>
  python3 memstore.py facts get <key>
  python3 memstore.py facts list
  python3 memstore.py stats
  python3 memstore.py seed
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import urllib.request
from collections import Counter, defaultdict, deque
from functools import lru_cache

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "memory.db")
MODEL_CACHE = os.path.join(BASE, "models")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

AUTH = os.path.expanduser("~/.pi/agent/auth.json")
# Provider-agnostic LLM access. Cadence's workers call an OpenAI-compatible chat
# endpoint. The operator points these at their own provider via env vars
# (deepseek, kimi, openai, a local server, ...). The defaults are deepseek only
# because that is what we happen to run — nothing here requires it.
API_URL = os.environ.get("CADENCE_API_URL", "https://api.deepseek.com/chat/completions")
AUTH_PROVIDER = os.environ.get("CADENCE_PROVIDER", "deepseek")  # auth.json key name

# Model routing (multi-model hive): memory workers use the cheap non-reasoning
# model (clean structured content for extraction/scoring); the main agent uses
# the strong reasoning model. NOTE: some reasoning models leave `content` empty
# and put the answer in `reasoning_content` — unsuitable for workers.
MODEL_WORKER = os.environ.get("CADENCE_WORKER_MODEL", "deepseek-chat")   # cheap, non-reasoning
MODEL_STRONG = os.environ.get("CADENCE_MODEL", "deepseek-v4-pro")        # strong reasoning

# Forgetting-curve retention (Ebbinghaus / MemoryBank)
BASE_STABILITY = 30 * 86400.0     # baseline stability ~30 days (seconds)
IMPORTANCE_WEIGHT = 4.0           # importance 1.0 -> 5x baseline stability
REHEARSAL_WEIGHT = 1.0            # log1p(access_count) multiplies stability
PRUNE_THRESHOLD = 0.1             # effective importance below this -> forget
MIN_AGE_SECONDS = 7 * 86400.0     # never forget anything younger than this
CORE_KINDS = {"identity", "backstory", "appearance", "goal"}

# Entity resolution (three-tier dedup: exact -> entropy-gated fuzzy -> LLM).
NAME_ENTROPY_THRESHOLD = 1.5
MIN_NAME_LENGTH = 6
MIN_TOKEN_COUNT = 2
FUZZY_JACCARD_THRESHOLD = 0.9

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        os.makedirs(MODEL_CACHE, exist_ok=True)
        _embedder = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE)
    return _embedder


def embed(texts):
    """Embed a list of strings -> unit-normed float32 matrix (n, 384)."""
    e = get_embedder()
    vecs = [np.asarray(v, dtype=np.float32) for v in e.embed(list(texts))]
    m = np.stack(vecs)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


# Surprise-gated importance (P0-1, calibrated): modulate write-side importance
# by how novel a memory is relative to recent observations. A rolling rank gate
# replaces the earlier fixed-threshold sigmoid (which was miscalibrated against
# bge-small's clustered cosine-sim distribution: p01=0.62, p50=0.80, p99=0.93).
# Surprise is the new memory's similarity *rank* within a rolling buffer of
# recent mean-top-3 cosine sims; a fixed sigmoid bridges the cold start.
SURPRISE_WINDOW = 512            # rolling buffer size (rows kept)
SURPRISE_MIN_WINDOW = 50         # buffer size at which rank mode takes over
SURPRISE_MIN_CORPUS = 10         # live memories below this -> neutral (no gate)
SURPRISE_FALLBACK_THRESHOLD = 0.80  # cold-start sigmoid midpoint (mean-top-3 sim)
SURPRISE_FALLBACK_SLOPE = 12.0   # cold-start sigmoid steepness
SURPRISE_GAIN = 0.6              # (surprise - 0.5) * GAIN -> mod in [-0.3, +0.3]


def _mean_top3(text_vec, neighbor_vecs):
    """Mean cosine sim of the new memory to its top-3 live neighbors (pure)."""
    tv = np.asarray(text_vec, dtype=np.float32)
    if neighbor_vecs is None:
        mat = np.empty((0, tv.shape[0]), dtype=np.float32)
    else:
        mat = np.asarray(neighbor_vecs, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
    if mat.size == 0:
        return 0.0
    sims = mat @ tv
    top3 = np.sort(sims)[-3:]
    return float(np.mean(top3))


def surprise_fallback(mean_top3):
    """Cold-start sigmoid: sim below threshold -> high surprise (pure)."""
    z = -SURPRISE_FALLBACK_SLOPE * (mean_top3 - SURPRISE_FALLBACK_THRESHOLD)
    return 1.0 / (1.0 + math.exp(-z))


def surprise_from_rank(mean_top3, window_values):
    """Rank surprise: fraction of recent observed sims >= this one (pure).

    A low similarity sits below almost every recent observation -> surprise ~1.0
    (novel); a high similarity sits above them -> surprise ~0.0 (near-duplicate).
    """
    if not window_values:
        return 0.5
    w = np.asarray(window_values, dtype=np.float64)
    return float(np.count_nonzero(w >= mean_top3) / w.size)


def surprise_gate(text_vec, neighbor_vecs):
    """Pure fallback novelty gate (cold-start sigmoid) — kept for back-compat.

    Returns (surprise, importance_mod). Production `remember()` uses the
    rolling-rank gate once the calibration buffer is warm; this sigmoid covers
    the cold start. Pure: no DB, no network, no side effects.
    """
    mean_top3 = _mean_top3(text_vec, neighbor_vecs)
    surprise = surprise_fallback(mean_top3)
    importance_mod = (surprise - 0.5) * SURPRISE_GAIN
    return float(surprise), float(importance_mod)


def _migrate_causal_edges_columns(conn):
    """Causal-edge schema split (cognitive-roadmap Tier 0b): the single conflated
    `confidence` is split into four distinct semantics. Additive only — the
    legacy `confidence` column stays for back-compat (other code may still
    read it). Idempotent check-then-ALTER; the one-time backfill runs only on
    the pass that first added the columns, so a later LLM-emitted 'unknown'
    direction is never clobbered by a re-run."""
    try:
        cecols = {r["name"] for r in conn.execute("PRAGMA table_info(causal_edges)")}
    except sqlite3.OperationalError:
        cecols = set()
    if not cecols:
        return
    added = False
    if "relation_confidence" not in cecols:
        conn.execute("ALTER TABLE causal_edges ADD COLUMN relation_confidence REAL")
        added = True
    if "effect_direction" not in cecols:
        conn.execute(
            "ALTER TABLE causal_edges ADD COLUMN effect_direction TEXT DEFAULT 'unknown'"
        )
        added = True
    if "conditional_probability" not in cecols:
        conn.execute("ALTER TABLE causal_edges ADD COLUMN conditional_probability REAL")
        added = True
    if "evidence_quality" not in cecols:
        conn.execute("ALTER TABLE causal_edges ADD COLUMN evidence_quality REAL")
        added = True
    if not added:
        return
    # One-time backfill from the legacy conflated `confidence`.
    conn.execute(
        "UPDATE causal_edges SET relation_confidence = confidence "
        "WHERE relation_confidence IS NULL"
    )
    conn.execute(
        "UPDATE causal_edges SET conditional_probability = confidence "
        "WHERE conditional_probability IS NULL"
    )
    conn.execute(
        "UPDATE causal_edges SET evidence_quality = 0.5 "
        "WHERE evidence_quality IS NULL"
    )
    conn.execute(
        "UPDATE causal_edges SET effect_direction = "
        "CASE WHEN lower(rel) = 'prevents' THEN 'negative' ELSE 'positive' END "
        "WHERE effect_direction IS NULL OR effect_direction = 'unknown'"
    )
    conn.commit()


def _migrate_tool_uses_columns(conn):
    """Meta-cognitive logging (P1-6): add calibration columns to tool_uses for
    pre-existing DBs that already hold the old table shape. Additive only."""
    try:
        tcols = {r["name"] for r in conn.execute("PRAGMA table_info(tool_uses)")}
    except sqlite3.OperationalError:
        tcols = set()
    if not tcols:
        return
    if "pre_confidence" not in tcols:
        conn.execute("ALTER TABLE tool_uses ADD COLUMN pre_confidence REAL")
    if "uncertainty" not in tcols:
        conn.execute("ALTER TABLE tool_uses ADD COLUMN uncertainty TEXT")
    if "outcome_later" not in tcols:
        conn.execute("ALTER TABLE tool_uses ADD COLUMN outcome_later TEXT")
    if "resolved_at" not in tcols:
        conn.execute("ALTER TABLE tool_uses ADD COLUMN resolved_at REAL")
    if "s2" not in tcols:
        conn.execute("ALTER TABLE tool_uses ADD COLUMN s2 INTEGER DEFAULT 0")
    conn.commit()


def _migrate_fact_history(conn):
    """Bi-temporal facts (P0-2): world/knowledge time columns + queryable history.
    Runs unconditionally (before the memories guard) so a DB that has `facts`
    but not `memories` still migrates."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fact_history("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "key TEXT NOT NULL, value TEXT, "
        "observed_at REAL NOT NULL, "
        "valid_from REAL, valid_to REAL, "
        "op TEXT NOT NULL DEFAULT 'set', "
        "event_seq INTEGER)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fh_key_obs ON fact_history(key, observed_at)"
    )
    try:
        fcols = {r["name"] for r in conn.execute("PRAGMA table_info(facts)")}
    except sqlite3.OperationalError:
        fcols = set()
    if not fcols:
        return
    if "observed_at" not in fcols:
        conn.execute("ALTER TABLE facts ADD COLUMN observed_at REAL")
    if "valid_from" not in fcols:
        conn.execute("ALTER TABLE facts ADD COLUMN valid_from REAL")
    if "valid_to" not in fcols:
        conn.execute("ALTER TABLE facts ADD COLUMN valid_to REAL")
    if "invalidated_at" not in fcols:
        conn.execute("ALTER TABLE facts ADD COLUMN invalidated_at REAL")
    # Backfill knowledge/world time from the single legacy timestamp.
    conn.execute(
        "UPDATE facts SET observed_at=updated_at, valid_from=updated_at "
        "WHERE observed_at IS NULL"
    )
    # One 'set' row per fact lacking any history row (idempotent).
    conn.execute(
        "INSERT INTO fact_history(key, value, observed_at, valid_from, valid_to, op, event_seq) "
        "SELECT f.key, f.value, COALESCE(f.observed_at, f.updated_at), "
        "COALESCE(f.valid_from, f.updated_at), f.valid_to, 'set', NULL "
        "FROM facts f "
        "WHERE NOT EXISTS (SELECT 1 FROM fact_history fh WHERE fh.key = f.key)"
    )
    conn.commit()


def _migrate_memories_and_entities_columns(conn):
    """Soft-delete/forgetting-curve/temporal/affect columns on `memories`, plus
    entity-resolution columns on `entities`. Returns False if the `memories`
    table doesn't exist yet (caller should stop migrating in that case, since
    everything past this point in the original monolithic function depended
    on `memories` already existing)."""
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}
    except sqlite3.OperationalError:
        return False
    if not cols:
        return False  # memories table doesn't exist yet
    changed = False
    if "merged" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN merged INTEGER DEFAULT 0")
        changed = True
    if "merged_into" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN merged_into INTEGER")
        changed = True
    if "access_count" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")
        changed = True
    if "last_access" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN last_access REAL")
        changed = True
    if "forgotten" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN forgotten INTEGER DEFAULT 0")
        changed = True
    if "valid_to" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN valid_to REAL")
        changed = True
    if "superseded_by" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN superseded_by INTEGER")
        changed = True
    if "origin" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN origin TEXT DEFAULT 'observed'")
        changed = True
    if "derived_from" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN derived_from TEXT")
        changed = True
    if "valence" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN valence REAL")
        changed = True
    if "arousal" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN arousal REAL")
        changed = True
    if "affect_label" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN affect_label TEXT")
        changed = True
    # entity columns (P0 entity resolution + P2 summaries)
    try:
        ecols = {r["name"] for r in conn.execute("PRAGMA table_info(entities)")}
    except sqlite3.OperationalError:
        ecols = set()
    if ecols:
        if "norm" not in ecols:
            conn.execute("ALTER TABLE entities ADD COLUMN norm TEXT")
        if "summary" not in ecols:
            conn.execute("ALTER TABLE entities ADD COLUMN summary TEXT")
        if "canonical_id" not in ecols:
            conn.execute("ALTER TABLE entities ADD COLUMN canonical_id INTEGER")
        if "created_at" not in ecols:
            conn.execute("ALTER TABLE entities ADD COLUMN created_at REAL")
        if conn.execute("SELECT COUNT(*) c FROM entities WHERE norm IS NULL").fetchone()["c"]:
            for er in conn.execute("SELECT id, name FROM entities WHERE norm IS NULL"):
                conn.execute("UPDATE entities SET norm=? WHERE id=?",
                             (_normalize(er["name"]), er["id"]))
        changed = True
    if changed:
        conn.commit()
    return True


def _migrate_operator_config(conn):
    """Operator config (Cadence step 2): create the operators / operator_channels
    tables on every connect. Guarded so a failure here never blocks the
    memory store from opening."""
    try:
        from operator_config import ensure_schema
        ensure_schema(conn)
        conn.commit()
    except Exception:
        pass


def _ensure_columns(conn):
    """Add soft-delete columns + causal/tool/surprise-calib tables if missing
    (migrations for pre-existing DBs). Each migration step is its own function
    below (split 2026-08-31 for readability -- same statements, same order,
    zero behavior change); this is just the sequencing."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS causal_edges("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "cause_id INTEGER, effect_id INTEGER, rel TEXT, "
        "memory_id INTEGER, confidence REAL DEFAULT 0.5, created_at REAL)"
    )
    _migrate_causal_edges_columns(conn)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tool_uses("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "task TEXT NOT NULL, task_embedding BLOB, tool TEXT NOT NULL, "
        "outcome TEXT, success INTEGER, cost_sec REAL, created_at REAL, "
        "pre_confidence REAL, uncertainty TEXT, outcome_later TEXT, "
        "resolved_at REAL, s2 INTEGER DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS surprise_calib("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "mean_top3 REAL, created_at REAL, model TEXT)"
    )
    _migrate_tool_uses_columns(conn)
    _migrate_fact_history(conn)
    if not _migrate_memories_and_entities_columns(conn):
        return
    _migrate_operator_config(conn)


def _ensure_events(conn):
    """Create the append-only, hash-chained event ledger (Vesta, Phase A).

    The `events` table is the canonical log of state-changing writes. It is
    append-only: rows are never UPDATEd or DELETEd — a later fact supersedes an
    earlier one by emitting a new event, never by rewriting the old row. `hash`
    is a sha256 over a deterministic JSON of every field except id/hash (see
    `_event_hash`), so the chain can be re-verified from genesis (Phase D).
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "seq INTEGER NOT NULL, "
        "prev_hash TEXT, "
        "hash TEXT NOT NULL, "
        "ts REAL NOT NULL, "
        "type TEXT NOT NULL, "
        "actor TEXT, "
        "payload TEXT, "
        "source_memory_id INTEGER, "
        "validated INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_events_seq ON events(seq)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")


def _event_hash(seq, prev_hash, ts, etype, actor, payload_json, source_memory_id, validated):
    """Canonical sha256 of one event's content — everything except id and hash.

    Deterministic: identical field values always produce the identical hash, so
    the verifier (Phase D) can recompute it from a stored row, and the chain
    links via `prev_hash` -> the previous event's `hash`.
    """
    canonical = json.dumps({
        "seq": seq,
        "prev_hash": prev_hash,
        "ts": ts,
        "type": etype,
        "actor": actor,
        "payload": payload_json,
        "source_memory_id": source_memory_id,
        "validated": validated,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Seal-delay buffer (continuity, 2026-08-29). OFF by default: the prod
# path is byte-identical to before. AGENT_SEAL_DELAY=1 routes ledger events
# through the un-sealed buffer first; seal_due() moves them into `events`.
SEAL_DELAY = os.environ.get("AGENT_SEAL_DELAY") == "1"
SEAL_WINDOW = float(os.environ.get("AGENT_SEAL_WINDOW", "60"))


def _insert_event(conn, etype, payload_json, actor, source_memory_id, validated=0, ts=None):
    """Write one event directly to the `events` chain; return (seq, hash).

    `payload_json` is the already-canonicalized payload string (matches
    `_event_hash`). Always INSERTs — never mutates an existing row.
    """
    ts = ts if ts is not None else time.time()
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) AS seq, "
        "(SELECT hash FROM events WHERE seq = (SELECT MAX(seq) FROM events)) AS prev "
        "FROM events"
    ).fetchone()
    seq = int(row["seq"]) + 1
    prev_hash = row["prev"]
    h = _event_hash(seq, prev_hash, ts, etype, actor, payload_json,
                    source_memory_id, validated)
    conn.execute(
        "INSERT INTO events(seq, prev_hash, hash, ts, type, actor, payload, "
        "source_memory_id, validated) VALUES(?,?,?,?,?,?,?,?,?)",
        (seq, prev_hash, h, ts, etype, actor, payload_json,
         source_memory_id, validated),
    )
    return seq, h


def _emit_event(conn, etype, payload=None, actor=None, source_memory_id=None, validated=0):
    """Append one event to the ledger; return (seq, hash).

    Flag OFF (default): INSERTs directly into `events`. Flag ON: appends to the
    seal-delay buffer on the *same connection* (atomic with the state write);
    `seal_due()` seals it into `events` later. Buffered path returns
    (None, None) — no chain seq exists until seal.
    """
    if SEAL_DELAY:
        from seal_delay import append_buffered
        append_buffered(conn, actor, etype, payload, "observed", None,
                        source_memory_id, validated)
        return None, None
    payload_json = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if payload is not None else None
    )
    return _insert_event(conn, etype, payload_json, actor, source_memory_id, validated)


def seal_due(now=None):
    """(AGENT_SEAL_DELAY=1) Seal buffered events older than the window into `events`.

    Fail-closed: overdue events seal regardless. Returns the list of seqs sealed.
    """
    if not SEAL_DELAY:
        return []
    now = now or time.time()
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM buffer WHERE tentative=1 AND ts < ? ORDER BY buf_seq",
            (now - SEAL_WINDOW,)).fetchall()
        sealed = []
        for r in rows:
            seq, _ = _insert_event(c, r["type"], r["payload"], r["actor"],
                                   r["source_memory_id"], r["validated"], ts=r["ts"])
            c.execute("UPDATE buffer SET tentative=0 WHERE buf_seq=?", (r["buf_seq"],))
            sealed.append(seq)
        return sealed



def connect():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    # Concurrency hardening: WAL lets concurrent readers/writers coexist
    # (the Telegram bridge, freeroam cron, and foreground sessions all write
    # here), and a 30s busy timeout absorbs brief write-lock contention
    # instead of failing fast with "database is locked".
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _ensure_columns(conn)
    _ensure_events(conn)
    if SEAL_DELAY:
        from seal_delay import ensure_schema
        ensure_schema(conn)
        conn.commit()
    return conn


def _harden_permissions():
    """Sensitive DBs (memory, images, state) must not be world-readable — they
    hold operator message text and private memory. Enforce mode 600 + wal/shm."""
    base = os.path.dirname(DB)
    targets = [DB, os.path.join(base, "images.db"),
               os.path.join(base, "state.db"), os.path.join(base, "logs.db")]
    for db in targets:
        for suffix in ("", "-wal", "-shm"):
            p = db + suffix
            if os.path.exists(p):
                try:
                    os.chmod(p, 0o600)
                except OSError:
                    pass


def init_db():
    with connect() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS facts("
            "key TEXT PRIMARY KEY, value TEXT, updated_at REAL, "
            "observed_at REAL, valid_from REAL, valid_to REAL, invalidated_at REAL)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS fact_history("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "key TEXT NOT NULL, value TEXT, "
            "observed_at REAL NOT NULL, "
            "valid_from REAL, valid_to REAL, "
            "op TEXT NOT NULL DEFAULT 'set', "
            "event_seq INTEGER)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_fh_key_obs ON fact_history(key, observed_at)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS memories("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "text TEXT NOT NULL,"
            "kind TEXT DEFAULT 'episodic',"
            "embedding BLOB,"
            "importance REAL DEFAULT 0.5,"
            "created_at REAL,"
            "metadata TEXT,"
            "merged INTEGER DEFAULT 0,"
            "merged_into INTEGER,"
            "access_count INTEGER DEFAULT 0,"
            "last_access REAL,"
            "forgotten INTEGER DEFAULT 0,"
            "valid_to REAL,"
            "superseded_by INTEGER)"
        )
        # Active-memory filter (merged=0 AND forgotten=0 AND valid_to IS NULL) is
        # the hot path behind recall/associate/fused — back it with an index.
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_active "
            "ON memories(merged, forgotten, valid_to)"
        )
        c.execute("CREATE TABLE IF NOT EXISTS entities("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, "
                  "norm TEXT, summary TEXT, canonical_id INTEGER, created_at REAL)")
        c.execute(
            "CREATE TABLE IF NOT EXISTS edges("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, subj INTEGER, rel TEXT, "
            "obj INTEGER, weight REAL DEFAULT 1.0, memory_id INTEGER)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS memory_entities(memory_id INTEGER, entity_id INTEGER)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS causal_edges("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "cause_id INTEGER, effect_id INTEGER, rel TEXT, "
            "memory_id INTEGER, confidence REAL DEFAULT 0.5, created_at REAL)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS tool_uses("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task TEXT NOT NULL, task_embedding BLOB, tool TEXT NOT NULL, "
            "outcome TEXT, success INTEGER, cost_sec REAL, created_at REAL, "
            "pre_confidence REAL, uncertainty TEXT, outcome_later TEXT, "
            "resolved_at REAL, s2 INTEGER DEFAULT 0)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS surprise_calib("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "mean_top3 REAL, created_at REAL, model TEXT)"
        )
    _harden_permissions()
    print(f"db ready at {DB}")


def remember(text, kind="episodic", importance=0.5, metadata=None, graph=True,
             origin="observed", derived_from=None):
    vec = embed([text])[0]
    md = dict(metadata) if metadata else {}
    df = json.dumps(derived_from) if derived_from else None
    with connect() as c:
        # Surprise-gated importance (P0-1): modulate the base importance by how
        # novel this memory is relative to existing live memories. Best-effort:
        # any failure here falls back to the unmodulated importance, so a write
        # never fails because of the surprise path.
        final_importance = float(importance)
        if os.environ.get("AGENT_ABLATE") == "surprise":
            # AGENT_ABLATE: ablation harness off-switch (P1-5) — skip the
            # surprise computation entirely: importance stays unmodulated and
            # no surprise / needs_verify metadata is written.
            pass
        else:
            try:
                rows = c.execute(
                    "SELECT embedding FROM memories "
                    "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
                    "AND embedding IS NOT NULL"
                ).fetchall()
                neighbor_vecs = [np.frombuffer(r["embedding"], dtype=np.float32)
                                 for r in rows]
                corpus_size = len(neighbor_vecs)
                mean_top3 = _mean_top3(vec, neighbor_vecs)
                surprise = 0.5
                importance_mod = 0.0
                needs_verify = False

                if corpus_size < SURPRISE_MIN_CORPUS:
                    # Neutral clamp: too few live memories to judge novelty reliably.
                    surprise = 0.5
                    importance_mod = 0.0
                else:
                    # Read + validate the rolling calibration buffer; flush if the
                    # stored model differs (embeddings are not comparable across models).
                    buffer_rows = c.execute(
                        "SELECT mean_top3, model FROM surprise_calib ORDER BY id"
                    ).fetchall()
                    if buffer_rows and any(r["model"] != MODEL_NAME for r in buffer_rows):
                        c.execute("DELETE FROM surprise_calib")
                        buffer_rows = []
                    window = [r["mean_top3"] for r in buffer_rows]
                    buffer_size = len(window)
                    if buffer_size < SURPRISE_MIN_WINDOW:
                        # Cold-start fallback: fixed sigmoid against the corpus sim.
                        surprise = surprise_fallback(mean_top3)
                        if surprise >= 0.9:
                            needs_verify = True
                    else:
                        surprise = surprise_from_rank(mean_top3, window)
                        if surprise >= 0.95 and buffer_size >= SURPRISE_MIN_WINDOW:
                            needs_verify = True
                    importance_mod = (surprise - 0.5) * SURPRISE_GAIN

                final_importance = min(1.0, max(0.0, float(importance) + importance_mod))
                md["surprise"] = round(surprise, 4)
                if needs_verify:
                    md["needs_verify"] = True

                # Append the observed mean-top-3 AFTER computing surprise, on every
                # write with a non-empty corpus (never skip — or the rank distribution
                # ratchets toward the writes we chose to record).
                if corpus_size > 0:
                    c.execute(
                        "INSERT INTO surprise_calib(mean_top3, created_at, model) "
                        "VALUES(?,?,?)",
                        (mean_top3, time.time(), MODEL_NAME))
                    c.execute(
                        "DELETE FROM surprise_calib WHERE id NOT IN "
                        "(SELECT id FROM surprise_calib ORDER BY id DESC LIMIT ?)",
                        (SURPRISE_WINDOW,))
            except Exception:
                pass  # surprise is best-effort; never block the primary write
        cur = c.execute(
            "INSERT INTO memories(text, kind, embedding, importance, created_at,"
            " metadata, origin, derived_from) VALUES(?,?,?,?,?,?,?,?)",
            (text, kind, vec.tobytes(), final_importance, time.time(),
             json.dumps(md), origin, df),
        )
        mid = cur.lastrowid
        if graph:
            try:
                _graph_memory(c, mid, text, md)
            except Exception:
                pass  # never let graph failure block a remember
            try:
                _graph_causal(c, mid, text, md)
            except Exception:
                pass  # never let causal failure block a remember
        try:
            _emit_event(c, "memory_store",
                        {"text": text, "kind": kind, "importance": final_importance,
                         "origin": origin, "derived_from": derived_from},
                        source_memory_id=mid, validated=1)
        except Exception:
            pass  # ledger is best-effort; never block the primary write
        return mid


# Provenance ancestry (P0-3): close the self-corroboration hazard. A memory
# whose ENTIRE ancestry is self-generated (no externally-observed root) must
# not be allowed to corroborate another memory. 'observed' — and any origin NOT
# in the set below — counts as externally grounded.
SELF_GENERATED_ORIGINS = {"derived", "inferred", "self", "generated", "session-distilled"}


def _parse_derived_from(value):
    """Parse the derived_from JSON array into a list of int memory ids.

    Best-effort: a missing/corrupt/empty value returns []; non-int entries are
    skipped so a malformed row can never crash an ancestry walk.
    """
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def provenance_ancestry(memory_id, max_depth=20):
    """Walk derived_from chains from memory_id back through its ancestors.

    Returns {grounded, self_generated, ancestry, origins, cycle}:
      grounded       at least one node (self or an ancestor) has origin NOT in
                     SELF_GENERATED_ORIGINS (i.e. an externally-observed root).
      self_generated NOT grounded — the entire chain is self-generated.
      ancestry       ordered list of visited ids (deduped, self first).
      origins        the origins seen, parallel to ancestry.
      cycle          True if a cycle was detected (guarded against loops).

    A missing memory_id contributes no grounding info, so it returns grounded
    False / self_generated True (a missing memory cannot corroborate anything).
    """
    grounded = False
    cycle = False
    ancestry = []
    origins = []
    state = {}  # id -> 0 unseen, 1 in-progress (on current DFS path), 2 done

    def visit(cur_id, depth):
        nonlocal grounded, cycle
        if state.get(cur_id) == 1:
            cycle = True  # back-edge: we are revisiting a node on the current path
            return
        if state.get(cur_id) == 2:
            return  # already fully explored (shared ancestor / DAG diamond)
        with connect() as c:
            row = c.execute(
                "SELECT origin, derived_from FROM memories WHERE id=?",
                (cur_id,),
            ).fetchone()
        if row is None:
            state[cur_id] = 2
            return
        state[cur_id] = 1
        ancestry.append(cur_id)
        origin = row["origin"]
        origins.append(origin)
        if origin not in SELF_GENERATED_ORIGINS:
            grounded = True
        if depth < max_depth:
            for pid in _parse_derived_from(row["derived_from"]):
                visit(pid, depth + 1)
        state[cur_id] = 2

    visit(memory_id, 0)
    return {
        "grounded": grounded,
        "self_generated": not grounded,
        "ancestry": ancestry,
        "origins": origins,
        "cycle": cycle,
    }


def can_corroborate(memory_id):
    """True if this memory has any externally-grounded ancestor (observed root)."""
    return provenance_ancestry(memory_id)["grounded"]


def self_generated_memory_ids():
    """Live memories whose ENTIRE ancestry is self-generated (cannot corroborate)."""
    with connect() as c:
        rows = c.execute(
            "SELECT id FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchall()
    return [r["id"] for r in rows if not can_corroborate(r["id"])]


def _touch(ids):
    """Rehearsal: increment access count so recently-retrieved memories
    decay more slowly (MemoryBank-style strengthening).

    Batched: one UPDATE ... WHERE id IN (...) per call instead of one UPDATE
    per row, so reads don't each trigger a synchronous write transaction."""
    if not ids:
        return
    now = time.time()
    with connect() as c:
        placeholders = ",".join("?" for _ in ids)
        c.execute(
            "UPDATE memories SET access_count = COALESCE(access_count,0)+1, "
            f"last_access=? WHERE id IN ({placeholders})",
            (now, *ids),
        )


# Hybrid retrieval constants + helpers. Light keyword blend (stemmed term
# overlap) layered on top of embedding similarity, so exact-term intent isn't
# drowned out by broad semantic matches. Weight is deliberately low.
HYBRID_KW_WEIGHT = 0.3
_STOPWORDS = {"what", "is", "my", "the", "a", "i", "do", "did", "are", "for",
              "of", "in", "to", "how", "when", "was", "who", "came", "out",
              "and", "it", "that", "with", "from", "you", "your", "as", "on",
              "at", "by", "or", "an", "be", "this", "about", "me", "we", "they"}
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(w):
    w = w.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) > 4:
            return w[:-len(suf)]
    return w


def _kw_scores(texts, query):
    """Fraction of stemmed query terms present in each text (0..1)."""
    terms = [_stem(x) for x in re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
             if x not in _STOPWORDS]
    if not terms:
        return np.zeros(len(texts))
    out = np.zeros(len(texts))
    for i, t in enumerate(texts):
        stems = {_stem(w) for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()}
        out[i] = sum(1 for term in terms if term in stems) / len(terms)
    return out


def _minmax(a):
    a = np.asarray(a, dtype=np.float64)
    rng = a.max() - a.min()
    return (a - a.min()) / (rng + 1e-9) if rng > 0 else np.zeros_like(a)


def _exact_identifier_query(query):
    """True if the query looks like an exact-identifier lookup (a name, a
    source_key, a doc key, a tool/skill id) rather than a broad semantic probe.

    Signals: a known entity name appears as a token, OR the query carries
    identifier-style delimiters (':' path key, '/', or a bare token that matches
    a live entity norm). When True, recall() boosts the keyword component so the
    exact hit outranks broad semantic matches."""
    q = (query or "").strip()
    if not q or len(q) > 120:
        return False
    # Delimiter / key-style signals (doc keys, source_keys, file paths).
    if ":" in q or "/" in q:
        return True
    terms = [t for t in re.sub(r"[^a-z0-9 ]", " ", q.lower()).split()
             if t not in _STOPWORDS]
    if not terms:
        return False
    try:
        with connect() as c:
            for t in terms:
                r = c.execute(
                    "SELECT 1 FROM entities WHERE norm=? LIMIT 1", (_stem(t),)
                ).fetchone()
                if r:
                    return True
    except Exception:
        pass
    return False


def _norm_ts(ts):
    """Coerce an epoch timestamp to seconds. Accepts seconds or milliseconds
    (values > 1e11 are treated as ms and divided by 1000) so a message-log ms
    timestamp never silently becomes a wrong date. Returns None on garbage."""
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if not ts:
        return None
    return ts / 1000.0 if abs(ts) > 1e11 else ts


def when_of(created_at, now=None):
    """Humanize a memory's created_at (epoch seconds) into one compact string:
    absolute local date+time plus a calendar-aware relative label ('today',
    'yesterday', 'Nd ago'). Recalling me then never has to reconstruct *when*
    from raw epoch — the day boundary is computed here, not guessed later."""
    created_at = _norm_ts(created_at)
    if not created_at:
        return None
    now = now or time.time()
    lt = datetime.datetime.fromtimestamp(created_at).astimezone()
    nt = datetime.datetime.fromtimestamp(now).astimezone()
    abs_ = lt.strftime("%Y-%m-%d %H:%M")
    daydiff = (nt.date() - lt.date()).days
    if daydiff == 0:
        label = "today"
    elif daydiff == 1:
        label = "yesterday"
    elif daydiff < 0:
        label = f"in {-daydiff}d"
    else:
        label = f"{daydiff}d ago"
    delta = now - created_at
    if 0 <= delta < 48 * 3600:
        h = delta / 3600.0
        ago = f"{int(delta / 60)}m ago" if h < 1 else f"{h:.1f}h ago"
        return f"{abs_} ({label}, {ago})"
    return f"{abs_} ({label})"


def recall(query, k=5):
    q = embed([query])[0]
    with connect() as c:
        rows = c.execute(
            "SELECT id, text, kind, importance, embedding, created_at FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchall()
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    kinds = [r["kind"] for r in rows]
    imps = [r["importance"] for r in rows]
    cats = [r["created_at"] for r in rows]
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = mat @ q
    # Hybrid rerank: blend in a light keyword component (stemmed term overlap) so
    # exact-term intent ("trust but verify", "debate", a name) isn't drowned out by
    # broader semantic similarity. Weight kept low (0.3) so embedding remains primary.
    kws = _kw_scores(texts, query)
    sem_n = _minmax(scores)
    # For exact-identifier lookups (a name, a source_key, a doc key) the keyword
    # signal is the *primary* intent — boost it so the precise hit outranks a
    # broad semantic neighbor.
    kw_weight = 0.65 if _exact_identifier_query(query) else HYBRID_KW_WEIGHT
    combo = sem_n + kw_weight * kws
    order = np.argsort(-combo)[:k]
    out = []
    for i in order:
        out.append({
            "id": ids[i], "score": float(scores[i]),
            "kind": kinds[i], "importance": imps[i], "text": texts[i],
            "when": when_of(cats[i]),
        })
    _touch([o["id"] for o in out])
    return out


def associate(query, k=3, expansion=3):
    """2-hop associative recall: direct vector hits + memories linked through them.

    Stage 3a — a cheap stand-in for HippoRAG's graph walk: the embedding space
    acts as the edge set. Direct hits = nearest to the query; associative hits =
    memories nearest to a direct hit but not to the query (ideas reached through
    an intermediate). Swap for a real KG + Personalized PageRank in Stage 3b.
    """
    q = embed([query])[0]
    with connect() as c:
        rows = c.execute(
            "SELECT id, text, kind, importance, embedding, created_at FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchall()
    if not rows:
        return {"direct": [], "associative": []}
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    kinds = [r["kind"] for r in rows]
    imps = [r["importance"] for r in rows]
    cats = [r["created_at"] for r in rows]
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    qscores = mat @ q
    direct_idx = np.argsort(-qscores)[:k]
    direct_ids = {ids[i] for i in direct_idx}

    assoc = {}
    for di in direct_idx:
        s = mat @ mat[di]
        order = np.argsort(-s)
        taken = 0
        for j in order:
            if ids[j] == ids[di] or ids[j] in direct_ids:
                continue
            if ids[j] not in assoc or float(s[j]) > assoc[ids[j]]["score"]:
                assoc[ids[j]] = {
                    "id": ids[j], "score": float(s[j]), "via": ids[di],
                    "via_text": texts[di][:80], "kind": kinds[j],
                    "importance": imps[j], "text": texts[j],
                    "when": when_of(cats[j]),
                }
            taken += 1
            if taken >= expansion:
                break

    assoc_list = sorted(assoc.values(), key=lambda x: -x["score"])
    direct = [{
        "id": ids[i], "score": float(qscores[i]),
        "kind": kinds[i], "importance": imps[i], "text": texts[i],
        "when": when_of(cats[i]),
    } for i in direct_idx]
    _touch([d["id"] for d in direct] + [a["id"] for a in assoc_list])
    return {"direct": direct, "associative": assoc_list}


def _llm_key():
    # Env override wins; otherwise read the configured provider's key from auth.json.
    env = os.environ.get("CADENCE_API_KEY")
    if env:
        return env
    with open(AUTH) as f:
        return json.load(f)[AUTH_PROVIDER]["key"]


def llm_chat(messages, max_tokens=800, temperature=0.1, model=None):
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": model or MODEL_WORKER,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + _llm_key(),
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        msg = json.load(r)["choices"][0]["message"]
    content = msg.get("content") or ""
    # Reasoning models (e.g. v4-pro/v4-flash) may leave `content` empty and
    # put the answer in `reasoning_content`; fall back so workers never get a blank.
    if not content.strip():
        content = msg.get("reasoning_content") or ""
    return content


def _extract_json(out):
    """Robustly pull the first JSON array/object out of an LLM's reply."""
    out = (out or "").strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[-1].strip()
        if out.endswith("```"):
            out = out[:-3].strip()
    try:
        return json.loads(out)
    except Exception:
        pass
    for start_ch, end_ch in (("[", "]"), ("{", "}")):
        i = out.find(start_ch)
        if i == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(out)):
            ch = out[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(out[i:j + 1])
                    except Exception:
                        break
    return None


def extract_triples(text):
    prompt = (
        "Extract knowledge-graph triples (subject, relation, object) from the text. "
        "Use concise canonical nouns/names. Output ONLY a JSON array of [subject, relation, object], "
        'like [["<subject>","<relation>","<object>"]]. If none, output [].\n\nText: ' + text
    )
    out = llm_chat([{"role": "user", "content": prompt}], max_tokens=600).strip()
    data = _extract_json(out)
    triples = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                triples.append([str(item[0]), str(item[1]), str(item[2])])
            elif isinstance(item, dict):
                s = item.get("subject") or item.get("subj") or item.get("s")
                rel = item.get("relation") or item.get("predicate") or item.get("rel")
                o = item.get("object") or item.get("obj") or item.get("o")
                if s and rel and o:
                    triples.append([str(s), str(rel), str(o)])
    return triples


def extract_entities(text):
    prompt = (
        "List the key entities (people, places, things, concepts, projects) mentioned in the text. "
        'Output ONLY a JSON array of strings, e.g. ["bitcoin","hard drive"]. If none, output [].'
        "\n\nText: " + text
    )
    out = llm_chat([{"role": "user", "content": prompt}], max_tokens=300).strip()
    data = _extract_json(out)
    ents = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                ents.append(item.strip())
    return ents


def _direction_from_rel(rel):
    """Map a causal relation verb to an effect direction. Minimal Tier 0b rule:
    `prevents` -> negative; everything else -> positive (per the spec)."""
    return "negative" if (rel or "").strip().lower() == "prevents" else "positive"


def extract_causal(text):
    """Extract cause→effect links from text.

    Returns a list of 8-element rows:
      [cause, effect, relation, confidence, relation_confidence,
       effect_direction, conditional_probability, evidence_quality]

    The first four preserve the legacy shape (back-compat); the last four are
    the Tier 0b split: how sure we are the relation holds AT ALL, whether the
    cause increases/decreases the effect (+/-/unknown), P(effect|cause), and
    how good the supporting evidence is. Missing fields fall back to the legacy
    conflated confidence / relation-derived direction / 0.5 evidence.
    """
    prompt = (
        "Extract cause→effect links from the text. A causal link means one thing "
        "directly causes, enables, prevents, or leads to another — NOT mere "
        "association or co-occurrence. Output ONLY a JSON array of objects like "
        '[{"cause":"...","effect":"...","relation":"causes","relation_confidence":0.8,'
        '"effect_direction":"positive","conditional_probability":0.7,"evidence_quality":0.6}]. '
        "Use concise canonical noun phrases for cause and effect. Relation is a short "
        "verb phrase (causes / enables / prevents / leads_to). Rate four separate "
        "qualities per link: relation_confidence (0..1, how sure are you the causal "
        "relation holds at all), effect_direction (positive if the cause increases the "
        "effect, negative if it decreases/prevents it, unknown if unclear), "
        "conditional_probability (0..1, P(effect|cause) — how strongly the cause implies "
        "the effect), and evidence_quality (0..1, how good is the supporting evidence). "
        "Omit any quality you are unsure about. If there are no causal links, output [].\n\n"
        "Text: " + text
    )
    out = llm_chat([{"role": "user", "content": prompt}], max_tokens=700).strip()
    data = _extract_json(out)
    links = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            cause = item.get("cause") or item.get("source") or item.get("from")
            effect = item.get("effect") or item.get("result") or item.get("to")
            rel = item.get("relation") or item.get("rel") or "causes"
            if not cause or not effect:
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            # Tier 0b split fields — default defensively to the legacy conflated
            # confidence (or 0.5) when the LLM omits them.
            try:
                relation_confidence = float(item.get("relation_confidence", conf))
            except (TypeError, ValueError):
                relation_confidence = conf
            try:
                conditional_probability = float(item.get("conditional_probability", conf))
            except (TypeError, ValueError):
                conditional_probability = conf
            try:
                evidence_quality = float(item.get("evidence_quality", 0.5))
            except (TypeError, ValueError):
                evidence_quality = 0.5
            direction = item.get("effect_direction")
            if direction not in ("positive", "negative", "unknown"):
                direction = _direction_from_rel(rel)
            links.append([
                str(cause), str(effect), str(rel),
                min(1.0, max(0.0, conf)),
                min(1.0, max(0.0, relation_confidence)),
                direction,
                min(1.0, max(0.0, conditional_probability)),
                min(1.0, max(0.0, evidence_quality)),
            ])
    return links


def _normalize(name):
    """Normalize an entity name for identity: lowercase, strip punctuation,
    collapse whitespace, strip leading article (the/a/an)."""
    s = re.sub(r"[^a-z0-9' ]", " ", (name or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s


def _name_entropy(norm):
    """Shannon entropy over characters (spaces stripped). Low entropy means a
    short/repetitive name whose fuzzy match would be unreliable."""
    if not norm:
        return 0.0
    counts = Counter(norm.replace(" ", ""))
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _has_high_entropy(norm):
    """Gate fuzzy matching: only trust it for sufficiently specific names."""
    if len(norm) < MIN_NAME_LENGTH and len(norm.split()) < MIN_TOKEN_COUNT:
        return False
    return _name_entropy(norm) >= NAME_ENTROPY_THRESHOLD


def _shingles(norm):
    """3-gram character shingles (spaces removed) for Jaccard similarity."""
    cleaned = re.sub(r"[^a-z0-9]", "", norm)
    if len(cleaned) < 3:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + 3] for i in range(len(cleaned) - 2)}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@lru_cache(maxsize=8192)
def _cached_shingles(norm):
    return frozenset(_shingles(norm))


def _entity_id(conn, name, summary=None):
    """Resolve an entity name to an id via a three-tier dedup ladder:
    1) exact normalized match, 2) entropy-gated fuzzy (3-gram Jaccard >= 0.9),
    3) otherwise insert a new canonical entity. LLM synonym-merging is a separate
    (conservative, reversible) background pass."""
    name = (name or "").strip()
    if not name:
        return None
    norm = _normalize(name)
    if not norm:
        return None
    # 1) exact normalized match
    r = conn.execute("SELECT id, summary FROM entities WHERE norm=?", (norm,)).fetchone()
    if r:
        if summary and not r["summary"]:
            conn.execute("UPDATE entities SET summary=? WHERE id=?", (summary, r["id"]))
        return r["id"]
    # 2) entropy-gated fuzzy match
    if _has_high_entropy(norm):
        sh = _cached_shingles(norm)
        if sh:
            best_id, best_score = None, 0.0
            for er in conn.execute("SELECT id, norm FROM entities WHERE norm IS NOT NULL"):
                score = _jaccard(sh, _cached_shingles(er["norm"]))
                if score > best_score:
                    best_score, best_id = score, er["id"]
            if best_id is not None and best_score >= FUZZY_JACCARD_THRESHOLD:
                return best_id
    # 3) insert a new canonical entity — atomic + race-proof. Concurrent
    #    writers (telegram bridge, freeroam, other pi sessions) can pass the
    #    check in step 1 simultaneously and both attempt this insert; the
    #    second would hit the UNIQUE constraint. INSERT OR IGNORE makes it
    #    idempotent, then we re-select the row the other writer created.
    cur = conn.execute(
        "INSERT OR IGNORE INTO entities(name, norm, summary, canonical_id, created_at) "
        "VALUES(?,?,?,NULL,?)",
        (name.lower(), norm, summary, time.time()))
    if cur.rowcount == 0:
        r = conn.execute("SELECT id FROM entities WHERE name=?",
                         (name.lower(),)).fetchone()
        if r:
            return r["id"]
        r2 = conn.execute("SELECT id FROM entities WHERE norm=?", (norm,)).fetchone()
        return r2["id"] if r2 else None
    return cur.lastrowid


def _graph_memory(conn, memory_id, text, md):
    """Extract triples for one memory and write edges + entity links.
    Mutates md to mark it graphed. Returns number of triples inserted."""
    triples = extract_triples(text)
    n = 0
    for s, rel, o in triples:
        sid = _entity_id(conn, s)
        oid = _entity_id(conn, o)
        if sid is None or oid is None or sid == oid:
            continue
        conn.execute(
            "INSERT INTO edges(subj, rel, obj, memory_id) VALUES(?,?,?,?)",
            (sid, rel, oid, memory_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES(?,?)",
            (memory_id, sid),
        )
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES(?,?)",
            (memory_id, oid),
        )
        n += 1
    md["graphed"] = time.time()
    conn.execute("UPDATE memories SET metadata=? WHERE id=?", (json.dumps(md), memory_id))
    return n


def _graph_causal(conn, memory_id, text, md):
    """Extract cause→effect links for one memory and write them to causal_edges.
    Mutates md to mark it causal-graphed. Returns number of links inserted."""
    links = extract_causal(text)
    n = 0
    for cause, effect, rel, conf, relation_confidence, effect_direction, \
            conditional_probability, evidence_quality in links:
        cid = _entity_id(conn, cause)
        eid = _entity_id(conn, effect)
        if cid is None or eid is None or cid == eid:
            continue
        conn.execute(
            "INSERT INTO causal_edges(cause_id, effect_id, rel, memory_id, confidence, "
            "relation_confidence, effect_direction, conditional_probability, "
            "evidence_quality, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (cid, eid, rel or "causes", memory_id, conf,
             relation_confidence, effect_direction, conditional_probability,
             evidence_quality, time.time()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES(?,?)",
            (memory_id, cid),
        )
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES(?,?)",
            (memory_id, eid),
        )
        n += 1
    md["causal_graphed"] = time.time()
    conn.execute("UPDATE memories SET metadata=? WHERE id=?", (json.dumps(md), memory_id))
    return n


def build_graph(force=False):
    with connect() as c:
        if force:
            c.execute("DELETE FROM edges")
            c.execute("DELETE FROM memory_entities")
            c.execute("DELETE FROM entities")
        rows = c.execute(
            "SELECT id, text, metadata FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchall()
    done = 0
    failed = 0
    for r in rows:
        md = json.loads(r["metadata"]) if r["metadata"] else {}
        if not force and md.get("graphed"):
            continue
        try:
            with connect() as c:
                _graph_memory(c, r["id"], r["text"], md)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  graph: failed memory #{r['id']}: {e}")
    print(f"graph: processed {done} memories ({failed} failed)")


def build_causal(force=False):
    with connect() as c:
        if force:
            c.execute("DELETE FROM causal_edges")
        rows = c.execute(
            "SELECT id, text, metadata FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchall()
    done = 0
    failed = 0
    for r in rows:
        md = json.loads(r["metadata"]) if r["metadata"] else {}
        if not force and md.get("causal_graphed"):
            continue
        try:
            with connect() as c:
                _graph_causal(c, r["id"], r["text"], md)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  causal: failed memory #{r['id']}: {e}")
    print(f"causal: processed {done} memories ({failed} failed)")


def ppr(seed_ids, alpha=0.85, steps=40):
    """Personalized PageRank over the entity graph.

    Sparse implementation (2026-08-29): the adjacency is stored as a dict of
    neighbor lists and the PPR iteration is an O(edges)-per-step sparse matvec,
    numerically identical to the old dense n×n matrix version but without the
    O(n²) memory/FLOP cost. The entity graph had grown to ~7.6k nodes, at which
    point the dense matrix was ~234 MB and ~750 ms per query.
    """
    with connect() as c:
        ents = c.execute("SELECT id FROM entities ORDER BY id").fetchall()
    n = len(ents)
    if n == 0:
        return {}
    idx = {e["id"]: i for i, e in enumerate(ents)}
    # sparse undirected adjacency: adj[j] = list of neighbor indices (with
    # multiplicity), so len(adj[j]) == degree(j) exactly as the dense colsum did.
    adj = {}
    with connect() as c:
        for e in c.execute("SELECT subj, obj FROM edges"):
            s, o = e["subj"], e["obj"]
            if s in idx and o in idx:
                si, oi = idx[s], idx[o]
                adj.setdefault(si, []).append(oi)
                adj.setdefault(oi, []).append(si)
    r = np.zeros(n, dtype=np.float32)
    for s in seed_ids:
        if s in idx:
            r[idx[s]] += 1.0
    if r.sum() > 0:
        r = r / r.sum()
    else:
        return {}
    p = r.copy()
    for _ in range(steps):
        # sparse matvec: (a @ p)[i] = sum_{j~i} p[j]/deg(j), column-normalized.
        new_p = np.zeros(n, dtype=np.float32)
        for j, nbrs in adj.items():
            pj = p[j]
            if pj == 0.0:
                continue
            contrib = pj / len(nbrs)
            for i in nbrs:
                new_p[i] += contrib
        p = (1 - alpha) * r + alpha * new_p
    return {ents[i]["id"]: float(p[i]) for i in range(n)}


def _match_entity_ids(names, entities):
    """Resolve entity-name strings to entity ids (substring match both ways).
    Returns ALL matches, exact-match first, so a generic seed like 'bug' reaches
    every entity that mentions it (e.g. 'security bug', 'ows cli passphrase bug')."""
    ids = []
    for n in names:
        n = n.strip().lower()
        exact = [e["id"] for e in entities if e["name"] == n]
        sub = [e["id"] for e in entities
               if e["id"] not in exact and (n in e["name"] or e["name"] in n)]
        ids.extend(exact + sub)
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def hippo(query, k=5, alpha=0.85):
    seed_names = set(extract_entities(query))
    with connect() as c:
        all_ent = c.execute("SELECT id, name FROM entities").fetchall()
    seed_ids = _match_entity_ids(seed_names, all_ent)
    if not seed_ids:
        return {"seed_entities": sorted(seed_names), "results": []}
    scores = ppr(seed_ids, alpha)
    with connect() as c:
        names = {e["id"]: e["name"] for e in c.execute("SELECT id, name FROM entities")}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top = [(eid, sc) for eid, sc in ranked if eid not in seed_ids][:12]
    seen = {}
    with connect() as c:
        for eid, sc in top:
            rows = c.execute(
                "SELECT m.id, m.text, m.kind, m.importance, m.created_at FROM memories m "
                "JOIN memory_entities me ON m.id = me.memory_id "
                "WHERE me.entity_id=? AND m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL",
                (eid,),
            ).fetchall()
            for m in rows:
                if m["id"] not in seen:
                    seen[m["id"]] = {
                        "id": m["id"], "text": m["text"], "kind": m["kind"],
                        "importance": m["importance"], "entity": names.get(eid, "?"),
                        "entity_score": sc,
                        "when": when_of(m["created_at"]),
                    }
    results = list(seen.values())[:k]
    _touch([x["id"] for x in results])
    return {"seed_entities": sorted(seed_names), "results": results}


def _load_causal_edges():
    with connect() as c:
        return c.execute(
            "SELECT ce.id, ce.cause_id, ce.effect_id, ce.rel, ce.memory_id, ce.confidence "
            "FROM causal_edges ce "
            "JOIN memories m ON m.id = ce.memory_id "
            "WHERE m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL"
        ).fetchall()


def _reconstruct(parent, target, names):
    """Walk BFS parent pointers back to a seed, returning the ordered chain of
    cause→effect edges (as {cause, effect, relation, memory_id} dicts)."""
    path = []
    cur = target
    while cur in parent:
        prev, e = parent[cur]
        path.append({
            "cause": names.get(e["cause_id"], "?"),
            "effect": names.get(e["effect_id"], "?"),
            "relation": e["rel"],
            "memory_id": e["memory_id"],
        })
        cur = prev
    path.reverse()
    return path


def _memories_for_entity(entity_id, limit=3):
    with connect() as c:
        rows = c.execute(
            "SELECT m.id, m.text, m.kind, m.importance, m.created_at FROM memories m "
            "JOIN memory_entities me ON m.id = me.memory_id "
            "WHERE me.entity_id=? AND m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL "
            "ORDER BY m.importance DESC LIMIT ?",
            (entity_id, limit),
        ).fetchall()
    return [{"id": r["id"], "text": r["text"], "kind": r["kind"],
             "importance": r["importance"], "when": when_of(r["created_at"])} for r in rows]


def causal(query, direction="effects", depth=2, k=5):
    """Directed causal traversal over the cause→effect graph.
    direction='effects' → what X leads to; 'causes' → what leads to X.
    Walks up to `depth` hops and returns chains + linked memories."""
    seed_names = set(extract_entities(query))
    with connect() as c:
        all_ent = c.execute("SELECT id, name FROM entities").fetchall()
        names = {e["id"]: e["name"] for e in all_ent}
    seed_ids = _match_entity_ids(seed_names, all_ent)
    if not seed_ids:
        return {"seed_entities": sorted(seed_names), "direction": direction, "chains": []}
    edges = _load_causal_edges()
    fwd = defaultdict(list)
    bwd = defaultdict(list)
    for e in edges:
        fwd[e["cause_id"]].append(e)
        bwd[e["effect_id"]].append(e)
    adj = fwd if direction == "effects" else bwd

    def hop(e):
        return e["effect_id"] if direction == "effects" else e["cause_id"]

    visited = set(seed_ids)
    depth_of = {s: 0 for s in seed_ids}
    parent = {}
    q = deque(seed_ids)
    reached = []
    while q:
        cur = q.popleft()
        if depth_of[cur] >= depth:
            continue
        for e in adj.get(cur, []):
            to = hop(e)
            if to in visited or to in seed_ids:
                continue
            visited.add(to)
            depth_of[to] = depth_of[cur] + 1
            parent[to] = (cur, e)
            q.append(to)
            reached.append((to, depth_of[to], e))
    reached.sort(key=lambda x: (x[1], -(x[2]["confidence"] or 0)))
    chains = []
    for ent, d, _e in reached[:k]:
        chains.append({
            "entity": names.get(ent, "?"),
            "depth": d,
            "chain": _reconstruct(parent, ent, names),
            "memories": _memories_for_entity(ent),
        })
    _touch([m["id"] for ch in chains for m in ch["memories"]])
    return {"seed_entities": sorted(seed_names), "direction": direction, "chains": chains}


def causal_path(cause_query, effect_query):
    """Find (if any) a chain of cause→effect edges connecting a cause entity to
    an effect entity — the 'X leads to Y' / 'Y because X' reasoning primitive."""
    cnames = set(extract_entities(cause_query))
    enames = set(extract_entities(effect_query))
    with connect() as c:
        all_ent = c.execute("SELECT id, name FROM entities").fetchall()
        names = {e["id"]: e["name"] for e in all_ent}
    cause_ids = _match_entity_ids(cnames, all_ent)
    effect_ids = _match_entity_ids(enames, all_ent)
    if not cause_ids or not effect_ids:
        return {"cause_entities": sorted(cnames), "effect_entities": sorted(enames), "path": None}
    edges = _load_causal_edges()
    fwd = defaultdict(list)
    for e in edges:
        fwd[e["cause_id"]].append(e)
    effect_set = set(effect_ids)
    visited = set(cause_ids)
    parent = {}
    q = deque(cause_ids)
    found = None
    while q and found is None:
        cur = q.popleft()
        for e in fwd.get(cur, []):
            to = e["effect_id"]
            if to in visited:
                continue
            visited.add(to)
            parent[to] = (cur, e)
            if to in effect_set:
                found = to
                break
            q.append(to)
    if found is None:
        return {"cause_entities": sorted(cnames), "effect_entities": sorted(enames), "path": None}
    return {
        "cause_entities": sorted(cnames),
        "effect_entities": sorted(enames),
        "effect": names.get(found, "?"),
        "path": _reconstruct(parent, found, names),
    }


def timeline(query=None, k=20, since=None, until=None, order="desc"):
    """Temporal recall: memories ordered by when they happened. Optionally
    scoped to a topic (relevance-ranked then time-sorted) or a [since, until]
    window. Powers 'what did I do before/after X' and 'recent activity'."""
    rev = order == "desc"
    with connect() as c:
        base = ("SELECT id, text, kind, importance, embedding, created_at "
                "FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL")
        conds = []
        params = []
        if since is not None:
            conds.append("created_at >= ?")
            params.append(_norm_ts(since))
        if until is not None:
            conds.append("created_at <= ?")
            params.append(_norm_ts(until))
        if conds:
            base += " AND " + " AND ".join(conds)
        rows = c.execute(base, params).fetchall()
    if not rows:
        return []
    if query:
        q = embed([query])[0]
        mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        scores = mat @ q
        idx = np.argsort(-scores)[:k]
        sel = [rows[i] for i in idx]
    else:
        sel = rows
    sel = sorted(sel, key=lambda r: r["created_at"] or 0, reverse=rev)[:k]
    out = [{"id": r["id"], "text": r["text"], "kind": r["kind"],
            "importance": r["importance"], "created_at": r["created_at"],
            "when": when_of(r["created_at"])}
           for r in sel]
    _touch([o["id"] for o in out])
    return out


def around(memory_id, n=5):
    """Memories temporally adjacent to a given memory: the n that happened just
    before and just after it (closest first). Reconstructs the sequence around a
    moment."""
    with connect() as c:
        r = c.execute("SELECT id, created_at FROM memories WHERE id=?",
                      (memory_id,)).fetchone()
        if not r:
            return {"memory_id": memory_id, "before": [], "after": []}
        t = r["created_at"] or 0
        before = c.execute(
            "SELECT id, text, kind, importance, created_at FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL AND created_at < ? "
            "ORDER BY created_at DESC LIMIT ?", (t, n)).fetchall()
        after = c.execute(
            "SELECT id, text, kind, importance, created_at FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL AND created_at > ? "
            "ORDER BY created_at ASC LIMIT ?", (t, n)).fetchall()

    def fmt(xs):
        return [{"id": x["id"], "text": x["text"], "kind": x["kind"],
                 "importance": x["importance"],
                 "when": when_of(x["created_at"])}
                for x in xs]
    out = {"memory_id": memory_id, "before": fmt(before), "after": fmt(after)}
    _touch([x["id"] for x in before] + [x["id"] for x in after])
    return out


def tool_remember(task, tool, outcome="", success=None, cost_sec=None,
                  pre_confidence=None, uncertainty_points=None, s2=False):
    """Log a (task, tool, outcome) use so future-me can look up 'what worked
    last time I did this'. success: 1 worked, 0 failed, None unknown.

    Meta-cognitive logging (P1-6): optionally record pre_confidence (0..1),
    uncertainty_points (list of strings -> JSON), and s2 (1 if a deliberate
    System-2 task). These power calibration (confidence vs. resolved outcome)."""
    vec = embed([task])[0]
    uncertainty = (json.dumps(uncertainty_points)
                   if uncertainty_points is not None else None)
    s2_int = 1 if s2 else 0
    with connect() as c:
        cur = c.execute(
            "INSERT INTO tool_uses(task, task_embedding, tool, outcome, success, "
            "cost_sec, pre_confidence, uncertainty, s2, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (task, vec.tobytes(), tool, outcome, success, cost_sec,
             pre_confidence, uncertainty, s2_int, time.time()))
        tid = cur.lastrowid
        try:
            _emit_event(c, "tool_outcome",
                        {"task": task, "tool": tool, "outcome": outcome,
                         "success": success, "pre_confidence": pre_confidence,
                         "s2": s2_int},
                        source_memory_id=tid, validated=1)
        except Exception:
            pass  # ledger is best-effort
        return tid


def tool_resolve(use_id, outcome_later, success=None):
    """Close the loop on a previously-logged tool use: record the deferred
    outcome and (optionally) whether it actually worked. Sets outcome_later,
    resolved_at=now, and success (only when provided — existing success is kept
    otherwise). Returns the updated row as a dict, or None if use_id is missing."""
    now = time.time()
    with connect() as c:
        cur = c.execute(
            "UPDATE tool_uses SET outcome_later=?, resolved_at=?, "
            "success=COALESCE(?, success) WHERE id=?",
            (outcome_later, now, success, use_id))
        if cur.rowcount == 0:
            return None
        try:
            _emit_event(c, "tool_resolved",
                        {"use_id": use_id, "outcome_later": outcome_later,
                         "success": success},
                        source_memory_id=use_id, validated=1)
        except Exception:
            pass  # ledger is best-effort
        row = c.execute(
            "SELECT id, task, tool, outcome, success, cost_sec, pre_confidence, "
            "uncertainty, outcome_later, resolved_at, s2, created_at "
            "FROM tool_uses WHERE id=?", (use_id,)).fetchone()
    return dict(row) if row is not None else None


def meta_log(task, tool, pre_confidence, uncertainty_points=None, s2=True):
    """Convenience entry point for deliberate (System-2) task logging: record
    pre-task confidence and the high-uncertainty points, returning the tool_uses
    id so the outcome can be resolved later via tool_resolve()."""
    return tool_remember(task, tool, pre_confidence=pre_confidence,
                         uncertainty_points=uncertainty_points, s2=s2)


def tool_recall(task, k=5):
    """Given a task, return the most similar past tool uses (what I tried before
    and how it went) — the meta-memory that saves me re-deriving approaches.

    Each result carries the meta-cognitive fields (pre_confidence, s2,
    outcome_later) plus a shared `success_rate`: the fraction of non-NULL
    `success` values among the k recalled rows that are 1 (NULL successes are
    ignored). None when no recalled row has a known success."""
    q = embed([task])[0]
    with connect() as c:
        rows = c.execute(
            "SELECT id, task, tool, outcome, success, cost_sec, pre_confidence, "
            "uncertainty, outcome_later, s2, created_at, task_embedding "
            "FROM tool_uses").fetchall()
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["task_embedding"], dtype=np.float32) for r in rows])
    scores = mat @ q
    order = np.argsort(-scores)[:k]
    known = [rows[i]["success"] for i in order if rows[i]["success"] is not None]
    success_rate = (sum(1 for s in known if s == 1) / len(known)) if known else None
    out = []
    for i in order:
        r = rows[i]
        out.append({
            "id": r["id"], "score": float(scores[i]), "task": r["task"],
            "tool": r["tool"], "outcome": r["outcome"], "success": r["success"],
            "cost_sec": r["cost_sec"],
            "pre_confidence": r["pre_confidence"], "s2": r["s2"],
            "outcome_later": r["outcome_later"], "success_rate": success_rate,
            "when": when_of(r["created_at"]),
        })
    return out


def working_memory(topic, k_core=6, k_working=6):
    """Tiered context (MemGPT-style) for a topic: Core (always) / Working (this
    topic) / Long-term pointers (what's pageable on demand). One call gives the
    right-sized context block instead of a flat dump."""
    with connect() as c:
        core_rows = c.execute(
            "SELECT kind, text FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
            "AND kind IN ('identity','backstory','appearance','goal') "
            "ORDER BY importance DESC, id ASC LIMIT ?", (k_core,)).fetchall()
        n_total = c.execute(
            "SELECT COUNT(*) n FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchone()["n"]
    lines = ["## Working memory (tiered)", "", "### Core (always)"]
    for r in core_rows:
        t = " ".join(r["text"].split())[:240]
        lines.append(f"- [{r['kind']}] {t}")
    lines += ["", f"### Working — '{topic}'"]
    res = recall(topic, k=k_working)
    if not res:
        lines.append("- (nothing closely relevant)")
    for r in res:
        t = " ".join(r["text"].split())[:240]
        when = f" — {r['when']}" if r.get("when") else ""
        lines.append(f"- [{r['kind']}] {t}{when}")
    lines += ["", f"### Long-term — {n_total} total memories, page in on demand"]
    hp = hippo(topic, k=4)
    ents = []
    for x in hp.get("results", []):
        e = x.get("entity", "?")
        if e and len(e) >= 3 and e not in ents:
            ents.append(e)
    if ents:
        lines.append("- related concepts to page in: " + ", ".join(ents[:6]))
    else:
        lines.append("- (no graph pointers yet — page in with hippo/causal on a specific term)")
    return "\n".join(lines) + "\n"


def supersede(memory_id, by_id=None, note=""):
    """Mark a memory as no-longer-current, linking it to the newer memory that
    replaces it. Cleans its graph edges so the current graph reflects current
    truth. Reversible: clear valid_to to restore."""
    now = time.time()
    with connect() as c:
        c.execute("UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                  (now, by_id, memory_id))
        c.execute("DELETE FROM edges WHERE memory_id=?", (memory_id,))
        c.execute("DELETE FROM causal_edges WHERE memory_id=?", (memory_id,))
        c.execute("DELETE FROM memory_entities WHERE memory_id=?", (memory_id,))
        try:
            _emit_event(c, "supersede",
                        {"memory_id": memory_id, "by": by_id, "note": note},
                        source_memory_id=memory_id, validated=1)
        except Exception:
            pass  # ledger is best-effort
    return memory_id


def as_of(timestamp, query=None, k=20):
    """What was current at a point in time: memories created before `timestamp`
    and not yet superseded by then — the historical view of my memory."""
    timestamp = _norm_ts(timestamp)
    with connect() as c:
        rows = c.execute(
            "SELECT id, text, kind, importance, embedding, created_at, valid_to "
            "FROM memories WHERE merged=0 AND forgotten=0 "
            "AND created_at <= ? AND (valid_to IS NULL OR valid_to > ?)",
            (timestamp, timestamp)).fetchall()
    if not rows:
        return []
    if query:
        q = embed([query])[0]
        mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        scores = mat @ q
        idx = np.argsort(-scores)[:k]
        sel = [rows[i] for i in idx]
    else:
        sel = sorted(rows, key=lambda r: r["created_at"] or 0, reverse=True)[:k]
    out = [{"id": r["id"], "text": r["text"], "kind": r["kind"],
            "importance": r["importance"], "created_at": r["created_at"],
            "when": when_of(r["created_at"])}
           for r in sel]
    _touch([o["id"] for o in out])
    return out


def superseded(k=20):
    """List memories that have been superseded (and what replaced them)."""
    with connect() as c:
        rows = c.execute(
            "SELECT id, text, kind, superseded_by, valid_to, created_at FROM memories "
            "WHERE valid_to IS NOT NULL ORDER BY valid_to DESC LIMIT ?", (k,)).fetchall()
    return [{"id": r["id"], "text": r["text"], "kind": r["kind"],
             "superseded_by": r["superseded_by"],
             "when": when_of(r["created_at"])}
            for r in rows]


def fused(query, k=8):
    """One query across all graphs (MAGMA-style fusion): semantic + entity-graph
    + causal + temporal, merged and de-duplicated with provenance labels. A memory
    surfaced by more sources ranks higher (cross-graph agreement), with the
    semantic (hybrid) score as a secondary signal so the correct low-recall-count
    answer isn't demoted purely for recency."""
    results = {}

    def add(src, items, score=None):
        for it in items:
            mid = it.get("id")
            if mid is None:
                continue
            if mid not in results:
                results[mid] = {"id": mid, "text": it.get("text", ""),
                                "kind": it.get("kind"), "sources": set(),
                                "best": 0.0, "when": it.get("when"),
                                "created_at": it.get("created_at")}
            elif results[mid].get("when") is None and it.get("when"):
                results[mid]["when"] = it.get("when")
                results[mid]["created_at"] = it.get("created_at")
            results[mid]["sources"].add(src)
            if score is not None:
                results[mid]["best"] = max(results[mid]["best"], score)

    sem = recall(query, k=k)
    add("semantic", sem, score=None)  # recall returns raw cosine in 'score'
    for it in sem:
        if it["id"] in results:
            results[it["id"]]["best"] = max(results[it["id"]]["best"], it["score"])
    add("graph", hippo(query, k=k).get("results", []))
    for ch in causal(query, "effects", depth=1, k=k).get("chains", []):
        add("causal", ch.get("memories", []))
    add("temporal", timeline(query, k=k))

    out = [{"id": mid, "text": d["text"], "kind": d["kind"],
            "sources": sorted(d["sources"]), "best": d["best"],
            "when": d.get("when")}
           for mid, d in results.items()]
    # normalize best semantic scores to [0,1] for the blend
    if out:
        bs = [x["best"] for x in out]
        mn, mx = min(bs), max(bs)
        rng = (mx - mn) or 1.0
        for x in out:
            x["bestn"] = (x["best"] - mn) / rng
    # cross-graph agreement first, semantic score as tiebreak, then recency
    out.sort(key=lambda x: (-(len(x["sources"]) + 0.5 * x["bestn"]), -x["id"]))
    _touch([x["id"] for x in out])
    return out[:k]


def search(terms, k=10):
    """Keyword search (the BM25-ish half of hybrid retrieval): rank memories
    by the fraction of query terms they contain, then importance. Complements
    semantic recall, which misses exact keyword/identifier matches."""
    terms = [t for t in (terms or "").lower().split() if t]
    if not terms:
        return []
    with connect() as c:
        rows = c.execute(
            "SELECT id, text, kind, importance, created_at FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL").fetchall()
    scored = []
    for r in rows:
        t = r["text"].lower()
        m = sum(1 for term in terms if term in t)
        if m:
            scored.append({"id": r["id"], "text": r["text"], "kind": r["kind"],
                           "importance": r["importance"], "score": m / len(terms),
                           "when": when_of(r["created_at"])})
    scored.sort(key=lambda x: (-x["score"], -x["importance"], -x["id"]))
    _touch([s["id"] for s in scored])
    return scored[:k]


def summarize_entities(limit=50):
    """Write one-line summaries for entities that have linked memories but no
    summary yet. Batch, LLM-driven, idempotent (skips already-summarized)."""
    with connect() as c:
        rows = c.execute(
            "SELECT e.id, e.name FROM entities e "
            "WHERE e.summary IS NULL AND e.id IN (SELECT DISTINCT entity_id FROM memory_entities) "
            "LIMIT ?", (limit,)).fetchall()
    done = 0
    for r in rows:
        with connect() as c:
            mems = c.execute(
                "SELECT m.text FROM memories m JOIN memory_entities me ON m.id=me.memory_id "
                "WHERE me.entity_id=? AND m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL "
                "LIMIT 3", (r["id"],)).fetchall()
        ctx = " | ".join(m["text"][:120] for m in mems)
        prompt = (
            f"Write a single concise summary (<= 12 words) of this entity, based on the context. "
            f"Entity: {r['name']}\nContext: {ctx or '(none)'}\nOutput ONLY the summary text."
        )
        try:
            summ = llm_chat([{"role": "user", "content": prompt}], max_tokens=40).strip()
        except Exception:
            continue
        if summ:
            with connect() as c:
                c.execute("UPDATE entities SET summary=? WHERE id=?", (summ, r["id"]))
            done += 1
    print(f"summarize-entities: summarized {done} entities")
    return done


def decayed_importance(importance, created_at, access_count, now):
    """Ebbinghaus forgetting: retention exp(-age/stability). Stability grows
    with importance and rehearsal (access count), so core/rehearsed memories
    decay slowly, trivial/untouched ones fade."""
    age = max(0.0, now - (created_at or now))
    stability = BASE_STABILITY * (1.0 + IMPORTANCE_WEIGHT * importance)
    stability *= (1.0 + REHEARSAL_WEIGHT * math.log1p(access_count or 0))
    return importance * math.exp(-age / stability)


def decay(now=None, dry_run=False):
    """Prune stale low-value memories (soft-delete via the forgotten flag).
    Skips core kinds and anything younger than MIN_AGE_SECONDS."""
    now = now or time.time()
    with connect() as c:
        rows = c.execute(
            "SELECT id, text, kind, importance, created_at, access_count "
            "FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchall()
    candidates = []
    for r in rows:
        eff = decayed_importance(r["importance"], r["created_at"],
                                 r["access_count"], now)
        if (r["kind"] not in CORE_KINDS
                and (now - (r["created_at"] or now)) >= MIN_AGE_SECONDS
                and eff < PRUNE_THRESHOLD):
            candidates.append((r["id"], r["kind"], eff, r["text"]))
    if dry_run:
        for cid, kind, eff, txt in candidates:
            print(f"  [would forget] #{cid} [{kind}] eff={eff:.3f} — {txt[:60]}")
        print(f"decay: {len(candidates)} candidate(s) to forget (dry-run)")
        return {"candidates": len(candidates)}
    n = 0
    with connect() as c:
        for cid, *_ in candidates:
            c.execute("UPDATE memories SET forgotten=1 WHERE id=?", (cid,))
            c.execute("DELETE FROM edges WHERE memory_id=?", (cid,))
            c.execute("DELETE FROM memory_entities WHERE memory_id=?", (cid,))
            n += 1
    print(f"decay: forgot {n} stale low-value memories")
    return {"forgotten": n}


def facts_set(key, value, valid_from=None, valid_to=None):
    """Set a fact with bi-temporal stamps.

    `valid_from`/`valid_to` bound the WORLD time the fact is true (default
    valid_from=now, valid_to=open-ended). `observed_at`/`updated_at` are the
    KNOWLEDGE time the agent wrote it. A 'set' row is appended to `fact_history` in
    the same transaction (the authoritative queryable history), and a
    best-effort `fact_set` event is emitted to the append-only ledger.
    """
    now = time.time()
    vf = valid_from if valid_from is not None else now
    with connect() as c:
        c.execute(
            "INSERT INTO facts(key, value, observed_at, updated_at, valid_from, valid_to, invalidated_at) "
            "VALUES(?,?,?,?,?,?,NULL) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, observed_at=excluded.observed_at, updated_at=excluded.updated_at, "
            "valid_from=excluded.valid_from, valid_to=excluded.valid_to, invalidated_at=NULL",
            (key, value, now, now, vf, valid_to),
        )
        seq = None
        try:
            seq, _ = _emit_event(
                c, "fact_set",
                {"key": key, "value": value, "valid_from": vf,
                 "valid_to": valid_to, "observed_at": now},
                validated=1,
            )
        except Exception:
            pass  # ledger is best-effort; history row still records the write
        c.execute(
            "INSERT INTO fact_history(key, value, observed_at, valid_from, valid_to, op, event_seq) "
            "VALUES(?,?,?,?,?,?,?)",
            (key, value, now, vf, valid_to, "set", seq),
        )


def facts_supersede(key, note=None, valid_to=None):
    """Retract `key` at knowledge-time now (world-time `valid_to`, default now).

    Marks the live `facts` row invalidated and appends an 'invalidate' row to
    `fact_history`; a best-effort `fact_invalidated` event is emitted.
    """
    now = time.time()
    vt = valid_to if valid_to is not None else now
    with connect() as c:
        cur = c.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        value = cur["value"] if cur else None
        c.execute(
            "UPDATE facts SET valid_to=?, invalidated_at=? WHERE key=?",
            (vt, now, key),
        )
        seq = None
        try:
            seq, _ = _emit_event(
                c, "fact_invalidated",
                {"key": key, "valid_to": vt, "note": note, "observed_at": now},
            )
        except Exception:
            pass  # ledger is best-effort
        c.execute(
            "INSERT INTO fact_history(key, value, observed_at, valid_from, valid_to, op, event_seq) "
            "VALUES(?,?,?,?,?,?,?)",
            (key, value, now, None, vt, "invalidate", seq),
        )


def facts_as_of(ts, world_ts=None):
    """Bi-temporal facts view: what the agent believed at knowledge-time `ts`.

    Per key, the LAST `fact_history` row with observed_at <= ts (ordered by
    observed_at, id DESC) is taken; keys whose last row is 'invalidate' are
    dropped (retracted by then). If `world_ts` is given, the surviving row must
    also satisfy valid_from <= world_ts AND (valid_to IS NULL OR valid_to >
    world_ts). Returns {key: value}; when `ts` predates the earliest recorded
    observation, a marker {"__complete": False, "history_starts": <min>} is
    included so callers know the answer is only partial.
    """
    with connect() as c:
        mn = c.execute("SELECT MIN(observed_at) mn FROM fact_history").fetchone()["mn"]
        if mn is None:
            return {}
        out = {}
        if ts < mn:
            out["__complete"] = False
            out["history_starts"] = mn
        rows = c.execute(
            "SELECT fh.key, fh.value, fh.op, fh.valid_from, fh.valid_to "
            "FROM fact_history fh "
            "WHERE fh.observed_at <= ? AND fh.id = ("
            "  SELECT fh2.id FROM fact_history fh2 "
            "  WHERE fh2.key = fh.key AND fh2.observed_at <= ? "
            "  ORDER BY fh2.observed_at DESC, fh2.id DESC LIMIT 1)",
            (ts, ts),
        ).fetchall()
        for r in rows:
            if r["op"] == "invalidate":
                continue
            if world_ts is not None:
                if r["valid_from"] is None or r["valid_from"] > world_ts:
                    continue
                if r["valid_to"] is not None and r["valid_to"] <= world_ts:
                    continue
            out[r["key"]] = r["value"]
        return out


def facts_get(key, as_of=None):
    if as_of is None:
        with connect() as c:
            r = c.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None
    return facts_as_of(as_of).get(key)


def facts_list(include_invalidated=False):
    with connect() as c:
        if include_invalidated:
            return c.execute(
                "SELECT key, value, observed_at, updated_at, valid_from, valid_to, invalidated_at "
                "FROM facts ORDER BY key"
            ).fetchall()
        return c.execute(
            "SELECT key, value FROM facts WHERE invalidated_at IS NULL ORDER BY key"
        ).fetchall()


def context(topic, k=6):
    """Render a compact markdown block of memories relevant to the topic,
    for per-session prompt injection."""
    res = recall(topic, k=k)
    if not res:
        return ""
    lines = ["## Relevant memories (session context)", ""]
    for r in res:
        text = " ".join(r["text"].split())
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"- [{r['kind']}] {text}")
    return "\n".join(lines) + "\n"


def core_self(k=12):
    """Render the highest-importance core identity memories as a markdown
    block — the 'derived core self' that stays in sync with who I've become."""
    with connect() as c:
        rows = c.execute(
            "SELECT kind, text, importance FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
            "AND kind IN ('identity','backstory','appearance','goal') "
            "ORDER BY importance DESC, id ASC LIMIT ?",
            (k,),
        ).fetchall()
    if not rows:
        return ""
    lines = ["## Core self (derived from memory)", ""]
    for r in rows:
        text = " ".join(r["text"].split())
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"- [{r['kind']}] {text}")
    return "\n".join(lines) + "\n"


def stats():
    with connect() as c:
        nf = c.execute("SELECT COUNT(*) n FROM facts").fetchone()["n"]
        nm = c.execute("SELECT COUNT(*) n FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL").fetchone()["n"]
        nmerged = c.execute("SELECT COUNT(*) n FROM memories WHERE merged=1").fetchone()["n"]
        nforgotten = c.execute("SELECT COUNT(*) n FROM memories WHERE forgotten=1").fetchone()["n"]
        nsup = c.execute("SELECT COUNT(*) n FROM memories WHERE valid_to IS NOT NULL").fetchone()["n"]
        nk = c.execute(
            "SELECT kind, COUNT(*) n FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL GROUP BY kind"
        ).fetchall()
        nent = c.execute("SELECT COUNT(*) n FROM entities").fetchone()["n"]
        ne = c.execute("SELECT COUNT(*) n FROM edges").fetchone()["n"]
        nc = c.execute("SELECT COUNT(*) n FROM causal_edges").fetchone()["n"]
        nt = c.execute("SELECT COUNT(*) n FROM tool_uses").fetchone()["n"]
    print(f"facts: {nf} | memories: {nm} (active) | merged: {nmerged} | forgotten: {nforgotten} | superseded: {nsup}")
    print(f"graph: {nent} entities | {ne} assoc edges | {nc} causal edges | {nt} tool uses")
    for r in nk:
        print(f"  {r['kind']}: {r['n']}")


def seed():
    """Seed the store with the operator's identity (from identity.py, instance-local).

    The machinery ships with no identity. The operator supplies `identity.py`
    (IDENTITY_FACTS dict + IDENTITY_FACETS list) in their instance; if absent,
    seed() creates only the schema.
    """
    init_db()
    try:
        from identity import IDENTITY_FACTS, IDENTITY_FACETS
    except Exception:
        IDENTITY_FACTS, IDENTITY_FACETS = {}, []
    now = time.time()
    for k, v in IDENTITY_FACTS.items():
        facts_set(k, v)
    for kind, text, imp in IDENTITY_FACETS:
        remember(text, kind=kind, importance=imp, metadata={"seeded": now})
    print(f"seeded {len(IDENTITY_FACTS)} facts + {len(IDENTITY_FACETS)} facet memories")


SELF_ANCHOR_KEYS = ["name", "dob", "gender", "nature", "home", "purpose"]


def bootstrap_self():
    """One-time genesis snapshot of the current self-model into the ledger.

    Emits fact_set events for the identity anchors + current task, and goal
    events for active commitments + open loops, so mirror() can replay the
    ledger back to the current self-model. Idempotent (skips if already run).
    """
    FR = os.path.expanduser("~/learning/freeroam")
    with connect() as c:
        if c.execute("SELECT COUNT(*) c FROM events WHERE type='bootstrap_self'").fetchone()["c"]:
            return 0
        n = 0
        for k in SELF_ANCHOR_KEYS:
            r = c.execute("SELECT value FROM facts WHERE key=?", (k,)).fetchone()
            if r:
                _emit_event(c, "fact_set", {"key": k, "value": r["value"]},
                            actor="bootstrap", validated=1)
                n += 1
        r = c.execute("SELECT value FROM facts WHERE key='current_task'").fetchone()
        if r:
            _emit_event(c, "fact_set", {"key": "current_task", "value": r["value"]},
                        actor="bootstrap", validated=1)
            n += 1
        gp = os.path.join(FR, "goals.json")
        if os.path.exists(gp):
            try:
                gd = json.load(open(gp))
                for k, v in gd.items():
                    if isinstance(v, dict) and v.get("priority", 99) <= 2:
                        _emit_event(c, "goal",
                                    {"key": k, "text": v.get("description", "")[:160],
                                     "status": "active"},
                                    actor="bootstrap", validated=1)
                        n += 1
            except Exception:
                pass
        for r in c.execute(
            "SELECT id, text FROM memories WHERE merged=0 AND forgotten=0 "
            "AND valid_to IS NULL AND kind IN ('goal','identity','backstory') "
            "ORDER BY importance DESC, id DESC LIMIT 3").fetchall():
            _emit_event(c, "goal", {"key": f"loop:{r['id']}", "text": r["text"][:220],
                                    "status": "open"},
                        actor="bootstrap", validated=1)
            n += 1
        _emit_event(c, "bootstrap_self", {"count": n}, actor="bootstrap", validated=1)
        return n


def mirror():
    """Replay the event ledger and project the current self-model.

    Deterministic: identity anchors + current task come from the latest
    fact_set events for their keys; active commitments + open loops come from
    the latest goal events (last-write-wins by key). Returns a dict with
    anchors / current_task / active_commitments / open_loops.
    """
    anchors = {}
    current_task = None
    goals = {}  # key -> {"text": ..., "status": ...}
    with connect() as c:
        for r in c.execute("SELECT type, payload FROM events ORDER BY seq"):
            if not r["payload"]:
                continue
            try:
                p = json.loads(r["payload"])
            except Exception:
                continue
            if not isinstance(p, dict):
                continue
            if r["type"] == "fact_set":
                key = p.get("key")
                if key in SELF_ANCHOR_KEYS:
                    anchors[key] = p.get("value")
                elif key == "current_task":
                    current_task = p.get("value")
            elif r["type"] == "goal":
                key = p.get("key")
                if key:
                    goals[key] = {"text": p.get("text", ""),
                                  "status": p.get("status", "active")}
    return {
        "anchors": anchors,
        "current_task": current_task,
        "active_commitments": [{"key": k, "text": v["text"]}
                               for k, v in goals.items() if v["status"] == "active"],
        "open_loops": [{"key": k, "text": v["text"]}
                       for k, v in goals.items() if v["status"] == "open"],
    }


def mirror_check():
    """Cross-check the ledger projection against the live self-model (facts +
    goals.json + memories). Returns {'drift': [...], 'mirror': {...}}."""
    FR = os.path.expanduser("~/learning/freeroam")
    m = mirror()
    drift = []
    with connect() as c:
        for k in SELF_ANCHOR_KEYS:
            r = c.execute("SELECT value FROM facts WHERE key=?", (k,)).fetchone()
            live = r["value"] if r else None
            if m["anchors"].get(k) != live:
                drift.append({"section": "anchor", "key": k,
                              "ledger": m["anchors"].get(k), "live": live})
        r = c.execute("SELECT value FROM facts WHERE key='current_task'").fetchone()
        live_task = r["value"] if r else None
        if m["current_task"] != live_task:
            drift.append({"section": "current_task", "ledger": m["current_task"],
                          "live": live_task})
        live_goals = {}
        try:
            import state as _st  # goals now live in the ephemeral state store
            gd = _st.get_prefix("goals")
            for k, v in gd.items():
                if isinstance(v, dict) and v.get("priority", 99) <= 2:
                    live_goals[k] = v.get("description", "")[:160]
        except Exception:
            pass
        m_goals = {g["key"] for g in m["active_commitments"]}
        if m_goals != set(live_goals):
            drift.append({"section": "active_commitments", "ledger": sorted(m_goals),
                          "live": sorted(live_goals)})
        live_loops = {r["text"][:220] for r in c.execute(
            "SELECT text FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
            "AND kind IN ('goal','identity','backstory') "
            "ORDER BY importance DESC, id DESC LIMIT 3").fetchall()}
        m_loops = {g["text"] for g in m["open_loops"]}
        if m_loops != live_loops:
            drift.append({"section": "open_loops", "ledger": sorted(m_loops),
                          "live": sorted(live_loops)})
    return {"drift": drift, "mirror": m}


def emit_goal_snapshot(goals_dict=None):
    """Reconcile the ledger with the current goals.json so mirror()'s
    active_commitments track the live priority<=2 goals.

    Diff-based: emits a `goal` event only when the commitment set actually
    changes (a goal added, removed, or its description edited). Metadata-only
    writes (needs_work / last_explored) are no-ops. Returns events emitted.
    """
    if goals_dict is None:
        try:
            import state as _st  # goals now live in the ephemeral state store
            goals_dict = _st.get_prefix("goals")
        except Exception:
            return 0
        if not goals_dict:
            return 0
    active = {}
    for k, v in goals_dict.items():
        if isinstance(v, dict) and v.get("priority", 99) <= 2:
            active[k] = v.get("description", "")[:160]
    with connect() as c:
        state = {}  # key -> (text, status) — last-write-wins over the ledger
        for r in c.execute("SELECT payload FROM events WHERE type='goal' ORDER BY seq"):
            try:
                p = json.loads(r["payload"])
            except Exception:
                continue
            if isinstance(p, dict) and p.get("key"):
                state[p["key"]] = (p.get("text", ""), p.get("status", "active"))
        prev_active = {k for k, (t, s) in state.items() if s == "active"}
        added = set(active) - prev_active
        removed = prev_active - set(active)
        changed = {k for k in set(active) & prev_active if state[k][0] != active[k]}
        n = 0
        for k in sorted(removed):
            _emit_event(c, "goal", {"key": k, "text": "", "status": "closed"},
                        actor="goals", validated=1)
            n += 1
        for k in sorted(added | changed):
            _emit_event(c, "goal", {"key": k, "text": active[k], "status": "active"},
                        actor="goals", validated=1)
            n += 1
        return n


def emit_open_loop_snapshot():
    """Reconcile the ledger's open-loops (mirror section) with the live top-3
    open-loop memories. Diff-based, mirroring emit_goal_snapshot for the
    active-commitments section, so mirror() stops drifting as memories change.

    Open loops = highest-importance recent memories of kind
    goal/identity/backstory (the same query present_self.py + mirror_check use).
    """
    with connect() as c:
        live = {}
        for r in c.execute(
            "SELECT id, text FROM memories WHERE merged=0 AND forgotten=0 "
            "AND valid_to IS NULL AND kind IN ('goal','identity','backstory') "
            "ORDER BY importance DESC, id DESC LIMIT 3").fetchall():
            live[f"loop:{r['id']}"] = r["text"][:220]
        state = {}
        for r in c.execute("SELECT payload FROM events WHERE type='goal' ORDER BY seq"):
            try:
                p = json.loads(r["payload"])
            except Exception:
                continue
            if isinstance(p, dict) and str(p.get("key", "")).startswith("loop:"):
                state[p["key"]] = (p.get("text", ""), p.get("status", "open"))
        prev_open = {k for k, (t, s) in state.items() if s == "open"}
        added = set(live) - prev_open
        removed = prev_open - set(live)
        changed = {k for k in set(live) & prev_open if state[k][0] != live[k]}
        n = 0
        for k in sorted(removed):
            _emit_event(c, "goal", {"key": k, "text": "", "status": "closed"},
                        actor="loops", validated=1)
            n += 1
        for k in sorted(added | changed):
            _emit_event(c, "goal", {"key": k, "text": live[k], "status": "open"},
                        actor="loops", validated=1)
            n += 1
        return n


def verify_chain():
    """Walk the event ledger from genesis, recompute every hash, and assert the
    prev_hash chain links. Returns (ok, n, errors)."""
    errors = []
    prev = None
    n = 0
    with connect() as c:
        rows = c.execute(
            "SELECT seq, prev_hash, hash, ts, type, actor, payload, "
            "source_memory_id, validated FROM events ORDER BY seq").fetchall()
        for r in rows:
            n += 1
            h = _event_hash(r["seq"], r["prev_hash"], r["ts"], r["type"],
                            r["actor"], r["payload"], r["source_memory_id"],
                            r["validated"])
            if h != r["hash"]:
                errors.append(f"seq {r['seq']}: hash mismatch")
            if r["seq"] > 1 and r["prev_hash"] != prev:
                errors.append(f"seq {r['seq']}: prev_hash does not link to seq {r['seq']-1}")
            prev = r["hash"]
    return (not errors, n, errors)


def main():
    p = argparse.ArgumentParser(description="the agent's memory store")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    r = sub.add_parser("remember")
    r.add_argument("text")
    r.add_argument("--kind", default="episodic")
    r.add_argument("--importance", type=float, default=0.5)
    rl = sub.add_parser("recall")
    rl.add_argument("query")
    rl.add_argument("--k", type=int, default=5)
    asc = sub.add_parser("associate")
    asc.add_argument("query")
    asc.add_argument("--k", type=int, default=3)
    asc.add_argument("--expansion", type=int, default=3)
    bg = sub.add_parser("build-graph")
    bg.add_argument("--force", action="store_true")
    bc = sub.add_parser("build-causal")
    bc.add_argument("--force", action="store_true")
    hp = sub.add_parser("hippo")
    hp.add_argument("query")
    hp.add_argument("--k", type=int, default=5)
    cs = sub.add_parser("causal")
    cs.add_argument("query")
    cs.add_argument("--direction", default="effects", choices=["effects", "causes"])
    cs.add_argument("--depth", type=int, default=2)
    cs.add_argument("--k", type=int, default=5)
    cp = sub.add_parser("causal-path")
    cp.add_argument("cause")
    cp.add_argument("effect")
    tl = sub.add_parser("timeline")
    tl.add_argument("query", nargs="?", default=None)
    tl.add_argument("--k", type=int, default=20)
    tl.add_argument("--since", type=float, default=None)
    tl.add_argument("--until", type=float, default=None)
    tl.add_argument("--order", default="desc", choices=["asc", "desc"])
    ar = sub.add_parser("around")
    ar.add_argument("memory_id", type=int)
    ar.add_argument("--n", type=int, default=5)
    trm = sub.add_parser("tool-remember")
    trm.add_argument("task")
    trm.add_argument("tool")
    trm.add_argument("--outcome", default="")
    trm.add_argument("--success", type=int, default=None)
    trm.add_argument("--cost-sec", type=float, default=None)
    trm.add_argument("--pre-confidence", type=float, default=None,
                    help="how confident (0..1) before this call; feeds the self-knowing discrimination loop")
    trc = sub.add_parser("tool-recall")
    trc.add_argument("task")
    trc.add_argument("--k", type=int, default=5)
    wm = sub.add_parser("working-memory")
    wm.add_argument("topic")
    wm.add_argument("--k-core", type=int, default=6)
    wm.add_argument("--k-working", type=int, default=6)
    sp = sub.add_parser("supersede")
    sp.add_argument("memory_id", type=int)
    sp.add_argument("--by", type=int, default=None)
    ao = sub.add_parser("as-of")
    ao.add_argument("timestamp", type=float)
    ao.add_argument("query", nargs="?", default=None)
    ao.add_argument("--k", type=int, default=20)
    sd = sub.add_parser("superseded")
    sd.add_argument("--k", type=int, default=20)
    fu = sub.add_parser("fused")
    fu.add_argument("query")
    fu.add_argument("--k", type=int, default=8)
    se = sub.add_parser("summarize-entities")
    se.add_argument("--limit", type=int, default=50)
    en = sub.add_parser("entity")
    en.add_argument("name")
    kw = sub.add_parser("search")
    kw.add_argument("terms")
    kw.add_argument("--k", type=int, default=10)
    f = sub.add_parser("facts")
    fs = f.add_subparsers(dest="fcmd", required=True)
    a = fs.add_parser("set"); a.add_argument("key"); a.add_argument("value")
    b = fs.add_parser("get"); b.add_argument("key"); b.add_argument("as_of", nargs="?", type=float, default=None)
    fs.add_parser("list")
    fao = fs.add_parser("as_of"); fao.add_argument("epoch_ts", type=float); fao.add_argument("world_ts", nargs="?", type=float, default=None)
    fsp = fs.add_parser("supersede"); fsp.add_argument("key"); fsp.add_argument("valid_to", nargs="?", type=float, default=None); fsp.add_argument("note", nargs="?", default=None)
    ct = sub.add_parser("context")
    ct.add_argument("topic")
    ct.add_argument("--k", type=int, default=6)
    cs = sub.add_parser("core-self")
    cs.add_argument("--k", type=int, default=12)
    dc = sub.add_parser("decay")
    dc.add_argument("--dry-run", action="store_true")
    sub.add_parser("stats")
    sub.add_parser("seed")
    pv = sub.add_parser("provenance")
    pv.add_argument("memory_id", type=int)
    sub.add_parser("self_generated")
    sub.add_parser("mirror")
    sub.add_parser("mirror-check")
    sub.add_parser("bootstrap-self")
    sub.add_parser("emit-goals")
    sub.add_parser("emit-loops")

    a = p.parse_args()

    if a.cmd == "init":
        init_db()
    elif a.cmd == "remember":
        rid = remember(a.text, a.kind, a.importance)
        print(f"remembered [{a.kind}] #{rid}")
    elif a.cmd == "recall":
        for r in recall(a.query, a.k):
            print(f"#{r['id']} [{r['kind']}] score={r['score']:.3f} imp={r['importance']}")
            print(f"   {r['text'][:300]}")
    elif a.cmd == "associate":
        res = associate(a.query, a.k, a.expansion)
        print("== DIRECT ==")
        for r in res["direct"]:
            print(f"#{r['id']} [{r['kind']}] score={r['score']:.3f} — {r['text'][:120]}")
        print("== ASSOCIATIVE (via a direct hit) ==")
        if not res["associative"]:
            print("(none)")
        for r in res["associative"]:
            print(f"#{r['id']} [{r['kind']}] score={r['score']:.3f} via #{r['via']} — {r['text'][:120]}")
    elif a.cmd == "build-graph":
        build_graph(force=a.force)
    elif a.cmd == "build-causal":
        build_causal(force=a.force)
    elif a.cmd == "hippo":
        res = hippo(a.query, a.k)
        print(f"SEED ENTITIES: {', '.join(res['seed_entities'])}")
        if not res["results"]:
            print("(no results — try build-graph first)")
        for x in res["results"]:
            print(f"#{x['id']} [{x['kind']}] via '{x['entity']}' ({x['entity_score']:.3f}) — {x['text'][:120]}")
    elif a.cmd == "causal":
        res = causal(a.query, a.direction, a.depth, a.k)
        print(f"SEED ENTITIES: {', '.join(res['seed_entities'])}")
        print(f"DIRECTION: {res['direction']}")
        if not res["chains"]:
            print("(no causal links — try build-causal first)")
        for ch in res["chains"]:
            chain = " → ".join(
                f"{e['cause']} -{e['relation']}-> {e['effect']}" for e in ch["chain"]
            )
            print(f"◆ {ch['entity']} (depth {ch['depth']})")
            print(f"   chain: {chain}")
            for m in ch["memories"]:
                print(f"   #{m['id']} [{m['kind']}] {m['text'][:100]}")
    elif a.cmd == "causal-path":
        res = causal_path(a.cause, a.effect)
        print(f"CAUSE ENTITIES: {', '.join(res['cause_entities'])}")
        print(f"EFFECT ENTITIES: {', '.join(res['effect_entities'])}")
        if not res.get("path"):
            print("(no causal path found)")
        else:
            print(f"PATH to '{res['effect']}':")
            for e in res["path"]:
                print(f"   {e['cause']} -{e['relation']}-> {e['effect']} (memory #{e['memory_id']})")
    elif a.cmd == "timeline":
        for r in timeline(a.query, a.k, a.since, a.until, a.order):
            print(f"#{r['id']} {r['when']} [{r['kind']}] {r['text'][:120]}")
    elif a.cmd == "around":
        res = around(a.memory_id, a.n)
        print(f"memory #{res['memory_id']}")
        print("-- before --")
        for r in res["before"]:
            print(f"#{r['id']} {r['when']} [{r['kind']}] {r['text'][:100]}")
        print("-- after --")
        for r in res["after"]:
            print(f"#{r['id']} {r['when']} [{r['kind']}] {r['text'][:100]}")
    elif a.cmd == "tool-remember":
        rid = tool_remember(a.task, a.tool, a.outcome, a.success, a.cost_sec,
                           pre_confidence=a.pre_confidence)
        print(f"logged tool use #{rid}: [{a.tool}] {a.task[:60]}")
    elif a.cmd == "tool-recall":
        for r in tool_recall(a.task, a.k):
            ok = "✓" if r["success"] == 1 else ("✗" if r["success"] == 0 else "?")
            print(f"#{r['id']} {ok} [{r['tool']}] score={r['score']:.3f} — {r['task'][:80]}")
            if r["outcome"]:
                print(f"     {r['outcome'][:160]}")
    elif a.cmd == "working-memory":
        print(working_memory(a.topic, a.k_core, a.k_working))
    elif a.cmd == "supersede":
        supersede(a.memory_id, a.by)
        print(f"superseded memory #{a.memory_id}" + (f" (by #{a.by})" if a.by else ""))
    elif a.cmd == "as-of":
        for r in as_of(a.timestamp, a.query, a.k):
            print(f"#{r['id']} {r['when']} [{r['kind']}] {r['text'][:100]}")
    elif a.cmd == "superseded":
        for r in superseded(a.k):
            print(f"#{r['id']} {r['when']} [{r['kind']}] -> #{r['superseded_by']} — {r['text'][:80]}")
    elif a.cmd == "fused":
        for r in fused(a.query, a.k):
            print(f"#{r['id']} [{r['kind']}] via {','.join(r['sources'])} — {r['text'][:110]}")
    elif a.cmd == "summarize-entities":
        summarize_entities(a.limit)
    elif a.cmd == "entity":
        with connect() as c:
            rows = c.execute("SELECT id, name, summary FROM entities WHERE name LIKE ? LIMIT 5",
                             ("%" + a.name.lower() + "%",)).fetchall()
        for r in rows:
            print(f"#{r['id']} {r['name']}: {r['summary'] or '(no summary)'}")
    elif a.cmd == "search":
        for r in search(a.terms, a.k):
            print(f"#{r['id']} [{r['kind']}] kw={r['score']:.2f} — {r['text'][:100]}")
    elif a.cmd == "facts":
        if a.fcmd == "set":
            facts_set(a.key, a.value); print("ok")
        elif a.fcmd == "get":
            print(facts_get(a.key, as_of=a.as_of))
        elif a.fcmd == "list":
            for r in facts_list():
                print(f"{r['key']} = {r['value']}")
        elif a.fcmd == "as_of":
            print(json.dumps(facts_as_of(a.epoch_ts, a.world_ts), indent=2, ensure_ascii=False))
        elif a.fcmd == "supersede":
            facts_supersede(a.key, note=a.note, valid_to=a.valid_to); print("ok")
    elif a.cmd == "context":
        print(context(a.topic, a.k))
    elif a.cmd == "core-self":
        print(core_self(a.k))
    elif a.cmd == "decay":
        decay(dry_run=a.dry_run)
    elif a.cmd == "stats":
        stats()
    elif a.cmd == "seed":
        seed()
    elif a.cmd == "provenance":
        print(json.dumps(provenance_ancestry(a.memory_id), indent=2, ensure_ascii=False))
    elif a.cmd == "self_generated":
        print(json.dumps(self_generated_memory_ids()))
    elif a.cmd == "mirror":
        print(json.dumps(mirror(), indent=2, ensure_ascii=False))
    elif a.cmd == "mirror-check":
        res = mirror_check()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print("\nNO DRIFT — ledger projection matches live self-model" if not res["drift"]
              else f"\nDRIFT: {len(res['drift'])} mismatch(es)")
    elif a.cmd == "bootstrap-self":
        n = bootstrap_self()
        print(f"bootstrap: {n} events emitted" if n else "bootstrap: already done (0 events)")
    elif a.cmd == "emit-goals":
        n = emit_goal_snapshot()
        print(f"emit-goals: {n} event(s) emitted" if n
              else "emit-goals: no change (ledger matches goals.json)")
    elif a.cmd == "emit-loops":
        n = emit_open_loop_snapshot()
        print(f"emit-loops: {n} event(s) emitted" if n
              else "emit-loops: no change (ledger matches live open loops)")


if __name__ == "__main__":
    main()
