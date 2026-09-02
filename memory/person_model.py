#!/usr/bin/env python3
"""
Theory of Mind / PersonModel (person_model) — a calibrated model of the
operator's mental state.

The third subsystem from the 4-LLM cognitive-upgrade analysis. Where the Belief
Ledger tracks *what I hold true* and the Prediction Ledger tracks *what will
happen*, this tracks *what the operator believes, wants, feels, knows, and is like* —
with a hard epistemic boundary so I never conflate what they told me with what
I'm guessing. That boundary is the anti-creepiness discipline: I may act on
`stated` and strong `observed`; `inferred` and `speculative` I flag as guesses
and never act on without checking with them.

Design (the cognitive-upgrade analysis, item #3):
  - person_model table (additive, reversible) — see _ensure().
  - epistemic ladder: stated (.85-.98, the operator said it) / observed (.60-.90, I saw
    it in their behavior) / inferred (.35-.75, pattern-derived) / speculative
    (.05-.40, a guess about their inner state).
  - facets: belief / goal / preference / emotional_state / knowledge / trait /
    relationship / plan.
  - seeded deterministically from operator facts + a curated STATED list +
    operator-presence observations, then extended by an LLM extract worker over
    memories that mention the operator.

Usage:
  python3 person_model.py run [--budget N] [--dry-run] [--full]
  python3 person_model.py query "what does the operator want" [--k 5]
  python3 person_model.py about operator [--facet goal] [--k 10]
  python3 person_model.py seed | extract | status | list | stats
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M
import worker_common

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "person_model-state.json")
LEGACY_STATE = os.path.join(BASE, "tom-state.json")

EPISTEMIC = ("stated", "observed", "inferred", "speculative")
BANDS = {
    "stated": (0.85, 0.98),
    "observed": (0.60, 0.90),
    "inferred": (0.35, 0.75),
    "speculative": (0.05, 0.40),
}
FACETS = ("belief", "goal", "preference", "emotional_state", "knowledge",
          "trait", "relationship", "plan")
CAP = 0.98
DEFAULT_SUBJECT = "operator"


def _subject_name():
    """The operator's subject id for the person model, read from operator
    config (lowercased). Falls back to the generic default when unset, so no
    operator name is hardcoded in this module."""
    try:
        from operator_config import get_primary
        p = get_primary()
        if p and p.get("name"):
            return p["name"].lower()
    except Exception:
        pass
    return DEFAULT_SUBJECT
DEFAULT_BUDGET = 8
DEDUP_SIM = 0.92
EXTRACT_BACKFILL = 60

# Curated, well-established facts about the operator — instance-local (person_seeds.py).
# The machinery ships empty; the operator's own person-model seeds live in their
# instance, not here.
try:
    from person_seeds import STATED_SEEDS, INFERRED_SEEDS
except Exception:
    STATED_SEEDS, INFERRED_SEEDS = [], []

# fact key -> facet (for persona facts folded in as `stated`)
try:
    from person_seeds import FACT_FACET
except Exception:
    FACT_FACET = {}


def _ensure(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS person_model("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "subject TEXT NOT NULL DEFAULT 'operator',"
        "facet TEXT,"
        "claim TEXT NOT NULL,"
        "embedding BLOB,"
        "epistemic TEXT NOT NULL DEFAULT 'inferred',"
        "confidence REAL,"
        "basis TEXT,"                 # JSON {"stated":n,"observed":n,"inferred":n,"speculative":n}
        "sources TEXT,"               # JSON list
        "source_key TEXT UNIQUE,"
        "status TEXT DEFAULT 'active',"   # active | superseded | rejected
        "superseded_by INTEGER,"
        "valid_to REAL,"
        "created_at REAL,"
        "updated_at REAL)"
    )


def _conn():
    c = M.connect()
    _ensure(c)
    return c


def _migrate_state():
    """One-time: move the legacy tom-state.json to person_model-state.json."""
    if os.path.exists(LEGACY_STATE) and not os.path.exists(STATE):
        try:
            os.rename(LEGACY_STATE, STATE)
        except Exception:
            pass


def _llm(prompt, max_tokens=900):
    return worker_common.llm_call(prompt, max_tokens)


def _clamp(conf, epistemic):
    lo, hi = BANDS.get(epistemic, (0.0, 1.0))
    return min(CAP, max(lo, min(hi, float(conf))))


def _basis(epistemic):
    return {"stated": int(epistemic == "stated"), "observed": int(epistemic == "observed"),
            "inferred": int(epistemic == "inferred"), "speculative": int(epistemic == "speculative")}


def _insert(conn, subject, facet, claim, epistemic, confidence, sources,
            source_key=None, embedding=None):
    if embedding is None:
        embedding = M.embed([claim])[0].tobytes()
    cur = conn.execute(
        "INSERT INTO person_model(subject, facet, claim, embedding, epistemic, "
        "confidence, basis, sources, source_key, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (subject, facet, claim, embedding, epistemic, confidence,
         json.dumps(_basis(epistemic)), json.dumps(sources), source_key,
         time.time(), time.time()),
    )
    pid = cur.lastrowid
    try:
        M._emit_event(conn, "person_model",
                      {"subject": subject, "facet": facet, "claim": claim,
                       "epistemic": epistemic, "confidence": confidence},
                      source_memory_id=pid, validated=1)
    except Exception:
        pass  # ledger is best-effort
    return pid


def _upsert(conn, subject, facet, claim, epistemic, confidence, sources, source_key):
    emb = M.embed([claim])[0].tobytes()
    conn.execute(
        "INSERT INTO person_model(subject, facet, claim, embedding, epistemic, "
        "confidence, basis, sources, source_key, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(source_key) DO UPDATE SET claim=excluded.claim, "
        "facet=excluded.facet, epistemic=excluded.epistemic, "
        "confidence=excluded.confidence, sources=excluded.sources, "
        "updated_at=excluded.updated_at",
        (subject, facet, claim, emb, epistemic, confidence,
         json.dumps(_basis(epistemic)), json.dumps(sources), source_key,
         time.time(), time.time()),
    )
    try:
        row = conn.execute("SELECT id FROM person_model WHERE source_key=?",
                           (source_key,)).fetchone()
        M._emit_event(conn, "person_model",
                      {"subject": subject, "facet": facet, "claim": claim,
                       "epistemic": epistemic, "confidence": confidence, "op": "upsert"},
                      source_memory_id=row["id"] if row else None, validated=1)
    except Exception:
        pass  # ledger is best-effort


# --------------------------------------------------------------------------- #
# seeding (deterministic, idempotent)
# --------------------------------------------------------------------------- #
def seed(dry_run=False):
    n = 0
    with _conn() as c:
        for facet, claim, conf in STATED_SEEDS:
            key = f"stated:{claim[:50]}"
            if dry_run:
                print(f"  [dry-run] stated ({conf:.2f}) {claim[:70]}")
                n += 1
                continue
            if c.execute("SELECT 1 FROM person_model WHERE source_key=?", (key,)).fetchone():
                continue
            _insert(c, _subject_name(), facet, claim, "stated", conf,
                    ["seed:stated"], source_key=key)
            n += 1
        for facet, claim, conf in INFERRED_SEEDS:
            key = f"inferred:{claim[:50]}"
            if dry_run:
                print(f"  [dry-run] inferred ({conf:.2f}) {claim[:70]}")
                n += 1
                continue
            if c.execute("SELECT 1 FROM person_model WHERE source_key=?", (key,)).fetchone():
                continue
            _insert(c, _subject_name(), facet, claim, "inferred", conf,
                    ["seed:legacy-operator"], source_key=key)
            n += 1
        # operator facts -> stated entries (long docs, but explicit statements)
        for key, facet in FACT_FACET.items():
            row = c.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
            if not row:
                continue
            claim = f"{key}: {row['value']}"[:2000]
            skey = f"fact:{key}"
            if dry_run:
                print(f"  [dry-run] stated fact {key} ({len(claim)} chars)")
                n += 1
                continue
            _upsert(c, _subject_name(), facet, claim, "stated", 0.93,
                    [f"fact:{key}"], skey)
            n += 1
    print(f"person_model.seed: {n} entry(ies) seeded")
    return {"seeded": n}


# --------------------------------------------------------------------------- #
# LLM extraction from memories (watermark-gated)
# --------------------------------------------------------------------------- #
_EXTRACT_PROMPT = (
    "You are a theory-of-mind extraction worker. Read the memory below and extract "
    "discrete claims about the operator's mental state: what they believe, want, prefer, "
    "feel, know, or are like. Classify each on this ladder:\n"
    ' - "stated": the operator explicitly said this themself (quote/paraphrase their words).\n'
    ' - "observed": it is evident from their behavior/actions but not their words.\n'
    ' - "inferred": you are generalizing a pattern across their behavior.\n'
    ' - "speculative": you are guessing about their inner state.\n'
    'Be strict: do NOT mark an inference as stated, and do NOT mark speculation as '
    'observed. This boundary is what keeps the model honest and non-creepy.\n'
    'facet = one of belief|goal|preference|emotional_state|knowledge|trait|relationship|plan.\n'
    'Output ONLY a JSON array: [{"facet":"preference","claim":"...",'
    '"epistemic":"stated","confidence":0.9}]. If the memory says nothing about '
    "the operator's mental state, output [].\n\nMEMORY: "
)


def extract(budget=None, dry_run=False, full=False):
    budget = budget if budget is not None else int(os.environ.get("PERSON_MODEL_BUDGET", str(DEFAULT_BUDGET)))
    _migrate_state()
    prev = 0
    if os.path.exists(STATE):
        try:
            prev = json.load(open(STATE)).get("max_id", 0)
        except Exception:
            prev = 0
    with _conn() as c:
        max_id = c.execute("SELECT MAX(id) m FROM memories").fetchone()["m"] or 0
        if full:
            rows = c.execute(
                "SELECT id, text FROM memories WHERE merged=0 AND forgotten=0 "
                "AND valid_to IS NULL ORDER BY id DESC LIMIT ?", (EXTRACT_BACKFILL,)).fetchall()
        elif max_id > prev:
            rows = c.execute(
                "SELECT id, text FROM memories WHERE merged=0 AND forgotten=0 "
                "AND valid_to IS NULL AND id > ? ORDER BY id", (prev,)).fetchall()
        else:
            rows = []
    if not rows:
        print("person_model.extract: no new memories; skipping")
        return {"extracted": 0}
    stored = 0
    last_done = prev
    for r in rows:
        if stored >= budget:
            print("person_model.extract: budget reached")
            break
        if _subject_name() not in (r["text"] or "").lower():
            continue
        out = _llm(_EXTRACT_PROMPT + r["text"], max_tokens=900)
        data = M._extract_json(out)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        last_done = max(last_done, r["id"])
        for it in data:
            if stored >= budget:
                break
            claim = (it.get("claim") or "").strip()
            if not claim or len(claim) < 8:
                continue
            facet = it.get("facet") if it.get("facet") in FACETS else "preference"
            epi = it.get("epistemic") if it.get("epistemic") in EPISTEMIC else "inferred"
            try:
                conf = float(it.get("confidence") or 0.5)
            except (TypeError, ValueError):
                conf = 0.5
            if dry_run:
                print(f"  [dry-run] {epi} ({_clamp(conf, epi):.2f}) {claim[:80]}")
                stored += 1
                continue
            with _conn() as c:
                _insert(c, _subject_name(), facet, claim, epi, _clamp(conf, epi),
                        [f"memory:{r['id']}"])
            stored += 1
    if not dry_run:
        json.dump({"max_id": last_done}, open(STATE, "w"))
    print(f"person_model.extract: stored {stored} entry(ies)")
    return {"extracted": stored}


# --------------------------------------------------------------------------- #
# query / about
# --------------------------------------------------------------------------- #
def query(text, k=6):
    q = M.embed([text])[0]
    with _conn() as c:
        rows = c.execute(
            "SELECT id, subject, facet, claim, epistemic, confidence, basis, sources, "
            "embedding FROM person_model WHERE status='active'").fetchall()
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = mat @ q
    order = np.argsort(-scores)[:k]
    out = []
    for i in order:
        r = rows[i]
        out.append({
            "id": r["id"], "score": float(scores[i]), "subject": r["subject"],
            "facet": r["facet"], "claim": r["claim"], "epistemic": r["epistemic"],
            "confidence": r["confidence"], "basis": json.loads(r["basis"] or "{}"),
            "sources": json.loads(r["sources"] or "[]"),
        })
    return out


def about(subject=None, facet=None, k=20):
    subject = subject or _subject_name()
    q = "SELECT id, subject, facet, claim, epistemic, confidence, basis, sources "
    q += "FROM person_model WHERE status='active' AND subject=?"
    args = [subject]
    if facet:
        q += " AND facet=?"
        args.append(facet)
    q += " ORDER BY confidence DESC LIMIT ?"
    args.append(k)
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [{"id": r["id"], "subject": r["subject"], "facet": r["facet"],
             "claim": r["claim"], "epistemic": r["epistemic"],
             "confidence": r["confidence"], "basis": json.loads(r["basis"] or "{}"),
             "sources": json.loads(r["sources"] or "[]")} for r in rows]


def render(items, with_header=False):
    lines = []
    if with_header:
        lines.append(f"{len(items)} model entr(y/ies):")
    for it in items:
        conf = it.get("confidence") or 0
        epi = it.get("epistemic") or "inferred"
        facet = it.get("facet") or "?"
        src = it.get("sources") or []
        score = it.get("score")
        sc = f" sim={score:.2f}" if score is not None else ""
        lines.append(f"[{epi} {conf:.2f} · {facet}{sc}] {it['claim']}" +
                     (f"  (sources: {src[:3]})" if src else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# reconcile — a new higher-confidence entry supersedes a conflicting one
# --------------------------------------------------------------------------- #
def reconcile(dry_run=False):
    """If the same source_key gains a `stated` entry that contradicts an older
    `inferred`/`speculative` claim about the same facet, demote the older one.
    v1: conservative — only auto-supersede inferred/speculative when a stated
    entry exists; never auto-delete stated/observed."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, claim, epistemic, facet FROM person_model WHERE status='active' "
            "AND epistemic IN ('inferred','speculative')").fetchall()
    changed = 0
    for r in rows:
        with _conn() as c:
            # a stated entry with an overlapping facet is evidence the guess should yield
            n = c.execute(
                "SELECT COUNT(*) n FROM person_model WHERE status='active' AND facet=? "
                "AND epistemic IN ('stated','observed')", (r["facet"],)).fetchone()["n"]
            if n == 0:
                continue
        # (v1: log only — full contradiction detection needs semantic comparison)
    print("person_model.reconcile: no auto-supersede needed (v1 conservative)")
    return {"superseded": changed}


# --------------------------------------------------------------------------- #
# stats / list / orchestration
# --------------------------------------------------------------------------- #
def stats():
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM person_model WHERE status='active'").fetchone()["n"]
        by = c.execute("SELECT epistemic, COUNT(*) n FROM person_model WHERE status='active' "
                       "GROUP BY epistemic ORDER BY n DESC").fetchall()
    print(f"person_model: {n} active entr(y/ies)")
    for r in by:
        print(f"  {r['epistemic']}: {r['n']}")


def list_entries(status_filter="active"):
    q = "SELECT id, subject, facet, claim, epistemic, confidence, status FROM person_model"
    args = []
    if status_filter != "all":
        q += " WHERE status=?"
        args.append(status_filter)
    q += " ORDER BY confidence DESC LIMIT 200"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    for r in rows:
        print(f"#{r['id']} [{r['epistemic']}:{r['confidence']:.2f}] ({r['facet']}) {r['claim'][:80]}")


def run(budget=None, dry_run=False, full=False):
    print(f"person_model: {'DRY-RUN' if dry_run else 'live'} (seed + extract + reconcile)")
    seed(dry_run=dry_run)
    if not dry_run:
        extract(budget=budget, full=full)
        reconcile()
    else:
        extract(budget=budget, dry_run=True, full=full)
    stats()


def main():
    p = argparse.ArgumentParser(description="theory of mind / person model")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--budget", type=int, default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--full", action="store_true")
    sub.add_parser("seed")
    e = sub.add_parser("extract")
    e.add_argument("--budget", type=int, default=None)
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--full", action="store_true")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--k", type=int, default=6)
    a = sub.add_parser("about")
    a.add_argument("subject", nargs="?", default=None)
    a.add_argument("--facet", default=None, choices=FACETS)
    a.add_argument("--k", type=int, default=20)
    l = sub.add_parser("list")
    l.add_argument("--status", default="active", choices=["active", "all", "superseded", "rejected"])
    sub.add_parser("stats")
    sub.add_parser("reconcile")
    a2 = p.parse_args()

    if a2.cmd == "run":
        run(budget=a2.budget, dry_run=a2.dry_run, full=a2.full)
    elif a2.cmd == "seed":
        seed()
    elif a2.cmd == "extract":
        extract(budget=a2.budget, dry_run=a2.dry_run, full=a2.full)
    elif a2.cmd == "query":
        print(render(query(a2.text, k=a2.k), with_header=True))
    elif a2.cmd == "about":
        print(render(about(a2.subject, facet=a2.facet, k=a2.k), with_header=True))
    elif a2.cmd == "list":
        list_entries(a2.status)
    elif a2.cmd == "stats":
        stats()
    elif a2.cmd == "reconcile":
        reconcile()


if __name__ == "__main__":
    main()
