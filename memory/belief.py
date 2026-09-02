#!/usr/bin/env python3
"""
Belief Ledger — the metacognition layer (locked 2026-08-25, with the operator).

The keystone subsystem from the 4-LLM cognitive-upgrade analysis: every thing I
hold true becomes a *belief* — a proposition tagged with an epistemic label
(know/remember/infer/suspect/guess), a calibrated confidence, its basis, and its
counterevidence. The point is to stop asserting and start qualifying, which is
the antidote to my core hazard ("confidently wrong at high fluency, folds under
pushback").

Design (the belief-ledger plan):
  - beliefs table (additive, reversible) — see _ensure().
  - 5-rung ladder: know (.90-.99, session-scoped) / remember (.70-.95) /
    infer (.30-.85) / suspect (.15-.50) / guess (.01-.30).
  - confidence propagation: causal chains multiply (A->B at c1, B->C at c2 =>
    A->C at c1*c2*damp), so .8 and .7 become .56, NEVER certain.
  - seeded from facts (remember), causal_edges + accepted reason-worker insights
    (infer), then extended by an LLM extract worker over memories, then
    transitively propagated over the causal graph.

Usage:
  python3 belief.py run [--budget N] [--dry-run] [--full]   # seed + extract + propagate
  python3 belief.py query "text" [--k 5]                     # retrieve + render beliefs
  python3 belief.py about "subject" [--k 10]
  python3 belief.py status | list [--status active|all] | rollback [--all] | stats
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
import state as st  # ephemeral state store

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "belief-state.json")

EPISTEMIC = ("know", "remember", "infer", "suspect", "guess")
BANDS = {
    "know": (0.90, 0.99),
    "remember": (0.70, 0.95),
    "infer": (0.30, 0.85),
    "suspect": (0.15, 0.50),
    "guess": (0.01, 0.30),
}
CAP = 0.99
DAMP = 0.95            # per-hop damping in transitive causal propagation
REL_CONF_GATE = 0.60   # gate: only chain through relations we believe actually hold
PROPAGATE_DEPTH = 2    # A->B->C (depth 2) for v1
DEFAULT_BUDGET = 12
DEDUP_SIM = 0.92       # embedding dot-product above which a belief is a duplicate
EXTRACT_BACKFILL = 40  # top-N highest-importance memories for the initial backfill


def _ensure(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS beliefs("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "text TEXT NOT NULL,"
        "embedding BLOB,"
        "epistemic TEXT NOT NULL DEFAULT 'infer',"
        "confidence REAL,"
        "basis TEXT,"                 # JSON {"observed","stated","inferred","speculative"}
        "counterevidence TEXT,"       # JSON list [{text, source_id}]
        "sources TEXT,"               # JSON list of source ids/keys
        "source_key TEXT UNIQUE,"     # idempotent seeding key
        "subject TEXT,"               # normalized subject (canonical KG link)
        "volatility TEXT DEFAULT 'medium',"
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


def _llm(prompt, max_tokens=800):
    return worker_common.llm_call(prompt, max_tokens)


def _clamp(conf, epistemic):
    lo, hi = BANDS.get(epistemic, (0.0, 1.0))
    return min(CAP, max(lo, min(hi, float(conf))))


def _insert(conn, text, epistemic, confidence, basis, sources, source_key=None,
            subject=None, embedding=None):
    if embedding is None:
        embedding = M.embed([text])[0].tobytes()
    cur = conn.execute(
        "INSERT INTO beliefs(text, embedding, epistemic, confidence, basis, sources, "
        "source_key, subject, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (text, embedding, epistemic, confidence, json.dumps(basis),
         json.dumps(sources), source_key, subject, time.time(), time.time()),
    )
    bid = cur.lastrowid
    try:
        M._emit_event(conn, "belief",
                      {"text": text, "epistemic": epistemic, "confidence": confidence},
                      source_memory_id=bid, validated=1)
    except Exception:
        pass  # ledger is best-effort
    return bid


def _upsert(conn, text, epistemic, confidence, basis, sources, source_key,
            subject=None, embedding=None):
    """Insert, or refresh confidence/trace if the source_key already exists."""
    if embedding is None:
        embedding = M.embed([text])[0].tobytes()
    conn.execute(
        "INSERT INTO beliefs(text, embedding, epistemic, confidence, basis, sources, "
        "source_key, subject, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(source_key) DO UPDATE SET confidence=excluded.confidence, "
        "text=excluded.text, embedding=excluded.embedding, sources=excluded.sources, "
        "updated_at=excluded.updated_at",
        (text, embedding, epistemic, confidence, json.dumps(basis),
         json.dumps(sources), source_key, subject, time.time(), time.time()),
    )
    try:
        row = conn.execute("SELECT id FROM beliefs WHERE source_key=?",
                           (source_key,)).fetchone()
        M._emit_event(conn, "belief",
                      {"text": text, "epistemic": epistemic, "confidence": confidence,
                       "op": "upsert"},
                      source_memory_id=row["id"] if row else None, validated=1)
    except Exception:
        pass  # ledger is best-effort


# --------------------------------------------------------------------------- #
# seeding (deterministic, idempotent)
# --------------------------------------------------------------------------- #
def _seed_facts(conn, dry_run=False):
    n = 0
    for r in conn.execute("SELECT key, value FROM facts"):
        text = f"{r['key']}: {r['value']}".strip()
        if not text:
            continue
        if not dry_run:
            _upsert(conn, text, "remember", 0.95,
                    {"observed": 0, "stated": 1, "inferred": 0, "speculative": 0},
                    [f"fact:{r['key']}"], f"fact:{r['key']}", subject=r["key"].lower())
        n += 1
    return n


def _entity_names(conn):
    return {e["id"]: e["name"] for e in conn.execute("SELECT id, name FROM entities")}


def _seed_causal(conn, dry_run=False):
    n = 0
    names = _entity_names(conn)
    for e in conn.execute(
            "SELECT ce.id, ce.cause_id, ce.effect_id, ce.rel, ce.memory_id, ce.confidence "
            "FROM causal_edges ce JOIN memories m ON m.id=ce.memory_id "
            "WHERE m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL"):
        cause = names.get(e["cause_id"], f"entity#{e['cause_id']}")
        effect = names.get(e["effect_id"], f"entity#{e['effect_id']}")
        text = f"{cause} {e['rel'] or 'leads_to'} {effect}"
        if not dry_run:
            _upsert(conn, text, "infer", _clamp(e["confidence"] or 0.5, "infer"),
                    {"observed": 0, "stated": 0, "inferred": 1, "speculative": 0},
                    [f"edge:{e['id']}", f"memory:{e['memory_id']}"],
                    f"edge:{e['id']}", subject=M._normalize(cause))
        n += 1
    return n


def _seed_derived(conn, dry_run=False):
    n = 0
    for r in conn.execute(
            "SELECT id, kind, text, confidence, source_ids, rel, subj_name, obj_name "
            "FROM derived WHERE kind IN ('insight','antecedent') AND status='accepted' "
            "AND text IS NOT NULL"):
        srcs = json.loads(r["source_ids"] or "[]")
        if r["kind"] == "antecedent":
            text = f"{r['subj_name']} {r['rel'] or 'leads_to'} {r['obj_name']}"
        else:
            text = r["text"]
        if not text:
            continue
        if not dry_run:
            _upsert(conn, text, "infer", _clamp(r["confidence"] or 0.5, "infer"),
                    {"observed": 0, "stated": 0, "inferred": 1, "speculative": 0},
                    srcs + [f"derived:{r['id']}"], f"derived:{r['id']}",
                    subject=(M._normalize(r["subj_name"]) if r["subj_name"] else None))
        n += 1
    return n


# --------------------------------------------------------------------------- #
# LLM extraction from memories (watermark-gated)
# --------------------------------------------------------------------------- #
_EXTRACT_PROMPT = (
    "You are a belief-extraction worker. Read the memory below and extract the "
    "discrete PROPOSITIONS it asserts as true (facts about the world, about the agent, "
    "about the operator, or about how things work). For each, classify it on this ladder:\n"
    ' - "remember": the memory states this directly as a durable fact.\n'
    ' - "infer": the memory implies it but does not state it outright.\n'
    ' - "suspect": the memory hints at it without committing.\n'
    ' - "guess": the memory speculates about it.\n'
    "Rate confidence 0..1 (your honest certainty). Estimate a basis as counts "
    '{"observed": int, "stated": int, "inferred": int, "speculative": int}.\n'
    'Output ONLY a JSON array: [{"text": "...", "epistemic": "remember", '
    '"confidence": 0.9, "basis": {"observed":0,"stated":1,"inferred":0,"speculative":0}}]. '
    "If nothing worth keeping, output [].\n\nMEMORY: "
)


def _near_duplicate(text):
    with _conn() as c:
        rows = c.execute(
            "SELECT embedding FROM beliefs WHERE status='active'").fetchall()
    if not rows:
        return False
    q = M.embed([text])[0]
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return bool((mat @ q).max() > DEDUP_SIM)


def extract(budget=None, dry_run=False, full=False):
    budget = budget if budget is not None else int(os.environ.get("BELIEF_BUDGET", str(DEFAULT_BUDGET)))
    prev = st.get("worker/belief_extract", {}).get("max_id", 0)
    with _conn() as c:
        max_id = c.execute("SELECT MAX(id) m FROM memories").fetchone()["m"] or 0
        if full:
            rows = c.execute(
                "SELECT id, text, importance FROM memories WHERE merged=0 AND forgotten=0 "
                "AND valid_to IS NULL ORDER BY importance DESC LIMIT ?",
                (EXTRACT_BACKFILL,)).fetchall()
        elif max_id > prev:
            rows = c.execute(
                "SELECT id, text, importance FROM memories WHERE merged=0 AND forgotten=0 "
                "AND valid_to IS NULL AND id > ? ORDER BY id", (prev,)).fetchall()
        else:
            rows = []
    if not rows:
        print("belief.extract: no new memories; skipping (use --full to backfill)")
        return {"extracted": 0}
    stored = 0
    last_done = prev
    for r in rows:
        if stored >= budget:
            print("belief.extract: budget reached")
            break
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
            text = (it.get("text") or "").strip()
            if not text or len(text) < 6:
                continue
            epi = it.get("epistemic") if it.get("epistemic") in EPISTEMIC else "infer"
            try:
                conf = float(it.get("confidence") or 0.5)
            except (TypeError, ValueError):
                conf = 0.5
            basis = it.get("basis") if isinstance(it.get("basis"), dict) else {}
            if _near_duplicate(text):
                continue
            if dry_run:
                print(f"  [dry-run] {epi} ({_clamp(conf, epi):.2f}): {text[:80]}")
                stored += 1
                continue
            with _conn() as c:
                _insert(c, text, epi, _clamp(conf, epi), basis,
                        [f"memory:{r['id']}"], None)
            stored += 1
    if not dry_run:
        st.set("worker/belief_extract", {"max_id": last_done}, durable=True)
    print(f"belief.extract: stored {stored} belief(s)")
    return {"extracted": stored}


# --------------------------------------------------------------------------- #
# transitive propagation over the causal graph
# --------------------------------------------------------------------------- #
def propagate(dry_run=False):
    with _conn() as c:
        edges = c.execute(
            "SELECT ce.id, ce.cause_id, ce.effect_id, ce.rel, ce.memory_id, ce.confidence, "
            "ce.relation_confidence, ce.conditional_probability "
            "FROM causal_edges ce JOIN memories m ON m.id=ce.memory_id "
            "WHERE m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL").fetchall()
        names = _entity_names(c)
    # one-hop map: cause -> list of edge rows
    fwd = {}
    for e in edges:
        fwd.setdefault(e["cause_id"], []).append(e)
    added = 0
    for b, b_edges in fwd.items():
        for e1 in b_edges:
            mid = e1["effect_id"]
            if mid not in fwd:
                continue
            for e2 in fwd[mid]:
                # relation_confidence is a GATE (only chain through relations we
                # actually believe hold); conditional_probability is the VALUE we
                # multiply (chain P(A->B)*P(B->C) = P(A->C)).
                if (e1["relation_confidence"] or 0) < REL_CONF_GATE:
                    continue
                if (e2["relation_confidence"] or 0) < REL_CONF_GATE:
                    continue
                conf = (e1["conditional_probability"] or 0.5) * (e2["conditional_probability"] or 0.5)
                conf = round(min(CAP, conf) * DAMP, 3)
                if conf < BANDS["infer"][0]:
                    continue
                cause = names.get(b, f"entity#{b}")
                effect = names.get(e2["effect_id"], f"entity#{e2['effect_id']}")
                text = f"{cause} leads_to {effect}"
                key = f"trans:{b}:{e2['effect_id']}"
                if dry_run:
                    print(f"  [dry-run] infer ({conf:.2f}): {text}")
                    added += 1
                    continue
                with _conn() as c:
                    _upsert(c, text, "infer", conf,
                            {"observed": 0, "stated": 0, "inferred": 2, "speculative": 0},
                            [f"edge:{e1['id']}", f"edge:{e2['id']}"], key,
                            subject=M._normalize(cause))
                added += 1
    print(f"belief.propagate: {added} transitive belief(s)")
    return {"propagated": added}


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #
def query(text, k=5):
    q = M.embed([text])[0]
    with _conn() as c:
        rows = c.execute(
            "SELECT id, text, epistemic, confidence, basis, counterevidence, sources, "
            "subject, embedding FROM beliefs WHERE status='active'").fetchall()
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = mat @ q
    order = np.argsort(-scores)[:k]
    out = []
    for i in order:
        r = rows[i]
        out.append({
            "id": r["id"], "score": float(scores[i]), "text": r["text"],
            "epistemic": r["epistemic"], "confidence": r["confidence"],
            "basis": json.loads(r["basis"] or "{}"),
            "counterevidence": json.loads(r["counterevidence"] or "[]"),
            "sources": json.loads(r["sources"] or "[]"),
            "subject": r["subject"],
        })
    return out


def about(subject, k=10):
    norm = M._normalize(subject)
    with _conn() as c:
        rows = c.execute(
            "SELECT id, text, epistemic, confidence, basis, sources FROM beliefs "
            "WHERE status='active' AND subject=? ORDER BY confidence DESC LIMIT ?",
            (norm, k)).fetchall()
    return [{"id": r["id"], "text": r["text"], "epistemic": r["epistemic"],
             "confidence": r["confidence"], "basis": json.loads(r["basis"] or "{}"),
             "sources": json.loads(r["sources"] or "[]")} for r in rows]


def render(items, with_header=False):
    lines = []
    if with_header:
        lines.append(f"{len(items)} belief(s):")
    for it in items:
        conf = it.get("confidence") or 0
        epi = it.get("epistemic") or "infer"
        basis = it.get("basis") or {}
        b = "/".join(f"{k[0]}{v}" for k, v in basis.items() if v)
        src = it.get("sources") or []
        lines.append(f"[{epi} {conf:.2f}] {it['text']}" +
                     (f"  (basis: {b})" if b else "") +
                     (f"  (sources: {src[:4]})" if src else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# orchestration + management
# --------------------------------------------------------------------------- #
def run(budget=None, dry_run=False, full=False):
    print(f"belief: {'DRY-RUN' if dry_run else 'live'} (seed + extract + propagate)")
    with _conn() as c:
        nf = _seed_facts(c, dry_run)
    with _conn() as c:
        nc = _seed_causal(c, dry_run)
    with _conn() as c:
        nd = _seed_derived(c, dry_run)
    print(f"belief: seeded ~{nf} fact + {nc} causal + {nd} derived")
    if not dry_run:
        extract(budget=budget, dry_run=dry_run, full=full)
        propagate(dry_run=dry_run)
    else:
        extract(budget=budget, dry_run=True, full=full)
        propagate(dry_run=True)
    stats()


def stats():
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM beliefs WHERE status='active'").fetchone()["n"]
        by = c.execute("SELECT epistemic, COUNT(*) n FROM beliefs WHERE status='active' "
                       "GROUP BY epistemic ORDER BY n DESC").fetchall()
    print(f"beliefs: {n} active")
    for r in by:
        print(f"  {r['epistemic']}: {r['n']}")


def list_beliefs(status_filter="active"):
    q = "SELECT id, text, epistemic, confidence, subject, status FROM beliefs"
    args = []
    if status_filter != "all":
        q += " WHERE status=?"
        args.append(status_filter)
    q += " ORDER BY confidence DESC LIMIT 200"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    for r in rows:
        print(f"#{r['id']} [{r['epistemic']}:{r['confidence']:.2f}] {r['text'][:90]}")


def rollback(all_rows=False):
    with _conn() as c:
        if all_rows:
            n = c.execute("DELETE FROM beliefs").rowcount
        else:
            n = c.execute("DELETE FROM beliefs WHERE status != 'active'").rowcount
    print(f"belief.rollback: cleared {n} belief row(s)")


def main():
    p = argparse.ArgumentParser(description="belief ledger (metacognition)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--budget", type=int, default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--full", action="store_true")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--k", type=int, default=5)
    a = sub.add_parser("about")
    a.add_argument("subject")
    a.add_argument("--k", type=int, default=10)
    sub.add_parser("status")
    ex = sub.add_parser("extract")
    ex.add_argument("--budget", type=int, default=None)
    ex.add_argument("--dry-run", action="store_true")
    ex.add_argument("--full", action="store_true")
    pr = sub.add_parser("propagate")
    pr.add_argument("--dry-run", action="store_true")
    l = sub.add_parser("list")
    l.add_argument("--status", default="active", choices=["active", "all", "superseded", "rejected"])
    rb = sub.add_parser("rollback")
    rb.add_argument("--all", action="store_true")
    sub.add_parser("stats")
    a2 = p.parse_args()

    if a2.cmd == "run":
        run(budget=a2.budget, dry_run=a2.dry_run, full=a2.full)
    elif a2.cmd == "query":
        print(render(query(a2.text, k=a2.k), with_header=True))
    elif a2.cmd == "about":
        print(render(about(a2.subject, k=a2.k), with_header=True))
    elif a2.cmd == "status":
        stats()
    elif a2.cmd == "extract":
        extract(budget=a2.budget, dry_run=a2.dry_run, full=a2.full)
    elif a2.cmd == "propagate":
        propagate(dry_run=a2.dry_run)
    elif a2.cmd == "list":
        list_beliefs(a2.status)
    elif a2.cmd == "rollback":
        rollback(all_rows=a2.all)
    elif a2.cmd == "stats":
        stats()


if __name__ == "__main__":
    main()
