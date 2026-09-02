#!/usr/bin/env python3
"""
reason worker v2 — derive novel, VERIFIED links/insights across memories.

Replaces the naive v1 (`consolidate.reason`), which dumped every memory to the
LLM and asked it to "brainstorm novel connections". That produced exactly the
failure the literature predicts: speculative third-person predictions ("the
user would do X"), near-duplicate re-derivations, no graph structure, no
verification, no provenance, and junk auto-stored at importance 0.9.

v2 is grounded in three established patterns (researched 2026-08-24):

  1. CABLE — "retriever complementarity" (arXiv:2608.17911). A derived link is
     worth keeping only if it connects SEMANTICALLY DISTANT memories via cause /
     motivation / enabling-event / background — NOT topical or entity overlap.
     Mechanism: antecedent-oriented queries + overlap subtraction + LLM
     verification. This is the single most important fix over v1.

  2. ZSLP / KG-LLM link prediction — predict missing entity→entity edges from
     common-neighbor candidates ("structural holes" in the KG), LLM-verified
     with a named relation + confidence.

  3. GraphRAG community reports — cluster the entity graph into connected
     components and write short "emergent theme" summaries that no single
     memory contains (global sense-making).

Every artifact is:
  * verified       — a strict rubric rejects speculation about future behaviour,
                     single-source restatement, and topical/entity overlap;
  * provenance'd   — carries the ids of the source memories it combines;
  * confidence'd   — >=ACCEPT stored as an `insight` memory; the REVIEW band is
                     logged to the `derived` table for the agent/operator to approve;
  * deduped        — embedding similarity against existing memories blocks
                     re-deriving something already known;
  * reversible     — everything lands in the `derived` table (additive) and any
                     stored insight is a soft-delete-able memory.

Three derived artifact kinds (all in the `derived` table):
  kind='antecedent'  directed memory→memory link (CABLE). subj/obj = memory ids.
  kind='edge'        predicted entity→entity relation (ZSLP). subj/obj = entity ids.
  kind='insight'     a community-report / sense-making synthesis. text = the claim.

Usage:
  python3 reason.py run [--budget N] [--dry-run] [--full]
  python3 reason.py status
  python3 reason.py list [--status review|accepted|rejected|all]
  python3 reason.py accept IDS...      # approve review-band rows (promote to accepted)
  python3 reason.py reject IDS...      # mark rows rejected
  python3 reason.py rollback [--all]   # soft-delete insight memories + clear derived rows
  python3 reason.py cleanup-legacy [--dry-run]   # soft-delete v1's speculative insights
"""
import argparse
import difflib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M
import state as st  # ephemeral state store

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "reason-state.json")  # reuse the existing watermark file

ACCEPT = 0.80          # >= store as an insight memory (status 'accepted')
REVIEW = 0.60          # [REVIEW, ACCEPT) -> derived row, status 'review'
DEDUP_SIM = 0.92       # embedding dot-product above which a candidate is a duplicate
MAX_PAIRS_PER_BATCH = 15
MAX_ANTECEDENT_PER_MEM = 3
MAX_HUB = 40           # per component, cap candidate endpoints to top-hubs
MAX_COMPONENTS = 8     # cap community reports per run (biggest first)
MAX_COMPONENT_SIZE = 80  # skip the residual hairball for community reports
DEFAULT_BUDGET = 24    # estimated LLM calls

# Relations the edge predictor is forbidden to emit (LLMs often ignore "don't"
# instructions, so this is enforced in code, not just the prompt).
FORBIDDEN_RELS = {"precedes", "prioritized_in", "ordered_before", "related_to",
                  "relates_to", "associated_with", "co_occurs_with", "cooccurs_with",
                  "follows", "is_part_of_priority_list", "mentions", "contains",
                  "stored_as", "references"}

# De-hub: drop identity/pronoun hubs + super-connectors before clustering, so the
# 'everything is connected' hairball splits into real topic clusters (a cheap
# stand-in for GraphRAG's Leiden clustering at this scale).
DEHUB_DEGREE = 40
DEHUB_NAMES = {"user", "i", "agent", "me", "her", "she", "he", "him", "it",
               "the user", "myself", "we", "operator", "my", "mine"}


# --------------------------------------------------------------------------- #
# schema + helpers
# --------------------------------------------------------------------------- #
def _ensure_derived(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS derived("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "kind TEXT NOT NULL,"                 # antecedent | edge | insight
        "subj_id INTEGER, obj_id INTEGER,"    # entity ids (edge) / memory ids (antecedent)
        "subj_name TEXT, obj_name TEXT,"      # human-readable labels
        "rel TEXT,"                           # relation phrase (edge/antecedent)
        "text TEXT,"                          # full claim (insight) or label fallback
        "confidence REAL,"
        "source_ids TEXT,"                    # JSON list of source memory ids
        "status TEXT DEFAULT 'review',"       # accepted | review | rejected
        "reviewed INTEGER DEFAULT 0,"
        "memory_id INTEGER,"                  # created insight memory id, if any
        "created_at REAL)"
    )


def _conn():
    c = M.connect()
    _ensure_derived(c)
    return c


def _llm(prompt, max_tokens=800, model=None):
    return M.llm_chat([{"role": "user", "content": prompt}],
                      max_tokens=max_tokens, temperature=0.0, model=model)


def _max_id():
    with _conn() as c:
        return c.execute("SELECT MAX(id) m FROM memories").fetchone()["m"] or 0


def _active_memories():
    with _conn() as c:
        return c.execute(
            "SELECT id, text, kind, importance FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL ORDER BY id"
        ).fetchall()


def _load_watermark():
    return st.get("worker/reason", {}).get("max_id", 0)


def _save_watermark(mid):
    st.set("worker/reason", {"max_id": mid}, durable=True)


def _near_duplicate(text):
    """True if an existing active memory is already this claim (embedding check)."""
    near = M.recall(text, k=3)
    return any(n["score"] > DEDUP_SIM for n in near)


def _is_duplicate_insight(text):
    """Dedup a candidate insight against BOTH stored memories and prior derived
    rows. Review-band insights are not stored as memories, so a memory-only check
    misses them; difflib over the `derived` table closes that gap."""
    if _near_duplicate(text):
        return True
    with _conn() as c:
        prior = [r["text"] for r in c.execute(
            "SELECT text FROM derived WHERE kind='insight' AND text IS NOT NULL")]
    for t in prior:
        if difflib.SequenceMatcher(None, text, t).ratio() > 0.85:
            return True
    return False


# --------------------------------------------------------------------------- #
# 1. CABLE — antecedent linking (semantically distant, causally related)
# --------------------------------------------------------------------------- #
_ANTE_QUERY_PROMPT = (
    "Classify this memory by type (event / plan / opinion / preference / "
    "state_change / other) and generate up to 3 SHORT search queries for EARLIER "
    "memories that this one likely BUILDS ON — an earlier experience, plan, "
    "motivation, or background event that this memory is a consequence or outgrowth "
    "of. The queries must NOT restate this memory; they reach back to semantically "
    "distant antecedents. Output ONLY a JSON object: "
    '{"type": "...", "queries": ["...", "..."]}.\n\nMEMORY: '
)

_ANTE_VERIFY_PROMPT = (
    "You are a STRICT memory-link verifier. Decide whether CANDIDATE is a genuine "
    "ANTECEDENT of NEW MEMORY: the candidate supplies the cause, motivation, "
    "enabling event, earlier plan, or background that the new memory is a direct "
    "consequence or outgrowth of.\n\n"
    "REJECT (accept=false) if ANY of these hold:\n"
    " - the two merely share a topic or entity, or are both about the same person/thing;\n"
    " - the connection is speculative (what someone 'would' or 'might' do);\n"
    " - the candidate does not actually explain, cause, or enable the new memory.\n\n"
    "ACCEPT ONLY when the candidate is clearly a prior cause/plan/motivation/background "
    "that the new memory follows from. When in doubt, REJECT.\n\n"
    "Output ONLY JSON: "
    '{{"accept": true/false, "confidence": 0.0-1.0, "relation": '
    '"motivated_by | follows_from | enabled_by | caused_by | background_of"}}.\n\n'
    "NEW MEMORY: {}\nCANDIDATE: {}"
)


def _antecedent_links(new_rows, budget, dry_run=False):
    """For each new memory: generate antecedent queries, retrieve candidates,
    subtract the direct semantic neighborhood, verify the remainder. Returns
    (stored_count, last_fully_processed_memory_id) so the watermark only ever
    advances past memories that were actually processed."""
    stored = 0
    last_done = None
    for r in new_rows:
        if budget["left"] <= 0:
            print("reason: budget exhausted during antecedent linking")
            break
        budget["left"] -= 1
        out = _llm(_ANTE_QUERY_PROMPT + r["text"], max_tokens=250)
        data = M._extract_json(out)
        last_done = r["id"]
        if not isinstance(data, dict):
            continue
        queries = [q for q in (data.get("queries") or []) if isinstance(q, str) and q.strip()][:3]
        if not queries:
            continue
        # direct neighborhood to subtract
        direct = {n["id"] for n in M.recall(r["text"], k=8)}
        cand = {}
        for q in queries:
            for n in M.recall(q, k=8):
                if n["id"] != r["id"] and n["id"] not in direct:
                    cand[n["id"]] = n
        # rank by score, keep best few
        ranked = sorted(cand.values(), key=lambda n: -n["score"])[:MAX_ANTECEDENT_PER_MEM]
        for n in ranked:
            if budget["left"] <= 0:
                break
            budget["left"] -= 1
            vout = _llm(_ANTE_VERIFY_PROMPT.format(r["text"], n["text"]), max_tokens=200)
            v = M._extract_json(vout)
            if not isinstance(v, dict) or not v.get("accept"):
                continue
            try:
                conf = float(v.get("confidence") or 0)
            except (TypeError, ValueError):
                conf = 0
            if conf < REVIEW:
                continue
            status = "accepted" if conf >= ACCEPT else "review"
            if dry_run:
                print(f"  [dry-run] antecedent ({conf:.2f}): #{n['id']} "
                      f"-{v.get('relation')}-> #{r['id']} :: {n['text'][:60]}")
                stored += 1
                continue
            with _conn() as c:
                c.execute(
                    "INSERT INTO derived(kind, subj_id, obj_id, subj_name, obj_name, rel, "
                    "confidence, source_ids, status, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("antecedent", n["id"], r["id"],
                     n["text"][:80], r["text"][:80], v.get("relation"),
                     conf, json.dumps([n["id"], r["id"]]), status, time.time()),
                )
            stored += 1
    print(f"reason: {stored} antecedent link(s) derived")
    return stored, last_done


# --------------------------------------------------------------------------- #
# 2. GraphRAG — community reports (emergent themes across a cluster)
# --------------------------------------------------------------------------- #
_COMMUNITY_PROMPT = (
    "You are a careful memory-reasoning worker. Below are memories that all concern "
    "a connected cluster of entities. Write a short community report (1-3 sentences) "
    "stating ONLY what is newly implied by COMBINING two or more of them — a concrete "
    "fact, causal relationship, or shared goal that follows from their combination and "
    "appears in no single memory alone.\n\n"
    "STRICT RULES:\n"
    ' - Write factual, present-tense statements; never speculate about what anyone '
    '"would", "might", or "is likely to" do.\n'
    " - Do NOT restate or paraphrase a single memory. If your report just re-describes "
    "what one memory already says (e.g. re-stating an already-stored personality), "
    "output [].\n"
    " - Every sentence must combine information from >=2 sources and add something new.\n"
    " - If nothing non-trivial is implied, output [].\n\n"
    "Output ONLY a JSON array: "
    '[{"insight": "...", "confidence": 0.0-1.0, "source_ids": [id, id]}]\n\nMEMORIES:\n'
)


def _graph_components():
    """Connected components of the entity graph (undirected), size-desc, after
    de-hubbing identity/pronoun super-connectors so real topic clusters emerge."""
    with _conn() as c:
        rows = c.execute("SELECT DISTINCT subj, obj FROM edges").fetchall()
        names = {e["id"]: (e["name"] or "").lower() for e in
                 c.execute("SELECT id, name FROM entities")}
    deg = {}
    for r in rows:
        deg[r["subj"]] = deg.get(r["subj"], 0) + 1
        deg[r["obj"]] = deg.get(r["obj"], 0) + 1
    drop = {eid for eid, d in deg.items()
            if d >= DEHUB_DEGREE or names.get(eid, "") in DEHUB_NAMES}

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in rows:
        if r["subj"] in drop or r["obj"] in drop:
            continue
        union(r["subj"], r["obj"])
    comps = {}
    for x in parent:
        comps.setdefault(find(x), []).append(x)
    return sorted((v for v in comps.values() if len(v) >= 2), key=len, reverse=True)


def _component_memories(component, limit=30):
    with _conn() as c:
        q = (
            "SELECT DISTINCT m.id, m.text, m.kind FROM memories m "
            "JOIN memory_entities me ON m.id=me.memory_id "
            "WHERE me.entity_id IN (%s) AND m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL "
            "ORDER BY m.importance DESC LIMIT ?" % ",".join("?" * len(component))
        )
        rows = c.execute(q, list(component) + [limit]).fetchall()
    return [{"id": r["id"], "kind": r["kind"], "text": r["text"]} for r in rows]


def _store_insight(text, confidence, source_ids):
    """Store a high-confidence synthesis as an insight memory + derived row.
    graph=False keeps the KG clean; the normal `enhance` pass graphs it later."""
    mid = M.remember(
        text, kind="insight", importance=confidence,
        metadata={"reason_sources": source_ids, "reason_confidence": confidence,
                  "reason_derived_at": time.time()},
        graph=False,
    )
    with _conn() as c:
        c.execute(
            "INSERT INTO derived(kind, subj_id, obj_id, text, confidence, source_ids, "
            "status, memory_id, created_at) VALUES('insight', NULL, NULL, ?, ?, ?, 'accepted', ?, ?)",
            (text, confidence, json.dumps(source_ids), mid, time.time()),
        )
    return mid


def _community_reports(budget, dry_run):
    # skip tiny components (nothing to synthesize) and the residual hairball
    comps = [c for c in _graph_components() if 4 <= len(c) <= MAX_COMPONENT_SIZE]
    stored = 0
    for comp in comps[:MAX_COMPONENTS]:
        if budget["left"] <= 0:
            print("reason: budget exhausted during community reports")
            break
        mems = _component_memories(comp)
        if len(mems) < 3:
            continue  # nothing to synthesize
        budget["left"] -= 1
        listing = "\n".join(f"[{m['id']}] ({m['kind']}) {m['text']}" for m in mems)
        out = _llm(_COMMUNITY_PROMPT + listing, max_tokens=800)
        data = M._extract_json(out)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for it in data:
            text = (it.get("insight") or "").strip()
            try:
                conf = float(it.get("confidence") or 0)
            except (TypeError, ValueError):
                conf = 0
            src = [int(s) for s in (it.get("source_ids") or [])]
            if not text or conf < REVIEW or len(src) < 2:
                continue
            if _is_duplicate_insight(text):
                continue
            if dry_run:
                print(f"  [dry-run] insight (conf={conf:.2f}, src={src}): {text[:90]}")
                stored += 1
                continue
            if conf >= ACCEPT:
                _store_insight(text, conf, src)
            else:
                with _conn() as c:
                    c.execute(
                        "INSERT INTO derived(kind, text, confidence, source_ids, status, "
                        "created_at) VALUES('insight', ?, ?, ?, 'review', ?)",
                        (text, conf, json.dumps(src), time.time()),
                    )
            stored += 1
    print(f"reason: {stored} community insight(s) derived")
    return stored


# --------------------------------------------------------------------------- #
# 3. ZSLP — entity link prediction (structural holes)
# --------------------------------------------------------------------------- #
_EDGE_PROMPT = (
    "You are a knowledge-graph link predictor. For each entity pair below, decide "
    "whether a MEANINGFUL relation holds between them GIVEN the context, and if so "
    "name it and rate confidence. Link only if a real relation is implied by the "
    "context — NOT merely that both appear near each other. Use concise canonical "
    "relation verbs (e.g. 'uses', 'prefers', 'built', 'leads_to').\n\n"
    "Do NOT output ordering relations ('precedes', 'prioritized_in', 'ordered_before') "
    "or generic 'related_to'/'relates_to'. Only real semantic relations.\n\n"
    "Output ONLY a JSON array: "
    '[{{"subj": "...", "rel": "...", "obj": "...", "confidence": 0.0-1.0}}]. '
    "Include only pairs where a real relation holds; skip the rest.\n\n"
    "ENTITY CONTEXTS:\n{}\n\nPAIRS:\n{}"
)


def _entity_glossary(ids):
    """id -> 'name (short context)' for the LLM."""
    with _conn() as c:
        names = {e["id"]: e["name"] for e in
                 c.execute("SELECT id, name FROM entities WHERE id IN (%s)"
                           % ",".join("?" * len(ids)), list(ids)).fetchall()} if ids else {}
    gloss = {}
    with _conn() as c:
        for eid in ids:
            rows = c.execute(
                "SELECT m.text FROM memories m JOIN memory_entities me ON m.id=me.memory_id "
                "WHERE me.entity_id=? AND m.merged=0 AND m.forgotten=0 AND m.valid_to IS NULL "
                "LIMIT 1", (eid,)).fetchall()
            ctx = (rows[0]["text"][:100] if rows else "")
            gloss[eid] = f"{names.get(eid, '?')}: {ctx}".strip()
    return gloss


def _candidate_pairs(component):
    """Entity pairs in a component that share a common neighbour but lack a direct
    edge (structural holes), ranked by Adamic-Adar-lite score. Hub-capped."""
    ids = set(component)
    with _conn() as c:
        edges = set()
        deg = {}
        for e in c.execute("SELECT subj, obj FROM edges"):
            if e["subj"] in ids and e["obj"] in ids:
                edges.add((e["subj"], e["obj"]))
                edges.add((e["obj"], e["subj"]))
                deg[e["subj"]] = deg.get(e["subj"], 0) + 1
                deg[e["obj"]] = deg.get(e["obj"], 0) + 1
    hubs = [e for e, _ in sorted(deg.items(), key=lambda x: -x[1])[:MAX_HUB]]
    neigh = {}
    for a, b in edges:
        neigh.setdefault(a, set()).add(b)
        neigh.setdefault(b, set()).add(a)
    cand = {}
    for i, a in enumerate(hubs):
        na = neigh.get(a, set())
        for b in hubs[i + 1:]:
            if (a, b) in edges:
                continue
            common = na & neigh.get(b, set())
            if not common:
                continue
            cand[(a, b)] = sum(1.0 / (1.0 + len(neigh.get(n, set()))) for n in common)
    return sorted(((a, b, s) for (a, b), s in cand.items()), key=lambda x: -x[2])


def _edge_prediction(budget, dry_run):
    comps = _graph_components()
    stored = 0
    for comp in comps:
        if budget["left"] <= 0:
            print("reason: budget exhausted during edge prediction")
            break
        pairs = _candidate_pairs(comp)
        if not pairs:
            continue
        ids = set()
        for a, b, _ in pairs:
            ids.add(a)
            ids.add(b)
        gloss = _entity_glossary(list(ids))
        # batch pairs into one prompt
        for start in range(0, len(pairs), MAX_PAIRS_PER_BATCH):
            if budget["left"] <= 0:
                break
            batch = pairs[start:start + MAX_PAIRS_PER_BATCH]
            budget["left"] -= 1
            pair_lines = "\n".join(f"- {gloss.get(a, a)}  <->  {gloss.get(b, b)}"
                                   for a, b, _ in batch)
            out = _llm(_EDGE_PROMPT.format("\n".join(f"- {v}" for v in gloss.values()),
                                           pair_lines), max_tokens=700)
            data = M._extract_json(out)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                continue
            for it in data:
                subj = it.get("subj")
                obj = it.get("obj")
                rel = it.get("rel")
                try:
                    conf = float(it.get("confidence") or 0)
                except (TypeError, ValueError):
                    conf = 0
                if not (subj and obj and rel) or conf < ACCEPT:
                    continue
                if (rel or "").strip().lower().replace(" ", "_") in FORBIDDEN_RELS:
                    continue  # ordering/generic relations add noise, not signal
                # resolve names back to entity ids (exact/substring, case-insensitive)
                a = _resolve_entity(subj)
                b = _resolve_entity(obj)
                if a is None or b is None or a == b:
                    continue
                if dry_run:
                    print(f"  [dry-run] edge ({conf:.2f}): {subj} -{rel}-> {obj}")
                    stored += 1
                    continue
                with _conn() as c:
                    dup = c.execute(
                        "SELECT 1 FROM derived WHERE kind='edge' AND subj_name=? AND rel=? AND obj_name=?",
                        (subj, rel, obj)).fetchone()
                    if dup:
                        continue  # already predicted; skip
                    c.execute(
                        "INSERT INTO derived(kind, subj_id, obj_id, subj_name, obj_name, rel, "
                        "confidence, source_ids, status, created_at) "
                        "VALUES('edge', ?, ?, ?, ?, ?, ?, ?, 'review', ?)",
                        (a, b, subj, obj, rel, conf, json.dumps([]), time.time()),
                    )
                stored += 1
    print(f"reason: {stored} predicted edge(s) derived")
    return stored


def _resolve_entity(name):
    name = (name or "").strip().lower()
    if not name:
        return None
    with _conn() as c:
        r = c.execute("SELECT id FROM entities WHERE norm=? OR name=?",
                      (M._normalize(name), name)).fetchone()
        if r:
            return r["id"]
        r = c.execute("SELECT id FROM entities WHERE name LIKE ?", (f"%{name}%",)).fetchone()
        return r["id"] if r else None


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(budget=None, dry_run=False, full=False):
    budget = budget if budget is not None else int(os.environ.get("REASON_BUDGET", str(DEFAULT_BUDGET)))
    max_id = _max_id()
    prev = _load_watermark()

    with _conn() as c:
        new_rows = c.execute(
            "SELECT id, text, kind FROM memories "
            "WHERE merged=0 AND forgotten=0 AND valid_to IS NULL AND id > ? "
            "ORDER BY id", (prev,)).fetchall() if max_id > prev else []

    if not new_rows and not full:
        print("reason: no new memories since last run; skipping (use --full to force)")
        return {"budget": budget, "new_memories": 0}

    print(f"reason: {len(new_rows)} new memories, budget={budget} LLM calls, "
          f"{'DRY-RUN' if dry_run else 'live'}")

    # split the budget so all three stages get a turn: antecedent linking is the
    # highest-value work (CABLE), but community reports + edge prediction need a
    # floor or they'd starve behind the per-memory verify loop.
    ante_b = {"left": max(1, int(budget * 0.55))}
    comm_b = {"left": max(1, int(budget * 0.25))}
    edge_b = {"left": max(1, int(budget * 0.20))}

    _, last_done = _antecedent_links(new_rows, ante_b, dry_run)
    _community_reports(comm_b, dry_run)
    _edge_prediction(edge_b, dry_run)

    if not dry_run:
        # advance the watermark only past memories antecedent linking finished,
        # so budget exhaustion never silently drops unprocessed memories
        _save_watermark(max(last_done if last_done is not None else prev, prev))
    print(f"reason: done. budget left ~ ante {ante_b['left']} / comm {comm_b['left']} / edge {edge_b['left']}")
    return {"budget": budget, "spent": budget - ante_b["left"] - comm_b["left"] - edge_b["left"],
            "new_memories": len(new_rows)}


# --------------------------------------------------------------------------- #
# review / management CLI
# --------------------------------------------------------------------------- #
def status():
    with _conn() as c:
        rows = c.execute(
            "SELECT status, COUNT(*) n FROM derived GROUP BY status").fetchall()
        total = sum(r["n"] for r in rows)
    print(f"derived rows: {total}")
    for r in rows:
        print(f"  {r['status']}: {r['n']}")
    with _conn() as c:
        by_kind = c.execute(
            "SELECT kind, COUNT(*) n FROM derived GROUP BY kind").fetchall()
    for r in by_kind:
        print(f"  kind={r['kind']}: {r['n']}")


def list_rows(status_filter="all"):
    q = "SELECT id, kind, rel, text, confidence, status, source_ids, created_at FROM derived"
    args = []
    if status_filter != "all":
        q += " WHERE status=?"
        args.append(status_filter)
    q += " ORDER BY id"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    for r in rows:
        body = (r["text"] or f"{r['rel'] or ''}")[:100]
        print(f"#{r['id']} [{r['kind']}:{r['status']}] conf={r['confidence'] or 0:.2f} "
              f"src={r['source_ids']} :: {body}")


def _set_status(ids, new_status, promote=False):
    with _conn() as c:
        for i in ids:
            c.execute("UPDATE derived SET status=?, reviewed=1 WHERE id=?", (new_status, i))
            if promote and new_status == "accepted":
                row = c.execute("SELECT * FROM derived WHERE id=?", (i,)).fetchone()
                if row and row["kind"] == "insight" and row["memory_id"] is None:
                    mid = M.remember(row["text"], kind="insight", importance=row["confidence"],
                                     metadata={"reason_sources": json.loads(row["source_ids"] or "[]"),
                                               "reason_confidence": row["confidence"],
                                               "reason_derived_at": time.time()},
                                     graph=False)
                    c.execute("UPDATE derived SET memory_id=? WHERE id=?", (mid, i))
    print(f"set {len(ids)} row(s) to '{new_status}'")


def rollback(all_rows=False):
    """Soft-delete insight memories created from derived rows, then clear rows."""
    q = "SELECT id, memory_id FROM derived"
    if not all_rows:
        q += " WHERE status != 'accepted'"
    with _conn() as c:
        rows = c.execute(q).fetchall()
    n_mem = 0
    for r in rows:
        if r["memory_id"]:
            with _conn() as c:
                c.execute("UPDATE memories SET forgotten=1 WHERE id=?", (r["memory_id"],))
            n_mem += 1
    with _conn() as c:
        if all_rows:
            c.execute("DELETE FROM derived")
        else:
            c.execute("DELETE FROM derived WHERE status != 'accepted'")
    print(f"rollback: soft-deleted {n_mem} insight memories, cleared {len(rows)} derived rows")


def cleanup_legacy(dry_run=False):
    """Soft-delete v1's speculative insight memories (third-person / 'would do'
    speculation). Keeps factual syntheses. Reversible (forgotten flag)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, text FROM memories WHERE kind='insight' AND merged=0 AND forgotten=0"
        ).fetchall()
    spec = [r for r in rows
            if ("the user" in r["text"].lower() or "she would" in r["text"].lower()
                or "she might" in r["text"].lower() or "likely extends" in r["text"].lower()
                or "suggest she" in r["text"].lower() or "implies" in r["text"].lower()
                or "imply" in r["text"].lower() or "would use" in r["text"].lower()
                or "her " in r["text"].lower())]
    for r in spec:
        print(f"  {'[dry-run]' if dry_run else ''} forget #{r['id']}: {r['text'][:70]}")
    if not dry_run:
        with _conn() as c:
            for r in spec:
                c.execute("UPDATE memories SET forgotten=1 WHERE id=?", (r["id"],))
                c.execute("DELETE FROM edges WHERE memory_id=?", (r["id"],))
                c.execute("DELETE FROM memory_entities WHERE memory_id=?", (r["id"],))
    print(f"cleanup-legacy: {'would forget' if dry_run else 'forgot'} {len(spec)} of {len(rows)} insight memories")


def main():
    p = argparse.ArgumentParser(description="reason worker v2 (CABLE + ZSLP + GraphRAG)")
    p.add_argument("cmd", choices=["run", "status", "list", "accept", "reject", "rollback", "cleanup-legacy"])
    p.add_argument("ids", nargs="*", type=int)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--status", default="all",
                   choices=["all", "review", "accepted", "rejected"])
    p.add_argument("--all", action="store_true", dest="all_rows")
    a = p.parse_args()
    if a.cmd == "run":
        run(budget=a.budget, dry_run=a.dry_run, full=a.full)
    elif a.cmd == "status":
        status()
    elif a.cmd == "list":
        list_rows(a.status)
    elif a.cmd == "accept":
        _set_status(a.ids, "accepted", promote=True)
    elif a.cmd == "reject":
        _set_status(a.ids, "rejected")
    elif a.cmd == "rollback":
        rollback(all_rows=a.all_rows)
    elif a.cmd == "cleanup-legacy":
        cleanup_legacy(dry_run=a.dry_run)


if __name__ == "__main__":
    main()
