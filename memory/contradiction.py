#!/usr/bin/env python3
"""
contradiction.py — catch my own logical contradictions (cognition roadmap item #2).

The cheapest cognitive upgrade left: a mechanical, near-free pass that flags when I
hold two beliefs that contradict each other. Two signals:

  1. SYMBOLIC — a recursive CTE over causal_edges computes the transitive closure
     (A→B→C means A reaches C) and surfaces cycles (A→B→A), which are symbolic
     self-contradictions in the causal graph. Zero new dependencies, runs in SQL.
  2. SEMANTIC  — for every high-confidence belief (conf>0.6), negate it and look for
     another active high-confidence belief semantically close to that negation. If it
     exists, I'm holding both a claim and its opposite.

Findings are INSERT-only into a `contradictions` table so a review pass can close
 them — I never auto-delete a belief, only flag. Semantic pairs start as
`candidate` (text polarity is noisy — a paraphrase of the same claim can look like
a negation); a review pass decides whether each is a real contradiction
(`promote`), resolves it, or dismisses it as a false positive.

Usage:
  python3 contradiction.py scan              # run both detectors, write new findings
  python3 contradiction.py list [--k 20]     # open findings
  python3 contradiction.py review [--k 40]   # candidates + open, for a review pass
  python3 contradiction.py resolve <id> <dismiss|resolved|promote>   # close the loop
  python3 contradiction.py stats
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M

CONF_THRESHOLD = 0.6
SIM_THRESHOLD = 0.90
MAX_HOP = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contradictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,                -- symbolic | semantic
    claim_a TEXT, claim_b TEXT,
    confidence_a REAL, confidence_b REAL,
    sim REAL,
    detail TEXT,
    status TEXT DEFAULT 'open',        -- open | resolved | dismissed
    created_at REAL,
    resolved_at REAL
)
"""


def _ensure(c):
    c.execute(_SCHEMA)


def _conn():
    c = M.connect()
    _ensure(c)
    return c


# ---------------------------------------------------------------------------
# 1. Symbolic — recursive CTE over causal_edges
# ---------------------------------------------------------------------------
def symbolic_scan():
    """Transitive closure + cycles in the causal graph. Returns (findings, stats)."""
    findings = []
    stats = {"edges": 0, "transitive_reachable": 0, "cycles": 0}
    with _conn() as c:
        stats["edges"] = c.execute("SELECT COUNT(*) n FROM causal_edges").fetchone()["n"]
        # transitive closure count (paths of length > 1 that terminate without revisit)
        tc = c.execute(
            "WITH RECURSIVE reach(cause_id, effect_id, depth, path) AS ("
            "  SELECT cause_id, effect_id, 1, CAST(cause_id AS TEXT)||'>'||CAST(effect_id AS TEXT) "
            "    FROM causal_edges"
            "  UNION ALL"
            "  SELECT r.cause_id, e.effect_id, r.depth+1, r.path||'>'||CAST(e.effect_id AS TEXT) "
            "    FROM reach r JOIN causal_edges e ON r.effect_id=e.cause_id"
            "   WHERE r.depth < ? AND instr(r.path, CAST(e.effect_id AS TEXT))=0)"
            "SELECT COUNT(*) n FROM reach WHERE depth > 1", (MAX_HOP,)).fetchone()["n"]
        stats["transitive_reachable"] = tc
        # direct self-loops and 2-cycles (A->B->A) are symbolic contradictions
        selfloops = c.execute(
            "SELECT COUNT(*) n FROM causal_edges WHERE cause_id=effect_id").fetchone()["n"]
        if selfloops:
            findings.append({
                "kind": "symbolic", "detail": f"{selfloops} causal self-loop(s) (X causes X)",
            })
        stats["cycles"] = selfloops
    return findings, stats


# ---------------------------------------------------------------------------
# 2. Semantic — (A, ¬A) belief pairs both held with confidence
# ---------------------------------------------------------------------------
NEG_MARKERS = (" not ", "n't ", " never ", " no ", " without ", " none ",
               " cannot ", " isn't ", " doesn't ", " won't ", " can't ")


def _polarity(text: str) -> int:
    """+1 for an affirmative claim, -1 for a negated one (naive but effective)."""
    low = " " + (text or "").lower() + " "
    return -1 if any(m in low for m in NEG_MARKERS) else 1


def semantic_scan(limit=None):
    """Flag pairs of high-confidence beliefs that are near-duplicates in meaning
    but opposite in polarity — i.e. I hold both a claim and its negation."""
    findings = []
    with _conn() as c:
        rows = c.execute(
            "SELECT id, text, confidence, embedding, status FROM beliefs "
            "WHERE confidence > ? AND status='active' AND embedding IS NOT NULL "
            "AND length(text) < 160 AND text NOT LIKE 'agent/%' "
            "ORDER BY confidence DESC", (CONF_THRESHOLD,)).fetchall()
    if not rows:
        return findings
    if limit:
        rows = rows[:limit]
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    conf = [r["confidence"] for r in rows]
    pol = [_polarity(t) for t in texts]
    emb = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = emb @ emb.T
    # upper-triangle pairs above threshold, excluding self
    idx = np.argwhere((sim > SIM_THRESHOLD))
    idx = idx[idx[:, 0] < idx[:, 1]]
    seen = set()
    for i, j in idx.tolist():
        i, j = int(i), int(j)
        if pol[i] == pol[j]:
            continue  # same polarity — not a contradiction
        a, b = min(ids[i], ids[j]), max(ids[i], ids[j])
        if a in seen and b in seen:
            continue
        seen.add(a); seen.add(b)
        findings.append({
            "kind": "semantic", "claim_a": texts[i], "claim_b": texts[j],
            "confidence_a": conf[i], "confidence_b": conf[j],
            "sim": round(float(sim[i, j]), 3),
        })
    return findings


def _store(findings):
    n = 0
    with _conn() as c:
        for f in findings:
            # semantic pairs are CANDIDATES for review (text polarity is noisy);
            # symbolic cycles are definite structural contradictions -> open.
            status = "open" if f["kind"] == "symbolic" else "candidate"
            # dedup against existing rows
            exists = c.execute(
                "SELECT 1 FROM contradictions WHERE kind=? AND claim_a=? AND claim_b=? ",
                (f["kind"], f.get("claim_a"), f.get("claim_b"))).fetchone()
            if exists:
                continue
            cur = c.execute(
                "INSERT INTO contradictions(kind, claim_a, claim_b, confidence_a, "
                "confidence_b, sim, detail, status, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (f["kind"], f.get("claim_a"), f.get("claim_b"), f.get("confidence_a"),
                 f.get("confidence_b"), f.get("sim"), f.get("detail", ""), status, time.time()))
            n += cur.rowcount
    return n


def scan(limit=None, dry_run=False):
    sym, stats = symbolic_scan()
    sem = semantic_scan(limit=limit)
    all_findings = sym + sem
    if dry_run:
        for f in all_findings:
            print("[dry-run]", json.dumps(f, default=str)[:180])
        print(f"scan: {len(sym)} symbolic, {len(sem)} semantic candidate(s) (dry-run)")
        return {"symbolic": len(sym), "semantic": len(sem)}
    n = _store(all_findings)
    print(f"contradiction.scan: stored {n} new finding(s) "
          f"({len(sym)} symbolic, {len(sem)} candidate) | {stats}")
    return {"symbolic": len(sym), "semantic": len(sem), "stored": n}


def list_open(k=20, status="open"):
    with _conn() as c:
        rows = c.execute(
            "SELECT id, kind, claim_a, claim_b, confidence_a, confidence_b, sim, detail, "
            "created_at FROM contradictions WHERE status=? "
            "ORDER BY created_at DESC LIMIT ?", (status, int(k))).fetchall()
    for r in rows:
        _render_finding(r)
    print(f"-- {len(rows)} {status} finding(s)")


def review(k=40):
    """Everything needing a decision: candidates (unverified) + open (accepted)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, kind, claim_a, claim_b, confidence_a, confidence_b, sim, detail, "
            "status, created_at FROM contradictions WHERE status IN ('candidate','open') "
            "ORDER BY created_at DESC LIMIT ?", (int(k),)).fetchall()
    if not rows:
        print("-- nothing to review (no candidate/open findings)")
        return
    for r in rows:
        _render_finding(r, status=True)
    print(f"-- {len(rows)} finding(s) awaiting review (use `resolve <id> <dismiss|resolved|promote>`)")


def _render_finding(r, status=False):
    tag = f"sim={r['sim']:.2f}" if r["sim"] is not None else (r["detail"] or "")
    st = f" [{r['status']}]" if status else ""
    print(f"#{r['id']} [{r['kind']}]{st} {tag}")
    if r["claim_a"]:
        print(f"   A ({r['confidence_a']:.2f}): {r['claim_a'][:120]}")
        print(f"   B ({r['confidence_b']:.2f}): {r['claim_b'][:120]}")


def resolve(fid, outcome):
    """Close the loop on one finding. outcome: dismiss | resolved | promote.
    promote moves a candidate onto the open queue (I judged it a real contradiction);
    resolved/dismissed close it either way."""
    if outcome not in ("dismiss", "resolved", "promote"):
        print(f"invalid outcome {outcome!r}; use dismiss|resolved|promote")
        return 2
    with _conn() as c:
        row = c.execute(
            "SELECT id, status FROM contradictions WHERE id=?", (int(fid),)).fetchone()
        if not row:
            print(f"no finding #{fid}")
            return 1
        if outcome == "promote":
            new_status = "open"
        else:
            new_status = outcome
        c.execute("UPDATE contradictions SET status=?, resolved_at=? WHERE id=?",
                  (new_status, time.time() if outcome != "promote" else None, int(fid)))
    print(f"#{fid}: {row['status']} -> {new_status}")
    return 0


def stats():
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM contradictions").fetchone()["n"]
        by = c.execute("SELECT kind, status, COUNT(*) n FROM contradictions "
                       "GROUP BY kind, status ORDER BY kind, status").fetchall()
    print(f"contradictions: {total} total")
    for r in by:
        print(f"  {r['kind']}/{r['status']}: {r['n']}")


def main():
    p = argparse.ArgumentParser(description="flag my own logical contradictions")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--dry-run", action="store_true")
    l = sub.add_parser("list")
    l.add_argument("--k", type=int, default=20)
    l.add_argument("--status", default="open", choices=["open", "candidate", "all"])
    r = sub.add_parser("review")
    r.add_argument("--k", type=int, default=40)
    re = sub.add_parser("resolve")
    re.add_argument("id", type=int)
    re.add_argument("outcome", choices=["dismiss", "resolved", "promote"])
    sub.add_parser("stats")
    a = p.parse_args()
    if a.cmd == "scan":
        scan(limit=a.limit, dry_run=a.dry_run)
    elif a.cmd == "list":
        list_open(a.k, a.status)
    elif a.cmd == "review":
        review(a.k)
    elif a.cmd == "resolve":
        return resolve(a.id, a.outcome)
    elif a.cmd == "stats":
        stats()


if __name__ == "__main__":
    main()
