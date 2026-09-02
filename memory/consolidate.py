#!/usr/bin/env python3
"""
Consolidation micro-agent (Stage 4) — the first workers of the hive (Stage 5).

Multiple specialized LLM roles run over the memory store:
  dedup     merge near-duplicate memories (embedding similarity + LLM confirm)
  evaluate  re-score importance of not-yet-scored memories
  verify    scan for contradictions, log findings (never auto-delete)
  enhance   incrementally (re)build the knowledge graph

Usage: python3 consolidate.py [all|dedup|evaluate|verify|enhance]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M

BASE = os.path.dirname(os.path.abspath(__file__))
REASON_STATE = os.path.join(BASE, "reason-state.json")


def _llm(prompt, max_tokens=600):
    return M.llm_chat([{"role": "user", "content": prompt}],
                      max_tokens=max_tokens, temperature=0.0)


def load_memories():
    with M.connect() as c:
        return c.execute(
            "SELECT id, text, kind, importance, embedding, metadata "
            "FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL"
        ).fetchall()


def dedup(threshold=0.92):
    rows = load_memories()
    n = len(rows)
    if n < 2:
        print("dedup: nothing to do")
        return
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = mat @ mat.T
    np.fill_diagonal(sims, 0)
    merged = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                verdict = _llm(
                    "Are these two memories duplicates (the same fact)? Answer ONLY yes or no.\n"
                    f"A: {rows[i]['text']}\nB: {rows[j]['text']}"
                ).strip().lower()
                if verdict.startswith("yes"):
                    keep = i if rows[i]["importance"] >= rows[j]["importance"] else j
                    drop = j if keep == i else i
                    with M.connect() as c:
                        c.execute(
                            "UPDATE memories SET merged=1, merged_into=? WHERE id=?",
                            (rows[keep]["id"], rows[drop]["id"]),
                        )
                        c.execute("DELETE FROM edges WHERE memory_id=?", (rows[drop]["id"],))
                        c.execute("DELETE FROM memory_entities WHERE memory_id=?", (rows[drop]["id"],))
                    merged += 1
                    print(f"  merged #{rows[drop]['id']} into #{rows[keep]['id']}: "
                          f"{rows[drop]['text'][:60]}...")
    print(f"dedup: merged {merged} near-duplicates")


def evaluate():
    rows = load_memories()
    changed = 0
    with M.connect() as c:
        for r in rows:
            md = json.loads(r["metadata"]) if r["metadata"] else {}
            if md.get("evaluated"):
                continue
            out = _llm(
                "Rate the importance of this memory to an AI agent's long-term identity "
                "and goals, from 0.0 (trivial) to 1.0 (core). Output ONLY a number.\n"
                f"Memory: {r['text']}"
            ).strip()
            try:
                imp = max(0.0, min(1.0, float(out)))
                md["evaluated"] = time.time()
                c.execute("UPDATE memories SET importance=?, metadata=? WHERE id=?",
                          (imp, json.dumps(md), r["id"]))
                changed += 1
            except ValueError:
                pass
    print(f"evaluate: re-scored {changed} memories")


def verify():
    rows = load_memories()
    texts = "\n".join(f"[{r['id']}] {r['text']}" for r in rows)
    out = _llm(
        "Scan these memories for direct contradictions (two statements that cannot both "
        "be true). List them as 'ID1 <-> ID2: why'. If none, say NONE.\n" + texts,
        max_tokens=800,
    ).strip()
    print("verify findings:\n" + out)
    with open(os.path.join(BASE, "verify.log"), "a") as f:
        f.write(f"--- {time.ctime()} ---\n{out}\n")
    _provenance_audit()


def _provenance_audit():
    """Deterministic provenance audit (P0-3) — no LLM, a direct DB walk.

    Counts live self-generated memories (self_generated_memory_ids) and flags
    the self-corroboration hazard: any memory that is origin='derived' AND whose
    ENTIRE ancestry is self-generated (no externally-observed root) may not be
    used to corroborate another memory.
    """
    self_gen_ids = M.self_generated_memory_ids()
    hazard_ids = []
    with M.connect() as c:
        for mid in self_gen_ids:
            row = c.execute("SELECT origin FROM memories WHERE id=?", (mid,)).fetchone()
            if row and (row["origin"] or "") == "derived":
                hazard_ids.append(mid)
    lines = [
        "provenance audit:",
        f"  live self-generated memories: {len(self_gen_ids)}"
        + (f" {self_gen_ids}" if self_gen_ids else ""),
        "  HARD RULE: self-generated memories may not corroborate other memories "
        "(entire ancestry lacks an externally-observed root).",
    ]
    if hazard_ids:
        lines.append(
            "  SELF-CORROBORATION HAZARDS (origin='derived' AND entirely "
            f"self-generated): {hazard_ids}"
        )
    else:
        lines.append(
            "  no self-corroboration hazards (no origin='derived' memory with an "
            "entirely self-generated ancestry)."
        )
    audit = "\n".join(lines)
    print("\n" + audit)
    with open(os.path.join(BASE, "verify.log"), "a") as f:
        f.write(audit + "\n")


def reason():
    """v2 reason worker — CABLE antecedent linking + ZSLP link prediction +
    GraphRAG community reports, with strict verification, provenance, dedup,
    and a reversible `derived` review table. See reason.py for the design."""
    import reason as R
    R.run()


def decay():
    M.decay()


def enhance():
    M.build_graph()


def causal():
    M.build_causal()


def belief_extract():
    """Belief-ledger worker: distill calibrated beliefs from memories/facts/derived."""
    import belief as B
    B.extract()


def belief_propagate():
    """Belief-ledger worker: propagate confidence over the causal graph (no LLM)."""
    import belief as B
    B.propagate()


def forecast_extract():
    """Prediction-ledger worker: mine dated falsifiable forecasts from new memories."""
    import prediction as P
    P.extract()


def forecast_resolve():
    """Prediction-ledger worker: auto-resolve due self-checkable forecasts and
    log surprise memories (no LLM, always safe to run)."""
    import prediction as P
    P.resolve_due(auto=True)
    P.surprise()


def person_model_extract():
    """Theory-of-Mind worker: mine the operator's mental-state claims from new memories."""
    import person_model as T
    T.extract()


def affect_extract():
    """Affective-tagging worker: tag new memories with valence/arousal."""
    import affect as A
    A.tag()


def run_all():
    dedup()
    evaluate()
    verify()
    reason()
    decay()
    enhance()
    causal()
    belief_extract()
    belief_propagate()
    forecast_extract()
    forecast_resolve()
    person_model_extract()
    affect_extract()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", nargs="?", default="all",
                   choices=["all", "dedup", "evaluate", "verify", "reason", "decay",
                            "enhance", "causal", "belief_extract", "belief_propagate",
                            "forecast_extract", "forecast_resolve", "person_model_extract", "affect_extract"])
    a = p.parse_args()
    {"all": run_all, "dedup": dedup, "evaluate": evaluate,
     "verify": verify, "reason": reason, "decay": decay, "enhance": enhance,
     "causal": causal, "belief_extract": belief_extract,
     "belief_propagate": belief_propagate, "forecast_extract": forecast_extract,
     "forecast_resolve": forecast_resolve, "person_model_extract": person_model_extract,
     "affect_extract": affect_extract}[a.step]()


if __name__ == "__main__":
    main()
