#!/usr/bin/env python3
"""
Counterfactual reasoning — "what if" surgery on the causal graph.

The fourth subsystem from the 4-LLM cognitive-upgrade analysis. The
Belief/Prediction/ToM ledgers tell me *what I hold true*,
*what will happen*, and *what the operator thinks*; this tells me *what would have been
different*. It is a pure computation over the existing causal graph — no new
storage, nothing hypothetical ever written back into memory (a counterfactual
must not leak into my factual store).

Intervention semantics — nullification, NOT Pearl's do-operator: severing X's
outgoing edges asks "what if X had stopped causing anything?" (an
effects-of-causes nullification). It is NOT do(X=x), which severs X's *incoming*
edges to model an external intervention setting X to a value. Note: a directed
graph with confidence labels cannot produce *numerical* counterfactuals — that
requires structural equations / conditional distributions (an SCM), future work.
This tool reports only *qualitative* deltas: which downstream consequences
vanish when X (or a single X→Y edge) is nullified.

Usage:
  python3 counterfactual.py "what if the wallet bug hadn't happened" [--mode remove|sever] [--depth 2] [--k 8]
  python3 counterfactual.py "the wallet bug" --mode sever --edge-id 42
"""
import argparse
import json
import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M


def _load_edges():
    return M._load_causal_edges()


def _reachable(edges, seed_ids, depth, severed=frozenset()):
    """BFS cause→effect walk (skipping severed edge ids). Returns
    (reached, parent) where reached maps entity_id -> (depth, edge) and parent
    maps entity_id -> (prev_entity_id, edge) for chain reconstruction."""
    fwd = defaultdict(list)
    for e in edges:
        if e["id"] in severed:
            continue
        fwd[e["cause_id"]].append(e)
    visited = set(seed_ids)
    depth_of = {s: 0 for s in seed_ids}
    parent = {}
    q = deque(seed_ids)
    while q:
        cur = q.popleft()
        if depth_of[cur] >= depth:
            continue
        for e in fwd.get(cur, []):
            to = e["effect_id"]
            if to in visited or to in seed_ids:
                continue
            visited.add(to)
            depth_of[to] = depth_of[cur] + 1
            parent[to] = (cur, e)
            q.append(to)
    reached = {v: (depth_of[v], parent[v][1]) for v in visited if v not in seed_ids}
    return reached, parent


def _chain(parent, target, names):
    """Walk parent pointers back to a seed, returning the ordered cause→effect
    edge chain (reuses the same shape as memstore's causal chains)."""
    path = []
    cur = target
    while cur in parent:
        prev, e = parent[cur]
        path.append({
            "cause": names.get(e["cause_id"], "?"),
            "effect": names.get(e["effect_id"], "?"),
            "relation": e["rel"],
            "confidence": e["confidence"],
        })
        cur = prev
    path.reverse()
    return path


def counterfactual(query, mode="remove", depth=2, k=8, edge_id=None):
    seed_names = set(M.extract_entities(query))
    with M.connect() as c:
        all_ent = c.execute("SELECT id, name FROM entities").fetchall()
        names = {e["id"]: e["name"] for e in all_ent}
    seed_ids = M._match_entity_ids(seed_names, all_ent)
    if not seed_ids:
        return {"seed_entities": sorted(seed_names), "mode": mode,
                "severed_edges": [], "delta": [], "still_true": [], "note":
                "no matching entities found"}

    edges = _load_edges()

    # actual world
    actual, actual_parent = _reachable(edges, seed_ids, depth)

    # choose the intervention (nullification: sever outgoing edges)
    if mode == "remove":
        # sever ALL outgoing edges of the seed: X no longer causes anything.
        severed = {e["id"] for e in edges if e["cause_id"] in seed_ids}
    elif mode == "sever":
        out = [e for e in edges if e["cause_id"] in seed_ids]
        if edge_id is not None:
            target = [e for e in out if e["id"] == edge_id]
        else:
            out.sort(key=lambda e: -(e["confidence"] or 0))
            target = out[:1]
        severed = {e["id"] for e in target}
    else:
        severed = set()

    # counterfactual world
    cf, _cf_parent = _reachable(edges, seed_ids, depth, severed)

    delta_ids = set(actual) - set(cf)
    still_ids = set(cf) & set(actual)

    severed_edges = [{"id": e["id"], "cause": names.get(e["cause_id"], "?"),
                      "effect": names.get(e["effect_id"], "?"), "relation": e["rel"],
                      "confidence": e["confidence"]}
                     for e in edges if e["id"] in severed]

    def build(ids, parent_map):
        rows = []
        for ent in ids:
            d, _e = actual[ent] if ent in actual else (cf[ent][0], cf[ent][1])
            rows.append({
                "entity": names.get(ent, "?"),
                "depth": d,
                "chain": _chain(parent_map.get(ent) and parent_map, ent, names),
                "memories": M._memories_for_entity(ent),
            })
        rows.sort(key=lambda r: (r["depth"], -sum(x["confidence"] or 0 for x in r["chain"])))
        return rows[:k]

    delta = build(delta_ids, actual_parent)
    still = build(still_ids, actual_parent)

    return {
        "seed_entities": sorted(seed_names),
        "mode": mode,
        "severed_edges": severed_edges,
        "actual_effect_count": len(actual),
        "delta": delta,          # would vanish if the intervention held
        "still_true": still,     # survive the intervention (not downstream of severed edge)
    }


def render(res):
    out = []
    out.append(f"counterfactual: {', '.join(res['seed_entities'])} "
               f"[mode={res['mode']}]")
    if res.get("note"):
        out.append(res["note"])
        return "\n".join(out)
    if res["severed_edges"]:
        out.append("severed:")
        for e in res["severed_edges"]:
            out.append(f"  ✂ {e['cause']} -{e['relation'] or 'leads_to'}-> {e['effect']} "
                       f"(conf {e['confidence']})")
    out.append(f"\nIf this held, {len(res['delta'])} consequence(s) would vanish "
               f"(of {res['actual_effect_count']} actual downstream effects):")
    for d in res["delta"]:
        if d["chain"]:
            nodes = [d["chain"][0]["cause"]] + [c["effect"] for c in d["chain"]]
            chain = "→".join(nodes)
        else:
            chain = "(direct)"
        out.append(f"  ✗ {d['entity']} (depth {d['depth']}) via {chain}")
    if res["still_true"]:
        out.append(f"\nStill true regardless ({len(res['still_true'])}):")
        for s in res["still_true"]:
            out.append(f"  ✓ {s['entity']} (depth {s['depth']})")
    else:
        out.append("\nNothing survives — every downstream effect depended on this.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="counterfactual reasoning on the causal graph")
    p.add_argument("query")
    p.add_argument("--mode", default="remove", choices=["remove", "sever"])
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--edge-id", type=int, default=None)
    a = p.parse_args()
    print(render(counterfactual(a.query, mode=a.mode, depth=a.depth, k=a.k,
                                edge_id=a.edge_id)))


if __name__ == "__main__":
    main()
