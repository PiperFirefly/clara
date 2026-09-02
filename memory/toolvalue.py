#!/usr/bin/env python3
"""
Tool-Value Oracle — generalize tool selection from past outcomes.

Phase 0 (instrumentation) + Phase 1 (value estimator), per
tool-value-oracle-DESIGN.md. Imports DQN's "lookup -> generalize" idea into my
meta-memory: instead of tool_recall()'s pure k-NN retrieval (find similar tasks,
pool their success), this estimates a *per-tool-family expected value* for a new
task and adds a UCB exploration bonus so under-tried tools get a turn.

No schema change: arms are coarse tool *families* stored in the existing
tool_uses.tool column. numpy-only, consistent with memstore (no torch).

Usage:
  toolvalue.py log  "<task>" "<tool-name>" [--ok 0|1] [--cost 12.3]
  toolvalue.py value "<task>" [--k 5]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M

# Coarse arms: canonical tool name -> family. Unknown tools map to "other"
# (logged for provenance, excluded from value estimation).
FAMILIES = {
    "memory_recall": {"recall", "associate", "hippo", "causal", "causal_path",
                      "counterfactual", "fused", "search", "timeline", "around",
                      "as_of", "working_memory", "feeling", "belief", "tom", "person_model",
                      "logquery", "facts"},
    "memory_write": {"remember", "supersede", "tool_remember"},
    "web": {"web_search", "source_check", "fetch_content", "get_search_content"},
    "shell": {"bash"},
    "file": {"read", "write", "edit"},
    "code": {"code_check", "show_test", "code_graph"},
    "reason": {"route", "abduct", "forecast", "curiosity"},
    "secret": {"list_secrets", "get_secret"},
}
_NAME2FAMILY = {name: fam for fam, names in FAMILIES.items() for name in names}

UCB_C = 0.3        # exploration bonus scale — bounded to [0, UCB_C], same scale as p_success
COST_WEIGHT = 0.5  # lambda: how much expected cost discounts value


def family_of(tool):
    """Map a canonical tool name to its family; 'other' when unmapped."""
    return _NAME2FAMILY.get((tool or "").strip().lower(), "other")


def log_use(task, tool, success=None, cost_sec=None):
    """Phase 0: log a tool use with a canonical family arm and honest success.

    - `tool` is a canonical tool name (e.g. 'recall', 'web_search', 'bash').
    - success: 1 worked, 0 failed, None unknown (honest default — not 1).
    - cost_sec: measured elapsed; pass it, don't leave None.
    Returns the tool_uses id, or None if the family is 'other' (not logged as an
    arm, to avoid polluting the action space with singleton recipes).
    """
    fam = family_of(tool)
    if fam == "other":
        return None
    return M.tool_remember(task, fam, outcome=tool, success=success,
                           cost_sec=cost_sec)


def _value_from_rows(q, rows, k, explore, c, cost_weight):
    """Core estimator over raw rows (list of dicts). Pure numpy/float math so it
    is testable with synthetic data without touching the DB."""
    import numpy as np
    out = []
    fams = list(FAMILIES.keys())
    N = len(rows)
    by_fam = {f: [] for f in fams}
    for r in rows:
        f = r.get("tool")
        if f in by_fam:
            by_fam[f].append(r)

    # Expected cost per family (median), for cost-discounting.
    costs = {}
    for f in fams:
        cs = [r["cost_sec"] for r in by_fam[f] if r.get("cost_sec") is not None]
        costs[f] = (sorted(cs)[len(cs) // 2] if cs else 0.0)
    max_cost = max(costs.values()) or 1.0

    for f in fams:
        rs = by_fam[f]
        n = len(rs)
        if rs and N > 0:
            vecs = np.stack([np.frombuffer(r["task_embedding"], dtype=np.float32)
                             for r in rs])
            sims = np.clip(vecs @ q, 0.0, None)  # cosine sim, floor at 0
            # top-k within family, similarity-weighted success mean
            order = np.argsort(-sims)[:k]
            top_sim = sims[order]
            top_succ = np.array([rs[i]["success"] for i in order],
                                dtype=np.float64)
            known = ~np.isnan(top_succ)
            if known.any() and top_sim[known].sum() > 0:
                p_success = float((top_sim[known] * top_succ[known]).sum()
                                  / top_sim[known].sum())
            else:
                p_success = 0.5
        else:
            p_success = 0.5
        explore_bonus = (c / math.sqrt(n + 1.0)) if explore else 0.0
        est_cost = costs[f]
        value = p_success + explore_bonus - cost_weight * (est_cost / (max_cost + 1.0))
        out.append({"family": f, "n": n, "p_success": round(p_success, 3),
                    "est_cost": round(est_cost, 1),
                    "explore": round(explore_bonus, 3),
                    "value": round(value, 4)})
    out.sort(key=lambda x: -x["value"])
    return out


def tool_value(task, k=5, explore=True, c=UCB_C, cost_weight=COST_WEIGHT):
    """Phase 1: rank tool families by expected value for a task.

    With no logged data this returns pure exploration (uniform families ranked
    by UCB) — the correct cold-start behaviour to begin collecting balanced
    outcomes."""
    q = M.embed([task])[0]
    fams = list(FAMILIES.keys())
    qs = ",".join("?" for _ in fams)
    with M.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            f"SELECT task_embedding, tool, success, cost_sec FROM tool_uses "
            f"WHERE tool IN ({qs})", fams).fetchall()]
    ranked = _value_from_rows(q, rows, k, explore, c, cost_weight)
    n_total = len(rows)
    return {"task": task, "n_total": n_total,
            "mode": "exploit+explore" if n_total else "cold-start (explore only)",
            "ranking": ranked}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    lg = sub.add_parser("log")
    lg.add_argument("task")
    lg.add_argument("tool")
    lg.add_argument("--ok", type=int, choices=[0, 1])
    lg.add_argument("--cost", type=float)
    vl = sub.add_parser("value")
    vl.add_argument("task")
    vl.add_argument("--k", type=int, default=5)
    a = p.parse_args()

    if a.cmd == "log":
        tid = log_use(a.task, a.tool, success=a.ok, cost_sec=a.cost)
        if tid is None:
            print(f"tool '{a.tool}' -> family 'other'; not logged as an arm")
        else:
            print(f"logged #{tid} as family '{family_of(a.tool)}'")
    elif a.cmd == "value":
        r = tool_value(a.task, k=a.k)
        print(json.dumps(r, indent=1))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
