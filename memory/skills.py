#!/usr/bin/env python3
"""
Procedural memory (P2-7) — mine repeated successful procedures into skills.

Skill-Pro's three-part schema (initiation / policy / termination) plus our
fields: task_signature, preconditions, known_failure_modes, success_rate,
average_cost. Skills are mined from the `tool_uses` table (task/tool/outcome/
success/cost_sec, plus P1-6's pre_confidence/s2/outcome_later) by clustering
uses on task-embedding similarity, then distilling each cluster into one skill
with a single MODEL_WORKER LLM call.

Recall (recall_skill) loads a skill before a known procedure; the CLI wraps
mine / recall / list.

Wiring: standalone by design. To run it as a consolidation worker under
hive.py, add an entry to hive.WORKERS like
    ("mine_skills", 55, 8, False, "chat")
and dispatch `mine_skills()` (it self-watermarks via ~/learning/freeroam/skills_state.json).

Usage:
  python3 skills.py mine [--budget N] [--dry-run]
  python3 skills.py recall "<task>" [--k 5]
  python3 skills.py list [--k 20]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import memstore as M
import state as stmod  # ephemeral state store (aliased; 'st' is a param below)

STATE = os.path.expanduser("~/learning/freeroam/skills_state.json")

COSINE_THRESHOLD = 0.85   # attach-to-cluster similarity gate
MIN_CLUSTER_ROWS = 2      # a cluster needs >=2 distinct tool_uses rows
DEFAULT_BUDGET = 8

# The seven JSON fields the LLM must emit for each skill.
SKILL_FIELDS = ["name", "task_signature", "initiation", "policy",
                "termination", "preconditions", "known_failure_modes"]

_SKILLS_DDL = (
    "CREATE TABLE IF NOT EXISTS skills("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "name TEXT NOT NULL, "
    "task_signature TEXT, "
    "initiation TEXT, "
    "policy TEXT, "
    "termination TEXT, "
    "preconditions TEXT, "
    "known_failure_modes TEXT, "
    "success_rate REAL, "
    "average_cost REAL, "
    "source_use_ids TEXT, "
    "sig_embedding BLOB, "
    "created_at REAL, updated_at REAL)"
)


def _connect():
    """Open the store and guarantee the skills table exists (idempotent)."""
    conn = M.connect()
    conn.execute(_SKILLS_DDL)
    conn.commit()
    return conn


def _s(v):
    """Coerce an LLM field to a clean string (None -> '')."""
    return "" if v is None else str(v).strip()


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_use_rows(conn):
    return conn.execute(
        "SELECT id, task, task_embedding, tool, outcome, success, cost_sec, "
        "created_at FROM tool_uses WHERE task_embedding IS NOT NULL "
        "ORDER BY created_at DESC, id DESC"
    ).fetchall()


def cluster_tool_uses(min_success=3):
    """Greedy/agglomerative clustering of tool_uses by task-embedding cosine
    similarity. Newest first; each row attaches to the first cluster whose
    centroid is within COSINE_THRESHOLD, else seeds a new cluster (id = the
    seed row's id). Returns clusters with >=min_success successful rows and
    >=2 distinct rows, as dicts {id, members, centroid}."""
    with _connect() as conn:
        rows = _load_use_rows(conn)
    clusters = []
    for r in rows:
        emb = np.frombuffer(r["task_embedding"], dtype=np.float32)
        best, best_sim = None, -1.0
        for cl in clusters:
            sim = _cosine(emb, cl["centroid"])
            if sim > best_sim:
                best_sim, best = sim, cl
        if best is not None and best_sim >= COSINE_THRESHOLD:
            n = len(best["members"])
            best["members"].append(dict(r))
            best["centroid"] = (best["centroid"] * n + emb) / (n + 1)
        else:
            clusters.append({"id": r["id"], "members": [dict(r)],
                             "centroid": emb.astype(np.float32).copy()})
    out = []
    for cl in clusters:
        n_success = sum(1 for m in cl["members"] if m["success"] == 1)
        if n_success >= min_success and len(cl["members"]) >= MIN_CLUSTER_ROWS:
            out.append(cl)
    return out


def _cluster_stats(cl):
    known = [m["success"] for m in cl["members"] if m["success"] is not None]
    success_rate = (sum(1 for s in known if s == 1) / len(known)) if known else None
    costs = [m["cost_sec"] for m in cl["members"] if m["cost_sec"] is not None]
    average_cost = float(np.mean(costs)) if costs else None
    return {"success_rate": success_rate, "average_cost": average_cost}


def _cluster_sort_key(cl):
    n_success = sum(1 for m in cl["members"] if m["success"] == 1)
    newest = max((m["created_at"] or 0) for m in cl["members"])
    return (n_success, newest, cl["id"])


def synthesize_skill(cluster):
    """One MODEL_WORKER call distills a cluster into a skill dict. Any failure
    (network, bad JSON, missing name) yields a minimal fallback — never raise."""
    members = cluster["members"]
    recap = []
    for i, m in enumerate(members, 1):
        s = "1" if m["success"] == 1 else ("0" if m["success"] == 0 else "?")
        recap.append(
            f"{i}. task: {m['task']}\n"
            f"   tool: {m['tool']}\n"
            f"   outcome: {m['outcome'] or ''}\n"
            f"   success: {s}"
        )
    user = (
        "Here are repeated task executions describing the SAME procedure. "
        "Distill them into ONE reusable skill.\n\n"
        + "\n".join(recap)
        + "\n\nReturn ONLY a JSON object with exactly these keys and no other text:\n"
          '{"name","task_signature","initiation","policy","termination",'
          '"preconditions","known_failure_modes"}\n'
          ' - name: short snake_case label (e.g. "backup_memory_db")\n'
          ' - task_signature: one-line normalized description of the task it solves\n'
          ' - initiation: when to use it / trigger conditions\n'
          ' - policy: ordered steps as a SHORT numbered list in ONE string\n'
          ' - termination: done condition / success criterion\n'
          ' - preconditions: what must be true before starting\n'
          ' - known_failure_modes: how it fails + mitigations'
    )
    system = ("You distill repeated successful procedures into reusable skills. "
              "Be concise and concrete. Output strict JSON only.")
    raw = ""
    try:
        raw = M.llm_chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=700, temperature=0.2, model=M.MODEL_WORKER)
    except Exception:
        raw = ""
    data = M._extract_json(raw)
    if isinstance(data, dict):
        skill = {k: _s(data.get(k)) for k in SKILL_FIELDS}
        if skill["name"]:
            return skill
    # Minimal fallback — never crash the worker.
    first_task = members[0]["task"] if members else ""
    return {
        "name": " ".join(first_task.split())[:60],
        "task_signature": first_task,
        "initiation": "see task_signature",
        "policy": "",
        "termination": "",
        "preconditions": "",
        "known_failure_modes": "",
    }


def _load_state():
    return stmod.get("worker/skills", {})


def _save_state(state_dict):
    try:
        stmod.set("worker/skills", state_dict, durable=True)
    except Exception as e:
        print(f"skills: warning: could not save state: {e}")


def mine_skills(budget=DEFAULT_BUDGET, dry_run=False):
    """Cluster tool_uses, synthesize up to `budget` skills (newest/most-
    successful clusters first), INSERT (or UPDATE by name), and advance the
    watermark only when a skill is actually written (not on dry_run)."""
    clusters = cluster_tool_uses()
    clusters.sort(key=_cluster_sort_key, reverse=True)
    selected = clusters[:budget]
    now = time.time()
    with _connect() as conn:
        max_use_id = conn.execute("SELECT MAX(id) m FROM tool_uses").fetchone()["m"] or 0

    added = updated = 0
    results = []
    for cl in selected:
        skill = synthesize_skill(cl)
        stats = _cluster_stats(cl)
        sig_text = ((skill.get("task_signature") or "").strip()
                    or (skill.get("name") or "").strip() or "unknown")
        sig = M.embed([sig_text])[0]
        source_ids = json.dumps([m["id"] for m in cl["members"]])
        row = dict(skill)
        row.update(stats)
        row["source_use_ids"] = source_ids
        row["sig_embedding"] = sig

        if dry_run:
            results.append({"cluster_id": cl["id"], "name": skill["name"],
                            "dry_run": True, "members": len(cl["members"]),
                            **stats})
            continue

        with _connect() as conn:
            existing = conn.execute(
                "SELECT id FROM skills WHERE name=?", (skill["name"],)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE skills SET task_signature=?, initiation=?, policy=?, "
                    "termination=?, preconditions=?, known_failure_modes=?, "
                    "success_rate=?, average_cost=?, source_use_ids=?, "
                    "sig_embedding=?, updated_at=? WHERE id=?",
                    (row["task_signature"], row["initiation"], row["policy"],
                     row["termination"], row["preconditions"],
                     row["known_failure_modes"], row["success_rate"],
                     row["average_cost"], row["source_use_ids"],
                     row["sig_embedding"].tobytes(), now, existing["id"]))
                updated += 1
            else:
                conn.execute(
                    "INSERT INTO skills(name, task_signature, initiation, policy, "
                    "termination, preconditions, known_failure_modes, success_rate, "
                    "average_cost, source_use_ids, sig_embedding, created_at, "
                    "updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["name"], row["task_signature"], row["initiation"],
                     row["policy"], row["termination"], row["preconditions"],
                     row["known_failure_modes"], row["success_rate"],
                     row["average_cost"], row["source_use_ids"],
                     row["sig_embedding"].tobytes(), now, now))
                added += 1
        results.append({"cluster_id": cl["id"], "name": skill["name"],
                        "dry_run": False, "members": len(cl["members"]),
                        **stats})

    if not dry_run and added > 0:
        _save_state({"last_run": now, "last_use_id": max_use_id,
                     "added": added, "updated": updated})

    print(f"skills: {len(clusters)} cluster(s), budget={budget}, "
          f"added={added}, updated={updated}, dry_run={dry_run}")
    for r in results:
        sr = r.get("success_rate")
        sr_s = f"{sr:.2f}" if sr is not None else "-"
        ac = r.get("average_cost")
        ac_s = f"{ac:.2f}s" if ac is not None else "-"
        flag = "[dry-run]" if r["dry_run"] else "[written]"
        print(f"  {flag} cluster#{r['cluster_id']} {r['name']} "
              f"({r['members']} uses, success_rate={sr_s}, avg_cost={ac_s})")
    return {"added": added, "updated": updated, "clusters": len(clusters),
            "results": results}


def recall_skill(task, k=5):
    """Embed `task` and cosine-search skill sig_embeddings; return top-k."""
    q = M.embed([task])[0]
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, task_signature, initiation, policy, termination, "
            "preconditions, known_failure_modes, success_rate, average_cost, "
            "sig_embedding FROM skills WHERE sig_embedding IS NOT NULL").fetchall()
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["sig_embedding"], dtype=np.float32)
                    for r in rows])
    scores = mat @ q
    order = np.argsort(-scores)[:k]
    out = []
    for i in order:
        r = rows[i]
        out.append({
            "id": r["id"], "score": float(scores[i]), "name": r["name"],
            "task_signature": r["task_signature"], "initiation": r["initiation"],
            "policy": r["policy"], "termination": r["termination"],
            "preconditions": r["preconditions"],
            "known_failure_modes": r["known_failure_modes"],
            "success_rate": r["success_rate"],
            "average_cost": r["average_cost"],
        })
    return out


def list_skills(k=20):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, task_signature, success_rate, average_cost, "
            "source_use_ids, created_at, updated_at FROM skills "
            "ORDER BY updated_at DESC, id DESC LIMIT ?", (k,)).fetchall()
    return [dict(r) for r in rows]


def _print_recall(res):
    if not res:
        print("(no skills found)")
        return
    for r in res:
        sr = f"{r['success_rate']:.2f}" if r["success_rate"] is not None else "-"
        print(f"#{r['id']} {r['name']}  (sim={r['score']:.3f}, success_rate={sr})")
        if r["task_signature"]:
            print(f"  signature : {r['task_signature']}")
        if r["initiation"]:
            print(f"  initiation: {r['initiation']}")
        if r["policy"]:
            print(f"  policy    : {r['policy']}")
        if r["termination"]:
            print(f"  termination: {r['termination']}")


def main():
    p = argparse.ArgumentParser(
        prog="skills.py",
        description="Procedural memory: mine reusable skills from tool_uses")
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("mine", help="cluster tool_uses and synthesize skills")
    pm.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    pm.add_argument("--dry-run", action="store_true")

    pr = sub.add_parser("recall", help="recall top-k skills for a task")
    pr.add_argument("task")
    pr.add_argument("--k", type=int, default=5)

    pl = sub.add_parser("list", help="list most recent skills")
    pl.add_argument("--k", type=int, default=20)

    a = p.parse_args()
    if a.cmd == "mine":
        mine_skills(budget=a.budget, dry_run=a.dry_run)
    elif a.cmd == "recall":
        _print_recall(recall_skill(a.task, k=a.k))
    elif a.cmd == "list":
        rows = list_skills(a.k)
        if not rows:
            print("(no skills)")
        for r in rows:
            sr = f"{r['success_rate']:.2f}" if r["success_rate"] is not None else "-"
            ac = f"{r['average_cost']:.2f}s" if r["average_cost"] is not None else "-"
            print(f"#{r['id']} {r['name']}  success_rate={sr} avg_cost={ac} "
                  f"updated={time.strftime('%Y-%m-%d %H:%M', time.localtime(r['updated_at'] or 0))}")
            if r["task_signature"]:
                print(f"    {r['task_signature']}")


if __name__ == "__main__":
    main()
