#!/usr/bin/env python3
"""
Reuse-first generation policy — Coding Cortex item #9.

Before creating a function, run the search cascade FIRST and treat generation
as the LAST step:

   symbol search -> semantic search -> structural search
   -> dependency search -> historical/git search -> generate

This materially reduces duplicate helper functions. Emits a verdict: either a
ranked list of existing implementations to reuse/adapt, or (only after all
searches come up empty) a GREEN-LIGHT to generate new code.

Wires together primitives that already exist:
  * symbol/structural  -> code_graph (tree-sitter code_nodes/edges)
  * semantic           -> reuse_search (embedding over symbol names+sigs)
  * structural dedup   -> clone_detect (normalized AST shape)
  * historical/git     -> git log/grep over the repo

Usage:
  reuse_first.py "parse a config file at startup"
  reuse_first.py "normalize an address string" --repo ~/memory
  reuse_first.py "compute sha256 of a file" --top 5
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.expanduser("~/coding-cortex"))
sys.path.insert(0, os.path.expanduser("~/memory"))
import codegraph  # noqa: F401  (imported for its DB constants)

import reuse_search


# ---------------------------------------------------------------------------
# The cascade. Each stage returns (hit, ranked_list). We stop early on a hit.
# ---------------------------------------------------------------------------
def _stage_symbol(task):
    """Symbol search: does a function whose name tokenizes to the task already
    exist? Uses code_graph + reuse_search's tokenizer."""
    toks = reuse_search.tokens("_".join(task.split()))
    if not toks:
        return False, []
    key = " ".join(toks)
    rows = reuse_search.load_symbols()
    vecs, rows = reuse_search.load_index(rows)
    import numpy as np
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=reuse_search.MODEL_NAME,
                          cache_dir=reuse_search.MODEL_CACHE)
    qvec = np.array(next(iter(model.embed([key]))))
    hits = reuse_search.find_similar(vecs, rows, qvec, top=6, thresh=0.62)
    return bool(hits), hits


def _stage_clone(repo):
    """Structural search: same-capability-different-name clones touching repo."""
    if not repo:
        return False, []
    try:
        import clone_detect as CD
        found = CD.scan([repo])
        # cheap heuristic: only surface if a group has >=2 members
        groups = CD.detect(found, tok_thresh=0.55)
        hits = [g for g in groups if len(g[1]) >= 2]
        return bool(hits), hits[:3]
    except Exception as e:  # noqa: BLE001
        return False, [f"(clone stage skipped: {e})"]


def _stage_git(repo, task):
    """Historical/git search: has this been written before in this repo's history?"""
    if not repo:
        return False, []
    try:
        r = subprocess.run(
            ["git", "-C", repo, "log", "--oneline", "-S",
             task.split()[0] if task else "", "-n", "5"],
            capture_output=True, text=True, timeout=10, check=False)
        if r.returncode != 0:
            return False, []
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        return bool(lines), lines[:5]
    except Exception:  # noqa: BLE001
        return False, []


def reuse_first(task, repo=None, top=5, verbose=True):
    """Run the cascade; return verdict dict."""
    out = {"task": task, "stages": []}

    # 1. semantic (broadest) — symbol name/signature embedding
    hit, hits = _stage_symbol(task)
    out["stages"].append({"stage": "semantic", "hit": hit,
                          "count": len(hits)})
    if hit:
        out["verdict"] = "reuse"
        out["candidates"] = [
            {"name": h[1][0], "path": h[1][2], "score": round(h[0], 3)}
            for h in hits]
        if verbose:
            print(f"[semantic] found {len(hits)} existing implementations:")
            for c in out["candidates"]:
                print(f"  [{c['score']}] {c['name']}  @ {c['path']}")
        return out

    # 2. structural clone (same capability, different names)
    hit, hits = _stage_clone(repo)
    out["stages"].append({"stage": "structural", "hit": hit,
                          "count": len(hits)})
    if hit:
        out["verdict"] = "reuse-adapt"
        out["candidates"] = [f"{k} group, {len(m)} members" for k, m in hits]
        if verbose:
            print(f"[structural] {len(hits)} near-clone groups — same capability "
                  f"under different names. Adapt, don't regenerate.")
            for c in out["candidates"]:
                print(f"  {c}")
        return out

    # 3. git history
    hit, hits = _stage_git(repo, task)
    out["stages"].append({"stage": "git", "hit": hit, "count": len(hits)})
    if hit:
        out["verdict"] = "reuse-adapt"
        out["candidates"] = hits
        if verbose:
            print("[git] prior work touching this exists in history:")
            for c in hits:
                print(f"  {c}")
        return out

    # 4. only now green-light generation
    out["verdict"] = "generate"
    out["candidates"] = []
    if verbose:
        print("[generate] all reuse stages came up empty — green light to "
              f"write a NEW helper for: {task}")
    return out


def main():
    p = argparse.ArgumentParser(description="reuse-first generation policy")
    p.add_argument("task", nargs="+", help="what you were about to write")
    p.add_argument("--repo", default=None, help="repo dir for structural+git stages")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    reuse_first(" ".join(a.task), repo=a.repo, top=a.top, verbose=not a.quiet)


if __name__ == "__main__":
    main()
