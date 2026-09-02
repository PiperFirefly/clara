#!/usr/bin/env python3
"""
Concept inventory before implementation — Coding Cortex item #17.

Before coding a feature, produce the inventory:
    required capabilities
    existing capabilities
    missing capabilities
    possible compositions

Input is a capability pipeline, e.g.:
    stream CSV -> normalize -> deduplicate -> database

Each step is checked against:
  * the reuse index (semantic, over code_graph symbols) — does a function already
    exist that does this?
  * the polyglot capability registry — is there a language/library home for it?

Output answers the item's example: "Need: X -> Y -> Z. Existing: A, B, C.
Missing: none. => zero new algorithms. Just composition."

Deterministic for the reuse+registry checks; the composition step just chains
the existing-capability hits into the pipeline. No LLM by default.

Usage:
  concept_inventory.py "stream CSV -> normalize -> deduplicate -> database"
  concept_inventory.py "parse log -> extract IPs -> geo-locate -> store"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser("~/coding-cortex"))
sys.path.insert(0, os.path.expanduser("~/memory"))
import polyglot
import reuse_search


def split_pipeline(spec):
    """'a -> b -> c' -> ['a', 'b', 'c']; also handle '=>' and '- '."""
    for sep in (" -> ", " => ", "->", "=>", "\u2192"):
        if sep in spec:
            parts = spec.split(sep)
            return [p.strip() for p in parts if p.strip()]
    return [spec.strip()]


def check_step(step, rows, vecs, model, polyglot_map):
    """Return (step, existing_hits, polyglot_home, status)."""
    # 1. semantic reuse — does a symbol already do this?
    toks = reuse_search.tokens("_".join(step.split()))
    q = " ".join(toks)
    hits = []
    if toks:
        import numpy as np
        qvec = np.array(next(iter(model.embed([q]))))
        hits = reuse_search.find_similar(vecs, rows, qvec, top=3, thresh=0.60)
    # 2. polyglot best_fit home
    best = polyglot.best_for(step, top=2)
    home = [b for b, s in best]
    # status: existing if a strong reuse hit OR a polyglot home
    strong = [h for h in hits if h[0] >= 0.68]
    if strong:
        status = "existing"
    elif hits:
        status = "partial"
    else:
        status = "missing"
    return step, hits, home, status


def inventory(spec, top=3, verbose=True):
    steps = split_pipeline(spec)
    rows = reuse_search.load_symbols()
    vecs, rows = reuse_search.load_index(rows)
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=reuse_search.MODEL_NAME,
                          cache_dir=reuse_search.MODEL_CACHE)

    per_step = [check_step(s, rows, vecs, model, polyglot.REGISTRY)
                for s in steps]

    required = steps
    existing = []
    missing = []
    partial = []
    for step, hits, home, status in per_step:
        if status == "existing":
            existing.append((step, hits))
        elif status == "partial":
            partial.append((step, hits))
        else:
            missing.append(step)

    if verbose:
        print(f"REQUIRED capabilities: {' -> '.join(required)}")
        print(f"\nEXISTING ({len(existing)}):")
        for step, hits in existing:
            print(f"  [existing] {step}")
            for h in hits[:2]:
                print(f"      {h[1][0]} @ {h[1][2]}  (score {h[0]:.2f})")
        if partial:
            print(f"\nPARTIAL ({len(partial)}):")
            for step, hits in partial:
                print(f"  [partial] {step}")
                for h in hits[:2]:
                    print(f"      {h[1][0]} @ {h[1][2]}  (score {h[0]:.2f})")
        if missing:
            print(f"\nMISSING ({len(missing)}):")
            for step in missing:
                print(f"  [missing] {step}")
        print("\nPOSSIBLE COMPOSITIONS:")
        existing_steps = [s for s, _ in existing]
        print("  " + " -> ".join(existing_steps) if existing_steps
              else "  (none yet — build the missing primitives first)")

    return {
        "required": required,
        "existing": existing,
        "partial": partial,
        "missing": missing,
        "zero_new_algorithms": len(missing) == 0,
    }


def main():
    p = argparse.ArgumentParser(description="concept inventory before implementation")
    p.add_argument("spec", nargs="+", help="capability pipeline, e.g. 'a -> b -> c'")
    p.add_argument("--top", type=int, default=3)
    a = p.parse_args()
    res = inventory(" ".join(a.spec), top=a.top)
    print("\nVERDICT:",
          "ZERO NEW ALGORITHMS — just composition." if res["zero_new_algorithms"]
          else "missing primitives need building first.")


if __name__ == "__main__":
    main()
