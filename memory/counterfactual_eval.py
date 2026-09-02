#!/usr/bin/env python3
"""
Counterfactual eval — verify the nullification surgery is correct.

Two parts:
  A. Self-test on a synthetic causal graph — does severing produce the exact
     expected delta? (catches a wrong traversal before it poisons reasoning)
  B. Real probes on the live causal graph — spot-check a few known seeds and
     eyeball that the vanished consequences are the right ones.

Usage: python3 counterfactual_eval.py [--self-test-only]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counterfactual as C


# Synthetic: A->B->C, A->D->E  (entity ids 1..5)
SYN_EDGES = [
    {"id": 1, "cause_id": 1, "effect_id": 2, "rel": "leads_to", "memory_id": 0, "confidence": 0.9},
    {"id": 2, "cause_id": 2, "effect_id": 3, "rel": "leads_to", "memory_id": 0, "confidence": 0.8},
    {"id": 3, "cause_id": 1, "effect_id": 4, "rel": "leads_to", "memory_id": 0, "confidence": 0.7},
    {"id": 4, "cause_id": 4, "effect_id": 5, "rel": "leads_to", "memory_id": 0, "confidence": 0.6},
]


def self_test():
    fails = 0
    seed = {1}
    # actual: B(2),C(3),D(4),E(5)
    actual, _ = C._reachable(SYN_EDGES, seed, 3)
    assert set(actual) == {2, 3, 4, 5}, f"actual wrong: {set(actual)}"
    print("   actual reachable: B,C,D,E  ✓")

    # remove: sever all outgoing of A (edges 1,3) -> nothing reachable
    severed = {e["id"] for e in SYN_EDGES if e["cause_id"] in seed}
    cf, _ = C._reachable(SYN_EDGES, seed, 3, severed)
    delta = set(actual) - set(cf)
    ok = set(cf) == set() and delta == {2, 3, 4, 5}
    if not ok:
        fails += 1
    print(f"   remove A: counterfactual={set(cf)} delta={delta}  "
          f"{'✓' if ok else '✗ (want cf=∅, delta=B,C,D,E)'}")

    # sever A->B (edge 1): D,E survive; B,C vanish
    severed = {1}
    cf, _ = C._reachable(SYN_EDGES, seed, 3, severed)
    delta = set(actual) - set(cf)
    still = set(cf) & set(actual)
    ok = delta == {2, 3} and still == {4, 5}
    if not ok:
        fails += 1
    print(f"   sever A->B: delta={delta} still={still}  "
          f"{'✓' if ok else '✗ (want delta=B,C, still=D,E)'}")
    return fails


def real_probes():
    probes = [
        "wallet passphrase bug",
        "open wallet standard bug",
        "trust but verify habit",
    ]
    for q in probes:
        print(f"\n=== counterfactual: '{q}' (remove) ===")
        try:
            res = C.counterfactual(q, mode="remove", depth=3, k=8)
            print(C.render(res))
        except Exception as e:
            print(f"   (error: {e})")


def run():
    only = "--self-test-only" in sys.argv
    print("counterfactual eval\n")
    print("A. self-test (synthetic graph A->B->C, A->D->E):")
    fails = self_test()
    print(f"   {'0 failures — surgery correct' if fails == 0 else str(fails) + ' FAILURE(S)'}\n")
    if only:
        return 0 if fails == 0 else 1
    print("B. real probes (eyeball the vanished consequences):")
    real_probes()
    return 0


if __name__ == "__main__":
    sys.exit(run())
