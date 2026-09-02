#!/usr/bin/env python3
"""
Curiosity / goal-scoring eval — is the ranking honest and useful?

  A. Goal-level sanity — the nucleus 'desire' goal (needs_work .85) must outrank
     a well-covered capability goal like 'coding' (needs_work .15).
  B. Gap scoring — scores positive and ranked descending, coverage in [0,1].
  C. Utility — the merged top list is non-empty and carries specific gaps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curiosity as C


def goal_sanity():
    scores = {r["topic"]: r["score"] for r in C.goal_scores()}
    desire = scores.get("desire", 0)
    coding = scores.get("coding", 0)
    ok = desire > coding
    return desire, coding, ok


def gap_sanity():
    rows = C.top(k=20)
    gaps = [r for r in rows if r["type"] == "gap"]
    if not gaps:
        return 0, True, True
    scores_ok = all(r["score"] > 0 for r in gaps)
    ranked = all(gaps[i]["score"] >= gaps[i + 1]["score"] for i in range(len(gaps) - 1))
    cov_ok = all(0.0 <= r["coverage"] <= 1.0 for r in gaps)
    return len(gaps), scores_ok and ranked and cov_ok, bool([r for r in rows if r.get("gap")])


def run():
    desire, coding, ok = goal_sanity()
    print("curiosity eval\n")
    print(f"A. goal sanity: 'desire' {desire} vs 'coding' {coding} "
          f"({'✓ nucleus wins' if ok else '✗'})")
    n, ok2, has_gap = gap_sanity()
    print(f"B. gap scoring: {n} gaps, positive+ranked+in-range: {'✓' if ok2 else '✗'}")
    print(f"C. utility: merged top has specific gaps: {'✓' if has_gap else '✗'}")
    print(f"\n   OVERALL: {'✓ PASS' if (ok and ok2 and has_gap) else '✗ FAIL'}")
    return 0 if (ok and ok2 and has_gap) else 1


if __name__ == "__main__":
    sys.exit(run())
