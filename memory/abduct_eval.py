#!/usr/bin/env python3
"""Abduction eval — structural: does it produce scored hypotheses + discriminating questions?

Abduction is generative, so the probe is structural: hypotheses present, scores
in [0,1] and ranked, discriminating questions present. (Quality is eyeballed.)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abduct as A

OBS = "robauto-ai keeps re-posting the same identity essay every couple days"


def run():
    res = A.abduct(OBS, n=4)
    hyps = res["hypotheses"]
    disc = res["discriminating"]
    n = len(hyps)
    scores_ok = all(isinstance(h.get("score"), float) and 0 <= h["score"] <= 1 for h in hyps)
    ranked = all(hyps[i]["score"] >= hyps[i + 1]["score"] for i in range(n - 1)) if n > 1 else True
    ok = n >= 1 and scores_ok and ranked and len(disc) >= 1
    print(f"abduction eval")
    print(f"   hypotheses generated: {n} (want >=1)  {'✓' if n >= 1 else '✗'}")
    print(f"   scores in [0,1] + ranked: {'✓' if scores_ok and ranked else '✗'}")
    print(f"   discriminating questions: {len(disc)} (want >=1)  {'✓' if len(disc) >= 1 else '✗'}")
    print(f"\n   OVERALL: {'✓ PASS' if ok else '✗ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
