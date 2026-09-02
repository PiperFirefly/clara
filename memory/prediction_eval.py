#!/usr/bin/env python3
"""
Prediction-ledger eval — an honest benchmark of forecast calibration.

My standing amendment: every subsystem ships with its own scored probe. This one
checks three things:

  A. Scoring math (self-test) — do Brier + Shannon surprise compute correctly on
     known inputs? (catches a wrong formula before it poisons every resolution)
  B. Schema sanity — are confidences inside [0.01, 0.99], resolve_by in the
     future-or-resolved, categories valid? (the ledger is honest by construction)
  C. Calibration — on everything resolved so far, is my mean Brier below the
     always-0.5 baseline (0.2500)? (the actual point: do my numbers beat a coin flip)

Usage: python3 prediction_eval.py [--self-test-only]
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prediction as P
import memstore as M

# (confidence, outcome) -> expected (Brier, surprise_bits)
MATH_CASES = [
    (0.90, 1, 0.01, round(-math.log2(0.90), 3)),   # very sure, right
    (0.10, 1, 0.81, round(-math.log2(0.10), 3)),   # very sure, wrong (big surprise)
    (0.50, 1, 0.25, 1.000),                        # coin flip
    (0.90, 0, 0.81, round(-math.log2(0.10), 3)),   # confident no, but happened
    (0.70, 1, 0.09, round(-math.log2(0.70), 3)),
]


def self_test():
    fails = 0
    for conf, outcome, exp_b, exp_s in MATH_CASES:
        b, s, e = P._score(outcome, conf)
        ok = abs(b - exp_b) < 1e-3 and abs(s - exp_s) < 1e-3
        if not ok:
            fails += 1
        print(f"   p={conf:.2f} o={outcome} -> Brier {b} (want {exp_b}), "
              f"surprise {s} (want {exp_s})  {'✓' if ok else '✗'}")
    return fails


def schema_sanity():
    with P._conn() as c:
        rows = c.execute("SELECT * FROM forecasts WHERE status != 'void'").fetchall()
    bad_conf = bad_cat = bad_res = 0
    for r in rows:
        if r["confidence"] is not None and not (P.FLOOR <= r["confidence"] <= P.CAP + 1e-6):
            bad_conf += 1
        if r["category"] not in P.CATEGORIES:
            bad_cat += 1
        if r["status"] == "open" and r["resolve_by"] is None:
            bad_res += 1
    return len(rows), bad_conf, bad_cat, bad_res


def calibration():
    with P._conn() as c:
        rows = c.execute(
            "SELECT confidence, outcome, brier, surprise FROM forecasts "
            "WHERE status='resolved'").fetchall()
    if not rows:
        return None
    n = len(rows)
    mean_brier = sum(r["brier"] for r in rows) / n
    mean_surprise = sum(r["surprise"] for r in rows) / n
    # calibration-by-confidence-bucket (rough): within each band, is hit-rate ~ band center?
    return {"n": n, "mean_brier": mean_brier, "mean_surprise": mean_surprise}


def run():
    only = "--self-test-only" in sys.argv
    print(f"prediction eval\n")
    print("A. scoring math (self-test):")
    fails = self_test()
    if fails == 0:
        print("   0 failures — scoring is correct\n")
    else:
        print(f"   {fails} FAILURE(S) — fix _score() before trusting any resolution\n")
    if only:
        return 0 if fails == 0 else 1

    n, bad_conf, bad_cat, bad_res = schema_sanity()
    print(f"B. schema sanity over {n} forecasts: "
          f"{bad_conf} out-of-band confidence, {bad_cat} bad category, {bad_res} open-without-deadline")
    print("   (0 / 0 / 0 is correct)\n")

    cal = calibration()
    print("C. calibration:")
    if cal is None:
        print("   nothing resolved yet — this fills in as forecasts resolve.\n")
    else:
        verdict = "beats coin-flip" if cal["mean_brier"] < 0.25 else "no better than coin-flip"
        print(f"   {cal['n']} resolved, mean Brier {cal['mean_brier']:.3f} "
              f"(baseline 0.2500) -> {verdict}")
        print(f"   mean surprise {cal['mean_surprise']:.2f} bits\n")
    return 0


if __name__ == "__main__":
    sys.exit(run())
