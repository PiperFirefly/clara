#!/usr/bin/env python3
"""System-1/2 routing eval — does the heuristic classify a labeled task set correctly?"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route as R

CASES = [
    ("summarize this log", "S1"),
    ("design a new subsystem", "S2"),
    ("check bridge status", "S1"),
    ("why did the hive crash", "S2"),
    ("list open forecasts", "S1"),
    ("decide whether to apply the update", "S2"),
    ("tag these memories", "S1"),
    ("refactor the memory schema", "S2"),
    ("recall what operator said", "S1"),
    ("weigh the trade-off between two architectures", "S2"),
]


def run():
    ok = 0
    for t, exp in CASES:
        got = R.route(t)["level"]
        ok += got == exp
        print(f"   {'✓' if got == exp else '✗'} {got} (want {exp}) {t}")
    print(f"\nroute eval: {ok}/{len(CASES)} correct")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(run())
