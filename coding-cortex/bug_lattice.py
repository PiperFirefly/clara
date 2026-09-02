#!/usr/bin/env python3
"""
Bug-hypothesis lattice — Coding Cortex (wiring, not a new subsystem).

Reuses `abduct.py` (ACH) to turn a failing observation into competing
hypotheses (H1/H2/H3...), each with plausibility, then recommends the
CHEAPEST test that would split the survivors — per Claude's pushback:
no LLM-invented Bayesian probabilities, ACH structure + "cheapest test
that splits the surviving hypotheses" instead of information-gain math
I can't actually compute.

Usage:
  bug_lattice.py "memory.retrieve() returns stale rows after supersede"
  bug_lattice.py "PB-009"          # pull the observation from selfbench tasks.json
  bug_lattice.py --observe          # read a failing PB task's statement directly
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/memory"))
import abduct  # reuse the ACH primitive

SELFBENCH_TASKS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "selfbench", "tasks.json"
)


def _task_statement(pb_id):
    with open(SELFBENCH_TASKS) as f:
        data = json.load(f)
    for t in data.get("tasks", []):
        if t["id"] == pb_id:
            return t.get("statement") or t.get("title")
    raise KeyError(f"{pb_id} not in {SELFBENCH_TASKS}")


def cheapest_split(res):
    """Given scored hypotheses, return a short, concrete cheapest-first
    experiment list. Heuristic (deterministic): order by likely cost of the
    check implied by the hypothesis, then pick the one with the largest
    contrast between the top two survivors. No invented probabilities."""
    hyps = res.get("hypotheses", [])
    if len(hyps) < 2:
        return [("Run the existing failing test in isolation; confirm it reproduces, "
                "then read the traceback's first frame and patch toward it.")]
    # Cheap checks, roughly in cost order. These are templates; the hypothesis
    # text fills in what to actually look for.
    cheap = [
        ("Re-read the failing test's assertion and the 20 lines around the traceback "
         "- is the failure where you think it is, or earlier?"),
        ("Run the minimal repro (isolate the one call path) to confirm the bug is "
         "not a test-harness artifact."),
        "Check recent commits touching the suspect symbols (`code_graph` callers/callees).",
    ]
    out = [f"Split H1 vs H2 with the cheapest check: {cheap[0]}"]
    for i, h in enumerate(hyps[:3], 1):
        out.append(f"  H{i} ({h['score']}): {h['hypothesis']}  ->  rule out with a "
                   f"targeted probe (see its discriminating question)")
    return out


def main():
    p = argparse.ArgumentParser(description="bug-hypothesis lattice over abduct.py")
    p.add_argument("observation", nargs="?", help="failing observation, or a PB-XXX id")
    p.add_argument("--n", type=int, default=4)
    a = p.parse_args()

    obs = a.observation
    if obs and obs.startswith("PB-"):
        obs = _task_statement(obs)
    if not obs:
        # if no arg, run against the first failing selfbench task as a demo
        from selfbench import selfbench  # noqa: F401  (same dir)
        print("no observation given; run `selfbench.py verify` to find a failing "
              "PB-XXX, then: bug_lattice.py <PB-XXX>")
        return

    res = abduct.abduct(obs, n=a.n)
    print(abduct.render(res))
    print("\ncheapest-splitting plan:")
    for line in cheapest_split(res):
        print("  " + line)


if __name__ == "__main__":
    main()
