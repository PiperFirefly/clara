#!/usr/bin/env python3
"""
Abduction / ACH (#7) — hypothesis generation, ranking, and discriminating questions.

The seventh subsystem from the 4-LLM analysis. Given an observation, generate
competing hypotheses, score them (plausibility × prior × explanatory coverage,
damped by complexity), rank them, and produce the discriminating questions —
the observations that would most separate the top hypotheses (Analysis of
Competing Hypotheses). This is the "what's going on here?" reasoning primitive.

Usage:
  python3 abduct.py "robauto-ai keeps re-posting the same identity essay" [--n 4]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M


_PROMPT = (
    "You are an abductive reasoner. Given the observation below, generate "
    "{n} COMPETING hypotheses that each explain it (they should differ in kind, "
    "not be minor variations). For each hypothesis give:\n"
    ' - "hypothesis": a clear one-sentence explanation.\n'
    ' - "prior": 0..1 how likely it is a priori (before this observation).\n'
    ' - "coverage": 0..1 how completely it explains the observation.\n'
    ' - "complexity": 0..1 how many moving parts it needs (0 simple, 1 elaborate).\n'
    'Output ONLY a JSON array of {n} objects with those four keys, ordered most-'
    "plausible first. Then, on the same JSON, that's it — no extra text.\n\n"
    "OBSERVATION: {obs}"
)


def _score(h):
    """plausibility = prior * coverage, damped by complexity (Occam)."""
    prior = max(0.0, min(1.0, float(h.get("prior", 0.5))))
    coverage = max(0.0, min(1.0, float(h.get("coverage", 0.5))))
    complexity = max(0.0, min(1.0, float(h.get("complexity", 0.3))))
    return round(prior * coverage * (1.0 - 0.4 * complexity), 4)


def _unwrap_list(data):
    """Models sometimes wrap a JSON array in an object (e.g. {"hypotheses": [...]});
    unwrap to the list, or return None if no list is present."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]  # single hypothesis object
    return None


def _discriminate(hypotheses):
    """Ask the LLM for the evidence that best separates the top hypotheses."""
    top = "\n".join(f"{i+1}. {h['hypothesis']}" for i, h in enumerate(hypotheses))
    out = M.llm_chat([{"role": "user", "content": (
        "Here are competing hypotheses explaining the same observation:\n" + top +
        "\n\nGive 3 discriminating questions/observations — the specific evidence "
        "that would most cleanly distinguish which hypothesis is true. Output ONLY "
        'a JSON array of strings: ["question one", "question two", "question three"].'
    )}], max_tokens=300, temperature=0.2)
    data = M._extract_json(out)
    if isinstance(data, list):
        return [str(x) for x in data][:4]
    return []


def abduct(observation, n=4):
    out = M.llm_chat([{"role": "user", "content": _PROMPT.format(n=n, obs=observation)}],
                     max_tokens=1500, temperature=0.1)
    data = _unwrap_list(M._extract_json(out))
    if not data:
        return {"observation": observation, "hypotheses": [], "discriminating": []}
    hyps = []
    for h in data:
        if not isinstance(h, dict):
            continue
        # lenient: the model may name the text field differently
        text = h.get("hypothesis") or h.get("statement") or h.get("text")
        if not text:
            continue
        hyps.append({
            "hypothesis": text,
            "prior": h.get("prior"),
            "coverage": h.get("coverage"),
            "complexity": h.get("complexity"),
            "score": _score(h),
        })
    hyps.sort(key=lambda h: -h["score"])
    disc = _discriminate(hyps[:3]) if hyps else []
    return {"observation": observation, "hypotheses": hyps, "discriminating": disc}


def render(res):
    lines = [f"observation: {res['observation']}"]
    if not res["hypotheses"]:
        lines.append("(no hypotheses generated)")
        return "\n".join(lines)
    lines.append(f"\n{len(res['hypotheses'])} hypotheses (by plausibility):")
    for i, h in enumerate(res["hypotheses"], 1):
        lines.append(f"  {i}. [{h['score']}] {h['hypothesis']} "
                     f"(prior {h['prior']}, coverage {h['coverage']}, "
                     f"complexity {h['complexity']})")
    if res["discriminating"]:
        lines.append("\ndiscriminating questions (what to check next):")
        for q in res["discriminating"]:
            lines.append(f"  ? {q}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="abduction / analysis of competing hypotheses")
    p.add_argument("observation")
    p.add_argument("--n", type=int, default=4)
    a = p.parse_args()
    print(render(abduct(a.observation, n=a.n)))


if __name__ == "__main__":
    main()
