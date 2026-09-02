#!/usr/bin/env python3
"""
Spec compiler — Coding Cortex item #4 (correctness-by-construction).

Before modifying code, transform a user request into a machine-oriented contract:
    USER INTENT
        ↓
    BEHAVIORAL REQUIREMENTS  (R1..Rn)
        ↓
    INVARIANTS               (logical constraints that must hold)
        ↓
    ACCEPTANCE TESTS         (T1..Tn, each a concrete scenario)

This prevents the most expensive agent failure: solving a slightly different
problem from the one requested. The compiled spec is structured JSON so a later
step (patch dossier, test generation, blind reviewer) can consume it.

The transformation is an LLM pass (it's a semantic re-expression of intent), but
the OUTPUT is strict JSON, auditable, and versioned — not free-form prose.

Usage:
  spec_compiler.py "Don't let Doctor run when I'm actively talking to Agent"
  spec_compiler.py "make memory.retrieve return the current (non-superseded) row"
  spec_compiler.py <intent> --out spec.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import memstore as M

PROMPT = (
    "You are a requirements engineer. Turn the USER INTENT below into a precise, "
    "machine-checkable specification. Output ONLY a JSON object with exactly these keys:\n"
    '  "requirements":  [array of "R1: ..." strings, each a single behavioral requirement]\n'
    '  "invariants":    [array of logical constraints, each a boolean expression in '
    'plain English that MUST always hold]\n'
    '  "acceptance_tests": [array of "T1: <scenario> — <expected result>" strings, each a '
    'concrete testable scenario]\n'
    '  "assumptions":   [array of assumptions you made about the environment/inputs]\n'
    "Do not invent requirements that aren't implied by the intent. Be precise and "
    "minimal — every requirement and test must be individually checkable.\n\n"
    "USER INTENT: {intent}"
)


def compile_spec(intent, model=None):
    out = M.llm_chat([{"role": "user", "content": PROMPT.format(intent=intent)}],
                     max_tokens=1600, temperature=0.1, model=model)
    data = M._extract_json(out)
    if not isinstance(data, dict):
        return {"intent": intent, "error": "could not parse structured spec"}
    return {
        "intent": intent,
        "requirements": data.get("requirements", []),
        "invariants": data.get("invariants", []),
        "acceptance_tests": data.get("acceptance_tests", []),
        "assumptions": data.get("assumptions", []),
    }


def render(spec):
    L = [f"SPEC for: {spec.get('intent','')}", ""]
    if "error" in spec:
        L.append(f"ERROR: {spec['error']}")
        return "\n".join(L)
    for label, key in [("BEHAVIORAL REQUIREMENTS", "requirements"),
                       ("INVARIANTS", "invariants"),
                       ("ACCEPTANCE TESTS", "acceptance_tests"),
                       ("ASSUMPTIONS", "assumptions")]:
        items = spec.get(key, [])
        L.append(f"{label} ({len(items)}):")
        for it in items:
            L.append(f"  - {it}")
        L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description="spec compiler (intent -> contract)")
    p.add_argument("intent", nargs="*", help="the user request, in your own words")
    p.add_argument("--intent", dest="intent_opt", nargs="+", help="intent (flag form)")
    p.add_argument("--out", default=None, help="write spec JSON to a path")
    p.add_argument("--model", default=None, help="override model")
    a = p.parse_args()
    intent = a.intent or a.intent_opt or []
    spec = compile_spec(" ".join(intent), model=a.model)
    print(render(spec))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(spec, f, indent=1)
        print(f"\n(wrote spec to {a.out})")


if __name__ == "__main__":
    main()
