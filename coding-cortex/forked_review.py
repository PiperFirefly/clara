#!/usr/bin/env python3
"""
Forked-perspective engineering review — Coding Cortex (ChatGPT's closing item).

Independent Agent instances with DELIBERATELY DIFFERENT OBJECTIVES review the
same proposed architectural change, then an adjudicator decides by EVIDENCE
STRENGTH, not majority vote.

This differs from debate.py (which argues for/against one position) and from a
council (which votes). Here each fork has a *different objective function* —
correctness, simplicity, security, reuse-first, performance — so they genuinely
disagree about what "good" means, and the adjudicator weighs their evidence.

Design:
  * each fork gets: the proposed change + its objective + "give concrete evidence,
    not vibes" (specific risk/finding, the code path it affects, severity).
  * adjudicator: reads all forks' findings, classifies each as
    confirmed | refuted | partial (by cross-fork agreement AND evidence quality),
    and gives an EVIDENCE-BASED verdict — NOT a count of who approved.
  * A finding confirmed by a security fork that the simplicity fork dismissed is
    ADOPTED if the security fork gave a concrete attack path. Votes don't decide;
    the strongest concrete evidence does.

Usage:
  forked_review.py "<proposed change>" [--forks correctness,simplicity,security,reuse,performance]
                     [--context "<context/transcript>"]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import memstore as M

# Each fork: a distinct objective. The prompt forces concrete evidence.
FORKS = {
    "correctness": (
        "You review for CORRECTNESS. Your objective: does this change actually do "
        "what it claims, under edge cases? Find concrete correctness failures: a "
        "specific input/state where it breaks, a race, an off-by-one, an invariant "
        "violated. Give the exact code path. Do NOT care about style or elegance."),
    "simplicity": (
        "You review for SIMPLICITY / maintainability. Your objective: is this the "
        "minimal change that solves the problem? Flag accidental complexity, "
        "needless abstraction, duplicated logic, over-engineering. Concrete: name "
        "the redundant part and the simpler alternative."),
    "security": (
        "You review for SECURITY. Your objective: does this create or expose an "
        "attack surface? Find concrete risks: injection, path traversal, untrusted "
        "input reaching a sink, missing auth, information leak, DoS. Give the "
        "attack path and severity. Concrete evidence only."),
    "reuse": (
        "You review for REUSE. Your objective: does this reinvent something that "
        "already exists in the codebase? Flag copy-pasted logic, parallel "
        "implementations, a capability that should be composed not rewritten. "
        "Name the existing implementation it should reuse."),
    "performance": (
        "You review for PERFORMANCE. Your objective: does this introduce a "
        "hot-path regression? Flag O(n^2) where O(n) is possible, unbounded "
        "growth, blocking calls in a loop, redundant work. Concrete: the code path "
        "and why it's slow."),
    "architecture": (
        "You review for ARCHITECTURE / dependency direction. Your objective: does "
        "this respect the dependency envelope and module boundaries? Flag new "
        "circular dependencies, wrong-layer imports, a component reaching into "
        "another's internals. Concrete: the violating edge."),
}


def _fork_prompt(fork_key, change, context):
    obj = FORKS[fork_key]
    ctx = f"\nCONTEXT:\n{context}\n" if context else ""
    return (
        "You are an independent reviewer with a SPECIFIC objective.\n"
        f"{obj}\n\n"
        "PROPOSED CHANGE:\n{change}\n"
        "{ctx}\n"
        "Output ONLY a JSON object:\n"
        '  - "verdict": "approve" | "flag" | "block"\n'
        '  - "findings": array of {{finding, evidence_path, severity(low|med|high)}}\n'
        '  - "reason": one-line summary.\n'
        "Base your verdict on concrete evidence for YOUR objective only. If there "
        "is nothing wrong from your perspective, approve with empty findings.\n"
    ).format(change=change, ctx=ctx)


ADJUDICATE_PROMPT = (
    "You are the adjudicator for an engineering review. Multiple independent "
    "reviewers, each with a DIFFERENT objective, reviewed the same proposed change. "
    "Decide by EVIDENCE STRENGTH, NOT by majority vote — a single security fork "
    "with a concrete attack path outweighs three forks that approved.\n\n"
    "PROPOSED CHANGE:\n{change}\n\n"
    "REVIEWER FINDINGS (JSON):\n{findings}\n\n"
    "Output ONLY JSON:\n"
    '  - "verdict": "adopt" | "revise" | "reject"\n'
    '  - "adopt": array of objective names whose evidence supports adoption\n'
    '  - "must_fix": array of {{objective, finding, severity}} that MUST be addressed\n'
    '  - "reason": short justification grounded in the concrete evidence.\n'
)


def run(change, forks=None, context=None, model=None):
    forks = forks or ["correctness", "simplicity", "security", "reuse", "performance"]
    findings = {}
    for key in forks:
        key = key.strip()
        if key not in FORKS:
            continue
        out = M.llm_chat([{"role": "user",
                           "content": _fork_prompt(key, change, context)}],
                         max_tokens=600, temperature=0.3, model=model)
        findings[key] = M._extract_json(out) or {"verdict": "error", "reason": str(out)[:200]}
    # adjudicate
    adj = M.llm_chat([{"role": "user",
                       "content": ADJUDICATE_PROMPT.format(
                           change=change, findings=json.dumps(findings))}],
                     max_tokens=700, temperature=0.1, model=model)
    verdict = M._extract_json(adj) or {"verdict": "error", "reason": str(adj)[:200]}
    return {"change": change, "forks": findings, "adjudication": verdict}


def main():
    p = argparse.ArgumentParser(description="forked-perspective engineering review")
    p.add_argument("change", nargs="+", help="the proposed architectural change")
    p.add_argument("--forks", default="correctness,simplicity,security,reuse,performance",
                   help="comma-separated objectives")
    p.add_argument("--context", default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    res = run(" ".join(a.change), forks=a.forks.split(","), context=a.context)
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(f"FORKS: {', '.join(a.forks.split(','))}")
    for k, f in res["forks"].items():
        v = f.get("verdict", "?")
        n = len(f.get("findings", []))
        print(f"  [{v.upper():<7}] {k:<14} {n} finding(s): {f.get('reason','')[:90]}")
    a_ = res["adjudication"]
    print(f"\nADJUDICATION: {a_.get('verdict','?').upper()}")
    for m in a_.get("must_fix", []):
        print(f"  MUST FIX [{m.get('severity','?')}] {m.get('objective','?')}: {m.get('finding','')[:100]}")
    print(f"  reason: {a_.get('reason','')}")


if __name__ == "__main__":

    main()
