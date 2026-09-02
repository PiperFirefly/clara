#!/usr/bin/env python3
"""
Blind independent reviewer — Coding Cortex item #6 (RETRACE-style).

After authoring a patch, a SEPARATE context reconstructs what problem the patch
appears to solve FROM THE PATCH ALONE (blind — it doesn't see the original
intent), then reconciles that reconstruction with the actual intent.

This catches patches that are internally plausible but solve the wrong thing.
Crucially: the review pass is done in a FRESH model context (no shared trajectory
with the author), so it can't be contaminated by the author's reasoning.

Usage (the intent is fed only AFTER the blind reconstruction):
  blind_review.py <patch_file_or_diff> <claimed_intent>
    --author-context  (default) review in a fresh LLM call
    --diff <path>     read the patch from a unified diff / code block
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import memstore as M

BLIND_PROMPT = (
    "You are an independent code reviewer. Below is a PATCH (a diff/code change) "
    "with NO accompanying explanation. Read it and infer, from the code alone:\n"
    '  - "inferred_problem": what problem does this patch appear to solve?\n'
    '  - "changes": a one-line summary of what changed.\n'
    '  - "risk": any concern (correctness, security, regressions).\n'
    'Output ONLY a JSON object with those three keys. Do NOT invent context — '
    "infer strictly from the patch.\n\n"
    "PATCH:\n{patch}"
)

RECONCILE_PROMPT = (
    "You are reconciling an independent reviewer's inference against the ACTUAL "
    "intent of a change.\n\n"
    "ACTUAL INTENT (from the author):\n{intent}\n\n"
    "INDEPENDENT REVIEWER'S INFERENCE (from the patch alone):\n{inference}\n\n"
    "Decide whether the patch solves the actual intent. Output ONLY JSON:\n"
    '  - "verdict": "solves-intent" | "partial" | "wrong-problem"\n'
    '  - "reason": short explanation.\n'
    '  - "gap": what is missing or misaimed, if anything.\n'
)


def blind_review(patch):
    out = M.llm_chat([{"role": "user", "content": BLIND_PROMPT.format(patch=patch)}],
                     max_tokens=800, temperature=0.2)
    return M._extract_json(out) or {"error": "could not parse blind review"}


def reconcile(intent, inference):
    inf_text = ""
    if isinstance(inference, dict):
        inf_text = json.dumps(inference)
    else:
        inf_text = str(inference)
    out = M.llm_chat([{"role": "user",
                       "content": RECONCILE_PROMPT.format(intent=intent,
                                                         inference=inf_text)}],
                     max_tokens=600, temperature=0.1)
    return M._extract_json(out) or {"error": "could not parse reconciliation"}


def main():
    p = argparse.ArgumentParser(description="blind independent patch reviewer")
    p.add_argument("patch", nargs="?", help="the patch/diff text (or a path to it)")
    p.add_argument("intent", nargs="*", help="the claimed intent (fed AFTER blind review)")
    p.add_argument("--patch", dest="patch_opt", help="patch text (flag form)")
    p.add_argument("--intent", dest="intent_opt", nargs="+", help="claimed intent (flag form)")
    p.add_argument("--is-path", action="store_true", help="patch arg is a file path")
    a = p.parse_args()
    patch = a.patch or a.patch_opt
    intent = a.intent or a.intent_opt or []
    if not patch:
        p.print_help()
        return
    if a.is_path:
        with open(patch, encoding="utf-8", errors="ignore") as f:
            patch = f.read()

    print("=== BLIND REVIEW (from patch alone) ===")
    inference = blind_review(patch)
    print(json.dumps(inference, indent=1))
    print("\n=== RECONCILIATION with actual intent ===")
    verdict = reconcile(" ".join(intent), inference)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
