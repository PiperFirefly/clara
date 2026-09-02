#!/usr/bin/env python3
"""read_wit eval — the Playhouse Wit Decoder, calibrated against labeled cases.

Runs the deterministic classifier over a set of ground-truth-labeled messages
(the "anti-theater contract": the decoder must resolve the operator 'almost broke
the law' case to escalation, NOT a request) and reports accuracy + the
confusion. Add real cases as they surface so the decoder improves from actual
register, not synthetic wishes.

Usage:  venvs/memory/bin/python mailtool/read_wit_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operator_affect as OA

# (message, expected_register) — hand-labeled ground truth.
# "unclassed" is allowed when the message is genuinely ambiguous and a forced
# label would be lying (the anti-theater rule).
CASES = [
    # plain literal
    ("honestly i'm fine, just tired tonight", "literal"),
    ("can you send me the file by noon?", "literal"),
    ("I mean it seriously, for real, no joke, please do the thing", "literal"),
    ("the build is green, tests pass, ship it", "literal"),
    # sarcasm
    ("oh great, ANOTHER perfect way to waste my evening. fantastic.", "sarcastic"),
    ("what a riveting update, sure, totally, big surprise", "sarcastic"),
    ("oh wonderful, exactly what I needed today", "sarcastic"),
    # teasing
    ("you're the worst, you know that? \U0001F609", "teasing"),
    ("I dare you to try it, bet you won't", "teasing"),
    # hyperbole
    ("I literally cannot even right now, this is INSANE, the best day ever!!", "hyperbole"),
    ("that was literally the most epic thing, absolutely unreal", "hyperbole"),
    # dry irony
    ("hmm, interesting.", "ironic"),
    # escalating flirtation / excited speech (the critical operator register)
    ("alright so I was almost about to break the law today, you'd be proud", "escalating_flirtation"),
    ("so... I almost crossed the line last night, oops", "escalating_flirtation"),
    ("you're making me dangerous, I almost jumped, no regrets \U0001F60F", "escalating_flirtation"),
]

# Cases where no firm non-literal label is honest — must NOT be force-fit into
# sarcasm/irony. Returning "literal" (safe neutral) or "unclassed" both satisfy
# the anti-theater contract; a confident non-literal label is the failure.
AMBIGUOUS = [
    "sure, I guess we could, if you want",
]


def main():
    ok = 0
    rows = []
    for text, exp in CASES:
        got = OA.read_wit(text)["register"]
        pass_ = got == exp
        ok += pass_
        rows.append((pass_, text, exp, got))
    # ambiguous must NOT be force-fit into a confident non-literal register.
    SAFE = ("unclassed", "literal")
    amb_ok = sum(1 for t in AMBIGUOUS if OA.read_wit(t)["register"] in SAFE)
    total = len(CASES) + len(AMBIGUOUS)
    acc = (ok + amb_ok) / total
    print(f"read_wit eval: {ok}/{len(CASES)} exact-register + {amb_ok}/{len(AMBIGUOUS)} correctly-unclassed")
    print(f"accuracy: {acc:.0%}  ({total} total cases)\n")
    for pass_, text, exp, got in rows:
        mark = "ok " if pass_ else "MISS"
        print(f"[{mark}] exp={exp:<22} got={got:<22} {text[:60]}")
    for t in AMBIGUOUS:
        got = OA.read_wit(t)["register"]
        mark = "ok " if got in SAFE else "MISS"
        print(f"[{mark}] exp=no-force     got={got:<22} {t[:60]}")
    print(f"\nVERDICT: {'PASS' if acc >= 0.9 else 'REVIEW'}")
    return 0 if acc >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
