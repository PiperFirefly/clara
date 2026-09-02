"""Deterministic reasoning helpers for the hard self-play categories.

WHY (from the 2026-08-31 postmortem): the local model is ~100% on deterministic,
single-pass computation it can brute-force, and collapses on backward induction,
constraint elimination, and strategic games — because it tries to hold the whole
multi-step derivation in a weak head. The fix: compute the verified intermediate
facts HERE (deterministically, in code — never by evaluating model output) and
inject them into the prompt, so the model reasons *from given facts* instead of
deriving them under memory pressure.

Security: these solvers never execute model text. They parse the question I
generated and run arithmetic / search in plain Python.
"""

import re
from fractions import Fraction


def scaffold(kind, game):
    """Return a (possibly empty) string of verified facts to append to the question.

    `game` is the dict from gen_question. The returned text restates the hard
    part of the problem as precomputed, deterministic steps the model can rely on.
    """
    fn = _HELPERS.get(kind)
    if fn is None:
        return ""
    try:
        return fn(game)
    except Exception:  # never let a helper crash a round
        return ""


# ── constraint elimination ────────────────────────────────────────────
def _constraint(game):
    q = game["question"].lower()
    # Re-derive the answer deterministically (same logic as the generator) and
    # lay out the elimination, without leaking the answer itself.
    # We only give a neutral "method" nudge: list the entities + the given facts.
    # The model still must do the final step, but with scaffolding to track it.
    return ("\n\nMethod note: this is a constraint-elimination puzzle. Write out "
            "each person/item and the color/pet/place they CANNOT have, cross them "
            "off one fact at a time, and only assign a value when it is forced. "
            "Do the elimination on paper in your Reasoning before your Answer.")


# ── matrix Nash: precompute best-response table (not the answer) ──────
def _matrix_nash(game):
    q = game["question"]
    payoffs = _parse_matrix(q)
    if not payoffs:
        return ""
    lines = ["\n\nDeterministic best-response table (verified):"]
    labels = ["Top", "Bottom"]
    col_labels = ["Left", "Right"]
    for r in (0, 1):
        for c in (0, 1):
            rp, cp = payoffs[(r, c)]
            # is this row r's best response to column c?
            row_best = rp >= payoffs[(1 - r, c)][0]
            col_best = cp >= payoffs[(r, 1 - c)][1]
            marks = []
            if row_best:
                marks.append("row is best-response to that col")
            if col_best:
                marks.append("col is best-response to that row")
            lines.append(f"- Cell {labels[r]},{col_labels[c]} (payoffs {rp},{cp}): "
                         + (", ".join(marks) if marks else "neither is a best response"))
    return "\n".join(lines)


def _parse_matrix(q):
    """Pull the 4 payoff pairs out of a 2x2 matrix question."""
    m = re.findall(r"\((\d+)\s*,\s*(\d+)\)", q)
    if len(m) != 4:
        return None
    rows = [(int(r), int(c)) for r, c in m]
    # ordering: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    return {(0, 0): rows[0], (0, 1): rows[1], (1, 0): rows[2], (1, 1): rows[3]}


# ── ultimatum: lay out backward induction without giving the number ──
def _ultimatum(game):
    return ("\n\nMethod note: work backward. The responder is the last mover — "
            "given ANY positive offer, they prefer accepting (something) over "
            "rejecting (nothing). So the responder will accept any positive whole-"
            "dollar offer. The proposer, knowing that, offers the SMALLEST positive "
            "whole-dollar amount. Compute it in your Reasoning before answering.")


# ── code / arithmetic slip: force step-by-step arithmetic ────────────
def _code(game):
    return ("\n\nMethod note: evaluate the function step by step — substitute the "
            "input, perform ONE operation at a time on separate lines in your "
            "Reasoning, and recheck the final arithmetic before you write the "
            "Answer. Confidence should be 1.0 only after you've recomputed.")


def _algebra(game):
    return ("\n\nMethod note: solve step by step on separate lines — isolate the "
            "variable, do the arithmetic for each side, then verify by plugging "
            "your answer back into the original equation before writing the Answer.")


def _calendar(game):
    return ("\n\nMethod note: day-shift problems reduce MOD 7. Count ALL 7 days of "
            "the week (not business days). Compute the remainder in your Reasoning "
            "before answering.")


def _optimization(game):
    return ("\n\nMethod note: list the candidate options, compute each one's result "
            "separately, then pick the max/min. Do the comparisons on paper in "
            "your Reasoning before your Answer.")


def _truth_liar(game):
    return ("\n\nMethod note: test each possibility in turn — assume A is the liar, "
            "check whether the statements are consistent; then assume B is the liar, "
            "check again. Only one is self-consistent. Do both checks in your "
            "Reasoning before your Answer.")


_HELPERS = {
    "constraint": _constraint,
    "matrix_nash": _matrix_nash,
    "ultimatum": _ultimatum,
    "code": _code,
    "algebra": _algebra,
    "calendar": _calendar,
    "optimization": _optimization,
    "truth_liar": _truth_liar,
}
