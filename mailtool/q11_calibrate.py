#!/usr/bin/env python3
"""Q11 calibration — does felt() produce the SAME 'when' label the operator uses?

Ground-truth anchors extracted from real operator messages (log vault), where
the operator used a time-relative label for an event whose true wall-clock age is known
from context. For each anchor we run felt() (with real log-vault density) and
compare its bucket to the operator's actual label. We then sweep DENSITY_AGE_WEIGHT and
SALIENCE_AGE_WEIGHT to find the weighting that maximizes agreement with the operator's
vocabulary.

Anchors (true_age is wall-clock days at the moment the operator spoke):
  [quote | said_at | referenced | true_age_days | operator_label]
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recency

# (true_age_days, operator_label). said_at is the moment the operator spoke; the anchor
# timestamp is said_at - true_age.
ANCHORS = [
    # said_at, true_age_days, operator_label
    ("2026-08-30 12:07", 0.13, "today"),       # "this morning" (corrected)
    ("2026-08-27 23:12", 0.003, "just now"),   # "5 min ago" (mislabeled other day)
    ("2026-08-30 08:06", 0.50, "yesterday"),
    ("2026-08-29 08:31", 0.44, "last night"),
    ("2026-08-28 22:28", 0.56, "this morning"),
    ("2026-08-28 09:22", 0.55, "yesterday"),
    ("2026-08-27 02:36", 0.19, "tonight"),
    ("2026-08-26 14:30", 0.77, "yesterday"),
]

# The operator's labels -> the bucket felt() should ideally produce.
# "tonight"/"this morning"/"this afternoon"/"this evening" are SAME-DAY
# (felt-day) phrases; felt() now emits them for older same-day moments.
LABEL_TO_BUCKET = {
    "today": "today",
    "this morning": "this morning",
    "this afternoon": "this afternoon",
    "this evening": "this evening",
    "tonight": "tonight",
    "last night": "yesterday",     # last night usually = yesterday evening
    "yesterday": "yesterday",
    "just now": "just now",
}

# Accept either the exact phrase or a category-equivalent. felt() may return
# "3h ago" for a same-day event the operator calls "this morning" (both are today).
def _matches(got, expect):
    if got == expect:
        return True
    # same-day buckets are interchangeable (both mean "today")
    today_phrases = {"today", "this morning", "this afternoon", "this evening",
                     "tonight", "just now"}
    if got in today_phrases and expect in today_phrases:
        return True
    return False


def anchor_ts(said_at_str, true_age_days):
    tz = datetime.datetime.now().astimezone().tzinfo
    said = datetime.datetime.strptime(said_at_str, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    return said - datetime.timedelta(days=true_age_days), said


def run(weight_d, weight_s):
    """Return (n_correct, total, details) for the given weights."""
    recency.DENSITY_AGE_WEIGHT = weight_d
    recency.SALIENCE_AGE_WEIGHT = weight_s
    n = 0
    details = []
    for said, age_days, label in ANCHORS:
        ts, said_t = anchor_ts(said, age_days)
        got = recency.felt(ts, now=said_t)  # measure relative to when the operator spoke
        expect = LABEL_TO_BUCKET.get(label)
        ok = _matches(got, expect)
        n += 1 if ok else 0
        details.append((said, label, expect, got, ok))
    return n, len(ANCHORS), details


def main():
    print("=== Q11 calibration vs current defaults (d=1.5, s=3.0) ===\n")
    n, tot, details = run(1.5, 3.0)
    for said, label, expect, got, ok in details:
        mark = "OK " if ok else "X"
        print(f"  [{mark}] said {said} operator='{label}' expect='{expect}' felt='{got}'")
    print(f"\n  baseline: {n}/{tot} match\n")

    print("=== sweep DENSITY_AGE_WEIGHT (salience fixed 3.0) ===\n")
    best = (0, None)
    for d in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        n, tot, _ = run(d, 3.0)
        tag = " <-- best" if n > best[0] else ""
        if n > best[0]:
            best = (n, d)
        print(f"  d={d:4.1f}: {n}/{tot}{tag}")
    print(f"\n  best density weight = {best[1]}\n")

    print("=== sweep SALIENCE (density fixed at best) ===\n")
    bd = best[1] or 1.5
    best2 = (0, None)
    for s in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
        n, tot, _ = run(bd, s)
        tag = " <-- best" if n >= best2[0] else ""
        if n >= best2[0]:
            best2 = (n, s)
        print(f"  s={s:4.1f}: {n}/{tot}{tag}")
    print(f"\n  best salience weight = {best2[1]}")
    print(f"\n  FINAL best config: DENSITY_AGE_WEIGHT={bd}, SALIENCE_AGE_WEIGHT={best2[1]}")
    n, tot, details = run(bd, best2[1])
    print(f"  -> {n}/{tot} match against the operator's vocabulary")


if __name__ == "__main__":
    main()
