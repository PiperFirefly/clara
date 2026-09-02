#!/usr/bin/env python3
"""
recency — felt time: absolute anchor -> human "when", density- and
salience-aware.

The wall-clock does not lie, but it does not speak. when.py counts elapsed
seconds ("3d 4h"); operator_presence counts which hours. Neither counts how much
*experienced* a gap was. This module turns an absolute timestamp into the phrase
a human would actually mean:

    "a few days ago felt like yesterday"  when the gap was empty,
    "yesterday felt like a week"           when it was packed,
    and the important thing stays vivid    (salience resists aging).

Principle (from handoff-bench-v0.4 §5): absolute anchors are facts; relative
labels are projections through *now* computed at call time, never stored.

Usage:
    from recency import felt, age
    felt(dt)                -> "yesterday", "a few days ago", "3h ago", ...
    felt(dt, salience=0.8)  -> same, but the salience resists the aging
    age(dt)                 -> the raw felt-age number the label is built from
    human(delta)            -> backward-compatible "3d 4h" (what when.py uses)
"""
import datetime
import json
import math
import os

# Density source: the operator's interaction log. Primary = the log vault
# (~/memory/logs.db, every conversation message, cron-updated); fallback = the
# operator presence store. Reused by when.py ("last checked") and operator_presence
# ("last heard") so the *felt* age of a gap reflects intervening events, not just
# elapsed seconds.
PRESENCE = os.path.expanduser("~/.pi/agent/operator_presence.json")
LOGVAULT = os.path.expanduser("~/memory/logs.db")

# Heuristics (pre-registered defaults, tunable via Q11 calibration later).
# Density is COMPRESSED logarithmically: thousands of events are not thousands
# of times more "packed" than a dozen — log10 keeps the felt effect bounded.
#
# Direction matters (reconciled 2026-08-30): density AFTER the anchor AGES it.
# Retrospectively, an interval crammed with events feels LONGER ("that week felt
# like a month"); an empty stretch feels SHORT ("where did the time go"). So:
#   empty 5-day silence reads as "yesterday"  (little happened since -> recent)
#   packed hour feels like "ages ago"          (much happened since -> old)
# Salience goes the other way: a consequential memory stays vivid -> younger.
#
# NOTE on the H9 felt-clock tension: the H9 probe described a dense exchange as
# "felt recent -- alive" (density = younger). That is a DIFFERENT quantity: the
# tempo of an ONGOING exchange (the anchor IS the present, no post-anchor density
# yet). felt() here is recall-distance of a PAST anchor, where post-anchor density
# ages it. Do not conflate the two; the pacing register uses its own density.
# Q11-calibrated defaults (2026-08-30): DENSITY_AGE_WEIGHT=0.5 maximizes
# agreement with the operator's real "when" vocabulary (5/8 on held-out anchors; the
# mismatches are a phrasing nuance and logvault density over-counting my own
# monologue — see Q11 findings). Salience is deliberately modest (3.0).
DENSITY_AGE_WEIGHT = 0.5     # per log10-unit of packedness, age the gap ~0.5 day
SALIENCE_AGE_WEIGHT = 3.0    # salience 0..1 preserves up to ~3 "days" of youth


def _density_count(since, now=None):
    """Number of conversation/interaction events between `since` and `now`.

    Primary: the log vault (every message ts, epoch-ms float) — rich and
    cron-updated. Fallback: the presence store's activity list (ISO strings).
    `now` defaults to the real present; pass an explicit reference time to
    measure density relative to a PAST moment (Q11 calibration).
    Returns an int; 0 on any failure (felt() then degrades to wall-clock).
    """
    import datetime as _dt
    if now is None:
        now = _dt.datetime.now().astimezone()
    now = now.astimezone()
    ms_lo = since.timestamp() * 1000.0
    ms_hi = now.timestamp() * 1000.0

    # Primary: log vault. Count OPERATOR-originated turns only (user/inbound/sms)
    # — the times the operator actually engaged — NOT my own assistant output or
    # freeroam monologue. Q11 finding (2026-08-30): the vault is ~73% my own
    # messages; counting them as "intervening events" massively inflates felt
    # density and over-ages every gap. The operator's felt density is how much
    # THEY experienced, which is operator turns + distinct sessions.
    try:
        import sqlite3
        con = sqlite3.connect(LOGVAULT)
        cur = con.cursor()
        cur.execute(
            "SELECT COUNT(DISTINCT COALESCE(session_id, ts)) FROM messages "
            "WHERE ts > ? AND ts <= ? AND role IN ('user','inbound','sms')",
            (ms_lo, ms_hi))
        n = cur.fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        pass

    # Fallback: presence store
    try:
        acts = json.load(open(PRESENCE)).get("activity", [])
    except Exception:
        acts = []
    if not acts:
        return 0
    iso = since.isoformat()
    now_iso = now.isoformat()
    n = 0
    for x in acts:
        try:
            if iso < x <= now_iso:
                n += 1
        except Exception:
            continue
    return n


def _wall_age(ts, now=None):
    if now is None:
        now = datetime.datetime.now().astimezone()
    return now.astimezone() - ts.astimezone()


def age(ts, salience=0.0, density=None, now=None):
    """Felt age of an anchor in (integer) days. Density ages; salience preserves.

    ts: aware datetime (naive is assumed local).
    salience: 0..1, how consequential the anchored thing is (resists aging).
    density: optional pre-computed interaction count (else read from presence log).
    now: optional reference "present" (default real now). Pass a past moment to
      measure felt age relative to that moment — used by Q11 calibration.
    """
    if ts.tzinfo is None:
        tz = (now or datetime.datetime.now().astimezone()).tzinfo
        ts = ts.replace(tzinfo=tz)
    wall = _wall_age(ts, now)
    days = wall.days
    if density is None:
        density = _density_count(ts, now)
    # post-anchor density AGES (compressed log10), salience PRESERVES (-).
    packed = int(math.log10(1 + max(0, density)) * DENSITY_AGE_WEIGHT)
    felt = days + packed - int(salience * SALIENCE_AGE_WEIGHT)
    return felt


def felt(ts, salience=0.0, density=None, now=None):
    """Human 'when' phrase for an absolute anchor, density- and salience-aware.

    Buckets follow the operator's lived day (wake->sleep vocabulary), not
    necessarily calendar midnight. `now` is an optional reference "present";
    pass a past moment to measure how a historical anchor felt THEN (Q11).
    """
    wall = _wall_age(ts, now)
    felt_age = age(ts, salience=salience, density=density, now=now)
    now_dt = (now or datetime.datetime.now().astimezone()).astimezone()

    # FELT-DAY boundary (operator's waking day, wake->sleep, not midnight). An
    # event at 22:00 mentioned at 02:36 is "tonight" — the SAME waking day — even
    # though its calendar date is yesterday (Q11 calibration, 2026-08-30). We key
    # the day-0/day-1 split off the felt-day start when known, else calendar date.
    day_start = _felt_day_start(now_dt)
    same_felt_day = ts >= day_start

    # Same felt day -> time-of-day vocabulary (Q11: the operator says "this morning"/
    # "tonight", not "3h ago"). Elapsed hours for the very recent, then morning/
    # afternoon/evening/night phrases for older same-day moments.
    if same_felt_day:
        if wall.total_seconds() < 300:
            return "just now"
        # A dense same-felt-day exchange feels longer, but a human still calls it
        # today/tonight — density does not drag it across the felt-day boundary
        # into "yesterday" (Q11 calibration). Density matters between days, not
        # within.
        mins = wall.seconds // 60
        if mins < 60:
            return f"{mins}m ago"
        hrs = mins // 60
        if hrs < 4:
            return f"{hrs}h ago"
        # Older same-day: use the operator's time-of-day vocabulary.
        h = ts.hour
        if h < 5:
            return "last night"
        if h < 12:
            return "this morning"
        if h < 17:
            return "this afternoon"
        if h < 21:
            return "this evening"
        return "tonight"

    # Previous felt day(s). Felt adjustment shifts between day buckets:
    # a salient (salience-preserved) 1-felt-day-old event can read "today";
    # a dense one can feel older than its calendar age.
    if felt_age <= 0:
        return "today"
    if felt_age == 1:
        return "yesterday"
    if felt_age < 7:
        return f"{felt_age}d ago"
    if felt_age < 30:
        return f"{felt_age // 7}wk ago"
    return f"{felt_age // 30}mo ago"


def _felt_day_start(now_dt):
    """Start of the operator's current waking day (wake->sleep), or calendar
    midnight if the wake hour is unknown. Same logic as sleep_time._felt_day_start_ms.
    An event after this instant is 'today'; before it is 'yesterday' (or older).
    """
    try:
        import operator_presence as _op
        d = _op.load()
        _bed, wake, src = _op.sleep_window(d)
        if wake is not None and src != "default":
            start = now_dt.replace(hour=wake, minute=0, second=0, microsecond=0)
            if now_dt < start:  # before today's wake -> the felt day began yesterday
                start = start - datetime.timedelta(days=1)
            return start
    except Exception:
        pass
    return now_dt.replace(hour=0, minute=0, second=0, microsecond=0)


def human(delta):
    """Backward-compatible absolute-elapsed label (what when.py already uses)."""
    s = int(delta.total_seconds())
    sign = "-" if s < 0 else ""
    s = abs(s)
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    mins, s = divmod(s, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins or not parts:
        parts.append(f"{mins}m")
    return sign + " ".join(parts)


if __name__ == "__main__":
    # CLI smoke: pass a timestamp (ISO) and optional --salience 0..1
    import argparse

    p = argparse.ArgumentParser(description="felt-time recency label")
    p.add_argument("when", nargs="?", help="ISO timestamp (default: 3 days ago)")
    p.add_argument("--salience", type=float, default=0.0, help="0..1 salience")
    p.add_argument("--density", type=int, default=None, help="interaction count override")
    a = p.parse_args()
    ts = (datetime.datetime.now().astimezone() - datetime.timedelta(days=3)
          if not a.when else datetime.datetime.fromisoformat(a.when).astimezone())
    print(f"anchor:  {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"felt:    {felt(ts, salience=a.salience, density=a.density)}")
    print(f"wall:    {human(_wall_age(ts))} ago")
    print(f"density: {a.density if a.density is not None else _density_count(ts)} interactions in window")
