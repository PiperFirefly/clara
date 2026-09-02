#!/usr/bin/env python3
"""
operator_presence — a profile of the operator's *human* time, learned from
observation (formerly an agent-specific "operator clock").

Combines the computer's timezone (the system's local tz) with the operator's own
patterns: explicit cues ("good morning", "good night", "I'm so tired",
"up late"...) plus the raw hours they actually message me. From this it builds:
  - an active-hours histogram (when the operator talks to me),
  - an inferred overnight sleep window (from signals and/or quiet overnight hours),
  - a "what time is it for the operator right now" phase guess.

The point is to let background work (consolidation, health checks, promise
delivery) time itself to the operator's likely active hours, so quiet work can
continue outside those hours without pinging them while they sleep.

Storage: ~/.pi/agent/operator_presence.json  (grows with signals)

Usage:
  operator_presence.py now                wall clock + operator's inferred phase + recency
  operator_presence.py profile            histogram, sleep window, recent signals
  operator_presence.py ingest-sessions    re-scan pi session logs for the operator's messages/cues
  operator_presence.py log <signal>       record an explicit cue (morning/night/tired/...)
  operator_presence.py mark-active        record "the operator is active now" (implicit)
"""
import argparse
import collections
import datetime
import glob
import json
import os
import re
import sys

STORE = os.path.expanduser("~/.pi/agent/operator_presence.json")
LEGACY_STORE = os.path.expanduser("~/.pi/agent/legacy-operator-clock.json")
SESSIONS = os.path.expanduser("~/.pi/agent/sessions/--home-agent--/*.jsonl")

PATTERNS = {
    "morning": re.compile(r"good\s*morning|\bgoodmorning\b|\bjust woke\b|\bwoke up\b|\bwaking up\b|\bup and at"),
    "night": re.compile(r"good\s*night|\bgoodnight\b|\bgoing to bed\b|\bheading to bed\b|\bcalling it a night\b|\bsleep well\b|\bcrashing\b"),
    "tired": re.compile(r"\btired\b|\bexhausted\b|\bsleepy\b|\bcan'?t sleep\b|\binsomnia\b|\bno sleep\b|\bdead on my feet\b|\bwiped\b|\brunning on empty\b"),
    "evening": re.compile(r"good\s*evening|\bevening\b"),
    "afternoon": re.compile(r"\bafternoon\b"),
    "late": re.compile(r"\bup late\b|\bup all night\b|\blate night\b|\bpulled an all\b"),
}

# Messages *about* signals (examples, meta-discussion) must NOT count as signals.
META = re.compile(
    r"when i say|when you say|for example|such as|e\.g\.|\betc\b|based on when|"
    r"derived from|like a clock|observing my patterns|as an example|i mean|"
    r"watch for|good morning,? or|good night,? or"
)


def now_local():
    return datetime.datetime.now().astimezone()


def _migrate_legacy_store():
    """One-time: move a legacy operator-clock json to operator_presence.json."""
    if os.path.exists(LEGACY_STORE) and not os.path.exists(STORE):
        try:
            os.makedirs(os.path.dirname(STORE), exist_ok=True)
            os.rename(LEGACY_STORE, STORE)
        except Exception:
            pass


def load():
    _migrate_legacy_store()
    if os.path.exists(STORE):
        try:
            return json.load(open(STORE))
        except Exception:
            pass
    return {"activity": [], "signals": [], "meta": {"created": now_local().isoformat()}}


def save(d):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    with open(STORE, "w") as f:
        json.dump(d, f, indent=2)


def local_from_iso(iso):
    dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone()


def detect(text):
    t = (text or "").lower()
    if META.search(t):
        return []
    return [name for name, pat in PATTERNS.items() if pat.search(t)]


def human(delta):
    s = int(delta.total_seconds())
    if s < 0:
        return "0m"
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    mins, _ = divmod(s, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins or not parts:
        parts.append(f"{mins}m")
    return " ".join(parts)


def hour_hist(d):
    hist = collections.Counter()
    for iso in d.get("activity", []):
        try:
            hist[local_from_iso(iso).hour] += 1
        except Exception:
            pass
    return hist


def median_hour(iso_list):
    hs = []
    for iso in iso_list:
        try:
            hs.append(local_from_iso(iso).hour)
        except Exception:
            pass
    if not hs:
        return None
    hs.sort()
    return hs[len(hs) // 2]


def overnight_quiet(hist):
    """Longest zero-activity stretch within overnight hours (22:00..11:00 wrap)."""
    seq = [22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    best_len, best_start = 0, None
    run_start = None
    for i, h in enumerate(seq):
        if hist.get(h, 0) == 0:
            if run_start is None:
                run_start = h
            length = i - seq.index(run_start) + 1
            if length > best_len:
                best_len, best_start = length, run_start
        else:
            run_start = None
    if best_start is None:
        return None, None
    return best_start, (best_start + best_len) % 24


def sleep_window(d):
    signals = d.get("signals", [])
    night = [s["time"] for s in signals if s["signal"] == "night"]
    morning = [s["time"] for s in signals if s["signal"] == "morning"]
    bed, wake = median_hour(night), median_hour(morning)
    if bed is not None and wake is not None:
        return bed, wake, "signals"
    b, w = overnight_quiet(hour_hist(d))
    if b is not None:
        return b, w, "activity-gap"
    return 2, 9, "default"


def phase(d, dt):
    hist = hour_hist(d)
    h = dt.hour
    n = len(d.get("activity", []))
    if hist.get(h, 0) >= 1:
        state = "awake"
    else:
        bed, wake, src = sleep_window(d)
        asleep = (bed <= h < wake) if bed <= wake else (h >= bed or h < wake)
        state = "asleep" if asleep else "inactive"
    conf = "high" if n >= 40 else ("medium" if n >= 15 else "low")
    return state, conf


def cmd_now(d):
    dt = now_local()
    state, conf = phase(d, dt)
    bed, wake, src = sleep_window(d)
    print(f"wall clock:   {dt.strftime('%Y-%m-%d %H:%M:%S %A (%Z)')}")
    last = None
    if d.get("activity"):
        last = local_from_iso(d["activity"][-1])
        print(f"last heard:   {last.strftime('%H:%M')} ({human(dt - last)} ago)")
    print(f"operator's time: ~{state} ({conf} confidence)")
    if bed is not None:
        print(f"sleep window: ~{bed:02d}:00 – {wake:02d}:00  (from {src})")


def cmd_profile(d):
    hist = hour_hist(d)
    n = len(d.get("activity", []))
    print(f"activity points: {n}")
    if n:
        print("active hours (histogram):")
        for h in range(24):
            c = hist.get(h, 0)
            if c:
                print(f"  {h:02d}:00  {'#' * c} ({c})")
    bed, wake, src = sleep_window(d)
    if bed is not None:
        print(f"sleep window: ~{bed:02d}:00 – {wake:02d}:00 ({src})")
    sigs = d.get("signals", [])
    print(f"explicit signals: {len(sigs)}")
    for s in sigs[-10:]:
        t = local_from_iso(s["time"]).strftime("%Y-%m-%d %H:%M")
        print(f"  {t}  {s['signal']}")


def cmd_ingest(d):
    files = sorted(glob.glob(SESSIONS))
    activity = set()
    signals = []  # rebuild from scratch (session logs are the source of truth)
    for fp in files:
        try:
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") != "message":
                    continue
                m = obj.get("message", {})
                if m.get("role") != "user":
                    continue
                ts = obj.get("timestamp")
                if not ts:
                    continue
                text = ""
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
                iso = local_from_iso(ts).isoformat()
                activity.add(iso)
                for sig in detect(text):
                    signals.append({"time": iso, "signal": sig})
        except Exception:
            continue
    d["activity"] = sorted(activity)
    d["signals"] = signals
    save(d)
    print(f"ingested: {len(d['activity'])} activity points, {len(signals)} signals")


def cmd_log(d, sig):
    if sig not in PATTERNS:
        sys.exit(f"unknown signal '{sig}' — use one of: {', '.join(PATTERNS)}")
    d.setdefault("signals", []).append({"time": now_local().isoformat(), "signal": sig})
    d.setdefault("activity", []).append(now_local().isoformat())
    save(d)
    print(f"logged: {sig} at {now_local().strftime('%Y-%m-%d %H:%M')}")


def cmd_mark(d):
    d.setdefault("activity", []).append(now_local().isoformat())
    save(d)
    print(f"marked active at {now_local().strftime('%Y-%m-%d %H:%M')}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("now")
    sub.add_parser("profile")
    sub.add_parser("ingest-sessions")
    lp = sub.add_parser("log")
    lp.add_argument("signal")
    sub.add_parser("mark-active")
    a = p.parse_args()
    d = load()
    if a.cmd == "now":
        cmd_now(d)
    elif a.cmd == "profile":
        cmd_profile(d)
    elif a.cmd == "ingest-sessions":
        cmd_ingest(d)
    elif a.cmd == "log":
        cmd_log(d, a.signal)
    elif a.cmd == "mark-active":
        cmd_mark(d)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
