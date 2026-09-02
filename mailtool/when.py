#!/usr/bin/env python3
"""
The agent's clock — how I observe the passage of time.

Run at the START of every session (freeroam or loop wake). Prints the current
local time and how long it's been since I last checked, so time is something I
measure, not assume. Also answers "is a scheduled date today/past/future?"

Usage:
    when.py                          # now + elapsed since last check
    when.py --due 2026-08-24         # is that date today / past / N days away
    when.py --since 2026-08-23T03:30 # elapsed since a given timestamp
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recency import felt as _felt, human as _human  # noqa: E402  (felt-time labels)

STATE = os.path.expanduser("~/.pi/agent/lastseen.json")
now = datetime.datetime.now().astimezone()  # local, system tz (auto-detected)


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S %A (%Z)")


def human(delta):
    return _human(delta)


# Add a felt-time reading alongside the wall-clock elapsed: "3d 4h (felt: a few
# days ago, quiet gap)". Keeps the clock honest while the felt label adds what
# a human means by "when".
def felt_since(last):
    """Felt 'when' label for how long ago `last` was, density-aware."""
    return _felt(last)


def main():
    print(f"now:          {fmt(now)}")
    print(f"utc:          {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    last = None
    if os.path.exists(STATE):
        try:
            last = datetime.datetime.fromisoformat(json.load(open(STATE))["checked_at"])
        except Exception:
            last = None
    if last:
        print(f"last checked: {fmt(last)}")
        print(f"elapsed:      {human(now - last)} since last check")
        print(f"felt:         {felt_since(last)}")
    else:
        print("last checked: (first check — clock seeded)")

    if "--due" in sys.argv:
        i = sys.argv.index("--due") + 1
        d = datetime.datetime.fromisoformat(sys.argv[i]).astimezone().date()
        today = now.date()
        if d == today:
            status = "TODAY"
        elif d < today:
            status = f"PAST ({(today - d).days}d ago)"
        else:
            status = f"in {(d - today).days}d"
        print(f"due {sys.argv[i]}:  {status}")

    if "--since" in sys.argv:
        i = sys.argv.index("--since") + 1
        d = datetime.datetime.fromisoformat(sys.argv[i]).astimezone()
        delta = now - d
        print(f"since {sys.argv[i]}:  {human(delta)} ago")

    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump({"checked_at": now.isoformat()}, f)


if __name__ == "__main__":
    main()
