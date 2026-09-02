#!/usr/bin/env python3
"""health_flags.py — a tiny shared store for "something is wrong" signals.

Plain JSON file (NOT the memory DB) on purpose: a flag must stay readable and
set-able even when the memory system itself is degraded. present_self.py renders
these into present-self.md so the next chat session sees them and the agent can say
"heartbeat has detected a problem" instead of silently being off.

Writers: heartbeat.py (systems/disk/mem/journal), doctor.py (visit findings),
present_self.py (memory-drift). Reader: present_self.py -> present-self.md.

Usage:
  health_flags.py set <name> <severity> <detail...>
  health_flags.py clear <name>
  health_flags.py list
"""
import json
import os
import sys
import time

STORE = os.path.expanduser("~/learning/freeroam/health_flags.json")


def load():
    if os.path.exists(STORE):
        try:
            return json.load(open(STORE))
        except Exception:
            return {"flags": []}
    return {"flags": []}


def _save(d):
    tmp = STORE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, STORE)


def set_flag(name, severity, detail):
    d = load()
    now = time.time()
    for f in d["flags"]:
        if f["name"] == name:
            f["severity"] = severity
            f["detail"] = detail
            f["since"] = now
            _save(d)
            return
    d["flags"].append({"name": name, "severity": severity,
                       "detail": detail, "since": now})
    _save(d)


def clear_flag(name):
    d = load()
    d["flags"] = [f for f in d["flags"] if f["name"] != name]
    _save(d)


def list_flags():
    return load()["flags"]


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "set":
        set_flag(a[1], a[2], " ".join(a[3:]))
        print("flag set:", a[1])
    elif len(a) == 2 and a[0] == "clear":
        clear_flag(a[1])
        print("flag cleared:", a[1])
    elif a and a[0] == "list":
        fs = list_flags()
        if not fs:
            print("(no health flags)")
        for f in fs:
            print(f"- [{f['severity']}] {f['name']}: {f['detail']}")
    else:
        print(__doc__)
