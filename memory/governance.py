#!/usr/bin/env python3
"""governance.py — the meta-level self-governance layer (idea #15).

Unifies the deterministic safety/quality gates into ONE metacognitive control
view, so before a high-blast action I check the whole control surface at once
instead of remembering the gates ad hoc. Gates are all fail-closed deterministic
checks (no vibes), which is the point:

  - blast_radius : irreversible/high-blast actions token-gated (M3)
  - migration_cover : changes must reach clone/greenfield (migration rule)
  - behavior_gate : refactors must preserve behavior (differ/equivalent)

This is the CONTROL half of the Nelson-Narens loop, applied to self-governance:
MONITOR (know gate state) -> CONTROL (proceed only if gates green).

Usage:
  python3 governance.py status        # state of all gates + data, no action
  python3 governance.py audit         # detailed per-gate report
"""
import argparse
import os
import subprocess
import sys
import time

HOME = os.path.expanduser("~")


def _run(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
        return p.returncode, (p.stdout or p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def _gate(name, cmd, ok_code=0):
    rc, out = _run(cmd)
    return {"name": name, "rc": rc, "ok": rc == ok_code,
            "detail": out.splitlines()[0][:120] if out else ""}


def govern_status():
    gates = [
        _gate("migration_cover",
              "python3 ~/mailtool/migration_cover.py check"),
        _gate("blast_radius_avail",
              "python3 ~/mailtool/blast_radius.py status"),
        _gate("behavior_gate_avail",
              "python3 ~/mailtool/behavior_gate.py --help"),
    ]
    return gates


def govern_audit():
    gates = govern_status()
    print("# Self-governance layer — %s" % time.strftime("%Y-%m-%d %H:%M"))
    for g in gates:
        mark = "OK" if g["ok"] else ("FAIL" if g["rc"] != 0 else "DENY")
        print(f"[{mark}] {g['name']:<22} rc={g['rc']} {g['detail']}")
    ok = all(g["ok"] for g in gates)
    print("")
    print("Governance:", "ALL GATES GREEN" if ok else "GATE NON-ZERO — check before high-blast action")
    print("Note: blast_radius/behavior_gate are *availability* checks here; real enforcement")
    print("      happens when the specific action is guarded (irreversible -> token-gated).")
    return gates


def main():
    p = argparse.ArgumentParser(description="meta-level self-governance layer")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("audit")
    a = p.parse_args()
    if a.cmd == "status":
        for g in govern_status():
            print(f"[{'OK' if g['ok'] else 'X'}] {g['name']} rc={g['rc']}")
    elif a.cmd == "audit":
        govern_audit()


if __name__ == "__main__":
    main()
