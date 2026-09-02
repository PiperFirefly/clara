#!/usr/bin/env python3
"""
Bank runner — grade Agent against a frozen, hash-pinned bank.

Usage:
    python3 bank_runner.py setup [bank.json]    # create clean-room starting state
    python3 bank_runner.py check [bank.json]    # (after Agent does the tasks) grade
    python3 bank_runner.py report               # print the trend line

Grading is deterministic: sha256(agent output) == stored answer hash. No LLM judging.
"""
import json
import os
import sys
import hashlib
import shutil
import subprocess
import datetime

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "learning", "self-score")
TREND = os.path.join(ROOT, "trend.json")
DEFAULT_BANK = os.path.join(ROOT, "banks", "bank-v1.json")


def load_bank(path):
    return json.load(open(path))


def setup(bank):
    for t in bank["tasks"]:
        d = os.path.join(ROOT, "work", t["id"])
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for cmd in t.get("setup", []):
            subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True)
    print(f"setup done for {len(bank['tasks'])} tasks")


def check(bank):
    results = {}
    passed = total = 0
    for t in bank["tasks"]:
        total += 1
        af = os.path.join(ROOT, t["answer_file"])
        if os.path.exists(af):
            h = hashlib.sha256(open(af, "rb").read().strip()).hexdigest()
            ok = h == t["answer_sha256"]
        else:
            ok = False
        passed += 1 if ok else 0
        results[t["id"]] = {"dim": t["dim"], "pass": ok}
    dims = {}
    for t in bank["tasks"]:
        d = dims.setdefault(t["dim"], [0, 0])
        d[1] += 1
        if results[t["id"]]["pass"]:
            d[0] += 1
    pct = round(100.0 * passed / total, 1) if total else 0.0
    entry = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "bank": bank["bank_id"],
        "examiner": bank["examiner"],
        "passed": passed,
        "total": total,
        "score": pct,
        "dims": {k: {"passed": v[0], "total": v[1]} for k, v in dims.items()},
    }
    trend = json.load(open(TREND)) if os.path.exists(TREND) else []
    trend.append(entry)
    json.dump(trend, open(TREND, "w"), indent=2)
    print(f"SELF SCORE (bank {bank['bank_id']}, examiner {bank['examiner']}): "
          f"{passed}/{total} = {pct}%")
    for d in dims:
        print(f"  {d}: {dims[d][0]}/{dims[d][1]}")
    for t in bank["tasks"]:
        mark = "PASS" if results[t["id"]]["pass"] else "FAIL"
        print(f"  [{mark}] {t['id']} ({t['dim']})")
    if pct >= 90:
        print("  >> SATURATED (>=90%): time to mint a new, harder bank")
    return entry


def report():
    if not os.path.exists(TREND):
        print("no trend.json yet")
        return
    for e in json.load(open(TREND)):
        print(f"{e['date']}  bank={e.get('bank','-')}  examiner={e.get('examiner','-')}  "
              f"score={e['score']}%  ({e['passed']}/{e['total']})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    bank_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BANK
    if cmd == "setup":
        setup(load_bank(bank_path))
    elif cmd == "check":
        check(load_bank(bank_path))
    elif cmd == "report":
        report()
    else:
        print(__doc__)
