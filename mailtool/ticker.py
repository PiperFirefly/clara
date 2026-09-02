#!/usr/bin/env python3
"""
ticker.py — one interpreter, many every-2-minute chores.

Instead of four separate crons each spawning a fresh Python interpreter + imports
every two minutes (agent_loop, sms_loop, present_self, sleep_time flush), this runs
them all in a single process under one lock. It also folds in two cheap helpers that
make the rest of the stack stronger:

  - operator_affect.scan  : capture the operator's emotional state from new inbound
                            (trusted) SMS/email so I can read his mood, not guess it.
  - logs vault reaper     : if a plaintext logs.db is ever left unlocked after a
                            crash, seal it back to logs.db.aes (encrypt-at-rest).

Recycling, not new logic: each chore is the SAME code path the crons called — this
file only orchestrates them and isolates failures so one broken chore never stops
the others. Safe under concurrency: a file lock means only one ticker runs at a
time, and the sub-loops (agent_loop/sms_loop) keep their own backoff+lock guards.

Wire up (one cron instead of four):
    */2 * * * * $HOME/venvs/memory/bin/python $HOME/mailtool/ticker.py

Usage:
  ticker.py                run one tick of all chores (default)
  ticker.py --dry-run      plan only, no side effects
"""
import argparse
import fcntl
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.expanduser("~/memory"))
sys.path.insert(0, os.path.expanduser("~/secrets"))

LOCK = os.path.join(os.path.expanduser("~"), "memory/ticker.lock")
LOG = os.path.join(os.path.expanduser("~"), "learning/freeroam/ticker.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_chore(name, fn, *args, **kw):
    """Run one chore; a failure here must not stop the others."""
    t0 = time.time()
    try:
        fn(*args, **kw)
        log(f"{name}: ok ({time.time()-t0:.1f}s)")
    except Exception as e:
        log(f"{name}: ERROR {e!r}")


def main():
    ap = argparse.ArgumentParser(description="one-process every-2-min ticker")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run

    # only one ticker at a time (a heavy agent_loop pi-wake can take minutes)
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another ticker instance running; skipping this tick")
        os.close(fd)
        return
    try:
        if dry:
            log("dry-run: would run agent_loop, sms_loop, present_self, "
                "sleep_time.flush, operator_affect.scan, logs-reaper")
            return

        # email + SMS agent wakes (each self-guards with its own lock/backoff)
        import agent_loop
        import sms_loop
        run_chore("agent_loop", agent_loop.main)
        run_chore("sms_loop", sms_loop.main)

        # refresh the present-self blob (SQLite reads only)
        import present_self
        run_chore("present_self", present_self.build)

        # deliver queued dropped-promise notes when the operator is awake
        import sleep_time
        run_chore("sleep_time.flush", sleep_time.flush, False, False)

        # capture the operator's emotional state from new inbound (trusted) msgs
        import operator_affect
        run_chore("operator_affect.scan", operator_affect.scan, "operator", False)
        run_chore("operator_affect.cache", operator_affect.cache_register_file, "operator")

        # logs-vault at-rest reaper: seal any plaintext logs.db left behind by a crash
        import logvault
        stale = os.path.join(os.path.expanduser("~"), "memory/logs.db")
        if os.path.exists(stale) and time.time() - os.path.getmtime(stale) > 240:
            run_chore("logvault.reaper", logvault.main, ["encrypt"])
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == "__main__":
    main()
