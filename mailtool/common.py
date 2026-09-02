#!/usr/bin/env python3
"""
mailtool/common.py — shared helpers factored out of independently-grown mailtool
scripts (2026-08-31 housekeeping refactor #2). These are the pieces that
`simplify`/`clone_detect` flagged as byte-identical (or same-shape-with-a-
parameter) across multiple cron scripts: logging, operator lookup, file
locking, "is this check due", and report display.

Deliberately dumb: pure functions or thin wrappers, no module-level state,
no side effects on import. Each call site keeps its own module-level
constants (LOG_PATH, LOCK_PATH, REPORT, MIN_INTERVAL, ...) and passes them
in — this file does not know about any particular script's file layout.
"""
import os
import sys
import time
import json
import fcntl
from datetime import datetime


def operator_email():
    """Primary operator's email from operator config; env override; else None.

    Identical across deepseek_balance.py, notify.py, package_guard.py,
    provider_failover.py before this refactor.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory"))
        from operator_config import primary_channel
        return primary_channel("email") or os.environ.get("OPERATOR_ALERT_EMAIL")
    except Exception:
        return os.environ.get("OPERATOR_ALERT_EMAIL")


def operator_asleep():
    """(is_asleep: bool, state_or_error: str) via operator_presence.

    Identical across doctor.py, sleep_time.py, warden.py before this refactor.
    Imports operator_presence itself (same directory as every caller).
    """
    try:
        import operator_presence
        d = operator_presence.load()
        state, _conf = operator_presence.phase(d, operator_presence.now_local())
        return state == "asleep", state
    except Exception as e:
        return False, f"operator_presence unavailable ({e})"


def is_due(last_ts, min_interval):
    """True if last_ts is falsy, or min_interval seconds have elapsed since it.

    Same shape as doctor.py/warden.py's `_due()`, generalized: caller reads
    their own state key and passes the timestamp + their own MIN_INTERVAL.
    """
    if not last_ts:
        return True
    return (time.time() - last_ts) >= min_interval


def show_report(report_path, empty_label="report"):
    """Pretty-print a JSON report file, or a placeholder if it doesn't exist yet.

    Same shape as doctor.py/warden.py's `show_report()`; label lets each
    caller keep its own "(no X report yet)" wording.
    """
    if not os.path.exists(report_path):
        return f"(no {empty_label} yet)"
    return json.dumps(json.load(open(report_path)), indent=2)


def acquire_lock(lock_path):
    """Non-blocking exclusive file lock; returns an fd, or None if already held.

    Identical across agent_loop.py/sms_loop.py before this refactor.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def release_lock(fd):
    """Release a lock fd from acquire_lock(); no-op if fd is None."""
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def log_hard(msg, log_path):
    """Timestamped log line: mkdir + append to file (raises on write failure) + print.

    Same behavior as agent_inbox.py/agent_loop.py/sms_loop.py's `log()`.
    """
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def log_soft(msg, log_path):
    """Timestamped log line: print first, best-effort append (swallows write errors).

    Same behavior as session_distiller.py/memory/promise_check.py's `log()`.
    """
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_print_only(msg):
    """Plain flushed print, no file. Same behavior as package_guard.py/pi_update_planner.py's `log()`."""
    print(msg, flush=True)


def load_json(path, default):
    """Best-effort JSON load: default if missing or unparseable.

    Identical across heartbeat.py, instance.py, research_drift.py before this refactor.
    """
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default
