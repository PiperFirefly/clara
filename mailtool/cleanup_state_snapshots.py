#!/usr/bin/env python3
"""cleanup_state_snapshots.py — archive the superseded *_state.json snapshot files.

The ephemeral state store (memory/state.db, via memory/state.py) is now the source
of truth. The old JSON files it was seeded from are redundant snapshots. This script
moves them to ~/archive/state-snapshots/<date>/ — REVERSIBLE (not rm), and only after
verifying the corresponding state.db keys exist so no data is lost.

Safety:
  - Only touches files in the curated SNAPSHOTS list (the superseded set; still-live
    files like health_flags.json / agent_loop_state.json are deliberately absent).
  - Requires proof (state.db keys) before archiving each file; missing proof = skip+warn.
  - Reversible: files go to ~/archive, not deleted.
  - Idempotent: already-archived/absent files are skipped.
  - Dry-run: --dry-run prints what would happen without moving anything.

Usage:
  cleanup_state_snapshots.py            archive superseded snapshots
  cleanup_state_snapshots.py --dry-run  preview only
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.expanduser("~/memory"))
import state as st  # noqa: E402

HOME = os.path.expanduser("~")
ARCHIVE_ROOT = os.path.join(HOME, "archive", "state-snapshots")
LOG = os.path.join(HOME, "learning", "freeroam", "cleanup_state.log")

# snapshot file -> state.db key(s) that must exist as proof of coverage
SNAPSHOTS = [
    (os.path.join("memory", "hive-state.json"), ["worker/hive"]),
    (os.path.join("memory", "reason-state.json"), ["worker/reason"]),
    (os.path.join("memory", "belief-state.json"), ["worker/belief_extract"]),
    (os.path.join("memory", "affect-state.json"), ["worker/affect_extract"]),
    (os.path.join("memory", "forecast-state.json"), ["worker/forecast_extract"]),
    (os.path.join("memory", "person_model-state.json"), ["worker/person_model_extract"]),
    (os.path.join("learning", "freeroam", "state.json"), ["freeroam/day", "freeroam/tokens_today"]),
    (os.path.join("learning", "freeroam", "goals.json"), ["goals/coding"]),
    (os.path.join("learning", "freeroam", "heartbeat_state.json"), ["heartbeat/escalated"]),
    (os.path.join("learning", "freeroam", "warden_state.json"), ["cron/warden"]),
    (os.path.join("learning", "freeroam", "doctor_state.json"), ["cron/doctor"]),
    (os.path.join("learning", "freeroam", "steward_state.json"), ["cron/steward"]),
    (os.path.join("learning", "freeroam", "promise_check_state.json"), ["cron/promise_check"]),
    (os.path.join("learning", "freeroam", "sleep_time_state.json"), ["cron/sleep_time"]),
    (os.path.join("learning", "freeroam", "skills_state.json"), ["worker/skills"]),
]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="preview only, move nothing")
    args = ap.parse_args()

    stamp = time.strftime("%Y-%m-%d")
    dest_dir = os.path.join(ARCHIVE_ROOT, stamp)
    archived = skipped_noproof = missing = 0

    for rel, required_keys in SNAPSHOTS:
        path = os.path.join(HOME, rel)
        if not os.path.exists(path):
            missing += 1
            continue
        # proof: every required state.db key must exist
        missing_keys = [k for k in required_keys if st.get(k, "<absent>") == "<absent>"]
        if missing_keys:
            log(f"[skip-noproof] {rel}: state.db missing {missing_keys}; not archiving")
            skipped_noproof += 1
            continue
        if args.dry_run:
            log(f"[dry-run] would archive {rel}")
            archived += 1
            continue
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.exists(dest):
            dest = dest + ".dup"
        shutil.move(path, dest)
        log(f"[archived] {rel} -> {dest}")
        archived += 1

    log(f"cleanup done: archived={archived} skip_noproof={skipped_noproof} already_missing={missing}"
        + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
