#!/usr/bin/env python3
"""migrate_state.py — seed state.db from the existing *_state.json files.

Copies current ephemeral state into the new kv store (idempotent; keys that already
exist are left unless --force). The source JSON files are kept as fallback during
transition. This is the data step of the state-consolidation; the *writers* are
converted separately to use memory/state.py as their choke point.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import state  # noqa: E402

HOME = os.path.expanduser("~")
MEM = os.path.join(HOME, "memory")
FREEROAM = os.path.join(HOME, "learning", "freeroam")


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def seed_scalar(key, value, durable=False, force=False):
    if force or state.get(key, "<absent>") == "<absent>":
        state.set(key, value, durable=durable)


def seed_dict_children(prefix, d, durable=False, force=False):
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        seed_scalar(f"{prefix}/{k}", v, durable=durable, force=force)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing keys")
    args = ap.parse_args()
    force = args.force

    # Worker watermark files (memory/*-state.json) -> worker/<name>
    worker_files = {
        "hive": "hive-state.json",
        "reason": "reason-state.json",
        "belief_extract": "belief-state.json",
        "affect_extract": "affect-state.json",
        "forecast_extract": "forecast-state.json",
        "person_model_extract": "person_model-state.json",
        "curiosity": "curiosity-state.json",
    }
    for name, fn in worker_files.items():
        v = load(os.path.join(MEM, fn))
        if v is not None:
            seed_scalar(f"worker/{name}", v, durable=True, force=force)

    # freeroam/state.json -> freeroam/*  (tokens_today durable)
    fs = load(os.path.join(FREEROAM, "state.json")) or {}
    for k in ("day", "checkpoint", "last_run"):
        if k in fs:
            seed_scalar(f"freeroam/{k}", fs[k], durable=(k == "tokens_today"), force=force)
    if "tokens_today" in fs:
        seed_scalar("freeroam/tokens_today", fs["tokens_today"], durable=True, force=force)

    # goals.json -> goals/<name>  (durable)
    goals = load(os.path.join(FREEROAM, "goals.json"))
    if isinstance(goals, dict):
        seed_dict_children("goals", goals, durable=True, force=force)

    # heartbeat_state.json -> heartbeat/*
    hb = load(os.path.join(FREEROAM, "heartbeat_state.json")) or {}
    for k, v in hb.items():
        seed_scalar(f"heartbeat/{k}", v, durable=(k == "escalated"), force=force)

    # per-cron markers -> cron/<name>
    cron_files = {
        "warden": "warden_state.json",
        "steward": "steward_state.json",
        "doctor": "doctor_state.json",
        "promise_check": "promise_check_state.json",
        "sleep_time": "sleep_time_state.json",
    }
    for name, fn in cron_files.items():
        v = load(os.path.join(FREEROAM, fn))
        if v is not None:
            seed_scalar(f"cron/{name}", v, durable=True, force=force)

    # skills (self-watermarking, lives in freeroam dir)
    sk = load(os.path.join(FREEROAM, "skills_state.json"))
    if sk is not None:
        seed_scalar("worker/skills", sk, durable=True, force=force)

    # health_flags -> health/flags
    hf = load(os.path.join(FREEROAM, "health_flags.json"))
    if hf is not None:
        seed_scalar("health/flags", hf, force=force)

    print(f"state.db seeded: {len(state.keys())} keys.")
    for k in state.keys():
        print(f"  {k}")


if __name__ == "__main__":
    main()
