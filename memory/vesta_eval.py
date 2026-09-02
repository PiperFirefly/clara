#!/usr/bin/env python3
"""Vesta eval — integrity + replay tests for the event ledger.

Four honest checks, mapping to the design's success criteria:
  1. hash-chain     — walk from genesis, recompute every hash, assert prev_hash
                      links (the log is tamper-evident and self-verifying).
  2. mirror-replay  — replay the ledger and assert the projection matches the
                      live self-model (identity anchors + commitments, no drift).
  3. append-only    — no UPDATE/DELETE code paths may target the `events` table.
  4. tamper-detect  — a fabricated identity change made OUTSIDE the ledger (a
                      raw facts UPDATE with no event) must NOT appear in the
                      replay, and mirror_check() must flag the drift.

Usage: python3 vesta_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M


def test_hash_chain():
    ok, n, errors = M.verify_chain()
    print(f"[1] hash-chain    {'PASS' if ok else 'FAIL'}  ({n} events)")
    for e in errors:
        print("     ", e)
    return ok


def test_mirror_replay():
    res = M.mirror_check()
    ok = not res["drift"]
    detail = "no drift" if ok else f"{len(res['drift'])} drift"
    print(f"[2] mirror-replay {'PASS' if ok else 'FAIL'}  ({detail})")
    for d in res["drift"]:
        print("     drift:", d.get("section"), d.get("key", ""))
    return ok


def test_append_only():
    src = open(os.path.join(os.path.dirname(__file__), "memstore.py")).read()
    bad = [ln for ln in src.splitlines()
           if "UPDATE events" in ln or "DELETE FROM events" in ln
           or "UPDATE  events" in ln or "DELETE  FROM events" in ln]
    ok = not bad
    print(f"[3] append-only   {'PASS' if ok else 'FAIL'}  ({len(bad)} forbidden statements)")
    for ln in bad:
        print("     ", ln.strip())
    return ok


def test_tamper_detection():
    key = "name"
    # Robustness (2026-08-28): an OOM-kill between the fake-write commit and the
    # finally-restore leaks the fake value into the LIVE facts table, and a
    # second crash compounds it (the fabricated actor name repeats). Two fixes:
    #   1. Baseline from the LEDGER (mirror), never the live table — a polluted
    #      live table must not become the "original" that gets restored.
    #   2. Self-heal first: strip any leaked [FABRICATED] suffix already sitting
    #      in the live table, so a crash can never compound or persist.
    orig = M.mirror().get("anchors", {}).get(key)
    if not orig:
        print("[4] tamper-detect SKIP  (no 'name' anchor in ledger)")
        return True
    try:
        c = M.connect()
        row = c.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        if row and row["value"] != orig and " [FABRICATED]" in (row["value"] or ""):
            c.execute("UPDATE facts SET value=? WHERE key=?", (orig, key))
            c.commit()
        c.close()
    except Exception:
        pass
    fake = orig + " [FABRICATED]"
    try:
        c = M.connect()
        c.execute("UPDATE facts SET value=? WHERE key=?", (fake, key))
        c.commit()
        c.close()
        m = M.mirror()
        kept_old = m["anchors"].get(key) == orig
        drift_seen = any(d.get("section") == "anchor" and d.get("key") == key
                         for d in M.mirror_check()["drift"])
        ok = kept_old and drift_seen
        print(f"[4] tamper-detect {'PASS' if ok else 'FAIL'}  "
              f"(ledger kept '{orig}'; drift flagged={drift_seen})")
        return ok
    finally:
        c = M.connect()
        c.execute("UPDATE facts SET value=? WHERE key=?", (orig, key))
        c.commit()
        c.close()


def main():
    results = [
        test_hash_chain(),
        test_mirror_replay(),
        test_append_only(),
        test_tamper_detection(),
    ]
    ok = all(results)
    print("\nVESTA EVAL:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
