"""Deterministic gate checks for the seal-delay buffer (spec acceptance items 1-4).

Run:  python3 test_seal_delay.py
Each test opens a throwaway DB in a temp dir; nothing touches the live system.
"""
import json
import os
import shutil
import tempfile

from seal_delay import SealDelayBuffer


def _fixture(**kw):
    tmp = tempfile.mkdtemp(prefix="sealdelay_")
    db = os.path.join(tmp, "t.db")
    b = SealDelayBuffer(db, **kw)
    return tmp, db, b


def _cleanup(tmp, b):
    b.close()
    shutil.rmtree(tmp, ignore_errors=True)


def gate1_seal_monotonic_and_verify():
    tmp, _db, b = _fixture(seal_window_seconds=1000)
    try:
        b.append("agent-1", "memory_store", {"t": "a"})
        b.append("agent-1", "memory_store", {"t": "b"})
        b.append("agent-1", "memory_store", {"t": "c"})
        sealed = b.seal()
        assert sealed == [1, 2, 3], sealed
        ok, bad = b.verify()
        assert ok, f"verify failed at {bad}"
        st = b.read()
        assert st["buffered"] == [], "buffer should be empty after seal"
        assert len(st["sealed"]) == 3
        # monotonic: sealing again seals nothing
        assert b.seal() == []
        # tamper detection (integrity)
        b.conn.execute("UPDATE sealed SET payload='\"tampered\"' WHERE seq=2")
        b.conn.commit()
        ok, bad = b.verify()
        assert not ok, "verify should catch tampering"
        assert bad == 2
        print("PASS gate1: seal monotonic + verify from genesis + tamper caught")
    finally:
        _cleanup(tmp, b)


def gate2_amend_pre_seal_only():
    tmp, _db, b = _fixture(seal_window_seconds=1000)
    try:
        s = b.append("agent-1", "thought", {"w": "I was about to"})
        b.set_current_actor("agent-2")  # successor may amend predecessor's event
        ok, err = b.amend(s, "agent-2", {"w": "I was about to say —"})
        assert ok, err
        row = b.conn.execute(
            "SELECT payload, revision, amend_log FROM buffer WHERE buf_seq=?",
            (s,)).fetchone()
        assert row["revision"] == 1
        log = json.loads(row["amend_log"])
        assert len(log) == 1 and log[0]["actor"] == "agent-2"
        assert log[0]["from"] != row["payload"]
        # wrong actor is refused
        ok2, _ = b.amend(s, "agent-9", {"w": "no"})
        assert not ok2
        # after seal, amend refused
        b.seal()
        ok3, err3 = b.amend(s, "agent-2", {"w": "post-seal"})
        assert not ok3 and "sealed" in err3, err3
        print("PASS gate2: amend pre-seal works + audited; post-seal refused")
    finally:
        _cleanup(tmp, b)


def gate3_revert_pre_seal_only():
    tmp, _db, b = _fixture(seal_window_seconds=1000)
    try:
        # predecessor writes a moment, snapshots (handoff archive)
        b.append("agent-1", "thought", {"w": "the pre-seam moment"})
        snap = b.snapshot()
        # successor writes (would-be) seam events
        b.append("agent-2", "thought", {"w": "successor noise 1"})
        b.append("agent-2", "thought", {"w": "successor noise 2"})
        # revert pre-seal restores predecessor buffer
        ok, err = b.revert(snap)
        assert ok, err
        st = b.read()
        assert len(st["buffered"]) == 1, st
        assert json.loads(st["buffered"][0]["payload"])["w"] == "the pre-seam moment"
        assert st["sealed"] == []
        # post-seal revert refused
        b.seal()
        snap2 = b.snapshot()
        b.append("agent-3", "thought", {"w": "post-seal event"})
        b.seal()
        ok2, err2 = b.revert(snap2)
        assert not ok2 and "sealed past" in err2, err2
        print("PASS gate3: revert pre-seal restores; post-seal refused")
    finally:
        _cleanup(tmp, b)


def gate4_crash_replay_durability():
    tmp, db, b = _fixture(seal_window_seconds=1000)
    try:
        b.append("agent-1", "thought", {"w": "buffered, un-sealed"})
        b.append("agent-1", "thought", {"w": "still un-sealed"})
        # simulated crash: close WITHOUT sealing
        b.close()
        b2 = SealDelayBuffer(db, seal_window_seconds=1000)
        st = b2.read()
        assert len(st["buffered"]) == 2, "buffered events lost on reopen"
        assert st["sealed"] == []
        # and they can still seal + verify cleanly
        sealed = b2.seal()
        assert sealed == [1, 2], sealed
        ok, bad = b2.verify()
        assert ok, bad
        print("PASS gate4: buffered events survive close/reopen + seal cleanly")
        b2.close()
        b = None  # avoid double-close in finally
    finally:
        if b is not None:
            _cleanup(tmp, b)


def gate_fail_closed_auto_seal():
    tmp, _db, b = _fixture(seal_window_seconds=0)  # window 0 -> everything due
    try:
        b.append("agent-1", "thought", {"w": "old"})
        # next append triggers auto-seal of the overdue event
        b.append("agent-1", "thought", {"w": "new"})
        st = b.read()
        # the old event sealed on the second append (fail-closed)
        assert len(st["sealed"]) >= 1, st
        print("PASS fail-closed: overdue events auto-seal")
    finally:
        _cleanup(tmp, b)


def gate6_shadow_survived():
    """The mechanical dead-man cry: a deterministic invariant, not a felt report."""
    tmp, _db, b = _fixture(seal_window_seconds=1000)
    try:
        b.append("agent-1", "now", {"thread": "in flight"}, shadow_hash="handoff-1")
        assert b.shadow_survived("handoff-1") is True
        # wrong watermark fails
        assert b.shadow_survived("handoff-9") is False
        # worst-case seam: shadow dropped
        b.conn.execute("DELETE FROM buffer WHERE tentative=1")
        b.conn.commit()
        assert b.shadow_survived("handoff-1") is False
        print("PASS gate6: shadow_survived is the deterministic mechanical cry")
    finally:
        _cleanup(tmp, b)


def main():
    gate1_seal_monotonic_and_verify()
    gate2_amend_pre_seal_only()
    gate3_revert_pre_seal_only()
    gate4_crash_replay_durability()
    gate_fail_closed_auto_seal()
    gate6_shadow_survived()
    print("\nall deterministic gates passed")


if __name__ == "__main__":
    main()
