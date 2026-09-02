"""Integration check: flag-ON path routes events through the buffer, then seals
them into the real `events` chain — against a throwaway DB, never the live one.

Run:  python3 test_seal_delay_integration.py
"""
import os
import shutil
import tempfile

os.environ["AGENT_SEAL_DELAY"] = "1"
os.environ["AGENT_SEAL_WINDOW"] = "0"   # seal immediately on seal_due()

import memstore
from memstore import connect, _emit_event, seal_due, _event_hash

tmp = tempfile.mkdtemp(prefix="sealdelay_int_")
memstore.DB = os.path.join(tmp, "memory.db")

try:
    # 1. emit under flag-on -> buffered, NOT sealed into events
    with connect() as c:
        _emit_event(c, "memory_store", {"t": "a"}, actor="agent",
                    source_memory_id=None, validated=1)
        _emit_event(c, "memory_store", {"t": "b"}, actor="agent",
                    source_memory_id=None, validated=1)
    with connect() as c:
        nb = c.execute("SELECT COUNT(*) FROM buffer WHERE tentative=1").fetchone()[0]
        ne = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert nb == 2, f"buffer={nb}"
        assert ne == 0, f"events={ne}"

    # 2. seal_due -> moved into events, chain verifies from genesis
    sealed = seal_due()
    assert sealed == [1, 2], sealed
    with connect() as c:
        nb = c.execute("SELECT COUNT(*) FROM buffer WHERE tentative=1").fetchone()[0]
        ne = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert nb == 0, f"buffer after seal={nb}"
        assert ne == 2, f"events={ne}"
        prev = None
        for r in c.execute("SELECT * FROM events ORDER BY seq").fetchall():
            h = _event_hash(r["seq"], r["prev_hash"], r["ts"], r["type"],
                            r["actor"], r["payload"], r["source_memory_id"],
                            r["validated"])
            assert h == r["hash"], f"hash mismatch seq {r['seq']}"
            assert r["prev_hash"] == prev, f"link mismatch seq {r['seq']}"
            prev = r["hash"]

    print("integration PASS: flag-on buffers then seals into the real events chain")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
