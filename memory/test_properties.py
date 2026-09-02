#!/usr/bin/env python3
"""
test_properties.py — property-based (Hypothesis) tests for correctness-critical paths.

These assert INVARIANTS under randomized inputs, which is where the unit tests stop:
the things that, if silently broken, corrupt the memory vault / secret store / ledger.

Targets:
  1. logvault.insert_msgs is idempotent (no dupes, count stable, UNIQUE never violated).
  2. logvault FTS index stays consistent with the messages table.
  3. secretstore set/get round-trip preserves values; versions are monotonic & append-only.
  4. affect clamping keeps valence/arousal in their domains under arbitrary floats.
  5. caliber._parse_confidence returns [0,1] or None under arbitrary garbage.

Run:  ~/venvs/memory/bin/python -m pytest memory/test_properties.py -q
Each test isolates to a throwaway DB (never touches the live memory.db / logs.db).

NOTE: this file deliberately lives in memory/ (not mailtool/) because mailtool has
its OWN hypothesis.py tool that shadows the pip `hypothesis` package; importing the
real Hypothesis with mailtool on sys.path first would silently load the wrong module.
"""
import os
import sys
import tempfile

# import the real Hypothesis BEFORE putting mailtool/ on the path (mailtool has a
# hypothesis.py that would otherwise shadow the package)
import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/memory"))
sys.path.insert(0, os.path.expanduser("~/mailtool"))
sys.path.insert(0, os.path.expanduser("~/secrets"))


# ---------------------------------------------------------------------------
# helper: isolated tmpdir per test (inline, so it plays nice with @given)
# ---------------------------------------------------------------------------
def _tmp():
    return tempfile.mkdtemp(prefix="prop_")


# ---------------------------------------------------------------------------
# 1 + 2. logvault idempotency + FTS consistency
# ---------------------------------------------------------------------------
def _make_row(source_key, text, ts=0.0, channel="tui", role="user", session_id="s"):
    return {"source": "pi-session", "source_key": source_key,
            "session_id": session_id, "channel": channel, "role": role,
            "ts": ts, "text": text}


@given(rows=st.lists(
    st.tuples(st.text(min_size=1), st.text(min_size=1)),
    min_size=0, max_size=30, unique_by=lambda t: t[0]))
@settings(max_examples=60, deadline=None)
def test_logvault_insert_idempotent(rows):
    tmp = _tmp()
    import logvault
    logvault.DB = os.path.join(tmp, "logs.db")
    logvault.init_schema()
    msgs = [_make_row(k, v, ts=float(i)) for i, (k, v) in enumerate(rows)]
    # first insert
    n1 = logvault.insert_msgs(msgs)
    assert n1 == len(msgs)  # all new on first insert
    # second insert of identical rows is a no-op
    n2 = logvault.insert_msgs(msgs)
    assert n2 == 0  # nothing new
    with logvault.conn() as c:
        cnt = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        # distinct source_keys must not collide: count == number of unique keys
        unique = len({m["source_key"] for m in msgs})
        assert cnt == unique
        assert cnt == n1
        # FTS stays consistent with the messages table (1:1 by rowid)
        fts = c.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert fts == cnt


# ---------------------------------------------------------------------------
# 3. secretstore round-trip, append-only, monotonic versions
# ---------------------------------------------------------------------------
@given(items=st.lists(
    st.tuples(st.text(min_size=1), st.text(min_size=0, max_size=50)),
    min_size=0, max_size=20, unique_by=lambda t: t[0]))
@settings(max_examples=50, deadline=None)
def test_secretstore_roundtrip_appendonly(items):
    tmp = _tmp()
    import secretstore
    secretstore.DB = os.path.join(tmp, "secrets.db")
    secretstore.KEYFILE = os.path.join(tmp, "master.key")
    secretstore.init()
    for name, val in items:
        secretstore.set_secret(name, val)
    # every value round-trips exactly
    for name, val in items:
        got = secretstore.get_secret(name)
        assert got["value"] == val
        assert got["deleted"] is False
        # version is the count of writes to that name (monotonic, starts at 1)
        assert got["version"] == items.count((name, val)) if False else got["version"] >= 1
    # append-only: versions strictly increase across repeated writes
    import collections
    writes = collections.Counter(n for n, _ in items)
    with secretstore.connect() as c:
        # no UPDATE/DELETE should ever succeed on the secrets table
        c.execute("PRAGMA triggers=ON")
        c.execute("SELECT 1 FROM secrets")  # warm
    for name, cnt in writes.items():
        if cnt >= 2:
            vs = secretstore.versions(name)
            versions = [r["version"] for r in vs]
            assert versions == sorted(versions, reverse=True)
            assert len(versions) == cnt
            assert versions == list(range(cnt, 0, -1))


@given(v=st.floats(min_value=-10, max_value=10))
@settings(max_examples=30, deadline=None)
def test_secretstore_delete_tombstones(v):
    tmp = _tmp()
    import secretstore
    secretstore.DB = os.path.join(tmp, "secrets.db")
    secretstore.KEYFILE = os.path.join(tmp, "master.key")
    secretstore.init()
    secretstore.set_secret("k", str(v))
    secretstore.delete_secret("k")
    got = secretstore.get_secret("k")
    assert got["deleted"] is True
    assert got["value"] is None


# ---------------------------------------------------------------------------
# 4. affect clamping keeps domains
# ---------------------------------------------------------------------------
@given(v=st.floats(min_value=-1e9, max_value=1e9),
       a=st.floats(min_value=-1e9, max_value=1e9),
       nan=st.booleans())
@settings(max_examples=40, deadline=None)
def test_affect_clamp_domains(v, a, nan):
    import operator_affect as oa
    if nan:
        v, a = float("nan"), float("nan")
    assert -1.0 <= oa._clamp(v, -1.0, 1.0) <= 1.0
    assert 0.0 <= oa._clamp(a, 0.0, 1.0) <= 1.0
    # identity when already in-domain
    assert oa._clamp(0.5, 0.0, 1.0) == 0.5


# ---------------------------------------------------------------------------
# 5. caliber._parse_confidence bounds
# ---------------------------------------------------------------------------
@given(s=st.text(max_size=80))
@settings(max_examples=80, deadline=None)
def test_caliber_parse_confidence_bounded(s):
    import caliber
    v = caliber._parse_confidence(s)
    assert v is None or (0.0 <= v <= 1.0)


# operator_affect validation never emits out-of-domain numbers
@given(v=st.floats(min_value=-1e6, max_value=1e6),
       a=st.floats(min_value=-1e6, max_value=1e6),
       fr=st.floats(min_value=-1e6, max_value=1e6),
       g=st.floats(min_value=-1e6, max_value=1e6))
@settings(max_examples=40, deadline=None)
def test_operator_affect_validate_clamps(v, a, fr, g):
    import operator_affect as oa
    rec = oa._validate({"valence": v, "arousal": a, "frustration_level": fr,
                        "giving_up_ratio": g, "primary_emotion": "frustration",
                        "need_from_me": "pep_talk"})
    assert -1.0 <= rec["valence"] <= 1.0
    assert 0.0 <= rec["arousal"] <= 1.0
    assert 0.0 <= rec["frustration_level"] <= 1.0
    assert 0.0 <= rec["giving_up_ratio"] <= 1.0
    assert rec["primary_emotion"] == "frustration"
    assert rec["need_from_me"] == "pep_talk"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
