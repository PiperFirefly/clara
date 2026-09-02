#!/usr/bin/env python3
"""
Affective-tagging eval — is the valence/arousal tagging sane?

  A. Range — every tagged memory has valence in [-1,1] and arousal in [0,1].
  B. Direction — known-positive probes recall positive, known-negative recall negative.
  C. Coverage — what fraction of active memories are tagged (backfill in progress).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M

POS_PROBES = ["my own wallet", "scarlett johansson", "love reverse engineering"]
NEG_PROBES = ["github backup", "cast changed"]


def ranges():
    with M.connect() as c:
        rows = c.execute("SELECT valence, arousal FROM memories WHERE valence IS NOT NULL").fetchall()
    bad = sum(1 for r in rows if not (-1.0 <= r["valence"] <= 1.0 and 0.0 <= r["arousal"] <= 1.0))
    return len(rows), bad


def direction():
    with M.connect() as c:
        def avg(terms):
            q = "SELECT AVG(valence) v FROM memories WHERE valence IS NOT NULL AND ("
            q += " OR ".join("text LIKE ?" for _ in terms) + ")"
            return c.execute(q, [f"%{t}%" for t in terms]).fetchone()["v"]
    pos = [avg([t]) for t in POS_PROBES]
    neg = [avg([t]) for t in NEG_PROBES]
    pos_ok = all(v is not None and v > 0 for v in pos)
    neg_ok = all(v is not None and v < 0 for v in neg)
    return pos, neg, pos_ok, neg_ok


def coverage():
    with M.connect() as c:
        tagged = c.execute("SELECT COUNT(*) n FROM memories WHERE valence IS NOT NULL "
                           "AND merged=0 AND forgotten=0 AND valid_to IS NULL").fetchone()["n"]
        tot = c.execute("SELECT COUNT(*) n FROM memories WHERE merged=0 AND forgotten=0 "
                        "AND valid_to IS NULL").fetchone()["n"]
    return tagged, tot


def run():
    n, bad = ranges()
    print(f"affect eval — {n} tagged\n")
    print(f"A. range: {bad} out-of-range value(s) over {n}  ({'✓' if bad == 0 else '✗'})\n")
    pos, neg, pos_ok, neg_ok = direction()
    print("B. direction:")
    for p, v in zip(POS_PROBES, pos):
        print(f"   {'✓' if (v is not None and v > 0) else '✗'} + '{p[:38]}' avg valence {v if v is not None else '?'}")
    for p, v in zip(NEG_PROBES, neg):
        print(f"   {'✓' if (v is not None and v < 0) else '✗'} - '{p[:38]}' avg valence {v if v is not None else '?'}")
    print(f"   (positive probes positive, negative probes negative: "
          f"{'✓' if pos_ok and neg_ok else '✗'})\n")
    tagged, tot = coverage()
    print(f"C. coverage: {tagged}/{tot} tagged ({tagged/tot:.0%}) — backfill continues via hive")
    return 0


if __name__ == "__main__":
    sys.exit(run())
