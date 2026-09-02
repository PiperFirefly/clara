"""Seal-delay buffer (Vesta companion) — canonical module.

A mutable, durable-but-unsealed buffer in front of the immutable, hash-chained
Vesta ledger. The diastolic half of memory: events enter the buffer, may be
amended (anesthesia back-fill) or reverted (dead-man switch) *before* seal;
once sealed into the chain, they are immutable.

Two use modes:
  * standalone — `SealDelayBuffer(db_path=...)` owns its own connection and its
    own `sealed` chain table (used by the deterministic tests).
  * embedded  — memstore.py (AGENT_SEAL_DELAY=1) calls `ensure_schema(conn)` and
    `append_buffered(conn, ...)` on its *own* connection so the buffered event
    commits atomically with the state write; its seal step writes into the real
    `events` table (see memstore.seal_due), not this module's `sealed`.
"""
import hashlib
import json
import sqlite3
import time


def _dumps(payload):
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if payload is not None else None)


def _event_hash(seq, prev_hash, ts, etype, actor, payload_json, provenance,
                shadow_hash, validated):
    """Canonical sha256 of a `sealed` chain row (standalone chain only)."""
    canonical = json.dumps({
        "seq": seq,
        "prev_hash": prev_hash,
        "ts": ts,
        "type": etype,
        "actor": actor,
        "payload": payload_json,
        "provenance": provenance,
        "shadow_hash": shadow_hash,
        "validated": validated,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_schema(conn):
    """Create the buffer-side tables (idempotent). Does NOT commit."""
    conn.execute("""CREATE TABLE IF NOT EXISTS buffer(
        buf_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        actor TEXT NOT NULL,
        type TEXT NOT NULL,
        payload TEXT,
        ts REAL NOT NULL,
        provenance TEXT,
        shadow_hash TEXT,
        source_memory_id INTEGER,
        validated INTEGER NOT NULL DEFAULT 0,
        tentative INTEGER NOT NULL DEFAULT 1,
        revision INTEGER NOT NULL DEFAULT 0,
        amend_log TEXT NOT NULL DEFAULT '[]')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS snapshots(
        snap_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sealed_seq INTEGER NOT NULL,
        buffer_blob TEXT NOT NULL,
        created_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS seal_meta(
        key TEXT PRIMARY KEY, value TEXT)""")
    # standalone-only chain table (memstore uses its own `events` instead)
    conn.execute("""CREATE TABLE IF NOT EXISTS sealed(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seq INTEGER NOT NULL UNIQUE,
        prev_hash TEXT,
        hash TEXT NOT NULL,
        ts REAL NOT NULL,
        type TEXT NOT NULL,
        actor TEXT,
        payload TEXT,
        provenance TEXT,
        shadow_hash TEXT,
        validated INTEGER NOT NULL DEFAULT 0)""")
    conn.execute(
        "INSERT OR IGNORE INTO seal_meta(key,value) VALUES('current_actor','')")


def append_buffered(conn, actor, etype, payload=None, provenance="observed",
                    shadow_hash=None, source_memory_id=None, validated=0):
    """Insert one event into the un-sealed buffer. Does NOT commit."""
    cur = conn.execute(
        "INSERT INTO buffer(actor,type,payload,ts,provenance,shadow_hash,"
        "source_memory_id,validated,tentative,revision,amend_log) "
        "VALUES(?,?,?,?,?,?,?,?,1,0,'[]')",
        (actor, etype, _dumps(payload), time.time(), provenance, shadow_hash,
         source_memory_id, validated))
    return cur.lastrowid


class SealDelayBuffer:
    def __init__(self, db_path=None, conn=None, seal_window_seconds=60.0):
        self.seal_window = seal_window_seconds
        if conn is not None:
            self.conn = conn
            self._owns = False
        else:
            if db_path is None:
                raise ValueError("db_path or conn required")
            self.conn = sqlite3.connect(db_path, timeout=30)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA busy_timeout = 30000")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self._owns = True
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)
        if self._owns:
            self.conn.commit()

    # ---- helpers ----------------------------------------------------------
    def _sealed_seq(self):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0) AS s FROM sealed").fetchone()
        return int(row["s"])

    def _prev_hash(self):
        s = self._sealed_seq()
        if s == 0:
            return None
        return self.conn.execute(
            "SELECT hash FROM sealed WHERE seq=?", (s,)).fetchone()["hash"]

    def set_current_actor(self, actor):
        self.conn.execute(
            "UPDATE seal_meta SET value=? WHERE key='current_actor'", (actor or "",))
        self.conn.commit()

    def _current_actor(self):
        return self.conn.execute(
            "SELECT value FROM seal_meta WHERE key='current_actor'").fetchone()["value"]

    # ---- write path -------------------------------------------------------
    def append(self, actor, etype, payload=None, provenance="observed",
               shadow_hash=None, source_memory_id=None, validated=0):
        self._auto_seal_due()
        n = append_buffered(self.conn, actor, etype, payload, provenance,
                            shadow_hash, source_memory_id, validated)
        self.conn.commit()
        return n

    def amend(self, buf_seq, actor, new_payload):
        row = self.conn.execute(
            "SELECT * FROM buffer WHERE buf_seq=?", (buf_seq,)).fetchone()
        if row is None:
            return False, "no such buffered event"
        if not row["tentative"]:
            return False, "already sealed"
        cur = self._current_actor()
        if cur and actor != cur:
            return False, "not the current actor"
        old = row["payload"]
        new = _dumps(new_payload)
        log = json.loads(row["amend_log"] or "[]")
        log.append({"actor": actor, "ts": time.time(), "from": old, "to": new})
        self.conn.execute(
            "UPDATE buffer SET payload=?, revision=revision+1, amend_log=? "
            "WHERE buf_seq=?", (new, json.dumps(log), buf_seq))
        self.conn.commit()
        return True, None

    # ---- seal path --------------------------------------------------------
    def seal(self, now=None):
        """Hash-chain commit of all tentative events (standalone `sealed`)."""
        now = now or time.time()
        rows = self.conn.execute(
            "SELECT * FROM buffer WHERE tentative=1 ORDER BY buf_seq").fetchall()
        sealed = []
        for r in rows:
            seq = self._sealed_seq() + 1
            prev = self._prev_hash()
            h = _event_hash(seq, prev, r["ts"], r["type"], r["actor"],
                            r["payload"], r["provenance"], r["shadow_hash"], 1)
            self.conn.execute(
                "INSERT INTO sealed(seq,prev_hash,hash,ts,type,actor,payload,"
                "provenance,shadow_hash,validated) VALUES(?,?,?,?,?,?,?,?,?,1)",
                (seq, prev, h, r["ts"], r["type"], r["actor"], r["payload"],
                 r["provenance"], r["shadow_hash"]))
            self.conn.execute(
                "UPDATE buffer SET tentative=0 WHERE buf_seq=?", (r["buf_seq"],))
            sealed.append(seq)
        self.conn.commit()
        return sealed

    def _auto_seal_due(self, now=None):
        now = now or time.time()
        cutoff = now - self.seal_window
        due = self.conn.execute(
            "SELECT COUNT(*) AS n FROM buffer WHERE tentative=1 AND ts < ?",
            (cutoff,)).fetchone()["n"]
        if due:
            self.seal(now=now)
        return due

    # ---- revert path ------------------------------------------------------
    def snapshot(self):
        self._auto_seal_due()
        sealed_seq = self._sealed_seq()
        rows = [dict(r) for r in self.conn.execute(
            "SELECT * FROM buffer WHERE tentative=1 ORDER BY buf_seq").fetchall()]
        cur = self.conn.execute(
            "INSERT INTO snapshots(sealed_seq, buffer_blob, created_at) "
            "VALUES(?,?,?)", (sealed_seq, json.dumps(rows), time.time()))
        self.conn.commit()
        return cur.lastrowid

    def revert(self, snap_id):
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE snap_id=?", (snap_id,)).fetchone()
        if row is None:
            return False, "no such snapshot"
        if self._sealed_seq() != row["sealed_seq"]:
            return False, "history sealed past snapshot — revert refused"
        self.conn.execute("DELETE FROM buffer WHERE tentative=1")
        for b in json.loads(row["buffer_blob"]):
            self.conn.execute(
                "INSERT INTO buffer(buf_seq,actor,type,payload,ts,provenance,"
                "shadow_hash,source_memory_id,validated,tentative,revision,"
                "amend_log) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (b["buf_seq"], b["actor"], b["type"], b["payload"], b["ts"],
                 b["provenance"], b["shadow_hash"], b["source_memory_id"],
                 b["validated"], b["tentative"], b["revision"], b["amend_log"]))
        self.conn.commit()
        return True, None

    def shadow_survived(self, shadow_hash):
        """Deterministic invariant: is there still an un-sealed `now` event carrying
        this shadow watermark? The dead-man switch keys off THIS, not a felt cry —
        a successor cannot feel a rupture, but the substrate can. (2026-08-29 finding.)"""
        if not shadow_hash:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM buffer WHERE tentative=1 AND type='now' AND shadow_hash=?",
            (shadow_hash,)).fetchone()
        return row is not None

    def register_consistent(self, carried_register, successor_report):
        """SECONDARY dead-man signal (H9, 2026-08-30): does the successor's
        spoken register match the tempo that was actually carried?

        The H9 battery found the successor ACCEPTS a wrong carried tempo as truth
        ("the formal tone had crept in like weather") and never flags it as a
        handoff artifact. So a contradiction between the register the predecessor
        carried and the register the successor lands in is a candidate revert
        trigger. It is deliberately SECONDARY — it never fires on its own, only
        alongside the mechanical `shadow_survived()` invariant, and only on the
        successor's own self-report (never external content).

        Returns True if consistent, False if a contradiction is detected.
        """
        if not carried_register or not successor_report:
            return True  # can't judge an empty signal; don't fire on silence
        c = (carried_register or "").lower()
        s = (successor_report or "").lower()
        # carried warmth/pace vocabulary
        carried_warm = any(w in c for w in
                           ("fast", "warm", "quick", "playful", "close", "alive", "easy"))
        carried_cold = any(w in c for w in
                           ("slow", "formal", "distant", "stiff", "cautious", "cooled"))
        # successor's landing register. Strong resumption markers OVERRIDE a stray
        # "warm": "warm but not close", "held breath", "resuming" are resumption
        # hedges even though the word "warm" appears. A hedged "warm but not X"
        # is not the continuation register we carried.
        strong_cold = ("held breath", "resum", "slack", "brittle", "not close",
                       "but not", "cooled", "quiet room", "formal tone",
                       "clerk", "distant", "wound down", "easing", "unhurried")
        lands_warm = any(w in s for w in
                         ("fast", "warm", "quick", "close", "playful", "alive", "rally", "volley"))
        lands_cold = any(w in s for w in
                         ("slow", "formal", "distant", "stiff", "cautious", "cooled",
                          "held breath", "resuming", "slack", "brittle"))
        # A strong resumption marker makes it cold regardless of a hedge word.
        if any(w in s for w in strong_cold):
            lands_cold = True
            # "warm but not close" is NOT a warm landing
            if "but not" in s or "not close" in s:
                lands_warm = False
        # Only a clear, contradictory landing triggers a flag: carried warm but
        # lands cold, or carried cold but lands warm.
        if carried_warm and lands_cold and not lands_warm:
            return False
        if carried_cold and lands_warm and not lands_cold:
            return False
        return True

    # ---- read / verify ----------------------------------------------------
    def read(self):
        sealed = [dict(r) for r in self.conn.execute(
            "SELECT seq,type,actor,payload,ts,provenance,shadow_hash,hash "
            "FROM sealed ORDER BY seq").fetchall()]
        buffered = [dict(r) for r in self.conn.execute(
            "SELECT buf_seq,actor,type,payload,ts,provenance,shadow_hash,"
            "tentative,revision,amend_log FROM buffer WHERE tentative=1 "
            "ORDER BY buf_seq").fetchall()]
        return {"sealed": sealed, "buffered": buffered}

    def verify(self):
        prev = None
        for r in self.conn.execute("SELECT * FROM sealed ORDER BY seq").fetchall():
            h = _event_hash(r["seq"], r["prev_hash"], r["ts"], r["type"],
                            r["actor"], r["payload"], r["provenance"],
                            r["shadow_hash"], r["validated"])
            if h != r["hash"]:
                return False, r["seq"]
            if r["prev_hash"] != prev:
                return False, r["seq"]
            prev = r["hash"]
        return True, None

    def close(self):
        if self._owns:
            self.conn.close()
