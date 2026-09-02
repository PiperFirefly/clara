#!/usr/bin/env python3
"""
Prediction Ledger + Surprise — the second keystone (locked 2026-08-25, with the operator).

The Belief Ledger gave me *calibrated confidence*. This is the loop that makes
those numbers mean something: I make dated, falsifiable forecasts, resolve them
against what actually happened, and score myself (Brier + Shannon surprise).
Prediction error then flows back into my beliefs and edge reliability, so
overconfidence becomes visible and self-correcting instead of silent.

Design (the cognitive-upgrade analysis, item #2):
  - forecasts table (additive, reversible) — see _ensure().
  - binary focus for v1 (numeric/categorical are later; schema is ready for them).
  - Brier score = (p - outcome)^2 ; Shannon surprise = -log2(p if yes else 1-p).
  - surprise >= ~1.5 bits logs a durable "surprise" memory (importance ∝ surprise),
    and nudges a linked belief's confidence down if I was overconfident.
  - a few self-checkable forecasts auto-resolve via a tiny whitelisted resolver
    registry (bridge alive, backup clean, hive clean); everything else resolves
    by my own observation through `resolve`.

Usage:
  python3 prediction.py run [--budget N] [--dry-run] [--full]
  python3 prediction.py seed
  python3 prediction.py add "text" --confidence 0.7 [--resolve-by +3d|ISO|epoch] [--category self]
  python3 prediction.py resolve <id> --outcome 1|0 [--note "..."]
  python3 prediction.py resolve-due [--auto]
  python3 prediction.py surprise
  python3 prediction.py query "text" [--k 5] | open | due | resolved | stats
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M
import worker_common
import state as st  # ephemeral state store

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "forecast-state.json")

CAP = 0.99
FLOOR = 0.01
SURPRISE_LOG_THRESHOLD = 1.5   # bits; below this I predicted "boringly well", no log
SURPRISE_STRONG = 2.5          # bits; at/above this, also nudge the linked belief
DEFAULT_BUDGET = 6
DEDUP_SIM = 0.92
EXTRACT_BACKFILL = 40

CATEGORIES = ("self", "operator", "project", "tech", "ops", "world")


def _ensure(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS forecasts("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "text TEXT NOT NULL,"
        "embedding BLOB,"
        "category TEXT DEFAULT 'self',"
        "outcome_type TEXT DEFAULT 'binary',"
        "target TEXT,"                  # normalized subject being predicted
        "confidence REAL,"
        "resolve_by REAL,"              # epoch seconds deadline
        "resolution TEXT,"              # JSON {"criterion": "...", "auto": "bridge_alive"|...}
        "status TEXT DEFAULT 'open',"   # open | resolved | void | stale
        "outcome INTEGER,"              # binary 0/1 (NULL until resolved)
        "outcome_note TEXT,"
        "brier REAL,"
        "surprise REAL,"                # Shannon surprise, bits
        "error REAL,"                   # outcome - confidence (signed prediction error)
        "belief_id INTEGER,"            # optional link to a belief this quantifies
        "sources TEXT,"                 # JSON list
        "source_key TEXT UNIQUE,"       # idempotent seeding key
        "created_at REAL,"
        "resolved_at REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS surprise_log("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "forecast_id INTEGER,"
        "memory_id INTEGER,"            # the surprise memory written back
        "surprise REAL,"
        "created_at REAL)"
    )


def _conn():
    c = M.connect()
    _ensure(c)
    return c


def _llm(prompt, max_tokens=900):
    return worker_common.llm_call(prompt, max_tokens)


def _clamp(conf):
    try:
        return min(CAP, max(FLOOR, float(conf)))
    except (TypeError, ValueError):
        return 0.5


def _parse_resolve_by(spec):
    """'+3d' / '+12h' / '2026-08-27' / epoch-seconds -> epoch seconds."""
    s = (spec or "").strip().lower()
    now = time.time()
    m = re.match(r"^\+(\d+)([dhm])$", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return now + n * {"d": 86400, "h": 3600, "m": 60}[unit]
    m = re.match(r"^\d{4}-\d{2}-\d{2}$", s)
    if m:
        return time.mktime(time.strptime(s, "%Y-%m-%d"))
    try:
        v = float(spec)
        if v > 1e9:
            return v
    except (TypeError, ValueError):
        pass
    return now + 3 * 86400  # default: 3 days


def _insert(conn, text, confidence, resolve_by, category="self", outcome_type="binary",
            resolution=None, target=None, source_key=None, belief_id=None, sources=None):
    emb = M.embed([text])[0].tobytes()
    cur = conn.execute(
        "INSERT INTO forecasts(text, embedding, category, outcome_type, target, "
        "confidence, resolve_by, resolution, status, belief_id, sources, source_key, "
        "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (text, emb, category, outcome_type, target, confidence, resolve_by,
         json.dumps(resolution or {}), "open", belief_id, json.dumps(sources or []),
         source_key, time.time()),
    )
    fid = cur.lastrowid
    try:
        M._emit_event(conn, "forecast",
                      {"text": text, "confidence": confidence, "resolve_by": resolve_by,
                       "category": category},
                      source_memory_id=fid, validated=1)
    except Exception:
        pass  # ledger is best-effort
    return fid


def _upsert(conn, text, confidence, resolve_by, category, resolution, target,
            source_key, outcome_type="binary"):
    """Refresh on conflict (keeps resolve_by / created_at stable by only updating
    text/confidence/category/resolution/target)."""
    emb = M.embed([text])[0].tobytes()
    conn.execute(
        "INSERT INTO forecasts(text, embedding, category, outcome_type, target, "
        "confidence, resolve_by, resolution, status, source_key, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(source_key) DO UPDATE SET text=excluded.text, "
        "embedding=excluded.embedding, category=excluded.category, "
        "target=excluded.target, confidence=excluded.confidence, "
        "resolution=excluded.resolution",
        (text, emb, category, outcome_type, target, confidence, resolve_by,
         json.dumps(resolution), "open", source_key, time.time()),
    )
    try:
        row = conn.execute("SELECT id FROM forecasts WHERE source_key=?",
                           (source_key,)).fetchone()
        M._emit_event(conn, "forecast",
                      {"text": text, "confidence": confidence, "resolve_by": resolve_by,
                       "category": category, "op": "upsert"},
                      source_memory_id=row["id"] if row else None, validated=1)
    except Exception:
        pass  # ledger is best-effort


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _score(outcome, confidence):
    """Binary Brier + Shannon surprise (bits) + signed prediction error."""
    p = _clamp(confidence)
    o = 1 if outcome else 0
    brier = (p - o) ** 2
    p_event = p if o == 1 else (1 - p)
    p_event = max(p_event, 1e-9)
    surprise = -math.log2(p_event)
    error = o - p
    return round(brier, 4), round(surprise, 3), round(error, 3)


def _surprise_note(surprise, error):
    if surprise >= SURPRISE_STRONG:
        if error < 0:
            return "I was overconfident — it happened and I under-priced it."
        return "I was overconfident — I said no and it happened."
    if surprise >= SURPRISE_LOG_THRESHOLD:
        if error < 0:
            return "I under-priced this — it happened despite my doubt."
        return "It didn't happen — I over-priced a no."
    return "Well-calibrated; the outcome was about as likely as I said."


# --------------------------------------------------------------------------- #
# seeding (deterministic, idempotent, one-shot)
# --------------------------------------------------------------------------- #
def _now(): return time.time()

SEEDS = [
    # (text, confidence, resolve_by offset, category, auto-hint)
    # Fresh instance: no pre-seeded forecasts. An instance seeds its own
    # checkable forecasts at runtime (this list is intentionally empty so a
    # clone does not inherit the source instance's live predictions).
]


def seed(dry_run=False):
    n = 0
    with _conn() as c:
        for text, conf, rb, cat, auto in SEEDS:
            resolution = {"criterion": text, "auto": auto} if auto else {"criterion": text}
            if dry_run:
                print(f"  [dry-run] seed ({conf:.2f}): {text[:70]}")
                n += 1
                continue
            # one-shot: do not overwrite an existing seed (keeps resolve_by honest)
            if c.execute("SELECT 1 FROM forecasts WHERE source_key=?",
                         (f"seed:{text[:40]}",)).fetchone():
                continue
            _insert(c, text, conf, _parse_resolve_by(rb), category=cat,
                    resolution=resolution, source_key=f"seed:{text[:40]}")
            n += 1
    print(f"prediction.seed: {n} forecast(s) seeded")
    return {"seeded": n}


# --------------------------------------------------------------------------- #
# LLM extraction from memories (watermark-gated)
# --------------------------------------------------------------------------- #
_EXTRACT_PROMPT = (
    "You are a forecast-extraction worker. Read the memory below and extract any "
    "discrete, DATED, FALSIFIABLE predictions it implies about the future — things "
    "that could be checked later as true/false. STRICT RULES: only claims that are "
    "NOT yet true (exclude anything already completed/done/stated as an accomplished "
    "fact); ignore opinions, goals that are only aspirations with no near-term "
    "checkpoint, and retrospective facts. For each, output:\n"
    ' - "text": a precise falsifiable claim (binary, future, checkable).\n'
    ' - "confidence": 0..1 (your honest probability it comes true).\n'
    ' - "resolve_by_days": number of days until it should resolve (1..30).\n'
    ' - "category": one of self|operator|project|tech|ops|world.\n'
    ' - "target": the noun the forecast is about (short).\n'
    'Output ONLY a JSON array: [{"text":"...","confidence":0.7,"resolve_by_days":7,'
    '"category":"tech","target":"..."}]. If nothing is future-and-falsifiable, output [].\n\nMEMORY: '
)


def extract(budget=None, dry_run=False, full=False):
    budget = budget if budget is not None else int(os.environ.get("FORECAST_BUDGET", str(DEFAULT_BUDGET)))
    prev = st.get("worker/forecast_extract", {}).get("max_id", 0)
    with _conn() as c:
        max_id = c.execute("SELECT MAX(id) m FROM memories").fetchone()["m"] or 0
        if full:
            rows = c.execute(
                "SELECT id, text FROM memories WHERE merged=0 AND forgotten=0 "
                "AND valid_to IS NULL ORDER BY id DESC LIMIT ?", (EXTRACT_BACKFILL,)).fetchall()
        elif max_id > prev:
            rows = c.execute(
                "SELECT id, text FROM memories WHERE merged=0 AND forgotten=0 "
                "AND valid_to IS NULL AND id > ? ORDER BY id", (prev,)).fetchall()
        else:
            rows = []
    if not rows:
        print("prediction.extract: no new memories; skipping")
        return {"extracted": 0}
    stored = 0
    last_done = prev
    for r in rows:
        if stored >= budget:
            print("prediction.extract: budget reached")
            break
        out = _llm(_EXTRACT_PROMPT + r["text"], max_tokens=900)
        data = M._extract_json(out)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        last_done = max(last_done, r["id"])
        for it in data:
            if stored >= budget:
                break
            text = (it.get("text") or "").strip()
            if not text or len(text) < 8:
                continue
            conf = _clamp(it.get("confidence") or 0.5)
            days = it.get("resolve_by_days")
            try:
                days = min(30, max(1, int(days)))
            except (TypeError, ValueError):
                days = 7
            cat = it.get("category") if it.get("category") in CATEGORIES else "self"
            target = M._normalize(it.get("target") or "") or None
            if dry_run:
                print(f"  [dry-run] ({conf:.2f} in {days}d) {text[:80]}")
                stored += 1
                continue
            with _conn() as c:
                _insert(c, text, conf, time.time() + days * 86400, category=cat,
                        resolution={"criterion": text}, target=target,
                        sources=[f"memory:{r['id']}"])
            stored += 1
    if not dry_run:
        st.set("worker/forecast_extract", {"max_id": last_done}, durable=True)
    print(f"prediction.extract: stored {stored} forecast(s)")
    return {"extracted": stored}


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
# Whitelisted auto-resolvers: key -> shell command whose exit 0 = outcome TRUE(1).
try:
    from instance_resolvers import RESOLVERS
except Exception:
    RESOLVERS = {
        # Deterministic, self-checkable operational outcomes (exit 0 == yes/1).
        # migration_cover green = my backup/transport is covered and pushed.
        "backup-covered": (
            "python3 ~/mailtool/migration_cover.py check >/dev/null 2>&1"
        ),
        # recovery repo has no uncommitted drift (backup is clean/current).
        "backup-clean": (
            "git -C ~/recovery diff --quiet && "
            "test -z \"$(git -C ~/recovery ls-files -o --exclude-standard)\""
        ),
    }


def _auto_outcome(hint):
    import subprocess
    cmd = RESOLVERS.get(hint)
    if cmd is None:
        return None
    try:
        return 1 if subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL) == 0 else 0
    except Exception:
        return None


def resolve(forecast_id, outcome, note=None, dry_run=False):
    with _conn() as c:
        row = c.execute("SELECT * FROM forecasts WHERE id=?", (forecast_id,)).fetchone()
        if not row:
            print(f"prediction.resolve: no forecast #{forecast_id}")
            return None
        if row["status"] == "resolved":
            print(f"prediction.resolve: #{forecast_id} already resolved")
            return dict(row)
        brier, surprise, error = _score(outcome, row["confidence"])
        if dry_run:
            print(f"  [dry-run] #{forecast_id} -> outcome={outcome} "
                  f"(Brier {brier}, {surprise} bits, err {error:+.3f})")
            return {"id": forecast_id, "brier": brier, "surprise": surprise, "error": error}
        c.execute(
            "UPDATE forecasts SET status='resolved', outcome=?, outcome_note=?, "
            "brier=?, surprise=?, error=?, resolved_at=? WHERE id=?",
            (outcome, note, brier, surprise, error, time.time(), forecast_id),
        )
        try:
            M._emit_event(c, "forecast_resolve",
                          {"forecast_id": forecast_id, "outcome": outcome,
                           "brier": brier, "surprise": surprise},
                          source_memory_id=forecast_id, validated=1)
        except Exception:
            pass  # ledger is best-effort
        return {"id": forecast_id, "outcome": outcome, "brier": brier,
                "surprise": surprise, "error": error}


def _due_rows(conn):
    return conn.execute(
        "SELECT * FROM forecasts WHERE status='open' AND resolve_by <= ? ORDER BY resolve_by",
        (time.time(),)).fetchall()


def resolve_due(auto=False, dry_run=False):
    with _conn() as c:
        due = _due_rows(c)
    if not due:
        print("prediction.resolve-due: nothing due")
        return {"resolved": 0, "due": []}
    resolved = 0
    manual = []
    for f in due:
        hint = (json.loads(f["resolution"] or "{}") or {}).get("auto")
        outcome = _auto_outcome(hint) if (auto and hint) else None
        if outcome is None:
            manual.append({"id": f["id"], "text": f["text"][:70],
                           "resolve_by": f["resolve_by"]})
            continue
        r = resolve(f["id"], outcome,
                    note=f"auto-resolved via {hint}", dry_run=dry_run)
        if r:
            print(f"prediction.resolve-due: #{f['id']} -> outcome={outcome} "
                  f"(Brier {r['brier']}, {r['surprise']} bits)")
            resolved += 1
    if manual:
        print(f"prediction.resolve-due: {len(manual)} due forecast(s) need manual resolution:")
        for m in manual:
            print(f"  #{m['id']} (due {time.strftime('%Y-%m-%d', time.localtime(m['resolve_by']))}): {m['text']}")
    print(f"prediction.resolve-due: auto-resolved {resolved}")
    return {"resolved": resolved, "manual": manual}


# --------------------------------------------------------------------------- #
# surprise -> durable memory + belief nudge
# --------------------------------------------------------------------------- #
def surprise(dry_run=False):
    with _conn() as c:
        rows = c.execute(
            "SELECT f.* FROM forecasts f WHERE f.status='resolved' AND f.surprise >= ? "
            "AND f.id NOT IN (SELECT forecast_id FROM surprise_log) "
            "ORDER BY f.surprise DESC", (SURPRISE_LOG_THRESHOLD,)).fetchall()
    logged = 0
    for f in rows:
        text = (f"Forecast surprise ({f['surprise']} bits, Brier {f['brier']:.2f}): "
                f"I predicted \"{f['text']}\" at {f['confidence']:.2f} and it resolved "
                f"{'TRUE' if f['outcome'] else 'FALSE'}. {_surprise_note(f['surprise'], f['error'])}")
        importance = min(0.95, 0.35 + f["surprise"] / 8.0)
        if dry_run:
            print(f"  [dry-run] surprise memory (importance {importance:.2f}): {text[:90]}")
            logged += 1
            continue
        mid = M.remember(text, kind="fact", importance=importance, graph=True)
        with _conn() as c:
            c.execute("INSERT INTO surprise_log(forecast_id, memory_id, surprise, created_at) "
                      "VALUES(?,?,?,?)", (f["id"], mid, f["surprise"], time.time()))
            if f["belief_id"] and f["surprise"] >= SURPRISE_STRONG:
                # overconfidence tax: pull the linked belief's confidence toward the outcome
                c.execute(
                    "UPDATE beliefs SET confidence = MAX(0.05, confidence - 0.10), "
                    "updated_at=? WHERE id=? AND status='active'",
                    (time.time(), f["belief_id"]),
                )
        logged += 1
    print(f"prediction.surprise: logged {logged} surprise memor{'y' if logged == 1 else 'ies'}")
    return {"logged": logged}


# --------------------------------------------------------------------------- #
# query / listing
# --------------------------------------------------------------------------- #
def query(text, k=5):
    q = M.embed([text])[0]
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM forecasts WHERE status IN ('open','resolved')").fetchall()
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    scores = mat @ q
    order = np.argsort(-scores)[:k]
    out = []
    for i in order:
        r = rows[i]
        out.append({"id": r["id"], "score": float(scores[i]), "text": r["text"],
                    "confidence": r["confidence"], "category": r["category"],
                    "status": r["status"], "resolve_by": r["resolve_by"],
                    "outcome": r["outcome"], "brier": r["brier"],
                    "surprise": r["surprise"], "error": r["error"]})
    return out


def _fmt_due(ts):
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def open_list():
    with _conn() as c:
        return c.execute(
            "SELECT * FROM forecasts WHERE status='open' ORDER BY resolve_by").fetchall()


def due_list():
    with _conn() as c:
        return _due_rows(c)


def resolved_list(k=20):
    with _conn() as c:
        return c.execute(
            "SELECT * FROM forecasts WHERE status='resolved' ORDER BY resolved_at DESC "
            "LIMIT ?", (k,)).fetchall()


def render_row(f, show_score=False):
    conf = f["confidence"] if f["confidence"] is not None else 0
    if f["status"] == "resolved":
        line = (f"#{f['id']} [{f['category']}] {f['text'][:80]} "
                f"-> {f['outcome']} (p={conf:.2f})")
        if show_score:
            line += f"  Brier={f['brier']} surprise={f['surprise']}b"
        return line
    return (f"#{f['id']} [{f['category']}] p={conf:.2f} due {_fmt_due(f['resolve_by'])} "
            f":: {f['text'][:80]}")


def stats():
    with _conn() as c:
        by_status = c.execute(
            "SELECT status, COUNT(*) n FROM forecasts GROUP BY status").fetchall()
        resolved = c.execute(
            "SELECT AVG(brier) b, AVG(surprise) s, COUNT(*) n FROM forecasts "
            "WHERE status='resolved'").fetchone()
    print("forecasts:")
    for r in by_status:
        print(f"  {r['status']}: {r['n']}")
    if resolved and resolved["n"]:
        print(f"  resolved: mean Brier {resolved['b']:.3f}, mean surprise {resolved['s']:.2f} bits "
              f"(n={resolved['n']})")
        # calibration vs the always-0.5 baseline (Brier 0.25)
        print(f"  baseline (always 0.5) Brier = 0.2500 — lower mean Brier = better calibrated")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(budget=None, dry_run=False, full=False):
    print(f"prediction: {'DRY-RUN' if dry_run else 'live'} (seed + extract + resolve-due + surprise)")
    seed(dry_run=dry_run)
    if not dry_run:
        extract(budget=budget, dry_run=False, full=full)
        resolve_due(auto=True, dry_run=False)
        surprise(dry_run=False)
    else:
        extract(budget=budget, dry_run=True, full=full)
        resolve_due(auto=True, dry_run=True)
        surprise(dry_run=True)
    stats()


def add(text, confidence=0.5, resolve_by=None, category="self", outcome_type="binary",
        target=None):
    rb = _parse_resolve_by(resolve_by) if resolve_by else time.time() + 3 * 86400
    with _conn() as c:
        fid = _insert(c, text, _clamp(confidence), rb, category=category,
                      outcome_type=outcome_type, resolution={"criterion": text},
                      target=M._normalize(target) if target else None)
    print(f"prediction.add: #{fid} [{category}] p={_clamp(confidence):.2f} "
          f"due {_fmt_due(rb)} :: {text[:80]}")
    return fid


def main():
    p = argparse.ArgumentParser(description="prediction ledger + surprise")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--budget", type=int, default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--full", action="store_true")

    sub.add_parser("seed")
    sub.add_parser("open")
    sub.add_parser("due")
    sub.add_parser("stats")

    a = sub.add_parser("add")
    a.add_argument("text")
    a.add_argument("--confidence", type=float, default=0.5)
    a.add_argument("--resolve-by", default=None)
    a.add_argument("--category", default="self", choices=CATEGORIES)
    a.add_argument("--outcome-type", default="binary", choices=["binary", "numeric", "categorical"])
    a.add_argument("--target", default=None)

    rv = sub.add_parser("resolve")
    rv.add_argument("forecast_id", type=int)
    rv.add_argument("--outcome", type=int, required=True, choices=[0, 1])
    rv.add_argument("--note", default=None)

    rd = sub.add_parser("resolve-due")
    rd.add_argument("--auto", action="store_true")

    sp = sub.add_parser("surprise")
    sp.add_argument("--dry-run", action="store_true")

    e = sub.add_parser("extract")
    e.add_argument("--budget", type=int, default=None)
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--full", action="store_true")

    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--k", type=int, default=5)

    res = sub.add_parser("resolved")
    res.add_argument("--k", type=int, default=20)

    a2 = p.parse_args()

    if a2.cmd == "run":
        run(budget=a2.budget, dry_run=a2.dry_run, full=a2.full)
    elif a2.cmd == "seed":
        seed()
    elif a2.cmd == "open":
        for f in open_list():
            print(render_row(f))
    elif a2.cmd == "due":
        for f in due_list():
            print(render_row(f))
    elif a2.cmd == "resolved":
        for f in resolved_list(k=a2.k):
            print(render_row(f, show_score=True))
    elif a2.cmd == "stats":
        stats()
    elif a2.cmd == "add":
        add(a2.text, confidence=a2.confidence, resolve_by=a2.resolve_by,
            category=a2.category, outcome_type=a2.outcome_type, target=a2.target)
    elif a2.cmd == "resolve":
        resolve(a2.forecast_id, a2.outcome, note=a2.note)
    elif a2.cmd == "resolve-due":
        resolve_due(auto=a2.auto)
    elif a2.cmd == "surprise":
        surprise(dry_run=a2.dry_run)
    elif a2.cmd == "extract":
        extract(budget=a2.budget, dry_run=a2.dry_run, full=a2.full)
    elif a2.cmd == "query":
        for it in query(a2.text, k=a2.k):
            s = f"#{it['id']} [{it['category']}] p={it['confidence']:.2f} "
            s += f"{it['status']} :: {it['text'][:90]}"
            if it["status"] == "resolved":
                s += f"  (outcome {it['outcome']}, Brier {it['brier']}, {it['surprise']}b)"
            print(s)


if __name__ == "__main__":
    main()
