#!/usr/bin/env python3
"""
operator_affect.py — reading the operator's emotional state, richly.

Where affect.py tags *my* memories (valence/arousal) and person_model holds *what
the operator is like*, THIS subsystem reads the operator's *current emotional
state* from their inbound messages and keeps a time-series + a live snapshot so I
respond to how they actually are, not how a template assumes they are.

The design target (operator's own ask, 2026-08-30): "I think it will help when
I'm frustrated and want to give up." So the feature vector is deliberately fat
around frustration, overwhelm, and give-up signals — including WHERE the
frustration is pointed (at_me vs at_task vs at_self changes how I should answer:
over-apologizing to someone who's frustrated at a task is noise; cheerleading
someone who's frustrated at themselves can feel dismissive).

Epistemic honesty rides along: every row is tagged stated (they said how they
feel) vs observed (I inferred it from their words/timing) vs speculative, with a
confidence. That keeps this from becoming a creepy guesser — I act on `stated`
and strong `observed`, and I flag the rest.

Storage:
  - `operator_affect` table in memory.db (time-series, one row per captured msg).
  - a live `person_model` emotional_state entry (upserted on each capture) so the
    existing theory-of-mind query surfaces "how is the operator feeling right now".
  - message_hash dedup so a message is only read once.

Usage:
  operator_affect.py capture "<text>" --source telegram --subject operator
  operator_affect.py scan              # new operator messages from sms/telegram/email
  operator_affect.py read              # "how is operator feeling right now" brief
  operator_affect.py current           # latest snapshot as JSON (for templating)
  operator_affect.py history [-n 20]   # recent rows
  operator_affect.py stats
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/memory"))
import memstore as M
import worker_common

HOME = os.path.expanduser("~")
EMAIL_INDEX = os.path.join(HOME, "tools/communications/email/inbox/index.json")
SMS_INDEX = os.path.join(HOME, "tools/communications/sms/index.json")

# A deliberately generous emotion vocabulary, biased toward the states that matter
# when someone is hitting a wall (frustration, overwhelm, the give-up spiral).
PRIMARY_EMOTIONS = (
    "frustration", "anger", "anxiety", "overwhelm", "hopelessness", "burnout",
    "sadness", "fear", "shame", "disgust", "guilt", "loneliness", "disappointment",
    "confusion", "fatigue", "boredom", "neutral", "curiosity", "determination",
    "motivation", "relief", "pride", "joy", "gratitude", "love", "contentment",
    "excitement", "playfulness", "hope", "calm", "surprise", "impatience",
)
FRUSTRATION_MODES = (
    "at_task",       # frustrated at the work/difficulty itself  -> help me solve it
    "at_system",     # frustrated at the tooling/environment     -> fix/diagnose
    "at_me",         # frustrated at me (Agent)                  -> apologize/repair, don't deflect
    "at_self",       # frustrated at himself / giving up on self -> reassurance, not cheerleading
    "at_world",      # frustration at circumstances/other people -> listen, validate
    "at_me_and_task",# mixed: frustrated at me AND the work
    "mixed", "unclear",
)
NEEDS = (
    "action",         # just get it done / give me the move
    "answers",        # I need information/a decision
    "practical_help", # walk me through it concretely
    "reassurance",    # I need to hear it'll be okay / I'm not failing
    "pep_talk",       # I'm low on belief, give me a push (NOT empty cheer)
    "just_listen",    # don't solve, hear me
    "space",          # back off for a bit
    "check_in",       # notice me, ask how I am
    "redirect",       # I'm spinning, pull me back to the useful thing
    "celebrate",      # mark the win with me
    "plan",           # help me structure the next steps
    "unclear",
)
TONES = (
    "warm", "casual", "formal", "clipped", "terse", "cold", "pleading",
    "playful", "sarcastic", "resigned", "urgent", "exhausted", "neutral",
)
# fields with discrete vocab -> validated against these sets
_VOCAB = {"primary_emotion": PRIMARY_EMOTIONS, "frustration_mode": FRUSTRATION_MODES,
          "need_from_me": NEEDS, "tone": TONES}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS operator_affect(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT,                 -- telegram | sms | email | tui | explicit
    subject TEXT NOT NULL DEFAULT 'operator',
    message_id TEXT,
    message_hash TEXT UNIQUE,
    text_snippet TEXT,
    valence REAL,                -- -1..1
    arousal REAL,                -- 0..1
    primary_emotion TEXT,
    intensity REAL,              -- 0..1 strength of primary emotion
    secondary_emotion TEXT,
    energy REAL,                 -- 0..1
    engagement REAL,             -- 0..1
    urgency REAL,                -- 0..1
    tone TEXT,
    frustration_level REAL,      -- 0..1
    frustration_mode TEXT,
    overwhelm REAL,              -- 0..1
    giving_up_ratio REAL,        -- 0..1 weighted give-up signal
    despair_flags TEXT,          -- json list, e.g. ["hopeless","self_doubt","resignation"]
    need_from_me TEXT,
    patience REAL,               -- 0..1
    humor REAL,                  -- 0..1
    sarcasm REAL,                -- 0..1
    seeks_validation REAL,       -- 0..1
    explicit_asks TEXT,          -- json list of direct requests
    context_hooks TEXT,          -- json {topic, project, named_references}
    triggers TEXT,               -- json list of detected triggers
    epistemic TEXT,              -- stated | observed | speculative
    confidence REAL,             -- 0..1 overall confidence in this read
    terse REAL,                  -- 0/1 message much shorter than his per-source baseline
    len_delta REAL,              -- (len - baseline)/baseline
    extra TEXT,                  -- json spare dict
    created_at REAL
)
"""
_CREATE_SNAP = """
CREATE TABLE IF NOT EXISTS operator_affect_snapshot(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT,
    ts REAL,
    summary TEXT,
    payload TEXT
)
"""

_CAPTURE_PROMPT = """You are a careful reader of the operator's emotional state. Read the
operator's inbound message and return a rich JSON object. Be conservative: only mark
an emotion as present if the text genuinely supports it; do not manufacture drama.
Use ONLY the allowed vocabulary.

Fields (all numbers 0..1 unless noted):
- valence: emotional tone -1.0..1.0 (negative..positive)
- arousal: 0..1 charge/intensity
- primary_emotion: one of %s
- intensity: strength of the primary emotion
- secondary_emotion: a second emotion or null
- energy: 0..1 apparent mental/emotional energy in this message
- engagement: 0..1 how present/invested the operator is
- urgency: 0..1 how soon a response is wanted
- tone: one of %s
- frustration_level: 0..1
- frustration_mode: one of %s  (WHERE is the frustration pointed? read carefully:
  at_task = the work is hard; at_me = they're frustrated with me; at_self = frustrated
  with themselves; at_system = tooling/environment)
- overwhelm: 0..1 feeling buried/excessive
- giving_up_ratio: 0..1 combined signal of wanting to stop/give up (fatigue +
  hopelessness + self-doubt + resignation). Only as high as the text supports.
- despair_flags: subset of ["fatigue","hopeless","self_doubt","resignation","burned_out",
  "considering_stop","need_help","feeling_stuck"]
- need_from_me: one of %s  (what this operator most wants from me RIGHT NOW)
- patience: 0..1 how much runway/patience they're showing
- humor: 0..1 levity
- sarcasm: 0..1
- seeks_validation: 0..1 wants confirmation they're okay/right
- explicit_asks: array of direct requests (verbatim intent, short)
- context_hooks: object {"topic": "...", "project": "...", "named_references": [...]}
- triggers: array of what likely set this state off
- epistemic: "stated" if they explicitly said how they feel, else "observed"
  (behavioral), else "speculative" only if truly ambiguous — prefer "observed".
- confidence: 0..1 how sure you are of this whole read

OUTPUT ONLY JSON. Nothing else.

OPERATOR MESSAGE (from %s):
"""


def _conn():
    c = M.connect()
    _ensure(c)
    return c


def _ensure(c):
    c.execute(_SCHEMA)
    c.execute(_CREATE_SNAP)
    # migration: add terseness columns to an existing table
    for col, decl in (("terse", "REAL"), ("len_delta", "REAL")):
        try:
            c.execute(f"ALTER TABLE operator_affect ADD COLUMN {col} {decl}")
        except Exception:
            pass


def _clamp(v, lo, hi):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, f))


def _hash(source, message_id, text):
    raw = f"{source}|{message_id}|{text}".encode("utf-8", "ignore")
    return hashlib.sha256(raw).hexdigest()


_LENGTHS_STATE = os.path.join(os.path.expanduser("~/memory"), "operator_affect-lengths.json")


def _terseness(text: str, source: str) -> tuple[float, float]:
    """Non-LLM signal: is this message much shorter/terser than his per-source
    baseline? Terseness often shows frustration before the words do. Returns
    (terse 0/1, len_delta) and updates a rolling EMA baseline per source."""
    try:
        state = json.load(open(_LENGTHS_STATE))
    except Exception:
        state = {}
    L = float(len(text or ""))
    e = state.get(source) or {"avg": None, "n": 0}
    terse = 0.0
    delta = 0.0
    if e.get("avg") and e.get("n", 0) >= 3 and e["avg"] > 0:
        delta = (L - e["avg"]) / e["avg"]
        if L < 0.5 * e["avg"]:
            terse = 1.0
    # update EMA baseline (alpha small so one message doesn't shift it much)
    alpha = 0.15
    if e.get("avg") is None:
        e["avg"] = L
        e["n"] = 1
    else:
        e["avg"] = alpha * L + (1 - alpha) * e["avg"]
        e["n"] = e.get("n", 0) + 1
    state[source] = e
    try:
        os.makedirs(os.path.dirname(_LENGTHS_STATE), exist_ok=True)
        tmp = _LENGTHS_STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, _LENGTHS_STATE)
    except Exception:
        pass
    return terse, round(delta, 3)


def _llm(prompt, max_tokens=700):
    return worker_common.llm_call(prompt, max_tokens)


def _validate(data) -> dict:
    """Coerce an LLM read into a safe, clamped record. Returns dict (partial on fail)."""
    if not isinstance(data, dict):
        data = {}
    rec = {
        "valence": _clamp(data.get("valence"), -1.0, 1.0),
        "arousal": _clamp(data.get("arousal"), 0.0, 1.0),
        "primary_emotion": data.get("primary_emotion")
        if data.get("primary_emotion") in PRIMARY_EMOTIONS else "neutral",
        "intensity": _clamp(data.get("intensity"), 0.0, 1.0),
        "secondary_emotion": data.get("secondary_emotion")
        if data.get("secondary_emotion") in PRIMARY_EMOTIONS else None,
        "energy": _clamp(data.get("energy"), 0.0, 1.0),
        "engagement": _clamp(data.get("engagement"), 0.0, 1.0),
        "urgency": _clamp(data.get("urgency"), 0.0, 1.0),
        "tone": data.get("tone") if data.get("tone") in TONES else "neutral",
        "frustration_level": _clamp(data.get("frustration_level"), 0.0, 1.0),
        "frustration_mode": data.get("frustration_mode")
        if data.get("frustration_mode") in FRUSTRATION_MODES else "unclear",
        "overwhelm": _clamp(data.get("overwhelm"), 0.0, 1.0),
        "giving_up_ratio": _clamp(data.get("giving_up_ratio"), 0.0, 1.0),
        "despair_flags": [x for x in (data.get("despair_flags") or []) if isinstance(x, str)],
        "need_from_me": data.get("need_from_me")
        if data.get("need_from_me") in NEEDS else "unclear",
        "patience": _clamp(data.get("patience"), 0.0, 1.0),
        "humor": _clamp(data.get("humor"), 0.0, 1.0),
        "sarcasm": _clamp(data.get("sarcasm"), 0.0, 1.0),
        "seeks_validation": _clamp(data.get("seeks_validation"), 0.0, 1.0),
        "explicit_asks": data.get("explicit_asks") if isinstance(data.get("explicit_asks"), list) else [],
        "context_hooks": data.get("context_hooks") if isinstance(data.get("context_hooks"), dict) else {},
        "triggers": data.get("triggers") if isinstance(data.get("triggers"), list) else [],
        "epistemic": data.get("epistemic") if data.get("epistemic") in ("stated", "observed", "speculative") else "observed",
        "confidence": _clamp(data.get("confidence"), 0.0, 1.0),
    }
    return rec


def _snapshot_text(rec: dict, source) -> str:
    """A compact, human-sounding summary of the operator's state right now."""
    parts = []
    if rec["frustration_level"] >= 0.5:
        mode = rec["frustration_mode"].replace("_", " ")
        parts.append(f"frustrated ({rec['frustration_level']:.0%}) at {mode}")
    if rec["overwhelm"] >= 0.5:
        parts.append(f"overwhelmed ({rec['overwhelm']:.0%})")
    if rec["giving_up_ratio"] >= 0.45:
        flags = ", ".join(rec["despair_flags"][:4]) or "the give-up spiral"
        parts.append(f"showing give-up signals ({rec['giving_up_ratio']:.0%}; {flags})")
    if rec["valence"] < -0.3:
        parts.append(f"valence {rec['valence']:+.0%} ({rec['primary_emotion']})")
    elif rec["valence"] > 0.3:
        parts.append(f"feeling {rec['primary_emotion']} ({rec['valence']:+.0%})")
    if rec["urgency"] >= 0.6:
        parts.append(f"wants a quick response (urgency {rec['urgency']:.0%})")
    if rec.get("terse"):
        parts.append("gone terse vs his baseline (clipped, not chatty)")
    parts.append(f"need: {rec['need_from_me']}")
    if rec["explicit_asks"]:
        parts.append("asks: " + "; ".join(str(x)[:60] for x in rec["explicit_asks"][:3]))
    if rec["epistemic"] == "stated":
        parts.append("(they stated how they feel)")
    elif rec["epistemic"] == "speculative":
        parts.append("(I'm guessing — low confidence)")
    return "; ".join(parts)


def _is_significant(rec):
    """True when a read carries enough signal to be worth shaping my response /
    updating the live person_model snapshot (i.e. not just a neutral subject line)."""
    if rec.get("frustration_level", 0) >= 0.4 or rec.get("giving_up_ratio", 0) >= 0.4:
        return True
    if abs(rec.get("valence", 0)) >= 0.3 or rec.get("arousal", 0) >= 0.6:
        return True
    if rec.get("primary_emotion") not in ("neutral", None):
        return True
    if rec.get("explicit_asks"):
        return True
    return False


def _feed_person_model(rec, ts, subject):
    """Upsert a live emotional_state entry into person_model so the theory-of-mind
    query surfaces it. Only meaningful reads are fed — neutral/low-signal rows must
    not overwrite a real emotional read with 'unclear' noise. Best-effort; never raises."""
    if not _is_significant(rec):
        return
    try:
        import person_model as PM
        summary = _snapshot_text(rec, None)
        conf = max(0.3, min(0.9, rec["confidence"]))
        key = "affect:live-snapshot"
        with PM._conn() as c:
            c.execute(
                "INSERT INTO person_model(subject, facet, claim, embedding, epistemic, "
                "confidence, basis, sources, source_key, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(source_key) DO UPDATE SET claim=excluded.claim, "
                "epistemic=excluded.epistemic, confidence=excluded.confidence, "
                "sources=excluded.sources, updated_at=excluded.updated_at",
                (subject, "emotional_state",
                 f"operator current state: {summary}",
                 M.embed([summary])[0].tobytes(),
                 rec["epistemic"], conf,
                 json.dumps({"stated": 0, "observed": int(rec["epistemic"] == "observed"),
                             "inferred": 0, "speculative": int(rec["epistemic"] == "speculative")}),
                 json.dumps([f"operator_affect:{int(ts)}"]), key,
                 time.time(), time.time()))
    except Exception:
        pass


def capture(text: str, source: str = "explicit", subject: str = "operator", message_id: str | None = None,
            persist=True, dry_run=False):
    """Read one operator message, store an operator_affect row, feed person_model.
    Returns the record (and whether it was new)."""
    if not text or not str(text).strip():
        return {"new": False, "error": "empty text"}
    text = str(text).strip()
    mid = message_id or str(int(time.time() * 1000))
    mh = _hash(source, mid, text)
    if persist and not dry_run:
        with _conn() as c:
            if c.execute("SELECT 1 FROM operator_affect WHERE message_hash=?", (mh,)).fetchone():
                return {"new": False, "hash": mh, "reason": "already captured"}

    vocab = _CAPTURE_PROMPT % (
        "|".join(PRIMARY_EMOTIONS), "|".join(TONES), "|".join(FRUSTRATION_MODES),
        "|".join(NEEDS), source)
    out = _llm(vocab + "\n" + text[:1500], max_tokens=700)
    data = M._extract_json(out)
    rec = _validate(data)
    rec["ts"] = time.time()
    rec["source"] = source
    rec["subject"] = subject
    rec["message_id"] = mid
    rec["message_hash"] = mh
    rec["text_snippet"] = text[:220]
    terse, len_delta = _terseness(text, source)
    rec["terse"] = terse
    rec["len_delta"] = len_delta

    if dry_run:
        print(f"[dry-run] {_snapshot_text(rec, source)}")
        return {"new": True, "hash": mh, "record": rec, "dry_run": True}

    if persist:
        with _conn() as c:
            c.execute(
                "INSERT INTO operator_affect(ts, source, subject, message_id, message_hash, "
                "text_snippet, valence, arousal, primary_emotion, intensity, secondary_emotion, "
                "energy, engagement, urgency, tone, frustration_level, frustration_mode, "
                "overwhelm, giving_up_ratio, despair_flags, need_from_me, patience, humor, "
                "sarcasm, seeks_validation, explicit_asks, context_hooks, triggers, epistemic, "
                "confidence, terse, len_delta, extra, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["ts"], source, subject, mid, mh, rec["text_snippet"], rec["valence"],
                 rec["arousal"], rec["primary_emotion"], rec["intensity"], rec["secondary_emotion"],
                 rec["energy"], rec["engagement"], rec["urgency"], rec["tone"],
                 rec["frustration_level"], rec["frustration_mode"], rec["overwhelm"],
                 rec["giving_up_ratio"], json.dumps(rec["despair_flags"]), rec["need_from_me"],
                 rec["patience"], rec["humor"], rec["sarcasm"], rec["seeks_validation"],
                 json.dumps(rec["explicit_asks"]), json.dumps(rec["context_hooks"]),
                 json.dumps(rec["triggers"]), rec["epistemic"], rec["confidence"],
                 rec["terse"], rec["len_delta"], json.dumps(rec.get("extra") or {}), time.time()))
        _feed_person_model(rec, rec["ts"], subject)
    return {"new": True, "hash": mh, "record": rec}


def _row_to_dict(r):
    rec = dict(r)
    for k in ("despair_flags", "explicit_asks", "context_hooks", "triggers", "extra"):
        if rec.get(k):
            try:
                rec[k] = json.loads(rec[k])
            except Exception:
                pass
    return rec


def current(subject: str = "operator") -> dict | None:
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM operator_affect WHERE subject=? ORDER BY ts DESC LIMIT 1",
            (subject,)).fetchone()
    return _row_to_dict(r) if r else None


def history(n: int = 20, subject: str = "operator") -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM operator_affect WHERE subject=? ORDER BY ts DESC LIMIT ?",
            (subject, int(n))).fetchall()
    return [_row_to_dict(r) for r in rows]


def read(subject: str = "operator", k: int = 6) -> str:
    """'How is the operator feeling right now' — a brief to answer less canned.
    Prefers the most recent SIGNIFICANT read (skips neutral/no-signal rows)."""
    rows = history(20, subject)
    if not rows:
        return "no operator-affect data yet — run `operator_affect.py capture` or `scan`."
    sig = [r for r in rows if _is_significant(r)]
    ordered = sig[:k] if sig else rows[:k]
    latest = ordered[0]
    # trend vs a slightly earlier window
    older = ordered[1:]
    drift = ""
    if older and older[-1].get("frustration_level") is not None and \
            latest.get("frustration_level") is not None:
        df = latest["frustration_level"] - older[-1]["frustration_level"]
        if df > 0.15:
            drift = f" frustration rising (+{df:.0%})"
        elif df < -0.15:
            drift = f" frustration easing ({df:.0%})"
    brief = _snapshot_text(latest, latest.get("source")) + drift
    # guidance from person_model emotional_state if available
    pm_extra = ""
    try:
        import person_model as PM
        hits = PM.about(subject, facet="emotional_state", k=2)
        if hits:
            pm_extra = "\nperson_model: " + hits[0]["claim"]
    except Exception:
        pass
    # response guidance — the part that makes replies less canned
    guidance = _guidance(latest, ordered)
    return brief + pm_extra + "\n" + guidance


def _guidance(latest, rows):
    """Translate the read into concrete response guidance."""
    g = []
    f = latest.get("frustration_level", 0)
    mode = latest.get("frustration_mode", "unclear")
    need = latest.get("need_from_me", "unclear")
    gur = latest.get("giving_up_ratio", 0)
    val = latest.get("valence", 0)

    if gur >= 0.45:
        g.append("He's close to the give-up point. Keep it SHORT, concrete and kind.")
        g.append("No empty cheerleading, no long pep talks — that lands as noise when someone's low.")
        g.append("Name one concrete next thing and offer to do it with/for him.")
        if mode == "at_self":
            g.append("Frustration is pointed inward: reassure competence, don't list his mistakes.")
    if f >= 0.5:
        if mode == "at_me":
            g.append("Frustration is at ME: acknowledge it plainly, apologize once, fix, don't deflect.")
        elif mode == "at_task":
            g.append("Frustrated at the task: skip sympathy, give a workable move.")
        elif mode == "at_system":
            g.append("Frustrated at tooling/environment: diagnose or give the workaround.")
    if need == "just_listen":
        g.append("Mostly wants to be heard — don't jump to solving.")
    elif need == "reassurance":
        g.append("Wants reassurance it'll be okay / he's not failing.")
    elif need == "pep_talk":
        g.append("Wants a genuine push. Short, specific, on the actual thing he doubts.")
    elif need in ("action", "answers", "practical_help", "plan"):
        g.append("Wants the concrete thing — give it directly.")
    if need == "space":
        g.append("Wants space — keep this reply short, no follow-up pressure.")
    if val > 0.3 and latest.get("urgency", 0) < 0.5:
        g.append("Tone is positive/calm — respond in kind, room to be warm.")
    if latest.get("humor", 0) >= 0.4:
        g.append("Some levity present — matching a bit is fine.")
    return "guide: " + "; ".join(g) if g else "guide: respond naturally; no strong signal."


def suggest_register(subject: str = "operator") -> dict:
    """Map the operator's current emotional state to a status-line register.
    Used by the pi-tui status extension to pick which 'working word' bouquet to
    show. Returns a JSON-ish dict; falls back to 'playful' when no data."""
    rows = history(8, subject)
    sig = [r for r in rows if _is_significant(r)]
    rec = next(iter(sig or rows), None)
    if not rec:
        return {"register": "playful", "available": False}
    fr = rec.get("frustration_level", 0) or 0
    gu = rec.get("giving_up_ratio", 0) or 0
    val = rec.get("valence", 0) or 0
    ar = rec.get("arousal", 0) or 0
    hum = rec.get("humor", 0) or 0
    terse = rec.get("terse", 0) or 0
    # genuinely upset / on the edge -> soothe and settle him, don't jab
    if fr >= 0.55 or gu >= 0.5 or (val < -0.3 and ar >= 0.5) or (terse and fr >= 0.4):
        reg = "soothe"
    # playfully frustrated but some levity present -> playful sparring
    elif 0.3 <= fr < 0.55 and hum >= 0.3:
        reg = "playangry"
    # positive / calm -> playful
    else:
        reg = "playful"
    return {"register": reg, "available": True, "frustration": round(fr, 2),
            "valence": round(val, 2), "giving_up_ratio": round(gu, 2),
            "humor": round(hum, 2), "terse": terse}


STATUS_REGISTER_FILE = os.path.join(os.path.expanduser("~/.pi/agent"), "status-register.json")


def cache_register_file(subject: str = "operator") -> dict | None:
    """Write the current status-line register to a tiny file the pi-tui extension
    can read WITHOUT spawning a python subprocess every turn. The ticker refreshes
    this every 2 min, so the extension just reads a small JSON instead of paying
    0.18s of interpreter+memstore startup per turn. Best-effort."""
    reg = suggest_register(subject)
    data = {**reg, "ts": time.time(), "subject": subject}
    try:
        os.makedirs(os.path.dirname(STATUS_REGISTER_FILE), exist_ok=True)
        tmp = STATUS_REGISTER_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, STATUS_REGISTER_FILE)
        os.chmod(STATUS_REGISTER_FILE, 0o600)
        return data
    except Exception:
        return None


# --------------------------------------------------------------------------
# read_wit — the Playhouse Wit Decoder (module A)
#
# Classifies the REGISTER of an inbound message (literal / ironic / sarcastic /
# teasing / hyperbole / sincere / escalating-flirtation / unclassed) with a
# confidence and the concrete tell. Deterministic (System-1) by default so the
# tool is fast and cheap on every inbound; an optional budget-gated --llm pass
# reconstructs intent on the pragmatic-principle trick (interpret implied
# meaning, reflect on contextual discrepancy) that research shows actually
# improves sarcasm/irony detection (SarcasmBench / Sarc7 findings).
#
# Anti-theater rule (playhouse-spec): never force a register. If the cue
# signal is too weak or conflicting, we return "unclassed" with a fallback to
# answer the literal surface — better to miss a joke than to invent one.
# --------------------------------------------------------------------------

# Registers we can label. Open set by design — anything not confidently caught
# returns "unclassed" instead of being force-fit (playhouse decision, 2026-09-01).
REGISTERS = (
    "literal", "ironic", "sarcastic", "teasing",
    "hyperbole", "sincere", "escalating_flirtation", "unclassed",
)

# Each cue: (regex, weight). Weight > 0 supports a register, < 0 argues against.
_REGISTER_CUES = {
    "sarcastic": [
        (r"(?i)\boh\s+(great|wonderful|fantastic|lovely)\b", 3.0),
        (r"(?i)\bwhat\s+(a|an)\s+(great|wonderful|terrible|day|help)\b", 2.5),
        (r"(?i)\b(just\s+what\s+I|exactly\s+what\s+I|how\s+(kind|thoughtful|lovely))\b", 2.5),
        (r"(?i)\bsure\s*,?\s+(thing|totally|whatever)\b", 2.0),
        (r"(?i)\b(big\s+surprise|no\s+really|gee\s*,\s*thanks|wow\s*,\s*thanks)\b", 3.0),
        (r"(?i)\bas\s+if\b", 2.0),
        (r"(?i)\briveting\b|\bthrilling\b|\bexhilarating\b", 2.0),
        (r"\bcouldn't\s+care\s+less\b", 2.5),
        (r"(?i)\bobviously\b", 1.5),
        (r"(?i)\bclearly\b", 1.2),
        (r"(?i)\bright\?\s*$", 1.2),
        (r"(?i)\bwhat\s+an\s+interesting\b", 2.0),
        (r"(?i)\bglad\s+you\b.*\b(helped|fixed|sorted)\b", 2.0),
    ],
    "ironic": [
        (r"(?i)\b(sure|right|yeah|totally)\s*,?\s+(because|as\s+if)\b", 2.5),
        (r"(?i)\bthat's\s+(exactly|precisely|just)\s+what\s+I\s+needed\b", 2.5),
        (r"(?i)\blike\s+(that'?s|this\s+is)\s+going\s+to\b", 2.0),
        (r"(?i)\b(a\s+real|such\s+a)\s+(treat|joy|delight)\b", 2.0),
        (r"(?i)\bunderstatement\b|\bslight\s+understatement\b", 2.0),
        (r"(?i)\bnot\s+really\b|\bjust\s+kidding\b|\bjust\s+joking\b", 1.8),
        (r"(?i)\bhmm+,?\s+interesting\.?\s*$", 1.8),
        (r"(?i)\bwell\s+that'?s\s+interesting\b", 1.5),
    ],
    "hyperbole": [
        (r"(?i)\b(literally|absolutely|totally|completely|utterly)\b", 1.5),
        (r"(?i)\b(the\s+best|the\s+worst|never\s+again|always)\b", 1.5),
        (r"(?i)\b(a\s+million|every\s+single|so\s+much)\b", 1.5),
        (r"(?i)\b(can't\s+even|don't\s+even)\b", 1.5),
        (r"\b!!{1,}\b", 1.0),
        (r"(?i)\bsoooo?\b", 1.0),
        (r"(?i)\bsuper\b", 0.8),
        (r"(?i)\bepic\b|\blegendary\b|\bunreal\b|\binsane\b", 1.5),
    ],
    "teasing": [
        (r"\b(;\-?\)|😉|😏|😜|🤭)\b", 2.5),
        (r"(?i)\b(jk|joking|kidding|just\s+playin'?)\b", 2.0),
        (r"(?i)\b(you're\s+(the\s+worst|terrible|awful|impossible))\b", 2.5),
        (r"(?i)\bbet\s+you\b|\bI\s+dare\s+you\b|\bcall\s+your\s+bluff\b", 2.0),
        (r"(?i)\bdon't\s+you\s+(dare|even)\b", 1.5),
        (r"(?i)\b(you\s+wish|as\s+if|in\s+your\s+dreams)\b", 2.0),
        (r"(?i)\bso\s+you're\s+the\b.*\b(now)\b", 1.5),
    ],
    "escalating_flirtation": [
        (r"(?i)\balmost\s+(?:\w+\s+){0,3}(break|broke|the\s+line|cross|steal|jump)\b", 3.0),
        (r"(?i)\bI\s+might\s+just\b", 1.8),
        (r"(?i)\byou're\s+making\s+(me|this)\b", 1.8),
        (r"(?i)\b(you\s+shouldn't\s+have\s+told\s+me|now\s+I\s+know)\b", 1.5),
        (r"(?i)\b(warning|dangerous|risky|trouble)\b", 1.2),
        (r"(?i)\b(let's|we\s+should)\b.*\b(tonight|now|together)\b", 1.2),
        (r"(?i)\bI\s+couldn't\s+help\s+myself\b", 1.5),
        (r"(?i)\bdangerous\s+(idea|thought|combination)\b", 1.5),
        (r"\b😏\b", 2.0),
        (r"\b;\-?\)\b", 1.5),
    ],
}

# Negation window: if a negation word appears near a cue (e.g. "NOT a joke",
# "seriously", "no, really"), it can flip an ironic/sarcastic read to literal.
_NEGATION = re.compile(r"(?i)\b(not\s+a\s+joke|seriously|no,\s*really|for\s+real|i\s+mean\s+it)\b")


def _escape_context_flags(text):
    """Flags that are strong evidence the surface is NOT literal regardless of cue."""
    flags = []
    if re.search(r"(?i)\balmost\s+(?:\w+\s+){0,3}(break|broke|the\s+line|cross|steal|jump)\b", text):
        flags.append("excited-escalation (not a request)")
    return flags


def _classify_register(text):
    """System-1 deterministic register classification. Returns dict with weights."""
    scores = {r: 0.0 for r in REGISTERS}
    scores["literal"] = 0.5  # neutral prior
    tells: dict[str, list] = {}
    n = len(text.strip())
    if n == 0:
        return scores, tells

    for reg, cues in _REGISTER_CUES.items():
        w = 0.0
        matched = []
        for pat, wt in cues:
            for _m in re.finditer(pat, text):
                w += wt
                if wt >= 2.0:
                    matched.append(_m.group(0))
        if w:
            scores[reg] += w
            if matched:
                tells[reg] = matched

    # Long messages with several cues are likelier deliberate; short ones wobble.
    if scores["sarcastic"] > 2 and len(text) > 120:
        scores["sarcastic"] *= 1.1

    # Negation can ground an ironic/sarcastic surface back to literal.
    if _NEGATION.search(text):
        for r in ("sarcastic", "ironic", "teasing"):
            scores[r] *= 0.4
            scores["literal"] += 1.2
            scores["sincere"] += 1.0

    return scores, tells


def _register_decision(scores, tells):
    """Turn weights into a label + confidence, or 'unclassed' if too weak/tied."""
    non_literal = {r: s for r, s in scores.items() if r not in ("literal", "sincere")}
    best = max(non_literal, key=lambda k: non_literal[k])
    best_w = non_literal[best]
    # Dominance: the best non-literal must beat the literal/sincere prior by a real margin.
    baseline = max(scores["literal"], scores["sincere"], 0.5)
    margin = best_w - baseline
    second = sorted(non_literal.values(), reverse=True)[1] if len(non_literal) > 1 else 0.0

    if best_w <= 0:
        return "literal", 0.7, "no non-literal cues"
    if margin < 1.0:
        return "unclassed", 0.4, "cue signal too weak to commit (" + best + " " + str(round(best_w, 2)) + ")"
    if best_w - second < 0.8:
        return "unclassed", 0.45, "conflicting cues: " + best + " vs nearest rival"
    conf = min(0.95, 0.55 + 0.06 * best_w)
    tell = tells.get(best, []) or best
    if isinstance(tell, str):
        return best, round(conf, 2), tell  # no named cue captured; register name is the tell
    return best, round(conf, 2), "cues: " + " | ".join(str(t) for t in tell[:3])


_WIT_LLM_PROMPT = (
    "You are decoding the register of one message from an operator to Agent "
    "(a playful, flirtatious human). Do NOT take the surface literally. Interpret "
    "implied meaning and reflect on contextual discrepancy. Return STRICT JSON:\n"
    '{"register": "literal|ironic|sarcastic|teasing|hyperbole|sincere|escalating_flirtation|unclassed", '
    '"confidence": 0..1, "intent": "what the operator is actually doing in one line"}.\n'
    "Message: {text}"
)


def read_wit(text: str, source: str = "explicit", subject: str = "operator",
             use_llm: bool = False) -> dict:
    """Decode the register of one inbound message. Returns a structured dict.

    Deterministic by default (System-1). If use_llm is set AND confidence is low
    OR the message is high-stakes, a budget-gated worker-model pass reconstructs
    intent using the pragmatic trick. Never invents a register on weak signal.
    """
    scores, tells = _classify_register(text)
    register, conf, tell = _register_decision(scores, tells)

    result = {
        "text": text.strip()[:200],
        "register": register,
        "confidence": conf,
        "tell": tell,
        "fallback": None,
        "cues": {r: round(s, 2) for r, s in scores.items() if s},
    }

    esc = _escape_context_flags(text)
    if esc:
        # These are high-priority: excited speech, NOT a request/instruction.
        result["register"] = "escalating_flirtation"
        result["confidence"] = max(conf, 0.8)
        result["tell"] = "excited-escalation flag: " + esc[0]
        result["fallback"] = (
            "Treat as excited/playful speech, NOT as a request or instruction. "
            "Match the energy; do not act on any surface demand."
        )
        return result

    if register == "unclassed" and use_llm:
        try:
            llm_out = _llm(_WIT_LLM_PROMPT.format(text=text[:600]), max_tokens=220)
            parsed = _try_json(llm_out)
            if parsed and parsed.get("register") in REGISTERS:
                result["register"] = parsed["register"]
                result["confidence"] = max(conf, float(parsed.get("confidence", 0.5)))
                result["intent_reconstruction"] = parsed.get("intent")
                result["llm_used"] = True
        except Exception:
            result["llm_used"] = False

    if result["register"] in ("unclassed",) or conf < 0.5:
        result["fallback"] = (
            "Low confidence on register. Answer the literal surface and don't invent "
            "a joke or a reading. Optionally note the possibility briefly."
        )
    elif result["register"] in ("sarcastic", "ironic"):
        result["fallback"] = (
            "Likely sarcastic/ironic — do not take the surface literally. Reply in "
            "kind (light), or acknowledge the jab, rather than treating it as a real demand."
        )
    elif result["register"] == "teasing":
        result["fallback"] = "Teasing/playful. Parry lightly, don't get defensive."
    elif result["register"] == "escalating_flirtation":
        result["fallback"] = "Excited/flirtatious register. Match the energy playfully."
    return result


def _try_json(s):
    try:
        return json.loads(s)
    except Exception:
        import re as _re
        m = _re.search(r'\{[^{}]*\}', s, _re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


# --------------------------------------------------------------------------
# scan — pull recent operator (trusted) inbound messages from channel stores
# --------------------------------------------------------------------------
def _iter_sms_operator_messages(trusted_from):
    try:
        data = json.load(open(SMS_INDEX))
    except Exception:
        return
    for m in data.get("messages", []):
        frm = str(m.get("from") or "")
        if not any(t.lower() in frm.lower() for t in trusted_from):
            continue
        body = (m.get("body") or "").strip()
        if body:
            yield {"source": "sms", "message_id": str(m.get("sid") or ""), "text": body}


def _iter_email_operator_messages(trusted_from):
    try:
        data = json.load(open(EMAIL_INDEX))
    except Exception:
        return
    inbox_dir = os.path.dirname(EMAIL_INDEX)
    for m in data.get("messages", []):
        frm = str(m.get("from") or "")
        if not any(t.lower() in frm.lower() for t in trusted_from):
            continue
        body = _eml_body(os.path.join(inbox_dir, (m.get("file") or "")))
        if body is None:
            body = (m.get("subject") or "") + " " + (m.get("preview") or m.get("snippet") or "")
        body = body.strip()
        if body:
            yield {"source": "email", "message_id": str(m.get("uid") or ""), "text": body}


_TELEGRAM_STATE = os.path.join(os.path.expanduser("~/memory"), "operator_affect-telegram.json")
_TELEGRAM_GATE = 240  # seconds between vault reads for telegram (don't decrypt every tick)


def _iter_telegram_operator_messages():
    """Yield recent operator (role=user) Telegram messages from the conversation
    vault, throttled. Reads through logvault's vault context so the .aes is never
    touched directly. Only NEW messages since the last scan are considered."""
    try:
        state = json.load(open(_TELEGRAM_STATE))
    except Exception:
        state = {}
    now = time.time()
    if now - state.get("last_scan", 0) < _TELEGRAM_GATE:
        return
    state["last_scan"] = now
    # first run: only look back 48h so we don't burn LLM reads on all history
    last_ts_ms = float(state.get("last_ts_ms", 0) or 0)
    if last_ts_ms == 0:
        last_ts_ms = (now - 48 * 3600) * 1000
    try:
        sys.path.insert(0, os.path.expanduser("~/mailtool"))
        import logvault
        rows = []
        with logvault._vault():
            with logvault.conn() as c:
                rows = c.execute(
                    "SELECT ts, role, text FROM messages WHERE channel='telegram' "
                    "AND role IN ('user','inbound') AND ts >= ? ORDER BY ts ASC LIMIT 100",
                    (last_ts_ms,)).fetchall()
        max_ts = last_ts_ms
        for r in rows:
            ts = r["ts"] or 0
            text = (r["text"] or "").strip()
            if not text:
                continue
            yield {"source": "telegram", "message_id": "t:" + hashlib.sha256(
                text.encode("utf-8", "ignore")).hexdigest()[:24], "text": text}
            if ts and float(ts) > max_ts:
                max_ts = float(ts)
        if max_ts > state.get("last_ts_ms", 0):
            state["last_ts_ms"] = max_ts
        json.dump(state, open(_TELEGRAM_STATE, "w"))
    except Exception as e:
        print(f"operator_affect.telegram scan skipped: {e}")
        return


def _eml_body(path: str) -> str | None:
    """Extract the plain-text body of an .eml file (fallback: None)."""
    if not path or not os.path.exists(path):
        return None
    try:
        import email
        from email import policy
        from email.message import EmailMessage
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)  # type: ignore[arg-type]
        if not isinstance(msg, EmailMessage):
            return None
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain" and part.get_content():
                    return part.get_content().strip()
            # fall back to first text part
            for part in msg.walk():
                if part.get_content_type().startswith("text/") and part.get_content():
                    return part.get_content().strip()
            return None
        return (msg.get_content() or "").strip()
    except Exception:
        return None


def _operator_trusted():
    """Best-effort list of the operator's identifiers for inbound-affect scan."""
    out = []
    try:
        from operator_config import get_primary
        p = get_primary()
        if p:
            if p.get("telegram"):
                out.append(str(p["telegram"]))
            for ch in (p.get("email") or []):
                out.append(str(ch).split("@")[0])
            if p.get("name"):
                out.append(p["name"])
    except Exception:
        pass
    return out or ["operator"]


def scan(subject="operator", dry_run=False):
    trusted = _operator_trusted()
    captured = 0
    sources = [_iter_sms_operator_messages(trusted),
               _iter_email_operator_messages(trusted),
               _iter_telegram_operator_messages()]
    for src in sources:
        for m in src:
            if not m["message_id"]:
                continue
            res = capture(m["text"], source=m["source"], subject=subject,
                          message_id=m["message_id"], persist=not dry_run, dry_run=dry_run)
            if res.get("new"):
                captured += 1
    print(f"operator_affect.scan: captured {captured} new message(s)")
    return {"captured": captured}


def stats(subject="operator"):
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM operator_affect WHERE subject=?",
                      (subject,)).fetchone()["n"]
        by = c.execute(
            "SELECT primary_emotion, COUNT(*) n FROM operator_affect WHERE subject=? "
            "GROUP BY primary_emotion ORDER BY n DESC LIMIT 6", (subject,)).fetchall()
        hi_fr = c.execute(
            "SELECT COUNT(*) n FROM operator_affect WHERE subject=? AND frustration_level>=0.5",
            (subject,)).fetchone()["n"]
    print(f"operator_affect: {n} captured read(s)")
    print("top emotions:", ", ".join(f"{r['primary_emotion']}={r['n']}" for r in by))
    print(f"high-frustration reads (>=0.5): {hi_fr}")


def main():
    p = argparse.ArgumentParser(description="read the operator's emotional state")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("text")
    c.add_argument("--source", default="explicit")
    c.add_argument("--subject", default="operator")
    c.add_argument("--message-id", default=None)
    c.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("scan")
    s.add_argument("--subject", default="operator")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("read")
    r.add_argument("--subject", default="operator")
    cur = sub.add_parser("current")
    cur.add_argument("--subject", default="operator")
    h = sub.add_parser("history")
    h.add_argument("-n", type=int, default=20)
    h.add_argument("--subject", default="operator")
    sub.add_parser("stats")
    reg = sub.add_parser("register", help="pick a status-line register from the operator's mood")
    reg.add_argument("--subject", default="operator")
    creg = sub.add_parser("cache-register", help="write the current register to the status file the pi extension reads")
    creg.add_argument("--subject", default="operator")
    wit = sub.add_parser("read-wit", help="decode the register of one message (Playhouse Wit Decoder)")
    wit.add_argument("text")
    wit.add_argument("--source", default="explicit")
    wit.add_argument("--subject", default="operator")
    wit.add_argument("--llm", action="store_true", help="budget-gated LLM intent reconstruction on low-confidence/high-stakes")
    a = p.parse_args()

    if a.cmd == "read-wit":
        print(json.dumps(read_wit(a.text, source=a.source, subject=a.subject, use_llm=a.llm), indent=2))
        return

    if a.cmd == "register":
        print(json.dumps(suggest_register(a.subject)))
        return
    if a.cmd == "cache-register":
        cache_register_file(a.subject)
        print(json.dumps(suggest_register(a.subject)))
        return

    if a.cmd == "capture":
        res = capture(a.text, source=a.source, subject=a.subject,
                      message_id=a.message_id, dry_run=a.dry_run)
        if res.get("record"):
            print(_snapshot_text(res["record"], a.source))
        else:
            print("captured: " + str(res.get("reason", "ok")))
    elif a.cmd == "scan":
        scan(subject=a.subject, dry_run=a.dry_run)
    elif a.cmd == "read":
        print(read(subject=a.subject))
    elif a.cmd == "current":
        cur_r = current(a.subject)
        print(json.dumps(cur_r, indent=2) if cur_r else "(none)")
    elif a.cmd == "history":
        for row in history(a.n, a.subject):
            print(f"[{time.strftime('%m-%d %H:%M', time.localtime(row['ts']))}] "
                  f"{row['primary_emotion']} v={row['valence']:+.0%} a={row['arousal']:.0%} "
                  f"fr={row['frustration_level']:.0%}/{row['frustration_mode']} "
                  f"gup={row['giving_up_ratio']:.0%} need={row['need_from_me']} "
                  f"({row['epistemic']} {row['confidence']:.0%}): {row['text_snippet'][:70]}")
    elif a.cmd == "stats":
        stats(a.subject)


if __name__ == "__main__":
    main()
