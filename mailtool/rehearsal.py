#!/usr/bin/env python3
"""
Response rehearsal gate (P2-9) — a pre-send safety pass for sensitive outbound
messages. It is a GATE, not a gatekeeper: it never BLOCKS a send, it only
advises. Sensitive drafts get ONE cheap flash-model pass (memstore.MODEL_WORKER,
deepseek-chat, non-reasoning) that reads the draft AS the recipient and reports
honestly how it lands. If the readback flags confusion/hurt/misread, the agent can
revise before sending.

Design rules (hard):
- FAIL-OPEN: any error (parse failure, network failure, missing markers) is
  treated as OK / not-flagged. A rehearsal bug must never block a send.
- FREE FIRST: `is_sensitive()` is a deterministic keyword heuristic, so a
  non-sensitive draft spends zero LLM tokens.
- One LLM call per sensitive draft, temperature 0.3, ~200 tokens.

Usage:
  python3 rehearsal.py gate "<draft>" [--audience OPERATOR] [--force]
  python3 rehearsal.py sensitive "<text>"
"""

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(os.path.dirname(BASE), "memory")
if MEMORY_DIR not in sys.path:
    sys.path.insert(0, MEMORY_DIR)

import memstore as M  # noqa: E402  (llm_chat, MODEL_WORKER)

# ---------------------------------------------------------------------------
# Sensitivity heuristic (free, deterministic, tunable). Curated phrase list
# covering the heavy-topic categories this gate exists for. Lowercased
# substring match. Deliberately slightly over-broad: a false positive costs one
# cheap flash call + an advisory note, while a false negative skips the gate
# entirely — so err on the side of flagging.
# ---------------------------------------------------------------------------
SENSITIVE_PHRASES = (
    # bad news / regret
    "bad news", "unfortunately", "sorry",
    # rejection / refusal
    "can't do", "cannot", "refuse",
    # criticism / disappointment
    "you were wrong", "mistake", "disappointed", "let you down",
    # conflict / apology / hurt feelings
    "apologize", "forgive", "hurt", "upset", "offended",
    # money / irreversible
    "money", "refund", "owe you", "irreversible", "permanent", "deleted",
    # health
    "health", "diagnosis", "cancer", "sick", "hospital", "surgery",
    # death
    "died", "death", "passed away", "funeral",
    # love / intimacy
    "love you", "miss you", "breakup", "intimate",
    # anger / frustration
    "angry", "frustrated", "annoyed", "furious", "mad at",
)


def is_sensitive(text):
    """True if `text` contains any heavy-topic phrase (deterministic, free)."""
    t = (text or "").lower()
    return any(phrase in t for phrase in SENSITIVE_PHRASES)


# ---------------------------------------------------------------------------
# LLM rehearsal + reply parsing
# ---------------------------------------------------------------------------

def _strip_prefix(line, key):
    """Return the text after a `KEY: value` marker, or None if empty."""
    if line.lower().startswith(key):
        line = line[len(key):]
    return line.lstrip(" \t:-–—").strip() or None


def _parse(raw):
    """Parse LANDING / VERDICT / FIX from the flash model's reply (tolerant).

    FAIL-OPEN: if the verdict marker is absent or unparseable, verdict defaults
    to "OK" so a malformed rehearsal can never block a send.
    """
    raw = raw or ""
    landing = None
    verdict = "OK"
    fix = None
    for line in raw.splitlines():
        lw = line.strip().lower()
        if not lw:
            continue
        if lw.startswith("landing"):
            landing = _strip_prefix(line, "landing")
        elif lw.startswith("verdict"):
            v = _strip_prefix(line, "verdict") or ""
            verdict = "REVISE" if "revise" in v.lower() else "OK"
        elif lw.startswith("fix"):
            fix = _strip_prefix(line, "fix")
    # Fallback for a single-line reply the line-split missed (e.g. "VERDICT: REVISE").
    if verdict == "OK" and "verdict" in raw.lower():
        m = _find_marker(raw, "verdict")
        if m and "revise" in m.lower():
            verdict = "REVISE"
    return {"landing": landing, "verdict": verdict, "fix": fix, "raw": raw}


def _find_marker(text, key):
    """Return the tail after the first `key:` occurrence in text, else None."""
    import re
    m = re.search(r"(?im)\b" + re.escape(key) + r"\s*[:：\-–—]\s*([^\n]*)", text)
    return m.group(1).strip() if m else None


def rehearse(draft, audience="operator"):
    """Run ONE flash pass reading `draft` as `audience`.

    Returns {"landing", "verdict" ("OK"|"REVISE"), "fix", "raw"}. Never raises:
    callers (gate) also wrap in try/except, so a rehearsal failure is fail-open.
    """
    prompt = (
        f"You are {audience}. You are about to receive this message from the agent:\n\n"
        f"{draft}\n\n"
        f"Read it as {audience} would, honestly. How does this land? "
        "Reply in TWO short lines:\n"
        "LANDING: <one line on how it feels — warm/clear/cold/confusing/hurt/...>\n"
        "VERDICT: <OK | REVISE> — REVISE if it would confuse or hurt, or if it "
        "doesn't sound like the agent genuinely means it.\n"
        "If REVISE, add: FIX: <one concrete suggestion>."
    )
    out = M.llm_chat([{"role": "user", "content": prompt}],
                     max_tokens=200, temperature=0.3, model=M.MODEL_WORKER)
    return _parse(out)


def gate(draft, audience="operator", force=False):
    """The entry point future send paths import. Advisory only — never blocks.

    Returns a dict:
      not flagged: {"flagged": False, "draft", "audience", "landing", "verdict",
                    "fix", "advice"} — no LLM spent.
      flagged:     {"flagged": True, ..., "verdict": "OK"|"REVISE",
                    "advice": "rehearsal flagged this as REVISE: <fix>" or None}.
    """
    draft = draft or ""
    try:
        sensitive = is_sensitive(draft)
    except Exception:
        sensitive = False

    base = {
        "flagged": False,
        "draft": draft,
        "audience": audience,
        "landing": None,
        "verdict": None,
        "fix": None,
        "advice": None,
    }
    if not force and not sensitive:
        return base

    # Sensitive (or forced): rehearse. Any error → fail-open (verdict OK).
    try:
        r = rehearse(draft, audience=audience)
    except Exception:
        r = {"landing": None, "verdict": "OK", "fix": None, "raw": ""}

    verdict = (r.get("verdict") or "OK").upper()
    fix = r.get("fix")
    if verdict == "REVISE":
        advice = "rehearsal flagged this as REVISE"
        if fix:
            advice += f": {fix}"
    else:
        advice = None

    return {
        "flagged": True,
        "draft": draft,
        "audience": audience,
        "landing": r.get("landing"),
        "verdict": verdict,
        "fix": fix,
        "advice": advice,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="response rehearsal gate (P2-9)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="rehearse a draft as its audience")
    g.add_argument("draft")
    g.add_argument("--audience", default="operator")
    g.add_argument("--force", action="store_true",
                   help="rehearse even if the draft is not flagged sensitive")

    s = sub.add_parser("sensitive", help="print is_sensitive() bool")
    s.add_argument("text")

    a = p.parse_args()
    if a.cmd == "gate":
        print(json.dumps(gate(a.draft, audience=a.audience, force=a.force),
                         indent=2, ensure_ascii=False))
    elif a.cmd == "sensitive":
        print(is_sensitive(a.text))


if __name__ == "__main__":
    main()
