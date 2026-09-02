#!/usr/bin/env python3
"""
Agent self-play / AI-AI test budget ledger.

A daily spend cap for the PAID portion of the self-test program (the "me on
DeepSeek" reflection / participation). Local ollama rounds are free and never
touch this ledger — it only tracks DeepSeek spend.

The same module is imported by:
  - selfplay.py   (enforces the cap before paid calls, records actual spend)
  - webapp/server.py  (GET/POST /api/budget — the agent-page control)

Default daily limit: $1.00 (see README for the per-call math; adjust freely).
"""

import json
import os
from datetime import datetime

BUDGET_PATH = os.path.expanduser("~/learning/selfplay/budget.json")

DEFAULT_LIMIT_USD = 1.00

# DeepSeek v4-pro pricing (USD per 1M tokens). Matches ~/.pi/agent/models-store.json.
PRO_IN = 0.435
PRO_OUT = 0.87

# Claude Fable (strongest) pricing (USD per 1M tokens).
FABLE_IN = 10.0
FABLE_OUT = 50.0


def _today():
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _load():
    """Return the raw dict, rolling 'used' over to 0 if the day changed."""
    d = {
        "daily_limit_usd": DEFAULT_LIMIT_USD,
        "date": _today(),
        "used_usd": 0.0,
        "history": [],  # [{date, used_usd}]
    }
    if os.path.exists(BUDGET_PATH):
        try:
            with open(BUDGET_PATH, "r", encoding="utf-8") as f:
                d.update(json.load(f))
        except Exception:  # noqa: S110, BLE001 — corrupt/unreadable budget file: fall back to defaults silently
            pass
    if d.get("date") != _today():
        if d.get("used_usd"):
            d.setdefault("history", []).append(
                {"date": d.get("date"), "used_usd": round(float(d.get("used_usd", 0)), 6)}
            )
        d["date"] = _today()
        d["used_usd"] = 0.0
    return d


def _save(d):
    os.makedirs(os.path.dirname(BUDGET_PATH), exist_ok=True)
    tmp = BUDGET_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, BUDGET_PATH)


def get_state():
    """Public read: what the web page shows."""
    d = _load()
    limit = float(d.get("daily_limit_usd", DEFAULT_LIMIT_USD))
    used = round(float(d.get("used_usd", 0.0)), 6)
    return {
        "daily_limit_usd": limit,
        "date": d.get("date", _today()),
        "used_today_usd": used,
        "remaining_usd": round(max(0.0, limit - used), 6),
    }


def set_limit(usd):
    """Set the daily cap. Enforced sane bounds: 0.01 .. 1000 USD/day."""
    usd = float(usd)
    if usd < 0.01 or usd > 1000.0:
        raise ValueError("daily limit must be between $0.01 and $1000")
    d = _load()
    d["daily_limit_usd"] = round(usd, 2)
    _save(d)
    return get_state()


def remaining():
    return get_state()["remaining_usd"]


def can_spend(usd):
    """True if spending `usd` more today stays within the cap."""
    return remaining() >= float(usd)


def record_spend(usd):
    """Add actual paid spend to today's total. Never blocks; caller checks first."""
    d = _load()
    d["used_usd"] = round(float(d.get("used_usd", 0.0)) + float(usd), 6)
    _save(d)
    return get_state()


def cost_usd(prompt_tokens, completion_tokens):
    """USD cost of a v4-pro call from token counts."""
    return (prompt_tokens * PRO_IN + completion_tokens * PRO_OUT) / 1_000_000


def cost_usd_claude(prompt_tokens, completion_tokens):
    """USD cost of a Claude Fable (strongest) call from token counts."""
    return (prompt_tokens * FABLE_IN + completion_tokens * FABLE_OUT) / 1_000_000


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "set":
        st = set_limit(sys.argv[2])
        print(json.dumps(st, indent=2))
    else:
        print(json.dumps(get_state(), indent=2))
