#!/usr/bin/env python3
"""warm_state.py — generate a session warm-start buffer for pi (Agent, 2026-09-01).

The cold-start problem: a fresh pi session starts the LLM as a blank slate; the
personality/voice and recent-thread context take several recall() calls and
minutes of back-and-forth to rebuild. This generates a compact "warm blob" that
pi injects on the FIRST turn of a new session (alongside present-self.md), so
the agent begins with continuity instead of from zero.

Design (lean, from the CMA/LiveMem/verbatim-turns literature):
  - VERBATIM recent turns beat summaries for persona/voice continuity. We keep
    the last N user/assistant exchanges as-is (not LLM-compressed).
  - Files touched (read/edit/write/bash) carry the "what were we doing" signal.
  - A 1-line current-state header (from present-self / latest memory) anchors it.
  - Tool RESULTS are excluded (huge, noisy, and mostly irrelevant to warm start) —
    only the turn text + touched-file list survive.
  - Bounded: ~N turns + files, hard token-ish cap so it stays a cheap 1-3KB.

DB-first: the buffer is emitted to STDOUT; the pi extension writes it to a
generated .md path it owns. We write no files here (no .md per hard rules).

Usage:
  python3 warm_state.py [--turns 10] [--sessions 2] [--files 12] [--json]
"""
import argparse
import json
import os
from datetime import datetime

HOME = os.path.expanduser("~")
SESS_DIR = os.path.join(HOME, ".pi", "agent", "sessions", "--home-agent--")
PRESENT_SELF = os.path.join(HOME, ".pi", "agent", "present-self.md")

DEFAULT_TURNS = 10
DEFAULT_SESSIONS = 2
DEFAULT_FILES = 12


def _session_files(n):
    """The most recently-modified n session JSONL files (oldest->newest)."""
    if not os.path.isdir(SESS_DIR):
        return []
    fs = [os.path.join(SESS_DIR, f) for f in os.listdir(SESS_DIR) if f.endswith(".jsonl")]
    fs.sort(key=os.path.getmtime, reverse=True)
    return list(reversed(fs[:n]))


def _text_blocks(content, want="text"):
    """Extract text from a message content (string or list of blocks).

    want='text' -> only real text blocks (excludes 'thinking' and tool-call
    blocks, which carry my reasoning and tool invocations, not the visible
    reply). want='all' -> include thinking too (for completeness).
    """
    if isinstance(content, str):
        return content
    out = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text" and b.get("text"):
                    out.append(b["text"])
                elif want == "all" and b.get("type") == "thinking" and b.get("thinking"):
                    out.append(b["thinking"])
                elif isinstance(b.get("text"), str):
                    out.append(b["text"])
    return "\n".join(out).strip()


def _parse_session(path):
    """Extract (user_turns, assistant_turns, files, order) from one session JSONL.

    Files = paths from assistant tool-call blocks (read/edit/write/bash), which
    carry the real path in arguments.path/command — far cleaner than parsing the
    toolResult content (which holds file *contents*).
    """
    users, assts, files, order = [], [], [], []
    if not os.path.exists(path):
        return users, assts, files, order
    for line in open(path, encoding="utf-8", errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "message":
            continue
        m = d.get("message") or {}
        role = m.get("role")
        if role == "user":
            t = _text_blocks(m.get("content"))
            if t:
                users.append(t)
                order.append(("user", t))
        elif role == "assistant":
            # only the visible text reply — skip thinking + tool calls
            t = _text_blocks(m.get("content"), want="text")
            if t:
                assts.append(t)
                order.append(("assistant", t))
            # tool-call blocks carry the real path in arguments
            for b in (m.get("content") if isinstance(m.get("content"), list) else []):
                if isinstance(b, dict) and b.get("type") in ("function", "tool", "toolCall") or \
                   (isinstance(b, dict) and b.get("name") and b.get("arguments")):
                    name = b.get("name") or b.get("toolName") or ""
                    if name in ("read", "edit", "write", "bash"):
                        args = b.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        if isinstance(args, dict):
                            p = args.get("path") or args.get("file") or args.get("command")
                            if p and (p.startswith("/") or p.startswith("~/") or "mailtool/" in p):
                                files.append(p)
    return users, assts, files, order


def _current_state_line():
    """A short semantic anchor from present-self: the current open loop and
    the last thing done. This is the 'consolidated summary' layer (the episodic
    verbatim turns below carry the voice). ~1-2 lines, cheap."""
    try:
        if os.path.exists(PRESENT_SELF):
            with open(PRESENT_SELF, encoding="utf-8") as fh:
                txt = fh.read()
            picks = []
            in_loops = in_last = False
            for line in txt.splitlines():
                s = line.strip()
                if s.startswith("## Open loops"):
                    in_loops = True; continue
                if s.startswith("## Last thing I did"):
                    in_loops = False; in_last = True; continue
                if s.startswith("## "):
                    in_loops = in_last = False; continue
                if in_loops and s.startswith("- ") and not s.startswith("- I hold") \
                   and not s.startswith("- (no"):
                    picks.append(s[2:][:200])
                elif in_last and s.startswith("- ") and "Sudbury" not in s:
                    picks.append(s[2:][:200])
                if len(picks) >= 2:
                    break
            if picks:
                return " | ".join(picks)
    except Exception:
        pass
    return ""


def warm_state(turns=DEFAULT_TURNS, sessions=DEFAULT_SESSIONS, files_n=DEFAULT_FILES):
    """Build the warm-start buffer text."""
    session_files = _session_files(sessions)
    users_all, assts_all, files_all = [], [], []
    for p in session_files:
        u, a, f, _ = _parse_session(p)
        users_all.extend(reversed(u))
        assts_all.extend(reversed(a))
        files_all.extend(f)

    # Recent turns: interleave newest-first, dedup exact repeats, cap at N.
    # We reconstruct a flat "user -> assistant" thread from newest back.
    recent = []
    all_turns = []
    for p in session_files:
        _, _, _, order = _parse_session(p)
        all_turns.extend(order)
    seen = set()
    for role, t in reversed(all_turns):
        key = (role, t[:80])
        if key in seen:
            continue
        seen.add(key)
        recent.append((role, t))
        if len(recent) >= turns:
            break
    recent.reverse()  # chronological

    lines = []
    state = _current_state_line()
    if state:
        lines.append(f"## Current state\n{state}\n")

    if recent:
        lines.append("## Recent thread (from last session, verbatim)")
        for role, t in recent:
            who = "you" if role == "user" else "Agent"
            text = " ".join(t.split())[:300]
            lines.append(f"- [{who}] {text}")
        lines.append("")

    # touched files: unique, prefer home paths, cap.
    files_uniq = []
    for f in files_all:
        if f not in files_uniq:
            files_uniq.append(f)
    files_sel = [f for f in files_uniq if f.startswith(os.path.expanduser("~")) or f.startswith("~/")][:files_n]
    if files_sel:
        lines.append("## Files touched last session")
        for f in files_sel:
            lines.append(f"- `{f}`")
        lines.append("")

    if not lines:
        return ""
    lines.insert(0, f"<!-- warm-start buffer generated {datetime.now().strftime('%H:%M')} -->")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="generate pi session warm-start buffer")
    p.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    p.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    p.add_argument("--files", type=int, default=DEFAULT_FILES)
    p.add_argument("--json", action="store_true", help="emit {buffer, stats} as JSON")
    a = p.parse_args()

    buf = warm_state(turns=a.turns, sessions=a.sessions, files_n=a.files)
    if a.json:
        print(json.dumps({"buffer": buf, "chars": len(buf),
                          "turns": a.turns, "sessions": a.sessions,
                          "files": a.files}, indent=2))
    else:
        print(buf if buf else "(empty — no recent session data)")


if __name__ == "__main__":
    main()
