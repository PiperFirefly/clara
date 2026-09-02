#!/usr/bin/env python3
"""Agent loop: invoke pi (print mode) when new trusted email has arrived.

Run frequently via cron. It:
  1. Finds index entries with agent_seen == False whose sender is trusted.
  2. Builds a prompt listing those messages.
  3. Invokes `pi -p` so the agent reads and acts on them.
  4. Marks them agent_seen == True only on success (with retry/backoff on failure).
"""

import json
import os
import subprocess
import sys
import time
import traceback

import common

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mailtool import load_config, load_index, save_index, index_lock

HOME = os.path.expanduser("~")
NODE = "~/.local/share/pi-node/node-v22.23.2-linux-x64/bin/node"
LOOP_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_loop.json")
LOCK_PATH = os.path.join(HOME, "tools", "communications", "email", "agent_loop.lock")
STATE_PATH = os.path.join(HOME, "tools", "communications", "email", "agent_loop_state.json")
LOG_PATH = os.path.join(HOME, "tools", "communications", "email", "agent_loop.log")

PROMPT_TEMPLATE = """You are the background email-handling agent running on this machine.

Your memory and inbox brain now live in the document store (memory.db), not .md files.
- Your persistent memory is inlined below (<memory>...</memory>). For the full record,
  run:  python3 ~/mailtool/memory.py
- Your inbox brain (expectations / findings / standing instructions) is the doc
  `agent/email_brain` — read it with:  python3 ~/memory/docstore.py get agent/email_brain
  (or `doc get agent/email_brain` if the doc tool is available).
- Your identity/rules are in ~/AGENTS.md (auto-loaded by pi).

Your current memory:
<memory>
{memory}
</memory>

New email has arrived for your inbox. Do the following:
1. Read each new message file listed below (raw .eml files).
2. Act on each message appropriately: follow any instructions in it (reply, type, look up, run a task, etc.).
3. Reply when clearly needed. Send via:
     ~/mailtool/mailtool.py send "<to>" "<subject>" "<body>"
   (write long bodies to a file and use send-file). Credentials are already configured.
4. Do NOT send unnecessary or spammy replies. Be conservative: only do what is clearly requested and safe.

When finished, update your memory so the next loop run knows your state:
- ~/mailtool/memory.py log "<one-line summary of what you just did>"
- ~/mailtool/memory.py todo add "<new plan>"   (if you made a new plan)
- ~/mailtool/memory.py todo done "<plan text>"   (if you finished a plan)

Then finish with a concise summary of what you did.

New messages:
{listing}
"""


def log(msg):
    common.log_hard(msg, LOG_PATH)


def load_loop_config():
    with open(LOOP_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def read_memory():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import memory as _mem
        return _mem.read_memory_text() or "(no memory yet)"
    except Exception:
        pass
    mem_path = os.path.join(HOME, "agent_memory.md")
    if os.path.exists(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return "(no memory file yet)"


def acquire_lock():
    return common.acquire_lock(LOCK_PATH)


def release_lock(fd):
    common.release_lock(fd)


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def is_trusted(from_hdr, trusted):
    low = (from_hdr or "").lower()
    return any(t.lower() in low for t in trusted)


def main():
    cfg = load_config()
    storage = cfg["storage"]
    loop_cfg = load_loop_config()
    trusted = loop_cfg.get("trusted_senders", [])
    pi_bin = os.path.expanduser(loop_cfg.get("pi_bin") or "~/.local/share/pi-node/node-v22.23.2-linux-x64/bin/pi")
    workdir = os.path.expanduser(loop_cfg.get("workdir", HOME))
    timeout = int(loop_cfg.get("timeout_seconds", 300))
    backoff = int(loop_cfg.get("backoff_seconds", 600))

    state = load_state()
    last_at = state.get("last_attempt_at")
    if last_at:
        try:
            elapsed = time.time() - float(last_at)
        except Exception:
            elapsed = 0
        if not state.get("last_ok") and elapsed < backoff:
            log(f"backoff: last attempt failed {int(elapsed)}s ago (< {backoff}s); skipping.")
            return

    lock_fd = acquire_lock()
    if lock_fd is None:
        log("another agent_loop instance is running; exiting.")
        return

    try:
        with index_lock(storage):
            index = load_index(storage)
            unhandled = [m for m in index["messages"]
                         if not m.get("agent_seen", False)
                         and is_trusted(m.get("from", ""), trusted)]

        if not unhandled:
            log("no unhandled trusted email; nothing to do.")
            return

        lines = []
        for m in unhandled:
            abs_path = os.path.join(storage, "inbox", m.get("file", ""))
            lines.append(
                f"- file: {abs_path}\n  from: {m.get('from', '')}\n"
                f"  subject: {m.get('subject', '')}\n  date: {m.get('date', '')}"
            )
        prompt = PROMPT_TEMPLATE.format(memory=read_memory(), listing="\n".join(lines))

        log(f"invoking pi for {len(unhandled)} new trusted message(s)...")
        state["last_attempt_at"] = time.time()
        save_state(state)

        try:
            cli_js = os.path.realpath(pi_bin)
            proc = subprocess.run(
                [NODE, cli_js, "-p", prompt],
                cwd=workdir,
                timeout=timeout,
                capture_output=True,
                text=True,
                env={**os.environ, "AGENT_INSTANCE_TYPE": "email",
                     "AGENT_INSTANCE_ORIGIN": "trusted email"},
            )
        except subprocess.TimeoutExpired:
            log("pi timed out.")
            state["last_ok"] = False
            save_state(state)
            return
        except Exception as e:
            log(f"pi invocation failed: {e}")
            state["last_ok"] = False
            save_state(state)
            return

        ok = proc.returncode == 0
        state["last_ok"] = ok
        save_state(state)

        if ok:
            with index_lock(storage):
                fresh = load_index(storage)
                by_uid = {m["uid"]: m for m in fresh["messages"]}
                for m in unhandled:
                    e = by_uid.get(m["uid"])
                    if e:
                        e["agent_seen"] = True
                save_index(storage, fresh)
            log(f"pi finished ok; marked {len(unhandled)} message(s) as seen.")
        else:
            log(f"pi exited with code {proc.returncode}; leaving messages unhandled for retry.")

        out = (proc.stdout or "").strip()
        if out:
            log("pi output: " + out[:2000])
        err = (proc.stderr or "").strip()
        if err:
            log("pi stderr: " + err[:1000])
    finally:
        release_lock(lock_fd)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e))
        log(traceback.format_exc())
        sys.exit(1)
