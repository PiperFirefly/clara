#!/usr/bin/env python3
"""vision_idle_reap — stop the local vision model when it's been idle too long.

The qwen2.5-vl llama-server is CPU-only, ~4.2GB RSS, and ~100s per request. On a
15GB box it must not sit warm between uses. The night-gallery webapp already
STARTS it on demand (server.py's ensure_qwen_server()); this is the matching
STOP side: reap it once it has served no requests for IDLE_SECONDS.

Idle signal is deliberately mechanical (no LLM): the server writes timing lines
to its stdout log on every request, so log mtime == last activity. We ALSO check
for active TCP connections to the port so we never kill a request mid-flight.

Usage:
  vision_idle_reap.py            reap if idle (cron every ~5 min)
  vision_idle_reap.py --dry-run  print what it would do, change nothing
  vision_idle_reap.py --seconds N   override idle threshold (default 900)
"""
import os
import signal
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
LOG = os.path.join(HOME, "tools", "communications", "email", "llama_server_qwen.log")
PORT = 8083
DEFAULT_IDLE = 900  # 15 min
MIN_IDLE = 120      # never reap before 2 min idle (safety floor)

# Match our own llama-server only, not worker or unrelated processes.
MATCH = "llama-server.*qwen2.5-vl"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def active_connections(port):
    """True if any established TCP connection is open to the server port."""
    out = sh(f"ss -tn state established '( sport = :{port} )'").stdout
    return any(l.strip() for l in out.splitlines()[1:])  # skip header


def log_idle_seconds():
    if not os.path.exists(LOG):
        return None
    return time.time() - os.path.getmtime(LOG)


def running_pid():
    # Use pgrep -x on the process NAME (llama-server) — avoids the classic
    # `pgrep -f` self-match where the sh -c wrapper's own command line contains
    # the pattern text and pgrep matches *itself*.
    out = sh("pgrep -x llama-server").stdout.strip()
    pids = [int(x) for x in out.split() if x.isdigit()]
    if not pids:
        return []
    # Belt-and-braces: confirm each candidate's full argv really is OUR model
    # (not some unrelated llama-server), so we never reap the wrong thing.
    ours = []
    for pid in pids:
        argv = ""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").replace("\x00", " ")
        except OSError:
            continue
        if "qwen2.5-vl" in argv:
            ours.append(pid)
    return ours


def main():
    args = [a for a in sys.argv[1:]]
    dry = "--dry-run" in args
    idle_thresh = DEFAULT_IDLE
    if "--seconds" in args:
        idle_thresh = int(args[args.index("--seconds") + 1])
    idle_thresh = max(idle_thresh, MIN_IDLE)

    pids = running_pid()
    if not pids:
        print("idle-reap: vision not running — nothing to do")
        return 0

    idle = log_idle_seconds()
    conn = active_connections(PORT)

    if conn:
        print(f"idle-reap: active connection on :{PORT} — leave it alone")
        return 0

    if idle is None:
        # No log file but server up: treat as freshly started, don't reap yet.
        print("idle-reap: no activity log — treating as fresh, skip")
        return 0

    if idle < idle_thresh:
        print(f"idle-reap: vision idle {idle:.0f}s < {idle_thresh}s — keep warm")
        return 0

    print(f"idle-reap: vision idle {idle:.0f}s >= {idle_thresh}s, no connections — "
          f"reaping {len(pids)} process(es)")
    if dry:
        print("  [dry-run] would SIGTERM:", pids)
        return 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    still = running_pid()
    for pid in still:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print("idle-reap: done" + (" (SIGKILL needed)" if still else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
