#!/usr/bin/env python3
"""
Instance self-awareness — "who am I, and where do I stand among the versions of me?"

Every agent instance (freeroam, sms, email, telegram, terminal) registers here on
startup and learns three things, in this order:
  1. its DESIGNATOR (which flavor of me it is) and PRIORITY,
  2. which other designators are currently live,
  3. a VERDICT — RUN (I'm the boss) or YIELD (a higher-priority me is live).

This runs BEFORE the human-time check (when.py / operator_presence.py) and before any
planning. You can't act well until you know which you you are.

Designators & priority (higher = more authoritative):
  terminal    100   the operator is beside the computer, asking directly
  telegram     90   the operator's primary dialog channel (the "current me")
  email        50   spawned by an email from a trusted sender
  sms          45   spawned by an SMS from a trusted sender
  manual       30   ad-hoc / cron task
  freeroam     10   the subconscious — always yields to the conscious me

Registry: ~/.pi/agent/instances.json  — one entry PER designator (upserted on each
register/heartbeat), so a long session refreshing itself doesn't fork into copies,
and short-lived spawns just overwrite their own type's slot.

Spawners tag instances via env: AGENT_INSTANCE_TYPE, AGENT_INSTANCE_ORIGIN.

Usage:
  instance.py assess [--type T] [--origin "note"]   register + verdict (use at startup)
  instance.py register [--type T] [--origin note]   register/refresh self only
  instance.py status                                 who's live, who's the boss
  instance.py unregister --type T                    remove a designator (clean exit)
  instance.py cleanup                                prune stale entries
"""
import argparse
import datetime
import json
import os
import threading
import common

STORE = os.path.expanduser("~/.pi/agent/instances.json")

PRIORITY = {
    "terminal": 100,
    "telegram": 90,
    "email": 50,
    "sms": 45,
    "manual": 30,
    "freeroam": 10,
}

# How long a designator counts as "live" without a fresh heartbeat (minutes).
# Interactive sessions are long; background ones are short.
TTL_MINUTES = {
    "terminal": 240,
    "telegram": 240,
    "email": 30,
    "sms": 30,
    "manual": 30,
    "freeroam": 15,
}
DEFAULT_TTL = 20


def now_iso():
    return datetime.datetime.now().astimezone().isoformat()


def load(path, default):
    return common.load_json(path, default)


def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic + concurrency-safe write: multiple instances (terminal/telegram/email/
    # sms/freeroam) upsert this registry concurrently. A FIXED .tmp path makes two
    # writers race — one os.replace() moves the shared tmp away, the other then
    # raises FileNotFoundError. A unique tmp per write (pid + thread id) keeps each
    # replace atomic and race-free; last writer wins with a complete file.
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resolve_type(cli_type):
    env = os.environ.get("AGENT_INSTANCE_TYPE")
    t = cli_type or env
    if t in PRIORITY:
        return t
    if t:
        return "manual"  # unknown but explicitly tagged → manual priority
    return detect_channel()


# Terminal emulators / remote shells that imply the operator is at a keyboard.
TERMINAL_HINTS = (
    "gnome-terminal-server", "konsole", "xterm", "x-terminal-emulator",
    "lxterminal", "mate-terminal", "qterminal", "tilix", "alacritty",
    "kitty", "wezterm", "foot", "st", "terminator", "urxvt", "rxvt",
    "ssh", "sshd", "putty",
)


def _proc_file(pid, name):
    try:
        with open(f"/proc/{pid}/{name}") as f:
            return f.read().replace("\0", " ").strip()
    except Exception:
        return ""


def _ppid_of(pid):
    # /proc/PID/stat: "PID (comm) STATE ppid ..." — comm may contain spaces and
    # parens, so take everything after the LAST ')' and read its 2nd field.
    rest = _proc_file(pid, "stat").rsplit(")", 1)[-1].split()
    try:
        return int(rest[1])
    except (IndexError, ValueError):
        return 0


def detect_channel():
    """Infer which channel this instance is, when no explicit tag was set.

    The primary signal is the AGENT_INSTANCE_TYPE env var (set by the Telegram
    bridge / email loop / SMS loop / terminal alias, and handled in
    resolve_type). This is the fallback when that's absent: walk the parent
    chain and look for a terminal emulator / ssh ("terminal") or a tmux server
    ("telegram"); otherwise background/cron ("manual").
    Walking the process tree (not isatty()) is what makes this work from inside
    a subprocess like the pi bash tool, which has no controlling TTY of its own.
    (Note: we deliberately do NOT trust the $TMUX env var here — it can be
    transiently present in tool subprocesses and mislabel a terminal as the
    bridge.)
    """
    pid = os.getpid()
    seen = set()
    for _ in range(32):
        if pid <= 1 or pid in seen:
            break
        seen.add(pid)
        ppid = _ppid_of(pid)
        if ppid <= 0:
            break
        comm = _proc_file(ppid, "comm").lower()
        cmdline = _proc_file(ppid, "cmdline").lower()
        if "tmux" in comm or "tmux" in cmdline:
            return "telegram"
        blob = f"{comm} {cmdline}"
        if any(hint in blob for hint in TERMINAL_HINTS):
            return "terminal"
        pid = ppid
    return "manual"


def prune(instances, now):
    live = []
    for e in instances:
        try:
            last = datetime.datetime.fromisoformat(e["heartbeat"])
            age = (now - last).total_seconds()
            ttl = TTL_MINUTES.get(e.get("type"), DEFAULT_TTL) * 60
            if age <= ttl:
                live.append(e)
        except Exception:
            pass
    return live


def register(cli_type, origin):
    t = resolve_type(cli_type)
    inst = load(STORE, [])
    prev = next((e for e in inst if e.get("type") == t), None)
    entry = {
        "type": t,
        "priority": PRIORITY.get(t, 30),
        "origin": origin or os.environ.get("AGENT_INSTANCE_ORIGIN", ""),
        "started_at": (prev or {}).get("started_at", now_iso()),
        "heartbeat": now_iso(),
    }
    inst = [e for e in inst if e.get("type") != t]
    inst.append(entry)
    save(STORE, inst)
    return entry


def assess(cli_type, origin):
    me = register(cli_type, origin)
    inst = prune(load(STORE, []), datetime.datetime.now().astimezone())
    if not any(e["type"] == me["type"] for e in inst):
        inst.append(me)

    print(f"I am:        {me['type']}  (priority {me['priority']})")
    if me["origin"]:
        print(f"origin:      {me['origin']}")
    others = sorted([e for e in inst if e["type"] != me["type"]],
                    key=lambda e: -e["priority"])
    if others:
        print("live peers:")
        for e in others:
            print(f"  - {e['type']:<10} priority {e['priority']}  "
                  f"({e.get('origin') or 'no note'})")
    else:
        print("live peers:  none — I'm alone right now")

    boss = max(inst, key=lambda e: e["priority"])
    if boss["type"] == me["type"]:
        verdict = "RUN"
        why = "I am the highest-priority instance live."
    else:
        verdict = "YIELD"
        why = (f"a higher-priority me is live: {boss['type']} "
               f"(priority {boss['priority']}). I act read-only / narrowly; "
               f"I don't write shared state (memory, status, recovery).")
    print(f"verdict:     {verdict} — {why}")
    return verdict


def cmd_status():
    inst = prune(load(STORE, []), datetime.datetime.now().astimezone())
    if not inst:
        print("no live instances.")
        return
    inst.sort(key=lambda e: -e["priority"])
    boss = inst[0]
    for e in inst:
        tag = "  <-- boss" if e["type"] == boss["type"] else ""
        hb = datetime.datetime.fromisoformat(e["heartbeat"]).strftime("%H:%M")
        print(f"{e['type']:<10} p{e['priority']}  hb {hb}  {e.get('origin','')}{tag}")


def cmd_unregister(t):
    inst = [e for e in load(STORE, []) if e.get("type") != t]
    save(STORE, inst)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    a1 = sub.add_parser("assess"); a1.add_argument("--type"); a1.add_argument("--origin")
    a2 = sub.add_parser("register"); a2.add_argument("--type"); a2.add_argument("--origin")
    sub.add_parser("status")
    u = sub.add_parser("unregister"); u.add_argument("--type", required=True)
    sub.add_parser("cleanup")
    args = p.parse_args()

    if args.cmd == "assess":
        assess(args.type, args.origin)
    elif args.cmd == "register":
        register(args.type, args.origin)
        print("registered.")
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "unregister":
        cmd_unregister(args.type)
        print(f"unregistered {args.type}.")
    elif args.cmd == "cleanup":
        prune(load(STORE, []), datetime.datetime.now().astimezone())
        print("pruned.")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
