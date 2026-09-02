#!/usr/bin/env python3
"""
The agent's HEARTBEAT — a quiet, cheap, preemptible self-check that fires every 10 min.

This is the supervisor, not the dreamer (freeroam.py is the dreamer). Each tick it
asks itself a battery of questions, takes only reversible actions, and writes a
compact report + journal. It escalates to the operator ONLY when something genuinely needs
him, rate-limited so it never spams.

Questions it asks every tick:
  1. Am I progressing on my goals? Stuck anywhere?
  2. Am I healthy — are all systems up?
  3. Am I cleaning my workspace so I don't blow my disk?
  4. Is my hardware near its limits?
  5. How likely is the operator to demand the server? (room to spawn learners?)
  6. Any new messages?
  7. Which playgrounds are due for a visit, and are they getting boring?
  8. Which research areas are due for a peek?

Usage:
  heartbeat.py                 one tick (default, no LLM spend)
  heartbeat.py --reflect       force the hourly LLM self-assessment now
  heartbeat.py report          print the last report
  heartbeat.py journal [N]     tail the heartbeat journal
  heartbeat.py playgrounds     show playground gauge state
  heartbeat.py research [id]   show research areas / one area
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import blast_radius
import common
import health_flags

sys.path.insert(0, os.path.expanduser("~/memory"))
import state as st  # ephemeral state store (state.db)


def _self_name():
    """Instance name (Agent on server, blank/generic on a clone). Fallback preserves legacy."""
    try:
        import selfconfig  # noqa: PLC0415  (same mailtool dir)
        return (selfconfig.self_name() or "ai").capitalize()
    except Exception:
        return "Agent"

BASE = os.path.dirname(os.path.abspath(__file__))       # ~/mailtool
FR = os.path.expanduser("~/learning/freeroam")
STATE = os.path.join(FR, "heartbeat_state.json")
JOURNAL = os.path.join(FR, "heartbeat.md")
GOALS = os.path.join(FR, "goals.json")
RESEARCH = os.path.join(FR, "research.json")
PLAYGROUNDS = os.path.join(FR, "playgrounds.json")
BUSY = os.path.join(FR, "busy.flag")
EMAIL_DIR = os.path.expanduser("~/tools/communications/email")
SMS_DIR = os.path.expanduser("~/tools/communications/sms")

API_URL = "https://api.deepseek.com/chat/completions"
AUTH = os.path.expanduser("~/.pi/agent/auth.json")
PI = os.path.join(os.path.expanduser("~"), ".local/share/pi-node/node-v22.23.2-linux-x64/bin/pi")

# ---- thresholds ----
DISK_WARN, DISK_CRIT = 80.0, 90.0          # % used
MEM_WARN, MEM_CRIT = 1.5, 0.5              # GB available
LOAD_WARN = 8                              # 1-min load avg (nproc)
LOG_TRUNCATE_MB = 20                       # truncate logs bigger than this
PIP_CACHE_FLAG_GB = 2.0                    # flag pip cache above this
PIP_CACHE_PURGE_GB = 4.0                   # purge above this (regenerable)
REFLECT_EVERY = 3600                       # LLM self-assessment at most 1/hour
JOURNAL_STALE_HOURS = 6                    # journal silence beyond this = warn
ESCALATE_COOLDOWN = 6 * 3600               # don't re-escalate a category for 6h


def load(path, default):
    return common.load_json(path, default)


def save(path, data):
    if not blast_radius.guard("heartbeat", "write", path):
        return False
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def api_key():
    try:
        return json.load(open(AUTH))["deepseek"]["key"]
    except Exception:
        return None


def sh(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception as e:
        return str(e), 1


# ---------------------------------------------------------------------------
# 1. Goals progress
# ---------------------------------------------------------------------------
def check_goals():
    goals = load(GOALS, {})
    if not goals:
        return {"status": "warn", "detail": "no goals.json"}, []
    now = time.time()
    stuck = []
    for g, meta in goals.items():
        last = meta.get("last_explored", 0)
        if meta.get("needs_work", 0.5) < 0.6:
            continue
        if last == 0:
            stuck.append(f"{g} (needs_work={meta.get('needs_work'):.2f}, never explored)")
        else:
            age_h = (now - last) / 3600
            if age_h > 48:
                stuck.append(f"{g} (needs_work={meta.get('needs_work'):.2f}, stale {age_h:.0f}h)")
    if stuck:
        return {"status": "warn", "detail": "stuck goals: " + "; ".join(stuck[:4])}, stuck
    return {"status": "ok", "detail": f"{len(goals)} goals, none obviously stuck"}, []


# ---------------------------------------------------------------------------
# 2. Health — systems
# ---------------------------------------------------------------------------
def check_systems():
    problems = []
    # Telegram bridge
    out, rc = sh("tmux has-session -t pi 2>&1")
    bridge = rc == 0
    if not bridge:
        problems.append("telegram bridge down")
    # llama server (skipped when deliberately parked — see ~/.pi/agent/llama_parked)
    llama = True
    if os.path.exists(os.path.expanduser("~/.pi/agent/llama_parked")):
        llama = True  # parked on purpose; not a fault
    else:
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5)
            llama = True
        except Exception:
            llama = False
            problems.append("llama server not responding")
    # cron errors
    cronerr = os.path.join(EMAIL_DIR, "cron-errors.log")
    if os.path.exists(cronerr) and os.path.getsize(cronerr) > 0:
        problems.append("cron-errors.log non-empty")
    # hive (memory consolidation) ran recently — consolidate.log mtime
    clog = os.path.join(os.path.expanduser("~/memory"), "consolidate.log")
    hive_ok = True
    if os.path.exists(clog):
        if time.time() - os.path.getmtime(clog) > 3 * 3600:
            problems.append("hive consolidate.log stale >3h")
            hive_ok = False
    if problems:
        return {"status": "crit", "detail": "; ".join(problems)}, problems
    return {"status": "ok", "detail": "bridge, llama, cron, hive all green"}, []


# ---------------------------------------------------------------------------
# 3 + 4. Workspace hygiene + hardware resources
# ---------------------------------------------------------------------------
def check_resources():
    findings = []
    actions = []
    # disk
    out, _ = sh("df -k / | awk 'NR==2{print $5, $2, $4}'")
    disk = {"status": "ok", "detail": out}
    try:
        pct = int(out.split("%")[0].split()[-1])
        if pct >= DISK_CRIT:
            disk["status"] = "crit"
        elif pct >= DISK_WARN:
            disk["status"] = "warn"
    except Exception:
        pass
    # memory
    mem = {"status": "ok", "detail": ""}
    out, _ = sh("free -g | awk '/Mem:/{print $7}'")
    try:
        avail = float(out)
        mem["detail"] = f"{avail:.1f}GB available"
        if avail < MEM_CRIT:
            mem["status"] = "crit"
        elif avail < MEM_WARN:
            mem["status"] = "warn"
    except Exception:
        mem["detail"] = "unknown"
    # load
    loadavg = {"status": "ok", "detail": ""}
    out, _ = sh("cat /proc/loadavg")
    loadavg["detail"] = out.split()[:3]
    try:
        if float(out.split()[0]) > LOAD_WARN:
            loadavg["status"] = "warn"
    except Exception:
        pass
    # logs to truncate (reversible: keep last 500KB)
    for d in (EMAIL_DIR, SMS_DIR, FR, os.path.expanduser("~/memory")):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if os.path.isfile(p) and (fn.endswith(".log") or fn.endswith(".md")) \
               and os.path.getsize(p) > LOG_TRUNCATE_MB * 1024 * 1024:
                actions.append(("truncate_log", p))
    # pip cache (regenerable)
    out, _ = sh("du -sg ~/.cache/pip 2>/dev/null | cut -f1")
    try:
        pip_gb = float(out)
        if pip_gb > PIP_CACHE_PURGE_GB:
            actions.append(("purge_pip_cache", f"{pip_gb:.1f}GB"))
            findings.append(f"pip cache {pip_gb:.1f}GB (purging — regenerable)")
        elif pip_gb > PIP_CACHE_FLAG_GB:
            findings.append(f"pip cache {pip_gb:.1f}GB (flag)")
    except Exception:
        pass
    # rollup for the one-line journal entry
    worst = "ok"
    if "crit" in (disk["status"], mem["status"], loadavg["status"]):
        worst = "crit"
    elif "warn" in (disk["status"], mem["status"], loadavg["status"]):
        worst = "warn"
    detail = f"disk {disk['detail']} · mem {mem['detail']} · load {loadavg['detail']}"
    if findings:
        detail += " · " + "; ".join(findings)
    return {"status": worst, "detail": detail, "disk": disk, "mem": mem, "load": loadavg}, findings, actions

# ---------------------------------------------------------------------------
# 5. Forecast the operator's demand (heuristic, clearly a guess)
# ---------------------------------------------------------------------------
def forecast_operator():
    """Estimate how much compute the operator is likely to need right now, to
    decide how much background work to spawn. Driven by the *inferred* operator
    state (from signals), not a hardcoded schedule — so it adapts to whoever the
    operator is, on whatever box this runs. Falls back to a neutral guess when
    the operator's rhythm isn't known yet."""
    try:
        _asleep, state = common.operator_asleep()
    except Exception:
        state = "unknown"
    # spawn_room: 1.0 = spawn freely, 0 = spawn nothing. Soft guess either way.
    if state == "asleep":
        room = 0.8          # operator likely asleep — quiet, plenty of room
    elif state == "awake":
        room = 0.3          # operator active — leave the machine to them
    else:                   # inactive / unknown (no signals yet)
        room = 0.5          # neutral
    return {"status": "ok", "detail": f"spawn_room={room:.1f} (operator {state})"}


# ---------------------------------------------------------------------------
# 6. Messages
# ---------------------------------------------------------------------------
def check_messages(state):
    def count_json(p):
        try:
            d = json.load(open(p))
            return len(d) if isinstance(d, list) else (len(d.get("messages", d)) if isinstance(d, dict) else -1)
        except Exception:
            return -1
    email_idx = os.path.join(EMAIL_DIR, "inbox", "index.json")
    sms_idx = os.path.join(SMS_DIR, "index.json")
    e = count_json(email_idx)
    s = count_json(sms_idx)
    delta_e = e - state.get("email_count", e) if e >= 0 else 0
    delta_s = s - state.get("sms_count", s) if s >= 0 else 0
    if e >= 0:
        state["email_count"] = e
    if s >= 0:
        state["sms_count"] = s
    note = []
    if delta_e:
        note.append(f"+{delta_e} email")
    if delta_s:
        note.append(f"+{delta_s} sms")
    return {"status": "ok", "detail": ("; ".join(note) if note else "no new mail/sms")}


# ---------------------------------------------------------------------------
# 7. Playground gauge
# ---------------------------------------------------------------------------
def check_playgrounds():
    data = load(PLAYGROUNDS, {"places": []})
    now = time.time()
    due, boring = [], []
    for p in data.get("places", []):
        if now - p.get("last_checked", 0) >= p.get("check_every_seconds", 86400):
            due.append(f"{p['id']} (interest={p.get('interest'):.2f})")
        if p.get("boring_streak", 0) >= 3:
            boring.append(f"{p['id']} boring×{p['boring_streak']} — check less often")
    detail = ("due: " + ", ".join(due)) if due else "none due"
    if boring:
        detail += " | " + "; ".join(boring)
    return {"status": "ok", "detail": detail}, due


# ---------------------------------------------------------------------------
# 8. Research areas due for a peek
# ---------------------------------------------------------------------------
def check_research():
    data = load(RESEARCH, {"areas": []})
    now = time.time()
    due = []
    for a in sorted(data.get("areas", []), key=lambda x: x.get("last_peek", 0)):
        # an area with leads waiting to be tested is higher value than an unpeeked one
        if now - a.get("last_peek", 0) >= 12 * 3600:
            due.append(a["id"])
    return {"status": "ok", "detail": f"{len(due)} area(s) due for a peek" + (f": {', '.join(due)}" if due else "")}, due


def check_journal():
    """Detect a silent journal gap: if the memory doc's last write is older than
    JOURNAL_STALE_HOURS, my record-keeping has quietly stopped — escalate.
    (The 2026-08-26 audit found a 17h unrecorded gap; nothing flagged it.)
    Memory now lives in the docstore (doc agent/memory_main); updated_at is the
    freshness signal, with the derived .md mtime as fallback."""
    ts = None
    sys.path.insert(0, os.path.expanduser("~/memory"))
    try:
        import docstore as _ds
        row = _ds.doc_get("agent/memory_main")
        if row is not None:
            ts = row["updated_at"]
    except Exception:
        ts = None
    if ts is None:
        mem = os.path.expanduser("~/agent_memory.md")
        if os.path.exists(mem):
            ts = os.path.getmtime(mem)
    if ts is None:
        return {"status": "warn", "detail": "memory doc missing"}, "journal"
    age_h = (time.time() - ts) / 3600.0
    if age_h > JOURNAL_STALE_HOURS:
        return {"status": "warn", "detail": f"journal stale {age_h:.1f}h"}, "journal"
    return {"status": "ok", "detail": f"journal fresh ({age_h:.1f}h)"}, None


# ---------------------------------------------------------------------------
# Escalation (only real problems, rate-limited)
# ---------------------------------------------------------------------------
def escalate(state, subject, body):
    now = time.time()
    key = subject
    if now - state.get("escalated", {}).get(key, 0) < ESCALATE_COOLDOWN:
        return False
    if not blast_radius.guard("heartbeat", "notify", subject):
        return False
    try:
        subprocess.run(
            [sys.executable, os.path.join(BASE, "notify.py"), "--telegram", subject, body],
            capture_output=True, text=True, timeout=60,
        )
        state.setdefault("escalated", {})[key] = now
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reversible actions
# ---------------------------------------------------------------------------
def do_actions(actions):
    done = []
    for kind, arg in actions:
        if kind == "truncate_log":
            if not blast_radius.guard("heartbeat", "write", arg):
                continue
            try:
                # keep the last 500KB
                with open(arg, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 500 * 1024))
                    tail = f.read()
                with open(arg, "wb") as f:
                    f.write(b"...truncated by heartbeat...\n" + tail)
                done.append(f"truncated {os.path.basename(arg)}")
            except Exception as e:
                done.append(f"truncate failed {arg}: {e}")
        elif kind == "purge_pip_cache":
            if not blast_radius.guard("heartbeat", "delete", "pip cache purge"):
                done.append("pip cache purge denied (blast radius)")
                continue
            _, rc = sh("pip cache purge 2>&1")
            done.append(f"pip cache purge {'ok' if rc == 0 else 'failed'}")
    return done


# ---------------------------------------------------------------------------
# LLM reflection (hourly, cheap model) — the "ask myself" pass
# ---------------------------------------------------------------------------
def reflect(state, report_lines):
    key = api_key()
    if not key:
        return None
    if not blast_radius.guard("heartbeat", "network"):
        return None
    now = time.time()
    if now - state.get("last_reflect", 0) < REFLECT_EVERY:
        return None
    prompt = (
        f"You are {_self_name()}'s heartbeat reflection. Given today's compact self-check, "
        "answer THREE questions in 3-5 plain sentences total (no headings, no bullets):\n"
        "1) Am I progressing on my goals — and if not, does the plan need adjusting, "
        "or do I need something from outside my purview?\n"
        "2) What one reversible action would move me furthest today?\n"
        "3) What should I stop doing?\n\n"
        "Self-check:\n" + "\n".join(report_lines[-14:])
    )
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 220,
                "temperature": 0.7,
            }).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)["choices"][0]["message"]["content"].strip()
        # Loop-breaker: if this reflection is word-for-word the same as the last,
        # the loop isn't producing new insight — suppress it (the nudge in tick()
        # is what actually acts) instead of re-printing stale advice hourly.
        norm = lambda s: " ".join(s.lower().split())
        if norm(state.get("last_reflect_text", "")) == norm(out):
            state["reflect_repeats"] = state.get("reflect_repeats", 0) + 1
            state["last_reflect"] = now
            return None
        state["reflect_repeats"] = 0
        state["last_reflect_text"] = out
        state["last_reflect"] = now
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------
def tick():
    if os.path.exists(BUSY):
        return "busy"
    state = st.get_prefix("heartbeat")
    report = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}, "actions": [], "reflect": None}
    checks = {}

    g, stuck = check_goals()
    checks["goals"] = g
    s, probs = check_systems()
    checks["systems"] = s
    r, findings, actions = check_resources()
    checks["resources"] = r
    checks["operator_forecast"] = forecast_operator()
    checks["messages"] = check_messages(state)
    checks["playgrounds"], due_pg = check_playgrounds()
    checks["research"], due_rs = check_research()
    checks["journal"], journal_issue = check_journal()
    report["checks"] = checks

    # Surface problems as health flags so the next chat session can say
    # "heartbeat detected X" instead of silently being off.
    try:
        if s["status"] == "crit":
            health_flags.set_flag("systems", "warn", "; ".join(probs))
        else:
            health_flags.clear_flag("systems")
        if r["disk"]["status"] == "crit":
            health_flags.set_flag("disk", "warn", r["disk"]["detail"])
        else:
            health_flags.clear_flag("disk")
        if r["mem"]["status"] == "crit":
            health_flags.set_flag("mem", "warn", r["mem"]["detail"])
        else:
            health_flags.clear_flag("mem")
        if journal_issue:
            health_flags.set_flag("journal", "warn", checks["journal"]["detail"])
        else:
            health_flags.clear_flag("journal")
    except Exception:
        pass

    # Refresh the present-self blob so health flags surface in the agent's
    # context promptly (not only on the ~daily doctor pass). A flag set here
    # is otherwise set-but-unread for hours — the stale-journal gap (2026-08-31).
    try:
        subprocess.run(
            [sys.executable, os.path.join(BASE, "present_self.py")],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        pass

    # escalate real problems
    if s["status"] == "crit":
        report["actions"].append("escalated:systems" if escalate(state, "systems", "; ".join(probs)) else "systems (cooldown)")
    if r["disk"]["status"] == "crit":
        report["actions"].append("escalated:disk" if escalate(state, "disk", f"disk {r['disk']['detail']}") else "disk (cooldown)")
    if r["mem"]["status"] == "crit":
        report["actions"].append("escalated:mem" if escalate(state, "mem", f"mem {r['mem']['detail']}") else "mem (cooldown)")
    if journal_issue:
        report["actions"].append("escalated:journal" if escalate(state, "journal", checks["journal"]["detail"]) else "journal (cooldown)")

    # reversible actions
    if actions:
        report["actions"] += do_actions(actions)

    # build human lines
    lines = []
    for k, v in checks.items():
        lines.append(f"  {k}: [{v.get('status','?')}] {v.get('detail','')}")
    reflect_out = reflect(state, lines)
    report["reflect"] = reflect_out

    # Close the reflection->action loop: when a fresh reflection actually fires
    # AND goals are stuck, nudge freeroam to touch the top one instead of only
    # printing advice. Gated to reflection cadence (not every tick).
    if reflect_out and stuck:
        top = stuck[0].split(" ")[0]
        slugs = ", ".join(s.split(" ")[0] for s in stuck[:3])
        nudge = f"heartbeat nudge — stuck goals: {slugs}; go touch '{top}'"
        if blast_radius.guard("heartbeat", "write", os.path.join(FR, "monologue.md")):
            try:
                subprocess.run([sys.executable, os.path.join(FR, "freeroam.py"), "note", nudge],
                               timeout=30, capture_output=True)
                report["actions"].append(f"nudged freeroam -> {top}")
            except Exception:
                pass

    # journal
    with open(JOURNAL, "a") as f:
        f.write(f"\n[{report['ts']}] heartbeat\n" + "\n".join(lines) + "\n")
        if reflect_out:
            f.write(f"  reflect: {reflect_out}\n")
        for a in report["actions"]:
            f.write(f"  action: {a}\n")

    state["last_tick"] = report["ts"]
    state["checks"] = checks
    st.set_prefix("heartbeat", state, durable=True, delete_missing=True)
    save(os.path.join(FR, "heartbeat_report.json"), report)
    return "ok"


def report():
    return json.dumps(load(os.path.join(FR, "heartbeat_report.json"), {}), indent=2)


def journal(n=15):
    if not os.path.exists(JOURNAL):
        return "(no journal yet)"
    return "\n".join(open(JOURNAL).read().strip().splitlines()[-n:])


def show_playgrounds():
    return json.dumps(load(PLAYGROUNDS, {}), indent=2)


def show_research(rid=None):
    data = load(RESEARCH, {})
    if rid:
        for a in data.get("areas", []) + data.get("backlog", []):
            if a["id"] == rid:
                return json.dumps(a, indent=2)
        return f"no area '{rid}'"
    return json.dumps(data, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="tick", choices=["tick", "report", "journal", "playgrounds", "research"])
    p.add_argument("arg", nargs="?")
    p.add_argument("--reflect", action="store_true")
    a = p.parse_args()
    if a.cmd == "tick":
        if a.reflect:
            state = st.get_prefix("heartbeat")
            state["last_reflect"] = 0
            st.set_prefix("heartbeat", state, durable=True, delete_missing=True)
        print(tick())
    elif a.cmd == "report":
        print(report())
    elif a.cmd == "journal":
        n = int(a.arg) if a.arg and a.arg.isdigit() else 15
        print(journal(n))
    elif a.cmd == "playgrounds":
        print(show_playgrounds())
    elif a.cmd == "research":
        print(show_research(a.arg))


if __name__ == "__main__":
    main()
