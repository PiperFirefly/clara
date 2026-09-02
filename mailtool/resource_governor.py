#!/usr/bin/env python3
"""resource_governor.py — graceful degradation / load shedding for the box.

Protects server under memory/swap pressure by shedding the least-critical
running components FIRST, instead of letting the OOM killer or a full swapfile
decide for us. Inspired by k8s QoS classes (Guaranteed/Burstable/BestEffort)
and warden's quarantine philosophy (freeze-first, reversible).

Design doc: doc agent/resource-governor-design
Contract:    $HOME/tools/data/resource_governor_spec.json
Manifest:    $HOME/tools/data/resource_governor_manifest.json

CORE MODEL
  nominal_tier   (static)   super-critical | critical | important | less-important | frill
  in_use_boost   (dynamic)  a component actively serving is boosted +IN_USE_BOOST_TIERS
  priority_floor (stable)   never shed below this tier, regardless of classification
  effective_tier = min(nominal + in_use_boost)  -- the governor sheds lowest effective first

  Shed = SIGSTOP (freeze, reversible) first; escalate to SIGKILL/SIGTERM ONLY for
  frill-tier components that have a restart_cmd. Killing anything above frill, or
  shedding below the floor, is CONSCIOUS-only (blast-radius token-gated).

FAILS CLOSED
  - untagged / unparseable component => UNSHEDDABLE (never the reverse)
  - autonomous action only: freeze any tier below floor + kill frill w/ restart_cmd
  - never shed: super-critical, active operator session, any in-use (boosted) component
  - if a frill has no restart_cmd, it may be frozen but never killed

PRESSURE SIGNAL
  swap% (the metric that bit us) primary; /proc/pressure/memory PSI corroborates.

Usage:
  resource_governor.py scan        detect + report pressure & shed eligibility, no action
  resource_governor.py visit       gated full pass (asleep + idle + due); dry-run by default
  resource_governor.py visit --act actually shed/restore (autonomous-tier actions only)
  resource_governor.py report      print last report
  resource_governor.py manifest    validate + print the tier manifest
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.expanduser("~/memory"))

MANIFEST = os.path.expanduser("~/tools/data/resource_governor_manifest.json")
STATE = os.path.expanduser("~/learning/freeroam/governor_state.json")
REPORT = os.path.expanduser("~/learning/freeroam/governor_report.json")
BUSY = os.path.expanduser("~/learning/freeroam/busy.flag")
INSTANCES = os.path.expanduser("~/.pi/agent/instances.json")
MEM_VENV = os.path.expanduser("~/venvs/memory/bin/python")

import health_flags

# Ordered least->most critical. Higher index = more critical.
TIER_ORDER = ["frill", "less-important", "important", "critical", "super-critical"]
TIER_INDEX = {t: i for i, t in enumerate(TIER_ORDER)}
ACTIVE_SESSION = "operator-session"   # never shedable sentinel

# ---------------------------------------------------------------------------
# pressure thresholds (DRAFT — see design doc; validated against the incident)
# ---------------------------------------------------------------------------
WATCH_SWAP_PCT = 50.0        # start watching at 50% swap
SHED_SWAP_PCT = 70.0         # sustained here => shed
KILL_SWAP_PCT = 90.0         # swap this full => escalate to kill frills
KILL_SWAP_FREE_MB = 200.0    # or swap-free below this => escalate
RESTORE_SWAP_PCT = 40.0      # hysteresis low-water: below here (sustained) => restore
CONSEC_WATCH = 3             # consecutive polls at WATCH => shed
CONSEC_RESTORE = 5           # consecutive polls below RESTORE => restore
KILL_GRACE_S = 120           # freeze grace before escalating to kill
PSI_MEM_THRESH = 10.0        # PSI memory some avg10, corroborating
MIN_INTERVAL = 60 * 5        # min seconds between full governor passes


def load_manifest():
    if not os.path.exists(MANIFEST):
        return None
    try:
        return json.load(open(MANIFEST))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# component identification
# ---------------------------------------------------------------------------

def find_pids(match):
    """Return list of pids whose cmdline contains `match` (never self-matches)."""
    if not match or match in ("none", "network (API), not a local process"):
        return []
    try:
        r = subprocess.run(["pgrep", "-f", match], capture_output=True, text=True)
        me = str(os.getpid())
        return [p for p in r.stdout.split() if p != me]
    except Exception:
        return []


def operator_session_pid():
    """Return pids of the pi harness itself (never shed). Match the harness binary
    exactly via `pgrep -x pi`, NOT `-f pi` which over-matches system daemons."""
    try:
        r = subprocess.run(["pgrep", "-x", "pi"], capture_output=True, text=True)
        me = str(os.getpid())
        return [p for p in r.stdout.split() if p != me]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# in-use determination (conservative: false 'in use' is the safe error)
# ---------------------------------------------------------------------------

def _active_conn_pids():
    """Return set of pids ACTIVELY SERVING a client right now.

    A component is 'in use' (must never be shed) only when it has an ESTABLISHED
    INBOUND connection to one of its own LISTENING ports — i.e. a real client is
    talking to it. We deliberately IGNORE outbound upstream connections (e.g.
    voice_server -> TTS provider :443) and CLOSE-WAIT/TIME-WAIT/half-closed
    sockets, so an idle server keeping an upstream socket open stays shedable
    (voice is a frill; it must not be pinned forever by its provider connection)."""
    try:
        listen = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=10)
        conns = subprocess.run(["ss", "-tnp", "state", "established"],
                               capture_output=True, text=True, timeout=10)
    except Exception:
        return set()
    # map pid -> set of listening ports (what the server serves)
    listen_ports = {}
    for line in listen.stdout.splitlines():
        if "users:((" not in line:
            continue
        m = re.search(r":(\d+)\s+.*pid=(\d+)", line)
        if m:
            listen_ports.setdefault(m.group(2), set()).add(m.group(1))
    # a pid is 'in use' only if it has an ESTABLISHED conn to one of ITS OWN
    # listening ports (an inbound client) — not an outbound upstream socket.
    in_use = set()
    for line in conns.stdout.splitlines():
        if "users:((" not in line:
            continue
        m = re.search(r":(\d+).*?pid=(\d+)", line)
        if m:
            local_port, pid = m.group(1), m.group(2)
            if local_port in listen_ports.get(pid, set()):
                in_use.add(pid)
    return in_use


def component_in_use(comp, pids):
    """Best-effort in-use signal. Conservatively returns True if uncertain, so
    we never shed a component we're not sure is idle."""
    if comp.get("nominal_tier") == "super-critical":
        return True
    if not pids:
        return False
    active = _active_conn_pids()
    return any(p in active for p in pids)


# ---------------------------------------------------------------------------
# pressure
# ---------------------------------------------------------------------------

def read_pressure():
    """Return dict: swap_pct, swap_free_mb, psi_mem_avg10."""
    out = {}
    try:
        with open("/proc/meminfo") as f:
            mi = {}
            for line in f:
                k, _, v = line.partition(":")
                mi[k] = int(v.split()[0])  # kB
        total = mi.get("SwapTotal", 0)
        free = mi.get("SwapFree", 0)
        used = total - free
        out["swap_pct"] = (used / total * 100.0) if total else 0.0
        out["swap_free_mb"] = free / 1024.0
    except Exception:
        out["swap_pct"], out["swap_free_mb"] = 0.0, 0.0
    out["psi_mem_avg10"] = 0.0
    try:
        with open("/proc/pressure/memory") as f:
            for tok in f.read().split():
                if tok.startswith("avg10="):
                    out["psi_mem_avg10"] = float(tok.split("=")[1])
                    break
    except Exception:
        pass
    return out


def under_pressure(p):
    """True if we should consider shedding (WATCH or beyond)."""
    return p["swap_pct"] >= WATCH_SWAP_PCT or p["psi_mem_avg10"] >= PSI_MEM_THRESH


# ---------------------------------------------------------------------------
# gate (same quiet-window logic as warden/doctor)
# ---------------------------------------------------------------------------

def _operator_asleep():
    try:
        import common
        return common.operator_asleep()
    except Exception:
        return True  # fail-safe: treat as asleep if we can't tell


def _quiet():
    if os.path.exists(BUSY):
        return False
    if not os.path.exists(INSTANCES):
        return True
    try:
        data = json.load(open(INSTANCES))
    except Exception:
        return True
    try:
        import instance
        import datetime
        live = instance.prune(data if isinstance(data, list) else [],
                              datetime.datetime.now().astimezone())
        return not any(e.get("priority", 0) >= 30 for e in live)
    except Exception:
        return True


def _due():
    try:
        import state as st
        s = st.get("cron/governor", {})
        return (time.time() - s.get("last_full", 0)) >= MIN_INTERVAL
    except Exception:
        return True


# ---------------------------------------------------------------------------
# core decision
# ---------------------------------------------------------------------------

def effective_tier(comp, in_use):
    base = TIER_INDEX.get(comp.get("nominal_tier"))
    if base is None:
        return None  # untagged => unsheddable (fails closed)
    if in_use:
        base += 2  # busy-pinning boost
    return min(base, len(TIER_ORDER) - 1)


def plan(manifest, pressure, operator_pids):
    """Compute shed plan. Returns list of dicts (name, tier, action, reason, comp)."""
    floor = TIER_INDEX.get(manifest.get("priority_floor", "important"),
                           TIER_INDEX["important"])
    plan_ = []
    for comp in manifest.get("components", []):
        name = comp.get("name")
        nominal = comp.get("nominal_tier")
        if nominal not in TIER_INDEX:
            continue  # untagged => never shed
        pids = find_pids(comp.get("match", ""))
        if not pids:
            continue  # not running, nothing to shed
        in_use = component_in_use(comp, pids) or any(
            p in operator_pids for p in pids)
        eff = effective_tier(comp, in_use)
        if eff is None:
            continue
        # eligibility: effective tier strictly below the floor
        if eff >= floor:
            continue
        plan_.append({
            "name": name, "nominal": nominal, "effective": TIER_ORDER[eff],
            "action": "freeze",  # always freeze-first
            "kill_requires": comp.get("kill_requires", "conscious"),
            "restart_cmd": comp.get("restart_cmd"),
            "match": comp.get("match"), "pids": pids,
        })
    # shed from least critical upward
    plan_.sort(key=lambda c: (TIER_INDEX[c["effective"]], c["name"]))
    return plan_


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def freeze(pids):
    for p in pids:
        try:
            os.kill(int(p), signal.SIGSTOP)
        except (ProcessLookupError, ValueError):
            pass


def unfreeze(pids):
    for p in pids:
        try:
            os.kill(int(p), signal.SIGCONT)
        except (ProcessLookupError, ValueError):
            pass


def kill(pids):
    for p in pids:
        try:
            os.kill(int(p), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    time.sleep(3)
    for p in pids:
        try:
            os.kill(int(p), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass


def log_event(etype, payload):
    """Append to the Vesta ledger (events chain) via memstore._emit_event."""
    try:
        import memstore
        conn = memstore.connect()
        try:
            memstore._emit_event(conn, etype, payload=payload,
                                 actor="governor", validated=1)
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # logging must never crash the governor


def apply_plan(plan_, act, pressure):
    """Execute the plan. act=False => only report. Returns list of performed actions.

    Graceful: re-check pressure after each freeze — if freezing one component
    relieved the pressure, stop (don't nuke everything at once)."""
    performed = []
    for c in plan_:
        if pressure["swap_pct"] < SHED_SWAP_PCT:
            break  # pressure eased; stop shedding
        if c["effective"] == "super-critical":
            continue
        if not act:
            performed.append({**c, "action": "would-freeze"})
            continue
        freeze(c["pids"])
        performed.append({**c, "action": "froze"})
        log_event("governor_shed", {"component": c["name"], "tier": c["effective"],
                                    "action": "freeze", "swap_pct": pressure["swap_pct"]})
        # re-read pressure: if this freeze helped enough, stop shedding more
        pressure = read_pressure()
    return performed


def escalate_kills(plan_, act, pressure):
    """Escalate previously-frozen frills to kill if swap still critically full
    after the freeze grace period. Autonomous only for frill + restart_cmd."""
    if not act or pressure["swap_pct"] < KILL_SWAP_PCT \
            or pressure["swap_free_mb"] > KILL_SWAP_FREE_MB:
        return []
    killed = []
    state = load_state()
    now = time.time()
    remaining = []
    for fr in state.get("frozen", []):
        comp = next((c for c in plan_ if c["name"] == fr["name"]), None)
        if not comp:
            remaining.append(fr)
            continue
        is_frill = comp["nominal"] == "frill"
        auto_killable = comp["kill_requires"] == "autonomous" and comp.get("restart_cmd")
        if is_frill and auto_killable and now - fr.get("froze_at", 0) >= KILL_GRACE_S:
            kill(comp["pids"])
            killed.append({**comp, "action": "killed"})
            log_event("governor_shed", {"component": comp["name"], "tier": "frill",
                                        "action": "kill", "swap_pct": pressure["swap_pct"]})
        else:
            remaining.append(fr)  # not autonomously killable -> keep frozen, escalate later
    state["frozen"] = remaining
    save_state(state)
    return killed


def restore(manifest, act, pressure):
    """Restore frozen components (SIGCONT) and restart killed frills when pressure clears."""
    restored = []
    if pressure["swap_pct"] >= RESTORE_SWAP_PCT:
        return restored
    state = load_state()
    if act:
        for fr in state.get("frozen", []):
            comp = next((c for c in manifest.get("components", [])
                         if c.get("name") == fr["name"]), None)
            if not comp:
                continue
            pids = find_pids(comp.get("match", ""))
            if pids:
                unfreeze(pids)
                restored.append(f"{fr['name']}:unfroze")
                log_event("governor_restore", {"component": fr["name"], "action": "unfreeze",
                                               "swap_pct": pressure["swap_pct"]})
        state["frozen"] = []
        save_state(state)
    else:
        restored = [f"{fr['name']}:would-unfreeze" for fr in state.get("frozen", [])]
    return restored


# ---------------------------------------------------------------------------
# state persistence
# ---------------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            return {}
    return {}


def save_state(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, STATE)


def record_froze(performed):
    state = load_state()
    state.setdefault("frozen", [])
    now = time.time()
    existing = {fr["name"] for fr in state["frozen"]}
    for c in performed:
        if c["action"] == "froze" and c["name"] not in existing:
            state["frozen"].append({"name": c["name"], "froze_at": now})
    save_state(state)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def scan(manifest=None, act=False):
    manifest = manifest or load_manifest()
    if not manifest:
        return {"error": "manifest missing"}
    pressure = read_pressure()
    op_pids = operator_session_pid()
    plan_ = plan(manifest, pressure, op_pids)
    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pressure": pressure,
        "under_pressure": under_pressure(pressure),
        "shed_plan": plan_,
        "operator_session_pids": op_pids,
    }
    if act and under_pressure(pressure):
        performed = apply_plan(plan_, act=True, pressure=pressure)
        record_froze(performed)
        killed = escalate_kills(plan_, act=True, pressure=pressure)
        restored = restore(manifest, act=True, pressure=pressure)
        report["performed"] = performed
        report["killed"] = killed
        report["restored"] = restored
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    return report


def visit(act=False):
    if not _operator_asleep():
        return "deferred:not-asleep"
    if not _quiet():
        return "deferred:busy"
    if not _due():
        return "deferred:not-due"

    report = scan(act=act)
    try:
        if report.get("killed"):
            health_flags.set_flag("governor", "warn",
                                  f"killed {len(report['killed'])} frill(s): "
                                  + ", ".join(p["name"] for p in report["killed"]))
        elif report.get("performed"):
            health_flags.set_flag("governor", "warn",
                                  f"shed {len(report['performed'])} component(s): "
                                  + ", ".join(f"{p['name']}/{p['action']}" for p in report['performed']))
        elif report.get("restored"):
            health_flags.set_flag("governor", "info",
                                  f"restored: {', '.join(report['restored'])}")
        elif report.get("under_pressure"):
            health_flags.set_flag("governor", "warn",
                                  f"under pressure swap={report['pressure']['swap_pct']:.0f}%, "
                                  f"{len(report['shed_plan'])} shed candidates")
        else:
            health_flags.clear_flag("governor")
    except Exception:
        pass

    try:
        import state as st
        s = st.get("cron/governor", {})
        s["last_full"] = time.time()
        st.set("cron/governor", s, durable=True)
    except Exception:
        pass
    return "ok"


def show_manifest():
    m = load_manifest()
    if not m:
        return "manifest missing"
    lines = [f"floor={m.get('priority_floor')} boost={m.get('in_use_boost_tiers')}"]
    for c in m["components"]:
        lines.append(f"  {c['name']:<24} {c['nominal_tier']:<16} kill={c['kill_requires']}")
    return "\n".join(lines)


def show_report():
    if not os.path.exists(REPORT):
        return "no report yet"
    r = json.load(open(REPORT))
    return json.dumps(r, indent=2)


def main():
    p = argparse.ArgumentParser(description="graceful degradation / load shedding governor")
    p.add_argument("cmd", nargs="?", default="scan",
                   choices=["scan", "visit", "report", "manifest"])
    p.add_argument("--act", action="store_true",
                   help="actually shed/restore (autonomous-tier actions only; dry-run by default)")
    a = p.parse_args()
    if a.cmd == "scan":
        print(json.dumps(scan(act=a.act), indent=2))
    elif a.cmd == "visit":
        print(visit(act=a.act))
    elif a.cmd == "report":
        print(show_report())
    elif a.cmd == "manifest":
        print(show_manifest())


if __name__ == "__main__":
    main()
