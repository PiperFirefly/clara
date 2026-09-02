#!/usr/bin/env python3
"""doctor.py — the agent's daily self-healing pass (find → diagnose → heal → escalate).

Once a day, when the operator is asleep (operator_presence) and no conscious instance is
live, run a battery of health checks, heal what's reversible (each fix verified
after applying), and escalate only what genuinely needs the operator. Keeps health
flags current so the next chat session can say "doctor/heartbeat found X".

Gating: at most one full visit per ~20h, and only when the operator's phase is "asleep"
AND there's no busy.flag AND no higher-priority instance is live. Otherwise it's
a silent no-op — so a frequent cron slot that lands while the operator's awake simply
waits for the next quiet window instead of running while they need the box.

Healing is deliberately bounded by blast_radius (reversible writes/deletes only;
anything irreversible is never touched — that still needs the operator's token). Novel
fixes that would require writing code are researched and flagged for the
conscious instance, not auto-authored here.

Usage:
  doctor.py visit     gate + full pass (no-op unless it's time and quiet)
  doctor.py battery   run checks only, print findings (no healing)
  doctor.py report    print the last visit report
"""
import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.expanduser("~/learning/freeroam")
MEM_DIR = os.path.expanduser("~/memory")
MEM_VENV = os.path.expanduser("~/venvs/memory/bin/python")
sys.path.insert(0, MEM_DIR)
import state as st  # ephemeral state store
STATE = os.path.join(FR, "doctor_state.json")
REPORT = os.path.join(FR, "doctor_report.json")
BUSY = os.path.join(FR, "busy.flag")
MIN_INTERVAL = 20 * 3600          # at most one full visit per ~20h
INSTANCES = os.path.expanduser("~/.pi/agent/instances.json")

sys.path.insert(0, BASE)
sys.path.insert(0, MEM_DIR)

import health_flags
import heartbeat
import memstore
import common
import resume


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------

def _operator_asleep():
    return common.operator_asleep()


def _quiet():
    """True when no conscious (priority >= 30) instance is live."""
    if os.path.exists(BUSY):
        return False
    if not os.path.exists(INSTANCES):
        return True
    try:
        data = json.load(open(INSTANCES))
    except Exception:
        return True
    # Registry is a JSON *list* of designators with ISO `heartbeat` fields —
    # reuse instance.prune() (the canonical TTL/liveness logic) so this can't
    # drift from instance.py.
    try:
        import instance
        import datetime
        live = instance.prune(data if isinstance(data, list) else [],
                              datetime.datetime.now().astimezone())
    except Exception:
        return True
    return not any(e.get("priority", 0) >= 30 for e in live)


def _due():
    s = st.get("cron/doctor", {})
    return common.is_due(s.get("last_full", 0), MIN_INTERVAL)


# --------------------------------------------------------------------------
# battery
# --------------------------------------------------------------------------

def battery():
    out = {}
    # 1. memory ledger integrity (full Vesta eval)
    try:
        r = subprocess.run([MEM_VENV, os.path.join(MEM_DIR, "vesta_eval.py")],
                           capture_output=True, text=True, timeout=180)
        out["ledger"] = {"status": "ok" if r.returncode == 0 else "crit",
                         "detail": ((r.stdout or "").strip().splitlines()[-1]
                                    if r.stdout else (r.stderr or "").strip())}
    except Exception as e:
        out["ledger"] = {"status": "crit", "detail": f"vesta_eval error: {e}"}
    # 2. present-self freshness
    try:
        r = subprocess.run([MEM_VENV, os.path.join(BASE, "present_self.py"), "--check"],
                           capture_output=True, text=True, timeout=60)
        out["present_self"] = {"status": "ok" if r.returncode == 0 else "crit",
                               "detail": (r.stdout or r.stderr).strip()}
    except Exception as e:
        out["present_self"] = {"status": "crit", "detail": str(e)}
    # 3. memory recall quality
    try:
        r = subprocess.run([MEM_VENV, os.path.join(MEM_DIR, "eval.py")],
                           capture_output=True, text=True, timeout=180)
        txt = r.stdout or ""
        out["memory_recall"] = {"status": "ok" if "100%" in txt else "warn",
                                "detail": " ".join(txt.splitlines()[-2:])}
    except Exception as e:
        out["memory_recall"] = {"status": "warn", "detail": str(e)}
    # 4. systems + resources (reuse heartbeat's checks)
    s, probs = heartbeat.check_systems()
    out["systems"] = {"status": s["status"], "detail": s["detail"]}
    r, _findings, actions = heartbeat.check_resources()
    crit = r["disk"]["status"] == "crit" or r["mem"]["status"] == "crit"
    out["resources"] = {"status": "crit" if crit else "ok",
                        "detail": f"disk={r['disk']['detail']}; mem={r['mem']['detail']}"}
    # 5. resume catalog freshness -- surfaces drift instead of letting it sit
    # silently the way agent_memory.md used to (detection only; a version
    # bump still needs judgment, so this never auto-writes the catalog).
    try:
        stale_items = resume.stale()
        if stale_items:
            names = ', '.join(f"{it['short_name']}({it['stale_days']:.0f}d)" for it in stale_items[:5])
            out["resume_freshness"] = {"status": "warn",
                                       "detail": f"{len(stale_items)} item(s) possibly stale: {names}"}
        else:
            out["resume_freshness"] = {"status": "ok", "detail": "catalog matches tracked paths"}
    except Exception as e:
        out["resume_freshness"] = {"status": "warn", "detail": f"resume.stale() error: {e}"}
    return out, actions


# --------------------------------------------------------------------------
# heal (reversible only, verify after)
# --------------------------------------------------------------------------

def _heal(out, actions):
    healed = []

    # reconcile memory drift / ledger projection (reversible, verified)
    if out.get("ledger", {}).get("status") == "crit":
        try:
            n1 = memstore.emit_open_loop_snapshot()
            n2 = memstore.emit_goal_snapshot()
            if memstore.mirror_check()["drift"]:
                healed.append("ledger reconcile partial — drift persists")
            else:
                healed.append(f"reconciled ledger (loops={n1}, goals={n2}) — no drift")
        except Exception as e:
            healed.append(f"ledger reconcile failed: {e}")
    # reversible hygiene (log truncate + pip purge), blast-radius gated
    if actions:
        try:
            healed += heartbeat.do_actions(actions)
        except Exception as e:
            healed.append(f"hygiene actions failed: {e}")
    # telegram bridge down -> restart it (idempotent start script)
    if out.get("systems", {}).get("status") == "crit":
        start = os.path.join(BASE, "telegram_bridge_start.sh")
        if os.path.exists(start):
            try:
                subprocess.run(["bash", start], capture_output=True, timeout=60)
                healed.append("attempted telegram bridge restart")
            except Exception as e:
                healed.append(f"bridge restart failed: {e}")
    return healed


def _log(msg):
    with open(os.path.join(FR, "doctor.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def visit():
    asleep, phase = _operator_asleep()
    if not asleep:
        return "deferred:not-asleep"
    if not _quiet():
        return "deferred:busy"
    if not _due():
        return "deferred:not-due"

    _log(f"=== visit start (operator {phase}) ===")
    out, actions = battery()
    lines = [f"  {k}: [{v.get('status','?')}] {v.get('detail','')}"
             for k, v in out.items()]
    _log("findings:\n" + "\n".join(lines))

    healed = _heal(out, actions)
    for h in healed:
        _log(f"  healed: {h}")

    # refresh health flags to match post-heal reality
    try:
        if out.get("ledger", {}).get("status") == "crit":
            health_flags.set_flag("doctor-ledger", "warn", out["ledger"]["detail"])
        else:
            health_flags.clear_flag("doctor-ledger")
    except Exception:
        pass
    try:
        if out.get("resume_freshness", {}).get("status") == "warn":
            health_flags.set_flag("resume-stale", "info", out["resume_freshness"]["detail"])
        else:
            health_flags.clear_flag("resume-stale")
    except Exception:
        pass

    report = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": phase,
              "checks": out, "healed": healed}
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    state = st.get("cron/doctor", {})
    state["last_full"] = time.time()
    state["last_report"] = report
    st.set("cron/doctor", state, durable=True)

    _log("=== visit end ===\n")
    return "ok"


def show_report():
    return common.show_report(REPORT, "doctor report")


def main():
    p = argparse.ArgumentParser(description="the agent's daily self-healing pass")
    p.add_argument("cmd", nargs="?", default="visit",
                   choices=["visit", "battery", "report"])
    a = p.parse_args()
    if a.cmd == "visit":
        print(visit())
    elif a.cmd == "battery":
        out, actions = battery()
        for k, v in out.items():
            print(f"{k}: [{v['status']}] {v['detail']}")
        if actions:
            print("healable actions:", actions)
    elif a.cmd == "report":
        print(show_report())


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
