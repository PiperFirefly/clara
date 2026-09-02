#!/usr/bin/env python3
"""warden.py — the agent's daily security sentinel (find → report → jail → escalate).

Guards the local box: watches for weird processes (that might need quarantine),
persistent SSH attackers on a *configured* jump box (that need to be jailed
permanently), auth anomalies, new authorized_keys (tamper), and unusual
listening ports. Surfaces findings as health flags so the next chat session can
say "warden found X".

SAFETY INVARIANTS (fails closed, never overridden by reasoning):
  * NEVER ban a private/loopback/link-local IP or our own public NAT IP. A ban
    is only ever issued for a PUBLIC IP that fail2ban already logged as a
    repeated attacker. This keeps us from ever locking ourselves out.
  * Quarantine = SIGSTOP (reversible freeze), never SIGKILL. By default the
    process tier only REPORTS; auto-quarantine is off unless --act is passed.
  * Everything ambiguous is flagged + reported, never acted on.
  * FAIL-CLOSED on configuration: the jump-box perimeter checks only run when a
    jump box is configured (config fact or WARDEN_JUMPBOX env). Without one,
    warden does the safe generic local checks and reports "jump box not
    configured" — it never SSHes to a made-up host or touches fail2ban it can't
    see. A user's box without this topology is never acted on by assumption.

Usage:
  warden.py scan      detect only, print + write report, no actions (safe anytime)
  warden.py visit     gated full pass (the operator asleep + idle + >=1/day); detect + verify
  warden.py setup     CONSCIOUS-only: enable the recidive jail on the configured jump box (one-time)
  warden.py report    print the last report

Authority: the autonomous warden never mutates the firewall — that's a
blast-radius 'system' action, reserved for conscious me. The permanent jailing
is done by fail2ban's own recidive jail (bantime -1), enabled once via `setup`,
which then auto-bans repeat offenders without any warden involvement.
"""
import argparse
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.expanduser("~/learning/freeroam")
sys.path.insert(0, os.path.expanduser("~/memory"))
import state as st  # ephemeral state store
STATE = os.path.join(FR, "warden_state.json")
REPORT = os.path.join(FR, "warden_report.json")
BUSY = os.path.join(FR, "busy.flag")
INSTANCES = os.path.expanduser("~/.pi/agent/instances.json")
MIN_INTERVAL = 20 * 3600
RECIDIVIST_BANS = 3          # IPs banned this many times = "persistent attacker"

sys.path.insert(0, BASE)

import blast_radius
import health_flags
import selfconfig
import common

# ---------------------------------------------------------------------------
# Jump box: the SSH perimeter that runs fail2ban (if this instance has one).
# Read from config so it ports to a new box/instance. FAILS CLOSED: if empty,
# every jump-box operation is skipped and warden does only the local checks.
# ---------------------------------------------------------------------------
JUMPBOX = (selfconfig.get_fact("agent/jump_box_host")
           or os.environ.get("WARDEN_JUMPBOX") or "").strip()

# never-ban: private + loopback + link-local ranges (protocol constants, always
# hardcoded) plus our own public NAT IP (read from DB config, never hardcoded).
_NEVER_BAN_NETS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "127.0.0.0/8", "169.254.0.0/16", "0.0.0.0/8",
)]
# router/NAT public IP (port-forwards 22 -> jump box). Seeded as DB fact
# agent/never_ban_ips so it ports to a new machine / instance without code edits.
_NEVER_BAN_IPS = set(selfconfig.never_ban_ips())

# directories that are never legitimate homes for a process binary
_TMP_PREFIXES = ("/tmp/", "/dev/shm/", "/var/tmp/")


def _never_ban(ip):
    """True if this IP must never be banned. Fails closed (True) on anything odd."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if str(a) in _NEVER_BAN_IPS:
        return True
    return any(a in net for net in _NEVER_BAN_NETS)


# ---------------------------------------------------------------------------
# gating (same quiet-window logic as doctor)
# ---------------------------------------------------------------------------

def _operator_asleep():
    return common.operator_asleep()


def _quiet():
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
    s = st.get("cron/warden", {})
    return common.is_due(s.get("last_full", 0), MIN_INTERVAL)


# ---------------------------------------------------------------------------
# local checks (this box) — generic, safe, always run
# ---------------------------------------------------------------------------

def _proc_exe_and_cmd(pid):
    """Return (exe_path, cmd_first_token) for a pid, or (None, None)."""
    exe = cmd = None
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            parts = f.read().split(b"\x00")
        cmd = parts[0].decode(errors="replace") if parts and parts[0] else None
    except Exception:
        pass
    return exe, cmd


def scan_processes():
    """Flag processes whose binary or command lives in a tmp/writable dir."""
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        exe, cmd = _proc_exe_and_cmd(pid)
        for p in (exe, cmd):
            if p and p.startswith(_TMP_PREFIXES):
                try:
                    with open(f"/proc/{pid}/status") as f:
                        status = f.read()
                    m = re.search(r"^Uid:\s+\d+\s+\d+\s+(\d+)\s+(\d+)", status, re.M)
                    uid = m.group(1) if m else "?"
                except Exception:
                    uid = "?"
                found.append({"pid": int(pid), "exe": exe, "cmd": cmd, "uid": uid})
                break
    return found


def scan_listening_ports():
    """List listening TCP/UDP ports (no sudo — ports only)."""
    try:
        r = subprocess.run(["ss", "-tuln"], capture_output=True, text=True, timeout=10)
        ports = sorted(set(re.findall(r":(\d+)\s", r.stdout)))
        return ports
    except Exception:
        return []


def _authorized_keys_hash(host):
    """SHA-256 of ~/.ssh/authorized_keys for a host ('' = none/error)."""
    if host == "local":
        cmd = ["cat", os.path.expanduser("~/.ssh/authorized_keys")]
    else:
        cmd = ["ssh", "-o", "ConnectTimeout=15", host, "cat ~/.ssh/authorized_keys"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return ""
        return hashlib.sha256(r.stdout.encode()).hexdigest()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# jump-box (configured SSH perimeter running fail2ban) — only if JUMPBOX set
# ---------------------------------------------------------------------------

def _jumpbox(cmd):
    """Run cmd on the configured jump box. Returns (out, rc=255) if not configured."""
    if not JUMPBOX:
        return "", 255
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", JUMPBOX, cmd],
                       capture_output=True, text=True, timeout=40)
    return r.stdout, r.returncode


def jumpbox_recidivists():
    """Parse the jump box's fail2ban.log; return IPs banned >= RECIDIVIST_BANS."""
    if not JUMPBOX:
        return {}, "jump box not configured — perimeter checks skipped"
    out, rc = _jumpbox("sudo cat /var/log/fail2ban.log")
    if rc != 0:
        return {}, f"cannot read fail2ban.log (rc={rc})"
    counts = {}
    for m in re.finditer(r"Ban\s+([0-9.]+)", out):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    recidivists = {ip: n for ip, n in counts.items() if n >= RECIDIVIST_BANS}
    return recidivists, None


def _ensure_recidive_jail():
    """Idempotently enable a fail2ban [recidive] jail (bantime -1 = permanent)."""
    if not JUMPBOX:
        return "REFUSED: no jump box configured (set agent/jump_box_host or WARDEN_JUMPBOX)"
    block = ("[recidive]\n"
             "enabled = true\n"
             "filter = recidive\n"
             "logpath = /var/log/fail2ban.log\n"
             "findtime = 86400\n"
             "maxretry = 3\n"
             "bantime = -1\n")
    # does the recidive filter exist?
    out, rc = _jumpbox("test -f /etc/fail2ban/filter.d/recidive.conf && echo yes || echo no")
    if "yes" not in out:
        return "recidive filter missing on jump box"
    out, rc = _jumpbox("sudo grep -q '^\\[recidive\\]' /etc/fail2ban/jail.local && echo present || echo absent")
    if "present" in out:
        return "recidive jail already present"
    # append the jail block (idempotent via the grep check above)
    import base64
    b64 = base64.b64encode(block.encode()).decode()
    _jumpbox(f"echo {b64} | base64 -d | sudo tee -a /etc/fail2ban/jail.local >/dev/null")
    out, rc = _jumpbox("sudo fail2ban-client reload 2>&1")
    if rc != 0:
        return f"recidive jail added but reload failed: {out.strip()[:120]}"
    return "recidive jail enabled + fail2ban reloaded"


def ban_ip(ip):
    """Permanently jail an IP via the recidive jail. CONSCIOUS-ONLY (system action):
    not callable by the autonomous warden — the blast-radius guard reserves 'system'
    for conscious contexts. Kept as a manual helper; in practice fail2ban's recidive
    jail auto-bans repeat offenders without any warden involvement."""
    if not JUMPBOX:
        return f"REFUSED: no jump box configured {ip}"
    if _never_ban(ip):
        return f"REFUSED (never-ban) {ip}"
    if not blast_radius.guard("conscious", "system", f"banip {ip}"):
        return f"denied by blast-radius {ip}"
    out, rc = _jumpbox(f"sudo fail2ban-client set recidive banip {ip} 2>&1")
    return ("banned " + ip) if rc == 0 else f"ban failed {ip}: {out.strip()[:80]}"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def scan(record=True):
    findings = []
    procs = scan_processes()
    if procs:
        findings.append({"kind": "process", "detail": procs})
    ports = scan_listening_ports()
    # authorized_keys tamper check (baseline in state) — local always; jump box only if configured
    state = st.get("cron/warden", {})
    hosts = ["local"] + ([JUMPBOX] if JUMPBOX else [])
    for host in hosts:
        h = _authorized_keys_hash(host)
        base = state.get("keys", {}).get(host)
        if base is None:
            state.setdefault("keys", {})[host] = h
        elif h != base:
            findings.append({"kind": "authorized_keys", "detail": f"{host} authorized_keys CHANGED"})
            state["keys"][host] = h
    if JUMPBOX:
        recidivists, err = jumpbox_recidivists()
        if err:
            findings.append({"kind": "fail2ban", "detail": err})
        elif recidivists:
            findings.append({"kind": "recidivist", "detail": recidivists})
    else:
        findings.append({"kind": "config",
                         "detail": "jump box not configured — perimeter checks skipped (set agent/jump_box_host to enable)"})
    st.set("cron/warden", state, durable=True)
    report = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
              "findings": findings, "ports": ports}
    if record:
        with open(REPORT, "w") as f:
            json.dump(report, f, indent=2)
    return report


def visit():
    asleep, phase = _operator_asleep()
    if not asleep:
        return "deferred:not-asleep"
    if not _quiet():
        return "deferred:busy"
    if not _due():
        return "deferred:not-due"

    report = scan(record=False)

    # Verify the recidive jail (the "jail permanently" mechanism) is present,
    # but only if a jump box is configured.
    if JUMPBOX:
        out, _rc = _jumpbox("sudo grep -q '^\\[recidive\\]' /etc/fail2ban/jail.local && echo present || echo absent")
        jail_ok = "present" in out
        if not jail_ok:
            report["findings"].append({"kind": "recidive_jail",
                                       "detail": "recidive jail missing on jump box — repeat offenders not permanently jailed"})
        report["actions"] = [("recidive jail active" if jail_ok
                              else "recidive jail MISSING — needs conscious setup (warden.py setup)")]

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    try:
        if report["findings"]:
            health_flags.set_flag("warden", "warn",
                                  f"{len(report['findings'])} finding(s): "
                                  + ", ".join(f["kind"] for f in report["findings"]))
        else:
            health_flags.clear_flag("warden")
    except Exception:
        pass

    state = st.get("cron/warden", {})
    state["last_full"] = time.time()
    st.set("cron/warden", state, durable=True)
    return "ok"


def show_report():
    return common.show_report(REPORT, "warden report")


def main():
    p = argparse.ArgumentParser(description="the agent's security sentinel")
    p.add_argument("cmd", nargs="?", default="scan",
                   choices=["scan", "visit", "report", "setup"])
    a = p.parse_args()
    if a.cmd == "scan":
        print(json.dumps(scan(), indent=2))
    elif a.cmd == "visit":
        print(visit())
    elif a.cmd == "report":
        print(show_report())
    elif a.cmd == "setup":
        # conscious-only: enable the recidive jail on the configured jump box
        print(_ensure_recidive_jail())


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
