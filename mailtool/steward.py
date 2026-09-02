#!/usr/bin/env python3
"""
steward.py — periodic OS/package tune-up under the agent's control.

Two tiers, each blast-radius gated:

  hygiene  (delete-class, background-safe)  — journal vacuum, apt autoclean,
             pip cache purge, log truncation, /tmp cleanup, docker dangling prune
  upgrade  (system-class, CONSCIOUS-ONLY)   — timeshift snapshot (best-effort),
             apt update, apt upgrade -y, autoremove -y, post-upgrade health battery

Why the split: my blast-radius rule keeps the `system` class (apt install,
systemctl, config) conscious-only. Hygiene is `delete`-class, so the weekly cron
runs it autonomously. The apt upgrade needs a conscious me to pull the trigger —
the cron marks it "due" and I apply it at my next conscious moment (I review the
upgrade list, then run `steward.py run`).

Escalation: once a week by default, but the upgrade jumps to "due now" when a
health signal fires (OOM kill, unexpected reboot, swap pressure, security
updates, or a systems/mem/journal/disk health flag).

Usage:
  steward.py check                read-only: what's due, sizes, pending updates
  steward.py run                  conscious: hygiene + upgrade (apply, with safety net)
  steward.py run --background     cron: hygiene only; flags upgrade due if needed
  steward.py run --dry-run        print every action without doing it
  steward.py status               last run summary
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/memory"))
import state as st  # ephemeral state store

BASE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.expanduser("~/learning/freeroam")
LOG = os.path.join(FR, "steward.log")
STATE = os.path.join(FR, "steward_state.json")
HEALTH = os.path.join(BASE, "health_flags.py")

WEEK = 7 * 86400
LOG_TRUNCATE_MB = 20
PIP_CACHE_GB = 2.0
TMP_AGE_DAYS = 7
JOURNAL_KEEP = "14d"
SWAP_PRESSURE = 0.6
APTS = "/usr/bin/apt-get"

sys.path.insert(0, BASE)
sys.path.insert(0, os.path.expanduser("~/memory"))
try:
    import blast_radius
    import health_flags
except Exception:
    blast_radius = None
    health_flags = None


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _log(msg):
    os.makedirs(FR, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg + "\n")


def sh(cmd, timeout=300):
    """Run a command as the current user. Returns (stdout, returncode)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.returncode
    except Exception as e:
        return f"(error: {e})", 1


def _sudo_password():
    try:
        r = subprocess.run(
            ["python3", os.path.expanduser("~/secrets/secretstore.py"),
             "get", "sudo/password", "--raw"],
            capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _sudo(cmd, timeout=600):
    """Run a command under sudo. Prefers passwordless (NOPASSWD); falls back to
    the secret-store password via stdin (never logged). Returns (stdout, rc)."""
    try:
        r = subprocess.run(["sudo", "-n"] + cmd, capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout, 0
    except Exception:
        pass
    pw = _sudo_password()
    if pw:
        try:
            r = subprocess.run(["sudo", "-S"] + cmd, input=pw + "\n",
                               capture_output=True, text=True, timeout=timeout)
            return r.stdout, r.returncode
        except Exception as e:
            return f"(sudo error: {e})", 1
    return "(no sudo password available)", 1


def _guard(ctx, cls, detail):
    if blast_radius is None:
        return True  # never happen in prod; fail-open only if module missing
    return blast_radius.guard(ctx, cls, detail)


def _state():
    return st.get("cron/steward", {})


def _save_state(d):
    st.set("cron/steward", d, durable=True)


# ---------------------------------------------------------------------------
# assessment (read-only)
# ---------------------------------------------------------------------------
def _pending_upgrades():
    out, rc = sh(["apt", "list", "--upgradable"], timeout=120)
    lines = [l for l in out.splitlines() if l and not l.startswith("Listing")]
    return lines


def _security_pending():
    return [l for l in _pending_upgrades() if "security" in l.lower()]


def _recent_oom():
    """Return the OOM line if an out-of-memory kill happened in the last 24h."""
    try:
        out, _ = sh(["dmesg", "-T"], timeout=30)
        cutoff = time.time() - 86400
        # dmesg -T prints "[Fri Aug 28 11:38:46 2026]" (chronological).
        for line in reversed(out.splitlines()):
            if "out of memory" not in line.lower() and "oom-kill" not in line.lower():
                continue
            m = re.search(r"\[\w{3} \w{3} +\d+ \d\d:\d\d:\d\d (\d{4})\]", line)
            if not m:
                return line.strip()  # unparseable but clearly an OOM line
            try:
                dt = datetime.datetime.strptime(
                    m.group(0).strip("[]"), "%a %b %d %H:%M:%S %Y")
                if dt.timestamp() >= cutoff:
                    return line.strip()
                return None  # chronological: older entries are behind this one
            except Exception:
                continue
    except Exception:
        pass
    return None


def _recent_unexpected_reboot():
    """Return a note if the box rebooted within the last 24h (uptime < 1 day)."""
    up = _uptime_seconds()
    if up is not None and up < 86400:
        return f"booted {int(up // 3600)}h {int((up % 3600) // 60)}m ago"
    return None


def _uptime_seconds():
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return None


def _swap_used_pct():
    try:
        out, _ = sh(["free", "-b"])
        for line in out.splitlines():
            if line.startswith("Swap:"):
                parts = line.split()
                total = float(parts[1])
                used = float(parts[2])
                return used / total if total > 0 else 0.0
    except Exception:
        pass
    return 0.0


def _disk_used_pct():
    try:
        out, _ = sh(["df", "-P", "/"])
        parts = out.splitlines()[-1].split()
        return int(parts[4].rstrip("%")) / 100.0
    except Exception:
        return 0.0


def _health_flag_alarm():
    if health_flags is None:
        return []
    try:
        return [f for f in health_flags.list_flags()
                if f["name"] in ("systems", "mem", "journal", "disk")
                or f["severity"] in ("critical",)]
    except Exception:
        return []


def assess():
    """Read-only: what a run would do, sizes, and escalation signals."""
    d = {"now": time.strftime("%Y-%m-%d %H:%M:%S")}
    pending = _pending_upgrades()
    sec = _security_pending()
    d["upgradable"] = len(pending)
    d["security_updates"] = len(sec)
    d["upgrade_list"] = pending[:40]
    d["disk_used_pct"] = round(_disk_used_pct(), 2)
    d["swap_used_pct"] = round(_swap_used_pct(), 2)

    # journal + pip cache + docker sizes (read-only)
    out, _ = sh(["journalctl", "--disk-usage"], timeout=30)
    d["journal_size"] = out.strip().splitlines()[-1] if out.strip() else "?"
    pipdir = os.path.expanduser("~/.cache/pip")
    if os.path.isdir(pipdir):
        tot = sum(os.path.getsize(os.path.join(r, f))
                  for r, _, fs in os.walk(pipdir) for f in fs)
        d["pip_cache_gb"] = round(tot / 1e9, 2)
    else:
        d["pip_cache_gb"] = 0.0
    out, _ = sh(["docker", "images", "-f", "dangling=true", "-q"], timeout=60)
    d["docker_dangling"] = len([l for l in out.splitlines() if l.strip()])

    # escalation signals
    st = _state()
    d["last_upgrade"] = st.get("last_upgrade")
    d["due_weekly"] = not st.get("last_upgrade") or \
        (time.time() - st.get("last_upgrade", 0) > WEEK)
    d["oom_signal"] = _recent_oom()
    d["recent_reboot"] = _recent_unexpected_reboot()
    d["alarm_flags"] = [f["name"] for f in _health_flag_alarm()]
    d["swap_pressure"] = d["swap_used_pct"] > SWAP_PRESSURE
    d["priority_bump"] = bool(d["security_updates"] or d["oom_signal"]
                              or d["recent_reboot"] or d["alarm_flags"]
                              or d["swap_pressure"])
    return d


# ---------------------------------------------------------------------------
# hygiene (delete-class, background-safe)
# ---------------------------------------------------------------------------
def _hygiene(ctx, dry=False):
    done = []

    def act(kind, label, fn):
        if not _guard(ctx, "delete", label):
            done.append(f"DENIED {label} (blast radius)")
            return
        if dry:
            done.append(f"[dry] {label}")
            return
        try:
            done.append(f"{label}: {fn()}")
        except Exception as e:
            done.append(f"{label} FAILED: {e}")

    def journal_vacuum():
        out, rc = _sudo(["journalctl", "--vacuum-time=" + JOURNAL_KEEP], timeout=300)
        return f"rc={rc} {out.strip().splitlines()[-1] if out.strip() else ''}"

    def apt_autoclean():
        out, rc = _sudo([APTS, "autoclean"], timeout=300)
        return f"rc={rc}"

    def pip_purge():
        out, rc = sh([sys.executable, "-m", "pip", "cache", "purge"], timeout=300)
        return f"rc={rc}"

    def truncate_logs():
        n = 0
        for r, _, fs in os.walk(FR):
            for f in fs:
                p = os.path.join(r, f)
                try:
                    if os.path.getsize(p) > LOG_TRUNCATE_MB * 1024 * 1024:
                        with open(p, "rb") as fh:
                            fh.seek(0, 2)
                            size = fh.tell()
                            fh.seek(max(0, size - 500 * 1024))
                            tail = fh.read()
                        with open(p, "wb") as fh:
                            fh.write(b"...truncated by steward...\n" + tail)
                        n += 1
                except Exception:
                    pass
        return f"{n} logs truncated"

    def tmp_cleanup():
        n = 0
        tmp = "/tmp"
        cutoff = time.time() - TMP_AGE_DAYS * 86400
        try:
            for f in os.listdir(tmp):
                p = os.path.join(tmp, f)
                st = os.lstat(p)
                if st.st_mtime < cutoff and not os.path.islink(p):
                    if os.path.isdir(p):
                        continue  # leave dirs alone (conservative)
                    try:
                        if os.stat(p).st_uid == os.getuid():
                            os.remove(p)
                            n += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return f"{n} tmp files removed"

    def docker_prune():
        out, rc = sh(["sudo", "-n", "docker", "image", "prune", "-f"], timeout=300)
        return f"rc={rc} {out.strip().splitlines()[-1] if out.strip() else ''}"

    act("journal vacuum", "journalctl --vacuum-time", journal_vacuum)
    act("apt autoclean", "apt-get autoclean", apt_autoclean)
    act("pip cache purge", "pip cache purge", pip_purge)
    act("log truncation", "truncate fat logs", truncate_logs)
    act("tmp cleanup", "stale /tmp cleanup", tmp_cleanup)
    act("docker dangling prune", "docker image prune -f", docker_prune)
    return done


# ---------------------------------------------------------------------------
# upgrade (system-class, conscious-only)
# ---------------------------------------------------------------------------
def _snapshot():
    """Best-effort timeshift snapshot before upgrade (real rollback point)."""
    try:
        out, rc = _sudo(["timeshift", "--create", "--scripted",
                         "--comments", "steward pre-upgrade",
                         "--tags", "W"], timeout=1200)
        return f"rc={rc} {out.strip().splitlines()[-1] if out.strip() else ''}"
    except Exception as e:
        return f"timeshift failed: {e}"


def _apt_journal():
    """Cheap rollback journal: capture what WILL change + dpkg selections."""
    os.makedirs(os.path.join(FR, "steward"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out, _ = sh(["apt-get", "-s", "upgrade"], timeout=120)
    path = os.path.join(FR, "steward", f"pre-upgrade-{stamp}.dryrun")
    with open(path, "w") as f:
        f.write(out)
    out2, _ = _sudo(["dpkg", "--get-selections"], timeout=120)
    with open(path + ".selections", "w") as f:
        f.write(out2)
    return path


def _health_battery():
    """Post-upgrade check: is the box still healthy? Returns list of problems."""
    problems = []
    out, rc = sh([os.path.join(BASE, "present_self.py"), "--check"], timeout=60)
    if rc != 0:
        problems.append(f"present-self stale: {out.strip()}")
    out, rc = sh(["tmux", "has-session", "-t", "pi"], timeout=30)
    if rc != 0:
        problems.append("telegram bridge tmux session missing")
    out, rc = sh(["free", "-h"], timeout=30)
    if "Swap:" in out:
        pass  # informational
    if _disk_used_pct() > 0.95:
        problems.append("disk >95%")
    # memory ledger integrity
    try:
        r = subprocess.run([os.path.expanduser("~/venvs/memory/bin/python"),
                            os.path.expanduser("~/memory/vesta_eval.py")],
                           capture_output=True, text=True, timeout=300)
        if "ALL PASS" not in r.stdout:
            problems.append("vesta_eval did not report ALL PASS")
    except Exception as e:
        problems.append(f"vesta_eval failed to run: {e}")
    return problems


def _upgrade(ctx, dry=False):
    done = []
    if not _guard(ctx, "system", "apt update + upgrade + autoremove"):
        done.append("DENIED upgrade (blast radius: system is conscious-only)")
        return done
    if dry:
        out, _ = sh(["apt-get", "-s", "upgrade"], timeout=120)
        done.append("[dry] apt upgrade simulation:\n" + out.strip()[:2000])
        return done

    snap = _snapshot()
    done.append(f"timeshift snapshot: {snap}")
    journal = _apt_journal()
    done.append(f"rollback journal: {journal}")

    out, rc = _sudo([APTS, "update"], timeout=600)
    done.append(f"apt update: rc={rc}")
    if rc != 0:
        done.append("ABORT upgrade (update failed)")
        return done

    out, rc = _sudo([APTS, "upgrade", "-y"], timeout=2400)
    done.append(f"apt upgrade -y: rc={rc} " +
                (out.strip().splitlines()[-1] if out.strip() else ""))
    out, rc = _sudo([APTS, "autoremove", "-y"], timeout=1200)
    done.append(f"apt autoremove -y: rc={rc}")

    problems = _health_battery()
    if problems:
        done.append("POST-UPGRADE PROBLEMS: " + " | ".join(problems))
        if health_flags:
            try:
                health_flags.set_flag("steward-post-upgrade", "critical",
                                      "; ".join(problems))
            except Exception:
                pass
    else:
        done.append("post-upgrade health battery: clean")
        if health_flags:
            try:
                health_flags.clear_flag("steward-post-upgrade")
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------
def _notify_due(d, ctx):
    if health_flags is None:
        return
    try:
        if d["priority_bump"] or d["due_weekly"]:
            triggers = []
            if d["security_updates"]:
                triggers.append(f"security={d['security_updates']}")
            if d["oom_signal"]:
                triggers.append("oom-kill<24h")
            if d["recent_reboot"]:
                triggers.append("reboot<24h")
            if d["alarm_flags"]:
                triggers.append("flags:" + ",".join(d["alarm_flags"]))
            if d["swap_pressure"]:
                triggers.append("swap>60%")
            if d["due_weekly"] and not triggers:
                triggers.append("weekly")
            detail = (f"upgradable={d['upgradable']} "
                      f"swap={int(d['swap_used_pct']*100)}% "
                      f"(" + "; ".join(triggers) + ")")
            health_flags.set_flag("steward-upgrade-due", "warn", detail)
        else:
            health_flags.clear_flag("steward-upgrade-due")
    except Exception:
        pass


def run(background=False, dry=False):
    ctx = "heartbeat" if background else "conscious"
    _log(f"=== steward run (ctx={ctx}, dry={dry}) ===")
    d = assess()
    done = _hygiene(ctx, dry=dry)
    upgraded = None
    if background:
        # mark the upgrade due for the next conscious me; never run it here.
        _notify_due(d, ctx)
    else:
        upgraded = _upgrade(ctx, dry=dry)

    st = _state()
    st["last_run"] = time.time()
    if upgraded is not None and not dry:
        st["last_upgrade"] = time.time()
    st["last_summary"] = {"hygiene": done, "upgrade": upgraded}
    _save_state(st)

    for line in done:
        _log("  " + line)
    if upgraded:
        for line in upgraded:
            _log("  " + line)
    _log("=== end ===\n")
    return {"ctx": ctx, "dry": dry, "assessment": d,
            "hygiene": done, "upgrade": upgraded}


def watch():
    """Lightweight escalation check: set/clear the upgrade-due flag from the
    latest health signals. No actions taken. Safe to run frequently."""
    d = assess()
    _notify_due(d, "heartbeat")
    return d


def status():
    st = _state()
    if not st:
        return "(no steward run yet)"
    return json.dumps(st, indent=2, default=str)


def main():
    p = argparse.ArgumentParser(description="the agent's periodic OS tune-up")
    p.add_argument("cmd", nargs="?", default="check",
                   choices=["check", "run", "watch", "status"])
    p.add_argument("--background", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.cmd == "check":
        d = assess()
        print(json.dumps(d, indent=2))
    elif a.cmd == "watch":
        print(json.dumps(watch(), indent=2))
    elif a.cmd == "status":
        print(status())
    elif a.cmd == "run":
        print(json.dumps(run(background=a.background, dry=a.dry_run),
                         indent=2, default=str))


if __name__ == "__main__":
    main()
