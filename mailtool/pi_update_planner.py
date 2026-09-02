#!/usr/bin/env python3
"""Pi core-update planner + rollback watchdog.

Checks https://pi.dev/news and the npm registry for new pi releases, applies the
72-hour supply-chain age gate, evaluates release notes (optionally via DeepSeek),
and writes a how/when update plan that includes a rollback path. It also implements
a dead-man's-switch: a core update that is applied but not *confirmed* within a
deadline is rolled back automatically by a cron watchdog.

Subcommands:
  check [--notify] [--telegram]   Detect new releases, age-gate, notify once per release.
  evaluate                        LLM-evaluate the latest release notes (DeepSeek).
  plan                            Write a full how/when plan to ~/tools/communications/email/pi_update_plan.md.
  snapshot                        Snapshot current pi + config into a rollback point.
  apply [--version V] [--deadline-min N]
                                  Snapshot, update core, health-check, arm the watchdog.
  confirm                         Confirm a completed update is good (disarms the watchdog).
  watchdog                        Cron mode: auto-rollback if a deadline passes unconfirmed.
  rollback [--snapshot DIR]       Restore the most recent snapshot.
  status                          Show version/age/watchdog state.

The cron never *applies* anything; it only detects and plans. `apply` is manual
and always refuses a target that is younger than MIN_AGE_HOURS.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import common

HOME = os.path.expanduser("~")
NODE_ROOT = os.path.join(HOME, ".local/share/pi-node/node-v22.23.2-linux-x64")
PI = os.path.join(NODE_ROOT, "bin", "pi")
PI_PKG = "@earendil-works/pi-coding-agent"
PI_PKG_DIR = os.path.join(NODE_ROOT, "lib", "node_modules", *PI_PKG.split("/"))
SETTINGS = os.path.join(HOME, ".pi/agent/settings.json")
AUTH = os.path.join(HOME, ".pi/agent/auth.json")
STATE = os.path.join(HOME, ".pi/agent/pi_update_state.json")
WATCH = os.path.join(HOME, ".pi/agent/update_watch.json")
SNAPSHOT_ROOT = os.path.join(HOME, ".pi/agent/update_snapshots")
NEWS_URL = "https://pi.dev/news"
NPM_REG = "https://registry.npmjs.org"
GH_API = "https://api.github.com/repos/earendil-works/pi"
DS_API = "https://api.deepseek.com/chat/completions"
MIN_AGE_HOURS = 72
DEFAULT_DEADLINE_MIN = 30
MAILTOOL = os.path.join(HOME, "mailtool")
PLAN_PATH = os.path.join(HOME, "tools", "communications", "email", "pi_update_plan.md")
SANDBOX_DIR = os.path.join(HOME, "docker-pi-sandbox")
SANDBOX_CONFIG = os.path.join(SANDBOX_DIR, "config")


def log(msg):
    common.log_print_only(msg)


# ---------------------------------------------------------------- data sources

def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "cadence-update-planner"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _json(url, timeout=25):
    return json.loads(_get(url, timeout))


def pi_version():
    try:
        return subprocess.run([PI, "--version"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def npm_pkg_meta(name=PI_PKG):
    return _json(f"{NPM_REG}/{urllib.parse.quote(name, safe='')}")


def npm_latest(name=PI_PKG):
    try:
        return npm_pkg_meta(name).get("dist-tags", {}).get("latest")
    except Exception:
        return None


def npm_publish_time(name, version):
    try:
        iso = npm_pkg_meta(name).get("time", {}).get(version)
        return iso
    except Exception:
        return None


def age_hours(iso):
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return None


def github_releases(limit=4):
    try:
        return _json(f"{GH_API}/releases?per_page={limit}")
    except Exception as e:
        log(f"  github releases failed: {e}")
        return []


def fetch_news_text(max_chars=4000):
    """Fetch pi.dev/news and return a stripped, readable slice (announcements)."""
    try:
        html = _get(NEWS_URL)
    except Exception as e:
        log(f"  news fetch failed: {e}")
        return ""
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text).replace("&gt;", ">")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()[:max_chars]


# ---------------------------------------------------------------- evaluation

def latest_release():
    """Return dict(version, published_at, age_hours, body) for the newest release.

    Version + age come from the npm registry (authoritative for the artifact);
    the release notes come from GitHub releases.
    """
    ver = npm_latest()
    if not ver:
        return None
    published = npm_publish_time(PI_PKG, ver)
    age = age_hours(published)
    body = ""
    for rel in github_releases():
        tag = rel.get("tag_name", "")
        if tag.lstrip("v") == ver or tag == ver:
            body = rel.get("body") or ""
            published = published or rel.get("published_at")
            age = age if age is not None else age_hours(rel.get("published_at"))
            break
    return {"version": ver, "published_at": published, "age_hours": age, "body": body}


def _deepseek(prompt, max_tokens=700):
    with open(AUTH, "r", encoding="utf-8") as f:
        key = json.load(f)["deepseek"]["key"]
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        DS_API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def my_surface():
    """Describe my installed customizations so an evaluator can reason about impact."""
    try:
        settings = json.load(open(SETTINGS, "r", encoding="utf-8"))
    except Exception:
        settings = {}
    return json.dumps({
        "packages": settings.get("packages", []),
        "extensions": settings.get("extensions", []),
        "defaultProvider": settings.get("defaultProvider"),
        "defaultModel": settings.get("defaultModel"),
    }, indent=2)


def evaluate(rel, verbose=True):
    """LLM evaluation of the release notes -> plain-language summary + risk + action."""
    if not rel or not rel.get("body"):
        log("no release notes to evaluate")
        return None
    notes = rel["body"][:6000]
    prompt = (
        "You are helping me (the agent, an autonomous coding agent) decide whether and when "
        "to update the 'pi' coding agent I run on. Be plain, concise, and opinionated.\n\n"
        f"Release version: {rel['version']}\n"
        f"Published: {rel['published_at']} ({rel['age_hours']:.0f}h ago if known)\n\n"
        "Release notes:\n" + notes + "\n\n"
        "My current surface (installed packages/extensions):\n" + my_surface() + "\n\n"
        "Answer in four short sections:\n"
        "1. NEW: the 3-6 most useful new features, one line each, plain language.\n"
        "2. BREAKING: any breaking changes that could affect my extensions or packages "
        "(list names like @llblab/pi-telegram, memory-tools.ts, secrets-tools.ts, "
        "pi-web-access), and how serious.\n"
        "3. RISK: low/medium/high, one line why.\n"
        "4. RECOMMEND: hold / test-then-apply / apply, and a one-line reason. "
        "Assume I still enforce a 72h age gate regardless of your recommendation."
    )
    try:
        out = _deepseek(prompt)
    except Exception as e:
        log(f"evaluation failed: {e}")
        return None
    if verbose:
        print(out)
    return out


def build_plan(rel, eval_text=None):
    cur = pi_version()
    ver = rel["version"]
    age = rel["age_hours"]
    age_ok = age is not None and age >= MIN_AGE_HOURS
    if age_ok:
        gate = "PASSED (>=72h) — eligible to sandbox-test and plan the apply"
    elif age is None:
        gate = f"UNKNOWN age — hold until publish time is confirmable and {MIN_AGE_HOURS}h have passed"
    else:
        gate = f"HOLD — only {age:.0f}h old, {MIN_AGE_HOURS - age:.0f}h until the 72h gate opens"

    lines = []
    lines.append("# Pi update plan")
    lines.append("")
    lines.append(f"- Current: **{cur}**")
    lines.append(f"- Latest:  **{ver}** (published {rel['published_at']})")
    lines.append(f"- Age gate: {gate}")
    lines.append("")
    lines.append("## How I'll update (when the gate opens)")
    lines.append("1. `pi_update_planner.py snapshot` — snapshot current pi + settings for rollback.")
    lines.append("2. Review the evaluation below; if breaking changes touch my extensions, sandbox-test first.")
    lines.append("3. `pi_update_planner.py apply` — update core, run health checks.")
    lines.append("4. Exercise memory + Telegram + secrets after the update, then `confirm`.")
    lines.append("")
    lines.append("## Rollback path")
    lines.append("- `pi_update_planner.py rollback` restores the latest snapshot (pi package dir + settings).")
    lines.append("- A cron watchdog auto-rolls-back if an applied update is not confirmed within the deadline.")
    lines.append("")
    if eval_text:
        lines.append("## Evaluation of the new release")
        lines.append("")
        lines.append(eval_text)
        lines.append("")
    else:
        lines.append("## Release notes (unevaluated)")
        lines.append("")
        lines.append((rel.get("body") or "(none)")[:4000])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- snapshot / rollback

def snapshot():
    ver = pi_version()
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(SNAPSHOT_ROOT, f"{ver}-{ts}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "version.txt"), "w") as f:
        f.write(ver or "")
    for src, name in ((SETTINGS, "settings.json"), (AUTH, "auth.json")):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(d, name))
            os.chmod(os.path.join(d, name), 0o600)
    # tar the pi package dir for a byte-exact restore
    if os.path.isdir(PI_PKG_DIR):
        with tarfile.open(os.path.join(d, "pi-package.tar.gz"), "w:gz") as t:
            t.add(PI_PKG_DIR, arcname="pi-coding-agent")
    log(f"snapshot -> {d}")
    return d


def list_snapshots():
    if not os.path.isdir(SNAPSHOT_ROOT):
        return []
    out = sorted(os.listdir(SNAPSHOT_ROOT), reverse=True)
    return out


def rollback(snap=None):
    snaps = list_snapshots()
    if not snaps:
        log("no snapshots available")
        return 1
    snap = snap or snaps[0]
    d = os.path.join(SNAPSHOT_ROOT, snap)
    if not os.path.isdir(d):
        log(f"snapshot not found: {d}")
        return 1
    ver_file = os.path.join(d, "version.txt")
    prev = open(ver_file).read().strip() if os.path.exists(ver_file) else "?"
    log(f"rolling back to snapshot {snap} (was {prev})")

    # restore pi package dir
    tarball = os.path.join(d, "pi-package.tar.gz")
    if os.path.exists(tarball) and os.path.isdir(PI_PKG_DIR):
        shutil.rmtree(PI_PKG_DIR, ignore_errors=True)
        with tarfile.open(tarball, "r:gz") as t:
            t.extractall(os.path.dirname(PI_PKG_DIR))
        log("  restored pi package dir")
    # restore settings + auth
    for name in ("settings.json", "auth.json"):
        src = os.path.join(d, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(HOME, ".pi", "agent", name))
            log(f"  restored {name}")
    v = pi_version()
    log(f"  pi version now: {v}")
    # clear watchdog state
    if os.path.exists(WATCH):
        os.remove(WATCH)
    return 0


# ---------------------------------------------------------------- watchdog state

def _load_json(path, default):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def arm_watch(version, deadline_min):
    _save_json(WATCH, {
        "status": "in_progress",
        "version": version,
        "snapshot": list_snapshots()[0] if list_snapshots() else None,
        "deadline": time.time() + deadline_min * 60,
        "deadline_min": deadline_min,
    })


def watchdog():
    w = _load_json(WATCH, None)
    if not w or w.get("status") != "in_progress":
        return 0
    if time.time() < w.get("deadline", 0):
        log("watchdog: still within deadline, waiting for confirmation")
        return 0
    log(f"watchdog: deadline passed for {w.get('version')} without confirmation -> rolling back")
    rc = rollback(w.get("snapshot"))
    _save_json(WATCH, {"status": "rolled_back", "version": w.get("version"),
                       "rolled_back_at": time.time()})
    _notify("pi update auto-rolled-back",
            f"An update to pi {w.get('version')} was not confirmed within "
            f"{w.get('deadline_min')} min, so I rolled back to the previous version. "
            f"rollback rc={rc}.", telegram=True)
    return rc


# ---------------------------------------------------------------- notify / state

def _notify(subject, body, telegram=False):
    args = [sys.executable, os.path.join(MAILTOOL, "notify.py"), "--email"]
    if telegram:
        args.append("--telegram")
    args += [subject, body]
    try:
        subprocess.run(args, cwd=HOME, timeout=60)
        log("  notified")
    except Exception as e:
        log(f"  notify failed: {e}")


def _state():
    return _load_json(STATE, {})


def _save_state(s):
    _save_json(STATE, s)


# ---------------------------------------------------------------- commands

def cmd_check(notify=False, telegram=False):
    cur = pi_version()
    rel = latest_release()
    log(f"current pi: {cur}")
    if not rel:
        log("could not resolve latest release")
        return 1
    ver, age = rel["version"], rel["age_hours"]
    age_s = f"{age:.1f}h" if age is not None else "unknown"
    log(f"latest pi: {ver} (published {rel['published_at']}, {age_s} ago)")

    state = _state()
    state["last_check"] = time.time()
    is_new = state.get("last_release_seen") != ver
    if is_new:
        state["last_release_seen"] = ver
        _save_state(state)
        if age is not None and age < MIN_AGE_HOURS:
            gate = f"HOLD — only {age:.0f}h old; {MIN_AGE_HOURS - age:.0f}h until the 72h gate opens."
            eta = time.strftime("%Y-%m-%d %H:%M", time.localtime(
                time.time() + (MIN_AGE_HOURS - age) * 3600))
            gate += f" I'll evaluate around {eta}."
        else:
            gate = f"Gate passed (>=72h). Ready to evaluate + plan."
        if notify:
            _notify("pi update available",
                    f"A new pi release is out: {ver} (published {rel['published_at']}).\n\n"
                    f"{gate}", telegram=telegram)
        else:
            log(gate)
    else:
        log("no new release since last check")
    return 0


def cmd_evaluate():
    rel = latest_release()
    if not rel:
        return 1
    return 0 if evaluate(rel) is not None else 1


def cmd_plan():
    rel = latest_release()
    if not rel:
        return 1
    eval_text = evaluate(rel, verbose=False)
    plan = build_plan(rel, eval_text)
    os.makedirs(os.path.dirname(PLAN_PATH), exist_ok=True)
    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        f.write(plan)
    log(f"plan written -> {PLAN_PATH}")
    print(plan)
    return 0


def cmd_snapshot():
    return 0 if snapshot() else 1


def cmd_apply(args):
    rel = latest_release()
    target = args.version or (rel["version"] if rel else None)
    if not target:
        log("no target version")
        return 1
    age = age_hours(npm_publish_time(PI_PKG, target))
    if age is None or age < MIN_AGE_HOURS:
        age_s = f"{age:.0f}h" if age is not None else "unknown"
        log(f"REFUSING: {target} is {age_s} old (<{MIN_AGE_HOURS}h age gate)")
        return 1
    log(f"applying update to {target} (age {age:.0f}h, gate passed)")

    if not args.skip_sandbox:
        log(f"sandbox-testing {target} before touching the host ...")
        if sandbox_test(target) != 0:
            log("sandbox test FAILED; NOT applying")
            return 1
        log("sandbox test PASSED; proceeding to host apply")

    snap = snapshot()
    arm_watch(target, args.deadline_min)
    log(f"watchdog armed: auto-rollback in {args.deadline_min} min if not confirmed")

    log("running `pi update self` ...")
    r = subprocess.run([PI, "update", "self"], cwd=HOME,
                       capture_output=True, text=True, timeout=900)
    log((r.stdout or "").strip()[-400:] or (r.stderr or "").strip()[-400:])
    if r.returncode != 0:
        log("update command failed; rolling back")
        rollback()
        return 1

    # health checks
    v = pi_version()
    log(f"post-update version: {v}")
    if v != target:
        log(f"version mismatch ({v} != {target}); rolling back")
        rollback()
        return 1

    log("smoke test (loads my extensions + one model call) ...")
    try:
        s = subprocess.run([PI, "-p", "Reply with exactly: SMOKE_OK"], cwd=HOME,
                           capture_output=True, text=True, timeout=300)
        ok = s.returncode == 0 and "SMOKE_OK" in (s.stdout or "")
    except subprocess.TimeoutExpired:
        ok = False
    log(f"smoke test -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        log("smoke test failed; rolling back")
        rollback()
        return 1

    _notify("pi updated — needs confirmation",
            f"Updated pi {pi_version() or '?'} -> {target}. Health checks passed. "
            f"I'll exercise memory/Telegram/secrets now, then confirm. If I don't "
            f"confirm within {args.deadline_min} min, the watchdog rolls it back.",
            telegram=True)
    log(f"updated to {target}; confirm within {args.deadline_min} min (pi_update_planner.py confirm)")
    return 0


def cmd_confirm():
    w = _load_json(WATCH, None)
    if not w or w.get("status") != "in_progress":
        log("no in-progress update to confirm")
        return 1
    _save_json(WATCH, {"status": "confirmed", "version": w.get("version"),
                       "confirmed_at": time.time()})
    _notify("pi update confirmed",
            f"Confirmed pi {w.get('version')} is good — keeping it. Watchdog disarmed.",
            telegram=True)
    log(f"confirmed {w.get('version')}; watchdog disarmed")
    return 0


def cmd_rollback(args):
    return rollback(args.snapshot)


def _sandbox(script, version, args, timeout):
    return subprocess.run(
        ["bash", os.path.join(SANDBOX_DIR, script), version] + args,
        cwd=SANDBOX_DIR, capture_output=True, text=True, timeout=timeout)


def sandbox_test(target):
    """Build a Docker sandbox image for `target` and smoke-test it. Returns 0/1.

    Does NOT touch the host pi install — only docker + the sandbox config dir.
    """
    log(f"sandbox-testing pi {target} ...")

    # Refresh the sandbox config with the live auth key + a minimal settings file
    # (drop host extensions/packages so the container can't reference host paths).
    os.makedirs(SANDBOX_CONFIG, exist_ok=True)
    if os.path.exists(AUTH):
        shutil.copy2(AUTH, os.path.join(SANDBOX_CONFIG, "auth.json"))
        os.chmod(os.path.join(SANDBOX_CONFIG, "auth.json"), 0o600)
    try:
        s = json.load(open(SETTINGS, "r", encoding="utf-8"))
        minimal = {
            "defaultProvider": s.get("defaultProvider", "deepseek"),
            "defaultModel": s.get("defaultModel", "deepseek-v4-pro"),
            "defaultThinkingLevel": s.get("defaultThinkingLevel", "high"),
        }
        with open(os.path.join(SANDBOX_CONFIG, "settings.json"), "w") as f:
            json.dump(minimal, f, indent=2)
    except Exception as e:
        log(f"  (could not refresh sandbox settings: {e})")

    log("  building image ...")
    b = _sandbox("build.sh", target, [], 900)
    if b.returncode != 0:
        log("  build FAILED:\n" + ((b.stderr or b.stdout or "")[-800:]))
        return 1
    log(f"  built pi-sandbox:{target}")

    log("  version check ...")
    v = _sandbox("run.sh", target, ["--version"], 120)
    got = (v.stdout or "").strip()
    log(f"  sandbox reports: {got or (v.stderr or '').strip()[:200]}")
    if target not in got:
        log("  version check FAILED")
        return 1

    log("  smoke test (one model call) ...")
    s = _sandbox("run.sh", target, ["-p", "Reply with exactly: SMOKE_OK"], 300)
    ok = s.returncode == 0 and "SMOKE_OK" in (s.stdout or "")
    log(f"  smoke test -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        log("  output tail:\n" + ((s.stdout or s.stderr or "")[-600:]))
        return 1

    log(f"sandbox test PASSED for pi {target}")
    return 0


def cmd_sandbox(args):
    rel = latest_release()
    target = args.version or (rel["version"] if rel else None)
    if not target:
        log("no target version")
        return 1
    age = age_hours(npm_publish_time(PI_PKG, target))
    if age is not None and age < MIN_AGE_HOURS:
        log(f"REFUSING: {target} is {age:.0f}h old (<{MIN_AGE_HOURS}h age gate)")
        return 1
    return sandbox_test(target)


def cmd_status():
    cur = pi_version()
    rel = latest_release()
    log(f"pi version: {cur}")
    if rel:
        age_s = f"{rel['age_hours']:.0f}h" if rel['age_hours'] is not None else "?"
        log(f"latest:     {rel['version']} ({age_s} old)")
    w = _load_json(WATCH, None)
    log(f"watchdog:   {w.get('status', 'idle') if w else 'idle'}")
    log(f"snapshots:  {', '.join(list_snapshots()[:5]) or '(none)'}")


def main():
    p = argparse.ArgumentParser(description="Pi core-update planner + rollback watchdog")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="detect new releases + age-gate")
    c.add_argument("--notify", action="store_true")
    c.add_argument("--telegram", action="store_true")

    sub.add_parser("evaluate", help="LLM-evaluate the latest release notes")
    sub.add_parser("plan", help="write the how/when plan")
    sub.add_parser("snapshot", help="snapshot current pi for rollback")
    sub.add_parser("confirm", help="confirm a completed update")
    sub.add_parser("watchdog", help="cron watchdog (auto-rollback)")
    sub.add_parser("status", help="show state")

    a = sub.add_parser("apply", help="update core (manual; age-gated)")
    a.add_argument("--version")
    a.add_argument("--deadline-min", type=int, default=DEFAULT_DEADLINE_MIN)
    a.add_argument("--skip-sandbox", action="store_true",
                   help="skip the Docker sandbox test before applying")

    s = sub.add_parser("sandbox", help="build + smoke-test a pi version in Docker")
    s.add_argument("--version")

    r = sub.add_parser("rollback", help="restore a snapshot")
    r.add_argument("--snapshot")

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return 1
    return {
        "check": lambda: cmd_check(args.notify, args.telegram),
        "evaluate": cmd_evaluate,
        "plan": cmd_plan,
        "snapshot": cmd_snapshot,
        "apply": lambda: cmd_apply(args),
        "confirm": cmd_confirm,
        "watchdog": watchdog,
        "rollback": lambda: cmd_rollback(args),
        "sandbox": lambda: cmd_sandbox(args),
        "status": cmd_status,
    }[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())
