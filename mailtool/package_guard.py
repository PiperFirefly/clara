#!/usr/bin/env python3
"""Package update guard for pi.

Looks for updates to pi and installed packages, tests candidates in an
isolated sandbox (a throwaway PI_CODING_AGENT_DIR) before installing.

Subcommands:
  check                 Compare installed packages + pi itself to npm latest;
                        list newly discoverable 'pi-package' packages.
  check --email-if-updates   Also email a report if updates are found.
  test <source>         Fetch <source> into a sandbox and run a smoke test.
  apply <source>        Sandbox-test <source>, then install into real config if it passes.
  update-all            For each outdated installed package, sandbox-test the new
                        version, then update the ones that pass.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import common

HOME = os.path.expanduser("~")
PI = os.path.join(HOME, ".local/share/pi-node/node-v22.23.2-linux-x64/bin/pi")
SETTINGS = os.path.join(HOME, ".pi/agent/settings.json")
AUTH = os.path.join(HOME, ".pi/agent/auth.json")
NPM_ROOT = os.path.join(HOME, ".pi/agent/npm/node_modules")
PI_PKG = "@earendil-works/pi-coding-agent"
def _operator_email():
    """Primary operator's email from operator config; env override; else None."""
    return common.operator_email()
MIN_AGE_HOURS = 72
MIN_STARS = 10


def log(msg):
    common.log_print_only(msg)


def npm_latest(name):
    enc = urllib.parse.quote(name, safe="")
    try:
        with urllib.request.urlopen(
            f"https://registry.npmjs.org/{enc}/latest", timeout=20
        ) as r:
            return json.load(r).get("version")
    except Exception:
        return None


def installed_version(name):
    p = os.path.join(NPM_ROOT, *name.split("/"), "package.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, "r", encoding="utf-8")).get("version")
        except Exception:
            return None
    return None


def pi_self_version():
    try:
        out = subprocess.run([PI, "--version"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return None


def parse_npm_source(src):
    """Return (name, version_spec_or_None) from an 'npm:...' source."""
    s = src[len("npm:"):]
    if s.startswith("@"):
        parts = s.split("@")
        name = "@" + parts[1]
        ver = parts[2] if len(parts) > 2 else None
    else:
        if "@" in s:
            name, ver = s.split("@", 1)
        else:
            name, ver = s, None
    return name, ver


def installed_npm_packages(settings):
    pkgs = settings.get("packages", [])
    result = []
    for p in pkgs:
        src = p if isinstance(p, str) else p.get("source", "")
        if isinstance(src, str) and src.startswith("npm:"):
            result.append(src)
    return result


def discover_pi_packages(limit=15):
    url = f"https://registry.npmjs.org/-/v1/search?text=keywords:pi-package&size={limit}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
        return [o["package"]["name"] for o in data.get("objects", [])]
    except Exception as e:
        log(f"discover failed: {e}")
        return []


def send_email(subject, body):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mailtool import load_config, send
    addr = _operator_email()
    if addr:
        send(load_config(), addr, subject, body)


def npm_meta(name):
    enc = urllib.parse.quote(name, safe="")
    try:
        with urllib.request.urlopen(f"https://registry.npmjs.org/{enc}", timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def _age_hours(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return None


def security_check(source):
    """Enforce supply-chain rules: >=72h old (hard), and stars -> scrutiny.
    Sub-10-star packages are deep-scanned by the security director instead of
    blanket-blocked (the operator, 2026-08-26). Returns (ok, reasons)."""
    reasons = []
    if not source.startswith("npm:"):
        return True, reasons
    name, ver = parse_npm_source(source)
    meta = npm_meta(name)
    if not meta:
        return True, reasons  # can't verify; let the sandbox test decide

    ver_key = ver if ver else meta.get("dist-tags", {}).get("latest")
    published = (meta.get("time") or {}).get(ver_key)
    if published:
        h = _age_hours(published)
        if h is not None and h < MIN_AGE_HOURS:
            reasons.append(f"{name}@{ver_key} published {h:.0f}h ago (<{MIN_AGE_HOURS}h)")
    elif ver:
        reasons.append(f"{name}@{ver}: unknown publish time")

    repo = meta.get("repository")
    repo_url = ""
    if isinstance(repo, dict):
        repo_url = repo.get("url", "") or ""
    elif isinstance(repo, str):
        repo_url = repo
    repo_url = (repo_url.replace("git+", "").replace("git://", "https://")
                .replace("ssh://git@github.com/", "https://github.com/").replace(".git", ""))
    m = re.search(r"github\.com[/:]([\w.-]+/[\w.-]+)", repo_url)
    if m:
        full = m.group(1)
        try:
            with urllib.request.urlopen(f"https://api.github.com/repos/{full}", timeout=20) as r:
                gi = json.load(r)
            stars = gi.get("stargazers_count", 0)
            if stars < MIN_STARS:
                # Stars are a popularity signal, not a safety one. Sub-threshold
                # -> deep scrutiny via the security director, not a blanket block.
                # Block only if the scan can't clear it (or can't run).
                log(f"  sub-threshold ({stars} < {MIN_STARS} stars): deep scrutiny via security director")
                try:
                    import security_director
                    res = security_director.screen(f"gh:{full}")
                    if res["verdict"] == "CLEAR":
                        log(f"    deep scrutiny CLEAR (score {res['score']}) — allow on scrutiny instead of stars")
                    else:
                        reasons.append(f"{full}: deep scrutiny {res['verdict']} (score {res['score']})")
                except Exception as e:
                    reasons.append(f"{full}: deep scrutiny unavailable ({e}) — blocked pending review")
            created = gi.get("created_at")
            if created:
                h = _age_hours(created)
                if h is not None and h < MIN_AGE_HOURS:
                    reasons.append(f"{full} created {h:.0f}h ago (<{MIN_AGE_HOURS}h)")
        except Exception as e:
            reasons.append(f"could not check {full}: {e}")
    return (len(reasons) == 0), reasons


def run_sandbox_test(source):
    """Install <source> into a throwaway config dir and run a smoke test."""
    sandbox = tempfile.mkdtemp(prefix="pi-sandbox-")
    shutil.copy(AUTH, os.path.join(sandbox, "auth.json"))
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = sandbox
    env["PI_SKIP_VERSION_CHECK"] = "1"
    cmd = [PI, "-e", source, "-p", "Reply with exactly: SMOKE_OK"]
    log(f"  sandbox test: {source} (dir {sandbox})")
    try:
        proc = subprocess.run(cmd, cwd=HOME, env=env, capture_output=True,
                              text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log("  TEST TIMED OUT")
        shutil.rmtree(sandbox, ignore_errors=True)
        return False
    ok = proc.returncode == 0 and "SMOKE_OK" in (proc.stdout or "")
    log(f"  exit={proc.returncode} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        tail = (proc.stdout or "")[-400:] or (proc.stderr or "")[-400:]
        if tail:
            log("  tail: " + tail.replace("\n", " ")[:400])
    shutil.rmtree(sandbox, ignore_errors=True)
    return ok


def cmd_check(email=False):
    report = []
    self_v = pi_self_version()
    latest_self = npm_latest(PI_PKG)
    if self_v and latest_self and self_v != latest_self:
        line = f"pi self: {self_v} -> {latest_self}"
        log("OUTDATED " + line); report.append(line)
    else:
        log(f"pi self: {self_v} (latest {latest_self})")

    settings = json.load(open(SETTINGS, "r", encoding="utf-8"))
    sources = installed_npm_packages(settings)
    log(f"installed npm packages: {len(sources)}")
    outdated = []
    for src in sources:
        name, _ = parse_npm_source(src)
        inst = installed_version(name)
        latest = npm_latest(name)
        status = f"{name}: installed={inst} latest={latest}"
        if inst and latest and inst != latest:
            log("OUTDATED " + status); outdated.append(status)
        else:
            log("ok " + status)
    report.extend(outdated)

    discover = [n for n in discover_pi_packages()
                if not any(n == parse_npm_source(s)[0] for s in sources)]
    if discover:
        line = "new pi-package(s) available: " + ", ".join(discover)
        log(line); report.append(line)

    if email and report:
        send_email("pi updates available", "Updates found:\n\n" + "\n".join(report))
        log("report emailed")
    return report


def cmd_test(source):
    return 0 if run_sandbox_test(source) else 1


def cmd_apply(source):
    ok, reasons = security_check(source)
    if not ok:
        log("SECURITY BLOCK: " + "; ".join(reasons))
        return 1
    log(f"testing {source} before install...")
    if not run_sandbox_test(source):
        log("test FAILED; not installing.")
        return 1
    log(f"installing {source} into real config...")
    r = subprocess.run([PI, "install", source], cwd=HOME, capture_output=True, text=True, timeout=300)
    log((r.stdout or "").strip()[-500:] or (r.stderr or "").strip()[-500:])
    return r.returncode


def cmd_update_all():
    settings = json.load(open(SETTINGS, "r", encoding="utf-8"))
    sources = installed_npm_packages(settings)
    rc = 0
    for src in sources:
        name, _ = parse_npm_source(src)
        inst = installed_version(name)
        latest = npm_latest(name)
        if not (inst and latest) or inst == latest:
            continue
        log(f"updating {name}: {inst} -> {latest}")
        ok, reasons = security_check("npm:" + name)
        if not ok:
            log(f"  SECURITY BLOCK: {'; '.join(reasons)}; skipping {name}")
            rc = 1
            continue
        if run_sandbox_test("npm:" + name):
            r = subprocess.run([PI, "update", "--extension", src],
                               cwd=HOME, capture_output=True, text=True, timeout=300)
            log(f"  update exit={r.returncode}")
            if r.returncode != 0:
                log("  " + ((r.stderr or r.stdout or "").strip()[-300:]))
                rc = 1
        else:
            log(f"  {name} failed sandbox test; skipping update.")
            rc = 1
    return rc


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "check":
        email = "--email-if-updates" in sys.argv[2:]
        sys.exit(1 if cmd_check(email=email) else 0)
    elif cmd == "test":
        sys.exit(cmd_test(sys.argv[2]))
    elif cmd == "apply":
        sys.exit(cmd_apply(sys.argv[2]))
    elif cmd == "update-all":
        sys.exit(cmd_update_all())
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
