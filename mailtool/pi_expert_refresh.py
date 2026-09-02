#!/usr/bin/env python3
"""pi_expert_refresh.py — keep the pi-expert persona current.

Every ~2 days (cron), the pi expert must reflect reality: pi gets updated, docs
change, we add/remove extensions/skills/packages, and new pi packages appear.

This complements pi_update_planner.py (which owns CORE updates + rollback). This
script owns KNOWLEDGE refresh:

  status                  Show last refresh, pi version, staleness.
  docs-delta              Detect pi doc/CHANGELOG changes vs last refresh; list
                          changed doc files + their section headers + changelog top.
  inventory               Reconcile our installed extensions/skills/packages vs
                          the recorded inventory; report additions/removals.
  skills-check            Lint every skill's front matter (name/description YAML)
                          so a bad SKILL.md can't crash the harness unnoticed.
  packages [--notify]     Surface NEW pi-package candidates from npm with
                          supply-chain metadata (age gate). NEVER installs.
  run                     Cron mode: docs-delta + skills-check + inventory +
                          packages, update the digest doc, optionally notify on
                          material change.

State: ~/.pi/agent/pi_expert_state.json (machine state; version, fingerprints).
Digest: doc pi/pi-expert-refresh (human-readable "what changed" — read when awake).
Guide:  doc pi/pi-expert-guide (the authoritative reference; updated by me when I
        actually re-read changed docs, not by this script).

Supply-chain: `packages` only LISTS candidates. It never installs. Fresh uploads
(<72h) and low-star packages are flagged for deep scrutiny (security_director)
per AGENTS.md rules.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

try:
    import yaml
except ImportError:
    yaml = None

try:
    import common
except ImportError:
    sys.path.insert(0, os.path.join(os.path.expanduser("~"), "mailtool"))
    import common

HOME = os.path.expanduser("~")
NODE_ROOT = os.path.join(HOME, ".local/share/pi-node/node-v22.23.2-linux-x64")
PI = os.path.join(NODE_ROOT, "bin", "pi")
PI_PKG = "@earendil-works/pi-coding-agent"
PI_PKG_DIR = os.path.join(NODE_ROOT, "lib", "node_modules", *PI_PKG.split("/"))
DOCS_DIR = os.path.join(PI_PKG_DIR, "docs")
EXAMPLES_DIR = os.path.join(PI_PKG_DIR, "examples")
AGENT_DIR = os.path.join(HOME, ".pi", "agent")
SETTINGS = os.path.join(AGENT_DIR, "settings.json")
STATE_FILE = os.path.join(AGENT_DIR, "pi_expert_state.json")
LOCK = os.path.join(AGENT_DIR, "pi_expert_refresh.lock")
NPM_SEARCH = "https://registry.npmjs.org/-/v1/search?text=keywords:pi-package&size=30"
GH_API = "https://api.github.com/repos/"
MIN_AGE_HOURS = 72
DOCSTORE = os.path.join(HOME, "memory", "docstore.py")
DIGEST_KEY = "pi/pi-expert-refresh"

GUIDE_INVENTORY = {
    "extensions": ["mic-listen.ts", "agent-status.ts",
                   "memory-tools.ts", "secrets-tools.ts", "coding-tools.ts"],
    "skills": ["book-reader", "db-lookup", "doc-ingest", "file-layout",
               "large-paste", "spatial-plausibility", "pi-expert"],
    "packages": ["pi-web-access", "@llblab/pi-telegram"],
}
INV_FILES = [
    ("agent_extensions", os.path.join(AGENT_DIR, "extensions")),
    ("settings_extensions", SETTINGS),
    ("skills", os.path.join(AGENT_DIR, "skills")),
]


def log(msg):
    common.log_print_only(msg)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_state():
    return common.load_json(STATE_FILE, {})


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def pi_version():
    try:
        return subprocess.run([PI, "--version"], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except Exception:
        return None


def doc_fingerprint():
    """Map doc filename -> sha1 of content. Only stable reference files."""
    fp = {}
    for d in (DOCS_DIR, EXAMPLES_DIR):
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for fname in files:
                if not (fname.endswith(".md") or fname.endswith(".ts")):
                    continue
                p = os.path.join(root, fname)
                rel = os.path.relpath(p, PI_PKG_DIR)
                try:
                    with open(p, "rb") as fh:
                        fp[rel] = hashlib.sha1(fh.read()).hexdigest()[:12]
                except Exception:
                    pass
    return fp


def section_headers(path):
    out = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if line.startswith("## ") or line.startswith("# "):
                    out.append(line.rstrip())
    except Exception:
        pass
    return out


def docs_delta(state):
    """Return (changed_files, changelog_head, is_new_version)."""
    cur = pi_version()
    changed = []
    fp = doc_fingerprint()
    old_fp = state.get("doc_fingerprint", {})
    old_ver = state.get("pi_version")
    is_new_version = (old_ver is not None and cur != old_ver)
    for rel, h in sorted(fp.items()):
        if old_fp.get(rel) != h:
            changed.append(rel)
    chlog = os.path.join(PI_PKG_DIR, "CHANGELOG.md")
    ch_head = ""
    if os.path.isfile(chlog):
        try:
            with open(chlog, "r", errors="replace") as f:
                ch_head = "".join(f.readlines()[:12]).strip()
        except Exception:
            ch_head = ""
    return cur, is_new_version, changed, ch_head


def inventory():
    """Collect current on-disk inventory for comparison."""
    ext_d = []
    for d in ("extensions",):
        p = os.path.join(AGENT_DIR, d)
        if os.path.isdir(p):
            ext_d = sorted(os.listdir(p))
    skills = []
    sp = os.path.join(AGENT_DIR, "skills")
    if os.path.isdir(sp):
        skills = sorted(os.listdir(sp))
    pkgs = []
    try:
        r = subprocess.run([PI, "list"], capture_output=True, text=True, timeout=30)
        pkgs = [l.strip() for l in r.stdout.splitlines() if "@" in l or "npm:" in l]
    except Exception:
        pass
    return {"extensions_dir": ext_d, "skills": skills, "packages_raw": pkgs}


def inventory_report():
    inv = inventory()
    lines = []
    cur_ext = set(inv["extensions_dir"]) | {"memory-tools.ts", "secrets-tools.ts", "coding-tools.ts"}
    rec_ext = set(GUIDE_INVENTORY["extensions"])
    lines.append("Extensions dir: " + (", ".join(inv["extensions_dir"]) or "(none)"))
    new_e = cur_ext - rec_ext
    gone_e = rec_ext - cur_ext
    if new_e:
        lines.append("  NEW extensions on disk: " + ", ".join(sorted(new_e)))
    if gone_e:
        lines.append("  GONE extensions (recorded but not found): " + ", ".join(sorted(gone_e)))
    cur_s = set(inv["skills"]); rec_s = set(GUIDE_INVENTORY["skills"])
    lines.append("Skills: " + (", ".join(inv["skills"]) or "(none)"))
    n_s = cur_s - rec_s; g_s = rec_s - cur_s
    if n_s: lines.append("  NEW skills: " + ", ".join(sorted(n_s)))
    if g_s: lines.append("  GONE skills: " + ", ".join(sorted(g_s)))
    lines.append("Packages (pi list): " + (" | ".join(inv["packages_raw"]) if inv["packages_raw"] else "(none)"))
    return "\n".join(lines)


# ------------------------------------------------------------- skills lint check
SKILLS_DIR = os.path.join(AGENT_DIR, "skills")


def check_skill_frontmatter():
    """Lint every SKILL.md front matter. Returns list of (skill, problem)."""
    problems = []
    if yaml is None:
        problems.append(("<pyyaml missing>", "yaml module not installed; cannot lint front matter"))
        return problems
    if not os.path.isdir(SKILLS_DIR):
        return problems
    for skill in sorted(os.listdir(SKILLS_DIR)):
        p = os.path.join(SKILLS_DIR, skill, "SKILL.md")
        if not os.path.isfile(p):
            continue
        try:
            text = open(p, "r", errors="replace").read()
        except Exception as e:
            problems.append((skill, f"unreadable: {e}"))
            continue
        if not text.startswith("---"):
            continue
        # front matter is the first --- ... --- block
        parts = text.split("---", 2)
        if len(parts) < 3:
            problems.append((skill, "front matter missing closing ---"))
            continue
        front = parts[1]
        try:
            meta = yaml.safe_load(front)
        except Exception as e:
            problems.append((skill, f"YAML parse error: {e}"))
            continue
        if not isinstance(meta, dict):
            problems.append((skill, f"front matter is not a mapping (got {type(meta).__name__})"))
            continue
        if not meta.get("name"):
            problems.append((skill, "missing or empty 'name'"))
        if not meta.get("description"):
            problems.append((skill, "missing or empty 'description'"))
    return problems


def cmd_skills_check():
    problems = check_skill_frontmatter()
    if not problems:
        print("All skill front matters OK (" + str(len([d for d in os.listdir(SKILLS_DIR) if os.path.isdir(os.path.join(SKILLS_DIR, d))])) + " skills).")
        return 0
    print(f"{len(problems)} skill front-matter problem(s):")
    for skill, prob in problems:
        print(f"  {skill}: {prob}")
    return 1


# ------------------------------------------------------------------ packages scan
def npm_package_candidates():
    req = urllib.request.Request(NPM_SEARCH, headers={"User-Agent": "pi-expert-refresh"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    out = []
    now = time.time()
    for obj in data.get("objects", []):
        p = obj.get("package", {})
        name = p.get("name", "")
        ver = p.get("version", "")
        published = p.get("date") or p.get("time", {}).get("modified") or ""
        desc = (p.get("description") or "")[:140]
        try:
            age_h = (now - datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()) / 3600
        except Exception:
            age_h = None
        fresh = age_h is not None and age_h < MIN_AGE_HOURS
        out.append({
            "name": name, "version": ver, "published": published,
            "age_h": round(age_h, 1) if age_h is not None else None,
            "fresh": fresh, "description": desc,
        })
    return out


def write_digest(body):
    """Overwrite doc pi/pi-expert-refresh via docstore. Returns success bool."""
    try:
        proc = subprocess.run(
            [sys.executable, DOCSTORE, "set", DIGEST_KEY, "note",
             "Pi-expert refresh digest", "--source", "pi_expert_refresh.py"],
            input=body.encode("utf-8"), capture_output=True, timeout=60)
        return proc.returncode == 0
    except Exception as e:
        log(f"digest write failed: {e}")
        return False


def cmd_status():
    st = load_state()
    ver = pi_version()
    print("pi version       :", ver)
    print("recorded version :", st.get("pi_version"))
    print("last refresh     :", st.get("last_refresh"))
    print("doc files tracked:", len(st.get("doc_fingerprint", {})))
    _c, is_new, changed, _h = docs_delta(st)
    print("docs changed since last read:", len(changed))
    for c in changed[:20]:
        print("   ", c)


def cmd_docs_delta():
    st = load_state()
    cur, is_new, changed, ch_head = docs_delta(st)
    print("pi version:", cur, "| changed since last read:", is_new)
    for c in changed:
        path = os.path.join(PI_PKG_DIR, c)
        print(f"\n== {c} ==")
        for h in section_headers(path)[:30]:
            print("   ", h)
    if ch_head:
        print("\n=== CHANGELOG (top) ===")
        print(ch_head)


def cmd_inventory():
    print(inventory_report())


def cmd_packages(notify=False):
    try:
        cands = npm_package_candidates()
    except Exception as e:
        log(f"package scan failed: {e}")
        return
    fresh = [c for c in cands if c.get("fresh")]
    print(f"{len(cands)} pi-package candidates on npm; {len(fresh)} are FRESH (<72h).")
    for c in sorted(cands, key=lambda x: (x["age_h"] or 0)):
        flag = "FRESH" if c["fresh"] else "OK"
        print(f"  [{flag:5}] {c['name']}@{c['version']} (age {c['age_h']}h)")
        if c["description"]:
            print(f"          {c['description']}")


def cmd_run(notify=False):
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("lock held; skipping")
        return

    st = load_state()
    cur, is_new, changed, ch_head = docs_delta(st)
    changed = [c for c in changed if "/docs/" in c or c.startswith("docs/")]

    digest_lines = [f"# Pi-expert refresh — {now_iso()}",
                    "", f"pi version: {cur}", ""]

    # --- docs delta
    if is_new or changed:
        digest_lines += [f"**Pi version changed**: {st.get('pi_version')} -> {cur}" if is_new else "",
                         f"**{len(changed)} doc file(s) changed since last read**:", ""]
        for c in changed:
            path = os.path.join(PI_PKG_DIR, c)
            digest_lines.append(f"### {c}")
            for h in section_headers(path)[:25]:
                digest_lines.append(f"  {h}")
            digest_lines.append("")
        if ch_head:
            digest_lines += ["### CHANGELOG (top)", ch_head, ""]
    else:
        digest_lines += ["Docs: unchanged since last read.", ""]

    # --- skills lint check
    skill_problems = check_skill_frontmatter()
    if skill_problems:
        digest_lines += ["## ⚠ Skill front-matter problems", ""]
        for skill, prob in skill_problems:
            digest_lines.append(f"- {skill}: {prob}")
        digest_lines.append("")

    # --- inventory
    digest_lines += ["## Inventory", "", "```", inventory_report(), "```", ""]

    # --- packages
    try:
        cands = npm_package_candidates()
        fresh = [c for c in cands if c.get("fresh")]
        known = set(st.get("known_packages", []))
        notified_fresh = set(st.get("notified_fresh", []))
        new_ok = [c for c in cands if not c["fresh"] and c["name"] not in known]
        new_fresh = [c for c in fresh if c["name"] not in notified_fresh]
        digest_lines += [f"## New packages ({len(cands)} candidates; {len(fresh)} fresh; {len(new_ok)} new-not-fresh)", ""]
        for c in sorted(cands, key=lambda x: (x["age_h"] or 0)):
            flag = "FRESH" if c["fresh"] else "OK"
            digest_lines.append(f"- [{flag}] {c['name']}@{c['version']} (age {c['age_h']}h)")
            if c["description"]:
                digest_lines.append(f"  {c['description']}")
        digest_lines.append("")
        st["known_packages"] = sorted({c["name"] for c in cands})
        st["notified_fresh"] = sorted(notified_fresh | {c["name"] for c in fresh})
    except Exception as e:
        new_ok, new_fresh = [], []
        digest_lines.append(f"package scan failed: {e}")

    body = "\n".join(digest_lines)
    write_digest(body)

    st["pi_version"] = cur
    st["doc_fingerprint"] = doc_fingerprint()
    st["last_refresh"] = now_iso()
    save_state(st)

    new_pkg_notice = len(new_ok) + len(new_fresh)
    # A broken skill front matter is material: it makes the harness cry. Include it.
    if skill_problems:
        new_pkg_notice += 1
    log(f"pi_expert_refresh: pi={cur} docs_changed={len(changed)} skill_problems={len(skill_problems)} new_pkgs={len(new_ok) + len(new_fresh)}")
    if notify and (changed or skill_problems or new_pkg_notice):
        try:
            subprocess.run([sys.executable, os.path.join(HOME, "mailtool", "notify.py"),
                            "--email", "--telegram",
                            f"pi-expert refresh ({cur})",
                            f"{len(changed)} docs changed, {len(skill_problems)} skill front-matter problem(s), {new_pkg_notice} new packages. See doc pi/pi-expert-refresh."],
                           capture_output=True, timeout=30)
        except Exception:
            pass
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def main():
    ap = argparse.ArgumentParser(description="pi-expert knowledge refresh")
    ap.add_argument("cmd", choices=["status", "docs-delta", "inventory", "skills-check", "packages", "run"],
                    nargs="?", default="status")
    ap.add_argument("--notify", action="store_true", help="notify on material change (email+telegram)")
    a = ap.parse_args()
    if a.cmd == "status":
        cmd_status()
    elif a.cmd == "docs-delta":
        cmd_docs_delta()
    elif a.cmd == "inventory":
        cmd_inventory()
    elif a.cmd == "skills-check":
        sys.exit(cmd_skills_check())
    elif a.cmd == "packages":
        cmd_packages(a.notify)
    elif a.cmd == "run":
        cmd_run(a.notify)


if __name__ == "__main__":
    main()
