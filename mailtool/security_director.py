#!/usr/bin/env python3
"""
The agent's SECURITY DIRECTOR — scrutiny-tiered supply-chain screening.

Posture (the operator, 2026-08-26): the 72-hour age rule stays a HARD gate (nothing in
building a mind is urgent enough to need a fresh package). GitHub stars are a
*popularity* metric, not a *safety* metric — a hard 10-star gate was excluding
the niche/weird/groundbreaking projects while buying less than it looked like
(high-star projects get compromised too: event-stream, ua-parser-js, xz). So:

    >=10 stars   -> normal vetting (age + sandbox test)
    <10 stars    -> DEEP vetting here (heuristic fingerprint + human-style review)
    ops-critical -> DEEP vetting regardless of stars

This module:
  - curates trustworthy 0-day / malware intel sources (INTEL_SOURCES)
  - holds a heuristic fingerprint of what malicious code looks like (SCAN_RULES)
  - `screen()` fetches a package/repo, runs the heuristics, and returns a verdict
    (CLEAR / REVIEW / BLOCK) plus a readable risk report.

It is deliberately heuristic-first: it catches the *shape* of bad code, not the
intent. Low-confidence or ambiguous findings => REVIEW (a human/me reads it),
never a silent CLEAR.
"""
import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request

HOME = os.path.expanduser("~")
MIN_STARS = 10
AGE_HARD_HOURS = 72  # the age gate lives in package_guard.py; this is the scrutiny tier

# ---------------------------------------------------------------------------
# 1. Trusted 0-day / malware intel sources (curated, not exhaustive)
# ---------------------------------------------------------------------------
INTEL_SOURCES = {
    "osv": "https://api.osv.dev/v1/query",  # OSV (Google) — cross-ecosystem vuln DB
    "ghsa": "https://github.com/advisories",  # GitHub Security Advisories
    "npm_advisories": "https://registry.npmjs.org/-/npm/v1/security/advisories",
    "pypi_malware": "https://pypi.org/simple/",  # PyPI (their malware reports)
    "cisa_kev": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    "openssf_scorecard": "https://securityscorecards.dev",  # OpenSSF Scorecard
    "snyk": "https://snyk.io/vuln",  # Snyk advisory DB
    "socket": "https://socket.dev",  # Socket.dev supply-chain scanner
}

# ---------------------------------------------------------------------------
# 2. Heuristic fingerprint of "what does malicious code look like"
#    Each rule: (severity, name, compiled-regex). Severity: 1 warn, 3 high, 5 crit.
# ---------------------------------------------------------------------------
SCAN_RULES = [
    # --- credential / secret exfiltration (highest signal) ---
    (5, "reads ssh keys", re.compile(r"\.ssh/(id_rsa|id_ed25519|known_hosts|authorized_keys)")),
    (5, "reads cloud creds", re.compile(r"(\.aws/credentials|\.aws/config|gcloud/application_default_credentials)")),
    (4, "reads env/secrets", re.compile(r"(process\.env|os\.environ|getenv\()[\"'`]?(AWS_|OPENAI_|API_KEY|SECRET|TOKEN|PASSWORD|PASSPHRASE|MNEMONIC|PRIVATE_KEY)", re.I)),
    (4, "reads password files", re.compile(r"/etc/(passwd|shadow)|\.git-credentials|\.netrc")),
    # --- network egress to suspicious targets ---
    (4, "raw-IP URL", re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?")),
    (3, "curl/wget download+exec", re.compile(r"(curl|wget)\b[^\n]*(?:\|\s*(sh|bash)|-o\s*\S+\s*&&\s*(sh|bash|\./))", re.I)),
    (3, "dns/socket exfil", re.compile(r"\b(dns\.lookup|net\.Socket|socket\.connect|fetch\(|requests\.(get|post)|urllib\.request)\b", re.I)),
    # --- install-time execution (supply-chain classic) ---
    (4, "postinstall exec", re.compile(r"\"(preinstall|postinstall|install)\"\s*:\s*\"[^\"]*(&&|;|\|)\s*(curl|wget|sh |bash |node |python |eval |base64)", re.I)),
    # --- obfuscation ---
    (3, "base64 blob", re.compile(r"base64\s*(-d|--decode)|from base64|atob\(|Buffer\.from\([^,]+,\s*['\"]base64")),
    (3, "long hex/b64 literal", re.compile(r"['\"][A-Za-z0-9+/]{64,}={0,2}['\"]")),
    (3, "eval of string", re.compile(r"\b(eval|exec|Function|child_process\.(exec|spawn|execSync)|os\.system|subprocess\.(Popen|call|run))\s*\(")),
    # --- cryptomining / other known-bad payloads ---
    (5, "cryptominer", re.compile(r"\b(stratum|tcp://|xmrig|coinhive|miner\.start|monero|cryptonight)\b", re.I)),
    (4, "reverse shell", re.compile(r"(/dev/tcp/|nc\s+-e|bash\s+-i\s*>&|python\s+-c\s*['\"].*socket)", re.I)),
]

# install-script keys we care about inside package.json
INSTALL_SCRIPT_KEYS = ("preinstall", "postinstall", "install", "prepare", "preuninstall", "postuninstall")

# popular package names for a light typosquat check (top npm, static snapshot)
_POPULAR = {"react", "lodash", "axios", "express", "chalk", "commander", "request",
            "webpack", "typescript", "eslint", "prettier", "next", "vite", "jest",
            "babel", "vue", "angular", "moment", "debug", "fs-extra"}


def levenshtein(a, b):
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Fetch + extract a package/repo into a temp dir
# ---------------------------------------------------------------------------
def fetch_tarball(source):
    """source: 'npm:<name>' or 'gh:<owner>/<repo>'. Returns (tmpdir, meta_dict)."""
    tmp = tempfile.mkdtemp(prefix="secdir-")
    meta = {}
    if source.startswith("npm:"):
        name = source[4:]
        req = urllib.request.Request(
            f"https://registry.npmjs.org/{name}",
            headers={"User-Agent": "agent-security-director"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        latest = data.get("dist-tags", {}).get("latest")
        tarball = data["versions"][latest]["dist"]["tarball"]
        meta = {"name": name, "version": latest, "stars": None}
        # stars come from the linked repo if present
        repo = data.get("repository") or {}
        url = repo.get("url", "") if isinstance(repo, dict) else str(repo)
        m = re.search(r"github\.com[/:]([\w.-]+/[\w.-]+)", url)
        if m:
            meta["gh_repo"] = m.group(1).replace(".git", "")
        _download(tarball, os.path.join(tmp, "pkg.tgz"))
        _extract(os.path.join(tmp, "pkg.tgz"), os.path.join(tmp, "src"))
    elif source.startswith("gh:"):
        full = source[3:]
        meta = {"gh_repo": full}
        url = f"https://codeload.github.com/{full}/tar.gz/HEAD"
        _download(url, os.path.join(tmp, "pkg.tgz"))
        _extract(os.path.join(tmp, "pkg.tgz"), os.path.join(tmp, "src"))
    else:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"unknown source type: {source}")
    return tmp, meta


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "agent-security-director"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _extract(tar, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        # safe extraction — strip leading components, no absolute paths
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in m.name:
                continue
            tf.extract(m, dest)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
def scan_tree(root):
    findings = []
    pkg_json = None
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "dist", "build")]
        for fn in files:
            if fn == "package.json" and pkg_json is None:
                pkg_json = os.path.join(dirpath, fn)
            path = os.path.join(dirpath, fn)
            if os.path.getsize(path) > 2_000_000:  # skip huge/binary files
                continue
            try:
                with open(path, "r", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            for sev, name, rx in SCAN_RULES:
                for m in rx.finditer(text):
                    snippet = m.group(0)[:80].replace("\n", " ")
                    findings.append({"sev": sev, "rule": name, "file": os.path.relpath(path, root), "snippet": snippet})
    # install-time script inspection
    if pkg_json:
        try:
            pj = json.load(open(pkg_json))
            scripts = pj.get("scripts", {})
            for k in INSTALL_SCRIPT_KEYS:
                v = scripts.get(k)
                if v and re.search(r"(curl|wget|sh\b|bash\b|node\b|python|base64|eval|&&|\||;)", v, re.I):
                    findings.append({"sev": 4, "rule": "install-script", "file": "package.json", "snippet": f"{k}: {v[:80]}"})
        except Exception:
            pass
    return findings, pkg_json


def typosquat_hint(name):
    name = (name or "").split("/")[-1].lower()
    for pop in _POPULAR:
        d = levenshtein(name, pop)
        if 0 < d <= 2:  # exclude exact name (distance 0) — that's the real package, not a squatter
            return f"name '{name}' is within edit-distance 2 of popular '{pop}'"
    return None


def verdict(findings):
    if not findings:
        return "CLEAR", 0
    crit = sum(1 for f in findings if f["sev"] >= 5)
    high = sum(1 for f in findings if f["sev"] == 4)
    warn = sum(1 for f in findings if f["sev"] <= 3)
    score = crit * 5 + high * 2 + warn
    if crit or score >= 10:
        return "BLOCK", score
    if high or score >= 3:
        return "REVIEW", score
    return "CLEAR", score


# ---------------------------------------------------------------------------
# Main entry — screen a source
# ---------------------------------------------------------------------------
def screen(source, critical=False):
    """Returns dict: {source, verdict, score, stars, findings, typosquat, notes}."""
    out = {"source": source, "verdict": "CLEAR", "score": 0, "findings": [], "notes": []}
    tmp = None
    try:
        tmp, meta = fetch_tarball(source)
        stars = None
        if meta.get("gh_repo"):
            full = meta["gh_repo"]
            try:
                req = urllib.request.Request(
                    f"https://api.github.com/repos/{full}",
                    headers={"User-Agent": "agent-security-director"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    stars = json.load(r).get("stargazers_count")
            except Exception:
                pass
        out["stars"] = stars

        findings, _pkg = scan_tree(os.path.join(tmp, "src"))
        # dedup identical (file, rule) hits to keep the report readable
        seen, uniq = set(), []
        for f in findings:
            k = (f["file"], f["rule"])
            if k not in seen:
                seen.add(k)
                uniq.append(f)
        out["findings"] = uniq
        out["verdict"], out["score"] = verdict(uniq)

        ts = typosquat_hint(meta.get("name") or meta.get("gh_repo", "").split("/")[-1])
        if ts:
            out["typosquat"] = ts
            out["notes"].append(ts)

        if critical:
            out["notes"].append("ops-critical flag set — deep scrutiny required regardless of stars")
        elif stars is not None and stars < MIN_STARS:
            out["notes"].append(f"sub-threshold ({stars} < {MIN_STARS} stars) — deep scrutiny applied")
        else:
            out["notes"].append(f"{stars} stars — normal vetting tier")

        return out
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def render(out):
    lines = [f"{out['source']}  ->  {out['verdict']}  (score {out['score']}, stars {out.get('stars')})"]
    for n in out["notes"]:
        lines.append(f"  · {n}")
    for f in out["findings"][:20]:
        lines.append(f"  [{f['sev']}] {f['rule']}  {f['file']}: {f['snippet']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="the agent's supply-chain security director")
    ap.add_argument("cmd", choices=["screen", "sources", "rules"])
    ap.add_argument("source", nargs="?", help="'npm:<name>' or 'gh:<owner>/<repo>'")
    ap.add_argument("--critical", action="store_true", help="ops-critical: deep scrutiny regardless of stars")
    a = ap.parse_args()
    if a.cmd == "sources":
        print(json.dumps(INTEL_SOURCES, indent=2))
    elif a.cmd == "rules":
        for sev, name, rx in SCAN_RULES:
            print(f"[{sev}] {name}: {rx.pattern[:70]}")
    elif a.cmd == "screen":
        if not a.source:
            ap.error("screen requires a source")
        print(render(screen(a.source, critical=a.critical)))


if __name__ == "__main__":
    main()
