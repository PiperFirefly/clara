#!/usr/bin/env python3
"""
CodeQL adversarial reviewer — Coding Cortex (the CodeQL item, made first-class).

Wraps the local CodeQL 2.26.4 install (+ the bundled python query packs at
/opt/codeql/qlpacks) so Agent can run the full security/correctness suite on any
repo and read the results as findings, not raw SARIF.

Confirmed working 2026-08-31:
  * `codeql database create --language=python` builds a program model (DB).
  * `codeql database analyze ... python-code-scanning.qls` runs 45+ security
    queries; catches e.g. SQL injection from a remote source.
  * IMPORTANT: default threat model is remote-facing, so local-only sources
    (sys.argv, plain function args) are NOT flagged unless --threat-model local
    (or you use a real remote source like request.args). That's correct behavior.

Usage:
  codeql_review.py <repo_dir> [--suite security-and-quality] [--threat-model X]
                   [--out DIR] [--format text|json]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

CODEQL = "/usr/local/bin/codeql"
PACKS = "/opt/codeql/qlpacks"
SUITES = {
    "code-scanning": f"{PACKS}/codeql/python-queries/1.8.9/codeql-suites/python-code-scanning.qls",
    "security-and-quality": f"{PACKS}/codeql/python-queries/1.8.9/codeql-suites/python-security-and-quality.qls",
    "security-extended": f"{PACKS}/codeql/python-queries/1.8.9/codeql-suites/python-security-extended.qls",
    "code-quality": f"{PACKS}/codeql/python-queries/1.8.9/codeql-suites/python-code-quality.qls",
    "lgtm": f"{PACKS}/codeql/python-queries/1.8.9/codeql-suites/python-lgtm.qls",
}


def review(repo, suite="code-scanning", threat_models=None, out_dir=None,
           timeout=900):
    """Build a CodeQL DB for repo, run the suite, return findings list."""
    repo = os.path.abspath(os.path.expanduser(repo))
    work = out_dir or tempfile.mkdtemp(prefix="codeql-review-")
    db = os.path.join(work, "db")
    sarif = os.path.join(work, "results.sarif")

    # 1. build DB
    r = subprocess.run(
        [CODEQL, "database", "create", db, "--language=python",
         "--source-root=" + repo, "--overwrite"],
        capture_output=True, text=True, timeout=timeout, check=False)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr[-2000:], "db": db}

    # 2. analyze
    cmd = [CODEQL, "database", "analyze", db, "--format=sarif-latest",
           "--output=" + sarif, "--search-path=" + PACKS, SUITES[suite]]
    if threat_models:
        for tm in threat_models:
            cmd.append("--threat-model=" + tm)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr[-2000:], "db": db}

    # 3. parse SARIF
    with open(sarif) as f:
        d = json.load(f)
    results = []
    for run in d.get("runs", []):
        for res in run.get("results", []):
            locs = res.get("locations", [])
            uri = ""
            line = None
            if locs:
                phys = locs[0].get("physicalLocation", {})
                uri = phys.get("artifactLocation", {}).get("uri", "")
                rg = phys.get("region", {})
                line = rg.get("startLine")
            results.append({
                "rule": res.get("ruleId", "?"),
                "severity": res.get("level", "?"),
                "message": res.get("message", {}).get("text", ""),
                "file": uri,
                "line": line,
            })
    return {"ok": True, "db": db, "sarif": sarif, "findings": results,
            "count": len(results)}


def render(res, suite):
    if not res["ok"]:
        return f"CodeQL review FAILED:\n{res['error']}"
    L = [f"CodeQL {suite} on repo — {res['count']} finding(s):", ""]
    sev = {"error": "!", "warning": "!", "note": "."}
    for f in sorted(res["findings"], key=lambda x: (x["file"] or "", x["line"] or 0)):
        loc = f"{f['file']}:{f['line']}" if f["line"] else (f["file"] or "?")
        L.append(f"  [{sev.get(f['severity'],'?')}] {f['rule']} @ {loc}")
        L.append(f"        {f['message'][:150]}")
    if not res["findings"]:
        L.append("  (no findings — clean under the default remote threat model)")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description="CodeQL adversarial reviewer")
    p.add_argument("repo", help="repo/source dir to analyze")
    p.add_argument("--suite", choices=list(SUITES), default="code-scanning")
    p.add_argument("--threat-model", nargs="*", default=None,
                   help="threat models to enable (e.g. local remote)")
    p.add_argument("--out", default=None, help="work dir (default tmp)")
    p.add_argument("--json", action="store_true", help="emit JSON findings")
    a = p.parse_args()
    res = review(a.repo, suite=a.suite, threat_models=a.threat_model,
                 out_dir=a.out)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(render(res, a.suite))
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
