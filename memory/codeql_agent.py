#!/usr/bin/env python3
"""CodeQL integration — adversarial reviewer / program-model query (item #11).

CodeQL is now licensed (GitHub Advanced Security) and installed. This is NOT
just a post-hoc scanner: the goal is to QUERY the program model (data flow,
taint, security) WHILE deciding how to make a change.

What it does:
  --build   : build CodeQL databases for the target repos (per-language).
  --analyze : run the standard security query suite (code-scanning) over a
              repo, output findings.
  --flow    : run a CUSTOM data-flow query — 'does value flow from source X
              to sink Y in this module?' — the adversarial-review use case.
              Write a QL dataflow query and point it at a DB.
  --query   : run an arbitrary .ql file against a DB.

Database locations: ~/.codeql/dbs/<repo>-<lang>. Build is idempotent-ish and
cheap for this tree. Databases are ~10-50MB each.
"""
import json
import os
import subprocess

CODEQL = "/usr/local/bin/codeql"
DBS_DIR = os.path.expanduser("~/.codeql/dbs")
REPOS = {
    "mailtool": os.path.expanduser("~/mailtool"),
    "memory": os.path.expanduser("~/memory"),
    "coding-cortex": os.path.expanduser("~/coding-cortex"),
}


def _run(args, timeout=600):
    r = subprocess.run([CODEQL] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"codeql {' '.join(args[:2])} failed rc={r.returncode}:\n"
                           f"{r.stderr[-2000:]}")
    return r.stdout


def build(repo, lang="python"):
    """Build a CodeQL database for a repo (default python)."""
    if repo not in REPOS:
        return {"ok": False, "error": f"unknown repo {repo!r}; known: {list(REPOS)}"}
    src = REPOS[repo]
    db = os.path.join(DBS_DIR, f"{repo}-{lang}")
    os.makedirs(DBS_DIR, exist_ok=True)
    # fresh build: remove stale db (CodeQL refuses to overwrite non-empty)
    if os.path.isdir(db):
        import shutil
        shutil.rmtree(db)
    _run(["database", "create", db, "--language", lang, "--source-root", src],
         timeout=900)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(db) for f in fs) // (1024 * 1024)
    return {"ok": True, "repo": repo, "db": db, "size_mb": size}


def analyze(repo, lang="python"):
    """Run the code-scanning security suite over a repo's DB."""
    db = os.path.join(DBS_DIR, f"{repo}-{lang}")
    if not os.path.isdir(db):
        return {"ok": False, "error": f"no DB for {repo}; run --build {repo} first"}
    sarif = os.path.join(DBS_DIR, f"{repo}-{lang}.sarif")
    _run(["database", "analyze", db, "--format=sarif-latest",
          "--output", sarif,
          "codeql/python-queries:codeql-suites/python-code-scanning.qls"],
         timeout=1800)
    # parse SARIF -> concise findings
    findings = _parse_sarif(sarif)
    return {"ok": True, "repo": repo, "sarif": sarif, "findings": findings}


def _parse_sarif(path):
    """Reduce a SARIF report to a compact findings list."""
    try:
        data = json.load(open(path))
    except Exception:
        return []
    out = []
    seen = set()
    for run in data.get("runs", []):
        for res in run.get("results", []):
            rule = res.get("ruleId", "?")
            msg = res.get("message", {}).get("text", "")
            locs = res.get("locations", [])
            path = locs[0].get("physicalLocation", {}).get("artifactLocation",
                                                           {}).get("uri", "") if locs else ""
            line = locs[0].get("physicalLocation", {}).get("region", {}).get("startLine") if locs else None
            key = (rule, path, line, msg[:60])
            if key in seen:
                continue
            seen.add(key)
            out.append({"rule": rule, "path": path, "line": line,
                        "severity": res.get("level", ""), "message": msg[:200]})
    return out


CUSTOM_PACK = os.path.expanduser("~/.codeql/custompack")
FLOW_QL_TEMPLATE = os.path.join(CUSTOM_PACK, "queries", "generic_flow.ql")


def _run_flow_query(db, source_pat, sink_pat, out_bqrs="/tmp/codeql_flow.bqrs"):
    """Run the generic taint-flow query with substituted SOURCE_PAT/SINK_PAT.
    The temp .ql must live INSIDE the custom pack so it resolves the pack's
    library context (DataFlow/AgentFlow). Returns list of flow tuples."""
    ql = open(FLOW_QL_TEMPLATE).read()
    ql = ql.replace("%SOURCE_PAT%", source_pat).replace("%SINK_PAT%", sink_pat)
    tmp = os.path.join(CUSTOM_PACK, "queries", "_agent_flow_run.ql")
    open(tmp, "w").write(ql)
    try:
        _run(["query", "run", "--database", db, "-o", out_bqrs, tmp],
             timeout=600)
    finally:
        os.remove(tmp)
    out = _run(["bqrs", "decode", "--format=json", out_bqrs], timeout=60)
    data = json.loads(out)
    return data.get("#select", {}).get("tuples", [])


def run_flow(repo, source, sink, lang="python"):
    """Answer 'does tainted value matching SOURCE reach a SINK node?' across
    the repo's program model — the adversarial-review primitive for deciding
    a change. source/sink are QL wildcard patterns (e.g. 'secret', '%log%')."""
    db = os.path.join(DBS_DIR, f"{repo}-{lang}")
    if not os.path.isdir(db):
        return {"ok": False, "error": f"no DB for {repo}; run --build {repo} first"}
    flows = _run_flow_query(db, source, sink)
    return {"ok": True, "repo": repo, "source_pat": source, "sink_pat": sink,
            "flows_found": len(flows), "flows": flows[:20]}


def list_dbs():
    if not os.path.isdir(DBS_DIR):
        return []
    return [d for d in os.listdir(DBS_DIR) if os.path.isdir(os.path.join(DBS_DIR, d))]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="repo to build a DB for (mailtool|memory|coding-cortex)")
    ap.add_argument("--analyze", help="repo to run security suite on")
    ap.add_argument("--flow", nargs=2, metavar=("SOURCE", "SINK"),
                    help="data-flow query: does SOURCE flow to SINK")
    ap.add_argument("--repo", default="mailtool", help="repo for --flow")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print(json.dumps({"dbs": list_dbs()}, indent=2))
    if a.build:
        print(json.dumps(build(a.build, a.lang), indent=2))
    if a.analyze:
        print(json.dumps(analyze(a.analyze, a.lang), indent=2)[:3000])
    if a.flow:
        print(json.dumps(run_flow(a.repo, a.flow[0], a.flow[1], a.lang), indent=2))
