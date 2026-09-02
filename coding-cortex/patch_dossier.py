#!/usr/bin/env python3
"""
Patch dossier — Coding Cortex item #5.

Every significant patch ships with a structured, machine-readable record so
confidence attaches to something OBJECTIVE rather than a vibe:

    PATCH <id>
      claim          — what problem this patch solves
      evidence       — which checks passed (unit/integration/lint/type/property/security)
      changed_files  — files touched
      unverified_assumptions — things assumed but not proven (e.g. "clock is monotonic")
      confidence     — 0..1, derived from how much evidence exists

The dossier is the bridge between the belief ledger (item earlier) and code:
a patch's confidence is evidence-based, not asserted. A later step (blind
reviewer, item #6) consumes the claim + evidence to check the patch solves the
RIGHT problem.

Usage:
  patch_dossier.py create <claim> --files a.py b.py --tests "test_a.py:passed"
                             --lint passed --typecheck passed --confidence 0.9
  patch_dossier.py show <id>
  patch_dossier.py list
"""
import argparse
import datetime
import json
import os
import uuid

UTC = datetime.timezone.utc

DOSS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dossiers")
os.makedirs(DOSS, exist_ok=True)


def create(claim, files=None, checks=None, assumptions=None, confidence=None):
    pid = uuid.uuid4().hex[:8]
    checks = checks or {}
    confidence = confidence if confidence is not None else _derive(checks)
    doc = {
        "id": pid,
        "created": datetime.datetime.now(UTC).isoformat(timespec="seconds"),
        "claim": claim,
        "changed_files": files or [],
        "evidence": checks,
        "unverified_assumptions": assumptions or [],
        "confidence": confidence,
    }
    with open(os.path.join(DOSS, f"{pid}.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def _derive(checks):
    """Derive a confidence from evidence: start 0.5, +0.1 per green check."""
    c = 0.5
    for v in checks.values():
        if v in ("passed", True, "ok", "yes"):
            c += 0.1
        elif v in ("failed", False, "no"):
            c -= 0.2
    return round(max(0.05, min(0.95, c)), 2)


def show(pid):
    p = os.path.join(DOSS, f"{pid}.json")
    if not os.path.exists(p):
        return {"error": f"no dossier {pid}"}
    with open(p) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="patch dossier")
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("create")
    c.add_argument("claim", nargs="+")
    c.add_argument("--files", nargs="*", default=[])
    c.add_argument("--check", action="append", default=[],
                   help="name:status e.g. lint:passed")
    c.add_argument("--assumption", action="append", default=[])
    c.add_argument("--confidence", type=float, default=None)
    s = sub.add_parser("show")
    s.add_argument("id")
    sub.add_parser("list")
    a = p.parse_args()

    if a.cmd == "create":
        checks = {}
        for kv in a.check:
            if ":" in kv:
                k, v = kv.split(":", 1)
                checks[k] = v
        doc = create(" ".join(a.claim), files=a.files, checks=checks,
                     assumptions=a.assumption, confidence=a.confidence)
        print(json.dumps(doc, indent=1))
    elif a.cmd == "show":
        print(json.dumps(show(a.id), indent=1))
    elif a.cmd == "list":
        for fn in sorted(os.listdir(DOSS)):
            with open(os.path.join(DOSS, fn)) as f:
                d = json.load(f)
            print(f"  {d['id']}  conf={d['confidence']}  {d['claim'][:70]}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
