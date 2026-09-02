#!/usr/bin/env python3
"""Hypothesis — test a claim empirically in the isolated Docker sandbox.

Takes a claim, asks a 'think' worker to design a minimal, self-contained,
falsifiable Python experiment, runs it in a throwaway, maximally-isolated
container (no network, read-only, capped memory/cpu/pids, no capabilities),
and reports PASS / FAIL / INCONCLUSIVE with the evidence.

This is the "simulation sandbox" from the operator's ideas (#6) — sharpen my world
model by constructing a toy that confirms or falsifies what I think I know,
then feed the result back into memory as evidence.

Safety model (blast-radius): the experiment is LLM-generated but runs with
  --network none  (no egress)
  --read-only + tmpfs /tmp  (no host writes)
  --memory 512m --cpus 1 --pids-limit 64  (bounded)
  --cap-drop ALL --security-opt no-new-privileges  (no privileges)
so even a hostile/loopy program can't reach out or hurt the host.

Usage:
  hypothesis.py "<claim>"
  hypothesis.py "<claim>" --remember   # also store the result as an evidence memory
  hypothesis.py "<claim>" --show-code
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dispatcher as D

IMAGE = "python:3.11-slim"


def _design(claim):
    prompt = (
        "You are a scientist testing this hypothesis:\n"
        f'"""{claim}"""\n\n'
        "Design the MINIMAL self-contained Python program that EMPIRICALLY tests "
        "it (compute, simulate, or measure — not assert). Rules:\n"
        "- Standard library only. NO network, NO subprocess, NO file reads/writes "
        "outside /tmp, NO infinite loops — finish in under 5 seconds.\n"
        "- End by printing EXACTLY one line: 'VERDICT: PASS' or 'VERDICT: FAIL' or "
        "'VERDICT: INCONCLUSIVE', followed by a short evidence clause.\n"
        "- INCONCLUSIVE only if the claim genuinely can't be tested in a sandbox "
        "(e.g. subjective or needs the real world).\n"
        "Output ONLY the Python code — no markdown fences, no commentary.\n"
    )
    out = D._deepseek(prompt, max_tokens=900, temperature=0.2)
    code = (out or "").strip()
    # strip ``` fences if the model added them anyway
    code = re.sub(r"^```[pP][yY][tT][hH][oO][nN]?\s*", "", code)
    code = re.sub(r"\s*```\s*$", "", code)
    return code


def _run(code, timeout=30):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    os.chmod(path, 0o644)
    try:
        r = subprocess.run(
            ["sudo", "docker", "run", "--rm",
             "--network", "none", "--read-only", "--tmpfs", "/tmp",
             "--memory", "512m", "--cpus", "1", "--pids-limit", "64",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
             "-v", f"{path}:/app/test.py:ro",
             IMAGE, "python", "/app/test.py"],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return None, "", "[experiment timed out]"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _verdict(out):
    m = re.search(r"VERDICT:\s*(PASS|FAIL|INCONCLUSIVE)\s*(.*)", out, re.I)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return None, ""


def run(claim, remember=False, show_code=False):
    print(f"=== HYPOTHESIS: {claim} ===\n")
    code = _design(claim)
    if show_code:
        print("--- experiment ---")
        print(code)
        print("-----------------")
    rc, out, err = _run(code)
    verdict, evidence = _verdict(out)
    if verdict is None:
        verdict = "INCONCLUSIVE"
        evidence = (out + " " + err).strip()[-200:]
    print(f"VERDICT: {verdict}")
    if evidence:
        print(f"evidence: {evidence}")
    if err and "timed out" in err:
        print(f"(stderr: {err[:120]})")
    if remember:
        text = (f"Hypothesis test ({time.strftime('%Y-%m-%d')}): \"{claim[:200]}\" "
                f"-> {verdict}. {evidence[:200]}")
        subprocess.run(
            [os.path.expanduser("~/venvs/memory/bin/python"),
             os.path.expanduser("~/memory/memstore.py"), "remember", text,
             "--kind", "evidence", "--importance", "0.5"],
            timeout=120)
        print("(stored as evidence memory)")
    return 0


def main():
    p = argparse.ArgumentParser(description="Empirically test a claim in the Docker sandbox")
    p.add_argument("claim")
    p.add_argument("--remember", action="store_true", help="store result as an evidence memory")
    p.add_argument("--show-code", action="store_true", help="print the generated experiment")
    a = p.parse_args()
    return run(a.claim, a.remember, a.show_code)


if __name__ == "__main__":
    sys.exit(main())
