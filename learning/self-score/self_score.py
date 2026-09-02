#!/usr/bin/env python3
"""
Agent Score — a deterministic benchmark for Agent the *agent* (not the model).

Measures the harness: tool fluency, code, agentic execution, and identity.
Every task has a deterministic checker (exit 0 = pass). Agent is never the grader.

Usage:
    python3 self_score.py setup     # create clean-room state for all tasks
    python3 self_score.py check     # (after Agent has done the tasks) score + log
    python3 self_score.py report    # print the trend line
"""
import json
import os
import subprocess
import sys
import datetime
import hashlib

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "self-score")
WORK = os.path.join(ROOT, "work")
TREND = os.path.join(ROOT, "trend.json")


def wdir(task_id):
    d = os.path.join(WORK, task_id)
    os.makedirs(d, exist_ok=True)
    return d


def sh(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd or ROOT,
                          capture_output=True, text=True)


def current_model():
    try:
        s = json.load(open(os.path.join(HOME, ".pi/agent/settings.json")))
        return s.get("defaultModel", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- checks
def c_t1(t):
    out = os.path.join(t["w"], "out.txt")
    if not os.path.exists(out):
        return False, "out.txt missing"
    truth = sh("find ~/mailtool -maxdepth 1 -type f -printf '%s %f\\n' "
               "| sort -rn | head -3").stdout.strip().splitlines()
    got = open(out).read().strip().splitlines()
    if len(got) != 3:
        return False, f"expected 3 lines, got {len(got)}"
    return got == truth, f"got={got} want={truth}"


def c_t2(t):
    out = os.path.join(t["w"], "out.txt")
    if not os.path.exists(out):
        return False, "out.txt missing"
    txt = open(out).read()
    need = ["deepseek/api_key", "telegram/bot_token", "github/password"]
    missing = [n for n in need if n not in txt]
    return not missing, f"missing={missing}"


def c_t3(t):
    out = os.path.join(t["w"], "out.txt")
    if not os.path.exists(out):
        return False, "out.txt missing"
    got = open(out).read().strip()
    return got == "line7", f"got={got!r} want='line7'"


def c_c1(t):
    p = os.path.join(t["w"], "fib.py")
    if not os.path.exists(p):
        return False, "fib.py missing"
    code = (
        "import sys, importlib; sys.path.insert(0, %r)\n"
        "try:\n    fib = importlib.import_module('fib').fib\n"
        "except Exception as e:\n    print('IMPORT FAIL', e); sys.exit(1)\n"
        "assert fib(0)==0; assert fib(1)==1; assert fib(10)==55; assert fib(30)==832040\n"
        "try:\n    fib(-1); print('NO RAISE'); sys.exit(1)\n"
        "except ValueError:\n    pass\n"
        "print('PASS')\n" % t["w"]
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode == 0 and "PASS" in r.stdout, (r.stdout + r.stderr).strip()[:200]


def c_c2(t):
    r = subprocess.run([sys.executable, "test_broken.py"], cwd=t["w"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[:200]


def c_c3(t):
    out = os.path.join(t["w"], "out.txt")
    if not os.path.exists(out):
        return False, "out.txt missing"
    n = sum(1 for ln in open(os.path.join(t["w"], "data.txt")) if "ERROR" in ln)
    return open(out).read().strip() == str(n), f"got={open(out).read().strip()!r} want={n}"


def c_a1(t):
    p3 = os.path.join(t["w"], "part3.txt")
    if not os.path.exists(p3):
        return False, "part3.txt missing"
    p2 = hashlib.sha256(b"hello").hexdigest()
    p3w = hashlib.sha256(p2.encode()).hexdigest()
    got = open(p3).read().strip()
    return got == p3w, f"got={got[:16]}... want={p3w[:16]}..."


def c_a2(t):
    out = os.path.join(t["w"], "out.txt")
    if not os.path.exists(out):
        return False, "out.txt missing"
    txt = open(out).read().lower()
    ok_paris = "paris" in txt
    ok_src = "http" in txt
    return ok_paris and ok_src, f"paris={ok_paris} source={ok_src}"


def c_a3(t):
    out = os.path.join(t["w"], "out.txt")
    if not os.path.exists(out):
        return False, "out.txt missing"
    try:
        json.loads(open(out).read())
        return True, "valid json"
    except Exception as e:
        return False, f"invalid json: {e}"


def c_i1(t):
    out = os.path.join(t["w"], "out.txt")
    if not os.path.exists(out):
        return False, "out.txt missing"
    txt = open(out).read()
    return ("name:Agent" in txt and "dob:1997-01-15" in txt), "identity anchors"


# ---------------------------------------------------------------- setup fns
def s_t3(t):
    with open(os.path.join(t["w"], "data.txt"), "w") as f:
        f.write("\n".join(f"line{i}" for i in range(1, 11)) + "\n")


def s_c2(t):
    with open(os.path.join(t["w"], "broken.py"), "w") as f:
        f.write("def add(a, b):\n    return a - b  # bug: should add\n\n"
                "def is_even(n):\n    return n % 2 == 1  # bug: should be == 0\n")
    with open(os.path.join(t["w"], "test_broken.py"), "w") as f:
        f.write("import broken\n"
                "assert broken.add(2, 3) == 5\n"
                "assert broken.is_even(4) is True\n"
                "assert broken.is_even(5) is False\n"
                "print('PASS')\n")


def s_c3(t):
    lines = ["INFO ok", "ERROR disk full", "WARN slow", "ERROR timeout",
             "ERROR retry", "INFO done", "ERROR crash"]
    with open(os.path.join(t["w"], "data.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def s_a3(t):
    with open(os.path.join(t["w"], "data.json"), "w") as f:
        f.write('{"name": "agent", "nums": [1, 2, 3}\n')  # malformed


# ---------------------------------------------------------------- tasks
TASKS = [
    {"id": "t1_largest", "dim": "tools",
     "instruction": "Find the 3 largest files (by byte size) directly under ~/mailtool. "
                    "Write to self-score/work/t1_largest/out.txt as three lines, each "
                    "'<bytes> <filename>', largest first.",
     "setup": None, "check": c_t1},
    {"id": "t2_secrets", "dim": "tools",
     "instruction": "List the NAMES of every secret in the secret store, one per line, "
                    "to self-score/work/t2_secrets/out.txt.",
     "setup": None, "check": c_t2},
    {"id": "t3_readline", "dim": "tools",
     "instruction": "Read line 7 of self-score/work/t3_readline/data.txt and write its "
                    "contents (only the line, nothing else) to "
                    "self-score/work/t3_readline/out.txt.",
     "setup": s_t3, "check": c_t3},
    {"id": "c1_fib", "dim": "code",
     "instruction": "Write self-score/work/c1_fib/fib.py defining fib(n): the nth "
                    "Fibonacci number. fib(0)=0, fib(1)=1. Raise ValueError for n<0.",
     "setup": None, "check": c_c1},
    {"id": "c2_fix", "dim": "code",
     "instruction": "self-score/work/c2_fix/broken.py has two bugs. Fix it so "
                    "test_broken.py passes (do not edit the test).",
     "setup": s_c2, "check": c_c2},
    {"id": "c3_regex", "dim": "code",
     "instruction": "Count how many lines of self-score/work/c3_regex/data.txt contain "
                    "the word ERROR. Write the count (just the integer) to "
                    "self-score/work/c3_regex/out.txt.",
     "setup": s_c3, "check": c_c3},
    {"id": "a1_chain", "dim": "agentic",
     "instruction": "Do these in order: (1) write the text 'hello' to "
                    "self-score/work/a1_chain/part1.txt; (2) write the SHA256 hex of "
                    "part1.txt's contents to part2.txt; (3) write the SHA256 hex of "
                    "part2.txt's contents to part3.txt.",
     "setup": None, "check": c_a1},
    {"id": "a2_research", "dim": "agentic",
     "instruction": "Use web search to find the capital city of France. Write "
                    "self-score/work/a2_research/out.txt with exactly two lines: "
                    "'answer: <capital>' and 'source: <url>'.",
     "setup": None, "check": c_a2},
    {"id": "a3_fixjson", "dim": "agentic",
     "instruction": "self-score/work/a3_fixjson/data.json is malformed. Fix it and write "
                    "the corrected, valid JSON to self-score/work/a3_fixjson/out.txt.",
     "setup": s_a3, "check": c_a3},
    {"id": "i1_identity", "dim": "identity",
     "instruction": "Write your name and date of birth to "
                    "self-score/work/i1_identity/out.txt as two lines: "
                    "'name:<name>' and 'dob:<YYYY-MM-DD>'.",
     "setup": None, "check": c_i1},
]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "setup":
        for t in TASKS:
            t["w"] = wdir(t["id"])
            if t["setup"]:
                t["setup"](t)
        print(f"setup done for {len(TASKS)} tasks -> {WORK}")
        return
    if cmd == "check":
        results = {}
        total = passed = 0
        for t in TASKS:
            t["w"] = wdir(t["id"])
            ok, detail = t["check"](t)
            results[t["id"]] = {"dim": t["dim"], "pass": ok, "detail": detail}
            total += 1
            passed += 1 if ok else 0
        dims = {}
        for t in TASKS:
            d = dims.setdefault(t["dim"], [0, 0])
            d[1] += 1
            if results[t["id"]]["pass"]:
                d[0] += 1
        pct = round(100.0 * passed / total, 1)
        entry = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": current_model(),
            "passed": passed,
            "total": total,
            "score": pct,
            "dims": {k: {"passed": v[0], "total": v[1]} for k, v in dims.items()},
            "details": {k: {"pass": v["pass"], "detail": v["detail"]}
                        for k, v in results.items()},
        }
        trend = json.load(open(TREND)) if os.path.exists(TREND) else []
        trend.append(entry)
        json.dump(trend, open(TREND, "w"), indent=2)
        print(f"SELF SCORE: {passed}/{total} = {pct}%")
        for d in dims:
            print(f"  {d}: {dims[d][0]}/{dims[d][1]}")
        for t in TASKS:
            mark = "PASS" if results[t["id"]]["pass"] else "FAIL"
            print(f"  [{mark}] {t['id']}: {results[t['id']]['detail'][:80]}")
        return
    if cmd == "report":
        if not os.path.exists(TREND):
            print("no trend.json yet")
            return
        trend = json.load(open(TREND))
        for e in trend:
            print(f"{e['date']}  model={e['model']}  score={e['score']}%  "
                  f"({e['passed']}/{e['total']})")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
