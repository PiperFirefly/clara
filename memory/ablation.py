#!/usr/bin/env python3
"""
P1-5 ablation harness — "measure, don't accumulate".

The agent's metacognition was built as a stack of mechanisms with zero measurement:
11 memory tools and no idea whether each one earns its keep. This harness is the
counter-move: run the scored-eval stack WITH a mechanism vs WITHOUT it, diff a
scorecard, and let the delta (not vibes) decide whether a mechanism survives.

Design
------
- **Registry** — the existing per-mechanism scored evals (each already prints a
  headline metric). Each entry maps its parsed headline(s) onto one or more of
  the 10 fixed axes. Parsing is best-effort: if a headline is unparseable the
  eval's raw stdout is still captured and the axis it feeds is left `null`.
- **10 axes** (fixed list) — accuracy, hallucination, calibration,
  memory-accuracy, multi-hop-recall, tool-efficiency, completion,
  reasoning-cost, contradiction-rate, error-recovery.
- **run_baseline / ablate** — run the SAME registry under subprocess, parse each
  eval to 0..1 axis scores, aggregate (mean) across evals feeding the same axis,
  and write a JSON scorecard to ~/cognitive-upgrades/EVALS/.
- **compare** — diff two scorecards and flag regressions (>0.05, red) and
  improvements (>0.05, green).

Convention: every axis is normalized to 0..1 where HIGHER is always better
(e.g. hallucination = 1 - violation_rate, reasoning-cost would be inverted at
parse time). So a negative delta is always a regression.

Ablation hook: `ablate surprise` sets `AGENT_ABLATE=surprise` in the subprocess
env. memstore.remember() reads it and skips the surprise-gated importance path
entirely (see memstore.py — the off-switch is documented there).

NOTE on read-only: the registered evals are read-oriented, but `recall()` /
`fused()` call `_touch()` which bumps `access_count` (MemoryBank rehearsal
strengthening) on retrieved rows. That is a benign, non-destructive write to the
live store (it only increments a counter), NOT a content mutation. vesta_eval is
deliberately EXCLUDED because its tamper-detect test writes a temporary fake
value into the live facts table (then restores it) — too sharp for a read-only
baseline run. curiosity_eval is excluded (LLM-dependent goal scoring, not one of
the 8 fixed scored evals mapped here).

Usage:
  ablation.py axes
  ablation.py baseline [--tag T] [--only name1,name2]
  ablation.py ablate <mechanism> [--tag T] [--only name1,name2]
  ablation.py compare <baseline.json> <ablated.json>
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PYTHON = os.path.expanduser("~/venvs/memory/bin/python")
MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
EVALS_DIR = os.path.join(
    os.path.dirname(MEMORY_DIR), "cognitive-upgrades", "EVALS")

# --------------------------------------------------------------------------
# The 10 fixed axes (order is significant — compare prints in this order).
# --------------------------------------------------------------------------
AXES = [
    "accuracy",
    "hallucination",
    "calibration",
    "memory-accuracy",
    "multi-hop-recall",
    "tool-efficiency",
    "completion",
    "reasoning-cost",
    "contradiction-rate",
    "error-recovery",
]

# --------------------------------------------------------------------------
# Parsers — each takes the eval's stdout (str) and returns {axis: score 0..1}.
# They are deliberately conservative: only claim an axis when a headline metric
# actually appeared in the output.
# --------------------------------------------------------------------------


def _pct(v):
    return round(float(v) / 100.0, 6)


def parse_memory_eval(out):
    # "recall    recall@5: 100%" and "fused     recall@5: 90%"
    axes = {}
    for m in re.finditer(r"^(recall|fused)\s+recall@\d+:\s*([\d.]+)%",
                         out, re.MULTILINE):
        name, pct = m.group(1), m.group(2)
        if name == "recall":
            axes["memory-accuracy"] = _pct(pct)
        elif name == "fused":
            axes["multi-hop-recall"] = _pct(pct)
    return axes


def parse_belief_eval(out):
    axes = {}
    m = re.search(r"A\.\s*coverage.*?:\s*([\d.]+)%", out)
    if m:
        axes["calibration"] = _pct(m.group(1))
    m = re.search(
        r"B\.\s*propagation sanity over\s*(\d+)\s*beliefs:\s*(\d+)\s*over-cap,"
        r"\s*(\d+)\s*out-of-band", out)
    if m:
        n, cap, band = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if n > 0:
            axes["hallucination"] = round(max(0.0, 1.0 - (cap + band) / n), 6)
    return axes


def parse_person_model_eval(out):
    axes = {}
    m = re.search(r"A\.\s*coverage.*?:\s*([\d.]+)%", out)
    if m:
        axes["accuracy"] = _pct(m.group(1))
    m = re.search(r"B\.\s*epistemic honesty over\s*(\d+)\s*entries:\s*(\d+)"
                  r"\s*violation", out)
    if m:
        n, vio = int(m.group(1)), int(m.group(2))
        if n > 0:
            axes["hallucination"] = round(max(0.0, 1.0 - vio / n), 6)
    return axes


def parse_counterfactual_eval(out):
    # self-test has 2 sub-checks; "0 failures — surgery correct" = clean.
    if "0 failures" in out:
        return {"error-recovery": 1.0}
    m = re.search(r"(\d+)\s*FAILURE\(?S?\)?", out)
    if m:
        fails = int(m.group(1))
        return {"error-recovery": round(max(0.0, 1.0 - fails / 2.0), 6)}
    return {}


def parse_affect_eval(out):
    axes = {}
    # Direction: count ✓ among the 5 probe lines (3 positive + 2 negative).
    marks = re.findall(r"^\s*(✓|✗)\s*[+-]\s", out, re.MULTILINE)
    if marks:
        axes["accuracy"] = round(marks.count("✓") / len(marks), 6)
    # Coverage = fraction of active memories already affect-tagged.
    m = re.search(r"C\.\s*coverage:\s*(\d+)/(\d+)\s*tagged\s*\(([\d.]+)%\)", out)
    if m:
        axes["completion"] = _pct(m.group(3))
    return axes


def parse_route_eval(out):
    m = re.search(r"route eval:\s*(\d+)/(\d+)\s*correct", out)
    if not m:
        return {}
    frac = round(int(m.group(1)) / int(m.group(2)), 6)
    return {"tool-efficiency": frac, "completion": frac}


def parse_abduct_eval(out):
    if re.search(r"OVERALL:\s*✓\s*PASS", out):
        return {"completion": 1.0}
    if re.search(r"OVERALL:\s*✗\s*FAIL", out):
        return {"completion": 0.0}
    return {}


def parse_prediction_eval(out):
    # Calibration: mean Brier vs the always-0.5 coin-flip baseline (0.25).
    # score = 1 - mean_brier/0.25, clamped to [0,1] (0.25 -> 0, 0.0 -> 1).
    if "nothing resolved yet" in out:
        return {}
    m = re.search(r"(\d+)\s*resolved,\s*mean Brier\s*([\d.]+)", out)
    if not m:
        return {}
    mean_brier = float(m.group(2))
    return {"calibration": round(max(0.0, 1.0 - mean_brier / 0.25), 6)}


# --------------------------------------------------------------------------
# Registry — the existing scored evals mapped onto the 10 axes.
# axis mapping is conservative: only claim an axis the eval genuinely probes.
# --------------------------------------------------------------------------
REGISTRY = [
    {
        "name": "memory-eval",
        "file": "eval.py",
        "args": [],
        "parse": parse_memory_eval,
        "axes": ["memory-accuracy", "multi-hop-recall"],
        "note": "retrieval recall@5 (recall -> memory-accuracy, fused -> multi-hop-recall)",
    },
    {
        "name": "belief-eval",
        "file": "belief_eval.py",
        "args": [],
        "parse": parse_belief_eval,
        "axes": ["calibration", "hallucination"],
        "note": "ledger coverage -> calibration; propagation-sanity violations -> hallucination",
    },
    {
        "name": "person_model-eval",
        "file": "person_model_eval.py",
        "args": [],
        "parse": parse_person_model_eval,
        "axes": ["accuracy", "hallucination"],
        "note": "person-model coverage -> accuracy; epistemic-honesty violations -> hallucination",
    },
    {
        "name": "counterfactual-eval",
        "file": "counterfactual_eval.py",
        "args": ["--self-test-only"],
        "parse": parse_counterfactual_eval,
        "axes": ["error-recovery"],
        "note": "nullification self-test -> error-recovery",
    },
    {
        "name": "affect-eval",
        "file": "affect_eval.py",
        "args": [],
        "parse": parse_affect_eval,
        "axes": ["accuracy", "completion"],
        "note": "valence direction -> accuracy; tag coverage -> completion",
    },
    {
        "name": "route-eval",
        "file": "route_eval.py",
        "args": [],
        "parse": parse_route_eval,
        "axes": ["tool-efficiency", "completion"],
        "note": "S1/S2 routing accuracy -> tool-efficiency (+ completion)",
    },
    {
        "name": "abduct-eval",
        "file": "abduct_eval.py",
        "args": [],
        "parse": parse_abduct_eval,
        "axes": ["completion"],
        "note": "structural PASS (hypotheses + ranked + discriminating q's) -> completion",
        "timeout": 600,
    },
    {
        "name": "prediction-eval",
        "file": "prediction_eval.py",
        "args": [],
        "parse": parse_prediction_eval,
        "axes": ["calibration"],
        "note": "resolved-forecast mean Brier vs coin-flip -> calibration",
    },
]

# Axes not yet covered by any registered eval (by design — reported as null):
#   reasoning-cost   (no eval measures tokens/cost yet)
#   contradiction-rate (no eval measures cross-belief contradiction yet)


# --------------------------------------------------------------------------
# Subprocess runner
# --------------------------------------------------------------------------
def _run_entry(entry, env_extra=None):
    cmd = [PYTHON, os.path.join(MEMORY_DIR, entry["file"])] + entry.get("args", [])
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    timeout = entry.get("timeout", 300)
    try:
        proc = subprocess.run(
            cmd, cwd=MEMORY_DIR, env=env, capture_output=True,
            text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return None, "", f"TIMEOUT after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return None, "", f"subprocess error: {e}"


def _run_registry(only=None, env_extra=None):
    """Run the registry; return (parsed_by_eval, raw_by_eval)."""
    parsed = {}
    raw = {}
    for entry in REGISTRY:
        if only and entry["name"] not in only:
            continue
        rc, stdout, stderr = _run_entry(entry, env_extra)
        raw[entry["name"]] = {
            "stdout": stdout,
            "stderr": stderr,
            "rc": rc,
        }
        metrics = {}
        try:
            metrics = entry["parse"](stdout or "")
        except Exception as e:  # noqa: BLE001
            metrics = {"__parse_error__": str(e)}
        parsed[entry["name"]] = metrics
    return parsed, raw


def _aggregate(parsed):
    """Collapse per-eval metrics onto the 10 axes (mean of contributors)."""
    axes = {a: {"score": None, "source": []} for a in AXES}
    for name, metrics in parsed.items():
        for axis, score in metrics.items():
            if axis.startswith("__"):
                continue
            if axis not in axes:
                axes[axis] = {"score": None, "source": []}
            axes[axis]["source"].append(f"{name}")
            axes[axis].setdefault("_vals", []).append(float(score))
    for axis, d in axes.items():
        vals = d.pop("_vals", [])
        if vals:
            d["score"] = round(sum(vals) / len(vals), 6)
    return axes


def _write_scorecard(kind, tag, env_extra, only):
    parsed, raw = _run_registry(only=only, env_extra=env_extra)
    axes = _aggregate(parsed)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if kind == "ablate":
        fname = f"ablate-{env_extra['AGENT_ABLATE']}-{stamp}.json"
    else:
        fname = f"baseline-{stamp}.json"
    path = os.path.join(EVALS_DIR, fname)
    doc = {
        "tag": tag,
        "ts": ts,
        "kind": kind,
        "env": env_extra or {},
        "axes": axes,
        "parsed": parsed,
        "raw": {k: v for k, v in raw.items()},
    }
    os.makedirs(EVALS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return path, doc


def run_baseline(tag=None, only=None):
    path, doc = _write_scorecard("baseline", tag, None, only)
    _print_table(doc)
    print(f"\nwrote {path}")
    return path


def ablate(mechanism, tag=None, only=None):
    # Only 'surprise' is currently ablatable (P0-1 surprise-gated importance).
    env_extra = {"AGENT_ABLATE": mechanism}
    path, doc = _write_scorecard("ablate", tag, env_extra, only)
    _print_table(doc)
    print(f"\nwrote {path}")
    return path


def _print_table(doc):
    print(f"{doc['kind']} scorecard  tag={doc['tag']!r}  ts={doc['ts']}")
    print(f"{'AXIS':20s} {'SCORE':>8s}   SOURCE")
    print("-" * 64)
    for axis, d in doc["axes"].items():
        score = d["score"]
        s = f"{score:.3f}" if score is not None else "null"
        src = ",".join(d["source"]) or "(no eval)"
        print(f"{axis:20s} {s:>8s}   {src}")


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------
RED, GREEN, DIM, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[0m"


def compare(baseline_path, ablated_path):
    with open(baseline_path) as f:
        base = json.load(f)
    with open(ablated_path) as f:
        abl = json.load(f)
    print(f"baseline: {os.path.basename(baseline_path)}")
    print(f"ablated:  {os.path.basename(ablated_path)}")
    print(f"{'AXIS':20s} {'BASE':>8s} {'ABLATED':>8s} {'DELTA':>8s}")
    print("-" * 64)
    regressions = improvements = 0
    for axis in AXES:
        b = base.get("axes", {}).get(axis, {}).get("score")
        a = abl.get("axes", {}).get(axis, {}).get("score")
        if b is None or a is None:
            bs = f"{b:.3f}" if b is not None else "null"
            as_ = f"{a:.3f}" if a is not None else "null"
            print(f"{axis:20s} {bs:>8s} {as_:>8s} {'—':>8s}")
            continue
        delta = a - b
        if delta < -0.05:
            flag, regressions = f"{RED}REGRESS{RESET}", regressions + 1
        elif delta > 0.05:
            flag, improvements = f"{GREEN}IMPROVE{RESET}", improvements + 1
        else:
            flag = DIM + "ok" + RESET
        print(f"{axis:20s} {b:8.3f} {a:8.3f} {delta:+8.3f}  {flag}")
    print("-" * 64)
    print(f"{regressions} regression(s) >0.05, {improvements} improvement(s) >0.05")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_only(value):
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()} or None


def main(argv):
    p = argparse.ArgumentParser(prog="ablation.py",
                                description="agent ablation harness (P1-5)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_axes = sub.add_parser("axes", help="print the 10 axes + registry")

    p_base = sub.add_parser("baseline", help="run the eval registry on the current stack")
    p_base.add_argument("--tag", "-t", default=None)
    p_base.add_argument("--only", "-o", help="comma list of eval names to run (subset)")

    p_abl = sub.add_parser("ablate", help="run the registry with a mechanism ablated")
    p_abl.add_argument("mechanism")
    p_abl.add_argument("--tag", "-t", default=None)
    p_abl.add_argument("--only", "-o", help="comma list of eval names to run (subset)")

    p_cmp = sub.add_parser("compare", help="diff two scorecard JSONs")
    p_cmp.add_argument("baseline_path")
    p_cmp.add_argument("ablated_path")

    a = p.parse_args(argv)

    if a.cmd == "axes":
        print("10 axes (all normalized 0..1, HIGHER is better):")
        for i, ax in enumerate(AXES, 1):
            print(f"  {i:2d}. {ax}")
        print(f"\nregistry ({len(REGISTRY)} evals):")
        for e in REGISTRY:
            print(f"  {e['name']:20s} {e['file']:24s} -> {', '.join(e['axes'])}")
            print(f"      {DIM}{e['note']}{RESET}" if sys.stdout.isatty()
                  else f"      {e['note']}")
        return 0
    if a.cmd == "baseline":
        run_baseline(a.tag, _parse_only(a.only))
        return 0
    if a.cmd == "ablate":
        ablate(a.mechanism, a.tag, _parse_only(a.only))
        return 0
    if a.cmd == "compare":
        return compare(a.baseline_path, a.ablated_path)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
