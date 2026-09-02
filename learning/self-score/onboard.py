#!/usr/bin/env python3
"""
onboard — test a newly-dropped model against the frozen screen-v1 bank, compute
its cost and workload-weighted composite, and emit a human decision sheet:
"switch vs stay" on the axes that actually matter (correctness, latency, $/day).

The decision is ALWAYS a recommendation to a human — never an auto-flip. If the
model is a serious contender, the deeper move is to run it as a shadow default
on real traffic before committing; the bank predicts, the real distribution decides.

Commands:
  onboard.py providers                          # list providers/models + cost
  onboard.py smoke <provider> <model>           # 1 cheap call, connectivity check
  onboard.py screen <provider> <model> [--n N]  # cheap subset (N atomic+N chained)
  onboard.py full <provider> <model>            # full 40-task screen-v1 + decision sheet
  onboard.py report                             # table of all logged runs
  onboard.py compare <modelA> <modelB>          # decision sheet between two logged runs

Workload profile: how Agent's real day splits between mechanical (atomic) and
multi-step (chained). Chained is weighted higher because real work (freeroam,
tool orchestration, email/tg triage, cognitive subsystems) is chained-heavy and
the engine-matrix found chained is the TRUE discriminator between engines.
Refine WORKLOAD + DAILY_VOLUME below from real logquery traffic as desired.

Results append to the same engine_matrix-results.jsonl that model_registry.py
ingests, so `model_registry.py ingest && list` picks them up automatically.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "learning", "self-score")
sys.path.insert(0, ROOT)
import providers as P  # noqa: E402

BANK = os.path.join(ROOT, "banks", "screen-v1.json")
RESULTS = os.path.join(ROOT, "work", "engine_matrix-results.jsonl")

# --- workload model (editable; refine from logquery if you want fidelity) ----
# fraction of real daily calls that are mechanical vs multi-step
WORKLOAD = {"atomic": 0.30, "chained": 0.70}
# estimated real daily token volume (active agent doing freeroam/triage/tools)
DAILY_VOLUME = {"prompt": 1_500_000, "completion": 300_000}

# the current default we're deciding whether to keep
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_THINKING = "low"

CHAINED_EPS = 3  # points of chained% regression we refuse to accept


# ----------------------------------------------------------------------------

def _grade(model_answer, t):
    return hashlib.sha256(model_answer.encode()).hexdigest() == t["answer_sha256"]


def _run_bank(provider, model, thinking, n=None):
    """Run up to n atomic + n chained tasks (n=None => all). Returns record."""
    bank = json.load(open(BANK))["tasks"]
    if n:
        atomic = [t for t in bank if t["kind"] == "atomic"][:n]
        chained = [t for t in bank if t["kind"] == "chained"][:n]
        tasks = atomic + chained
    else:
        tasks = bank
    at = sum(1 for t in tasks if t["kind"] == "atomic")
    ch = sum(1 for t in tasks if t["kind"] == "chained")
    at_ok = ch_ok = 0
    ptok = ctok = reasoning_hits = 0
    t0 = time.time()
    for t in tasks:
        ok = False
        try:
            ans, reas, pt, ct = P.ask(provider, model, t["instruction"], thinking)
            ok = _grade(ans, t)
            if t["kind"] == "atomic":
                at_ok += 1 if ok else 0
            else:
                ch_ok += 1 if ok else 0
            ptok += pt
            ctok += ct
            reasoning_hits += 1 if reas else 0
        except Exception as e:
            print(f"    [err] {t['id']}: {str(e)[:80]}", file=sys.stderr)
        time.sleep(0.3)  # human cadence, no rapid-fire
    ap = round(100 * at_ok / at) if at else 0
    cp = round(100 * ch_ok / ch) if ch else 0
    cost = P.cost_usd(provider, model, ptok, ctok)
    composite = round(WORKLOAD["atomic"] * ap + WORKLOAD["chained"] * cp, 1)
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider, "model": model, "thinking": thinking,
        "mode": "screen" if n else "full",
        "atomic_ok": at_ok, "atomic_total": at, "atomic_pct": ap,
        "chained_ok": ch_ok, "chained_total": ch, "chained_pct": cp,
        "composite": composite,
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "reasoning_hits": reasoning_hits,
        "cost_usd": round(cost, 4), "bank": os.path.basename(BANK),
        "sec": round(time.time() - t0, 1),
    }
    print(f"  {model:22s} thinking={thinking:4s}  atomic {at_ok}/{at} ({ap}%)  "
          f"chained {ch_ok}/{ch} ({cp}%)  composite={composite}  "
          f"${cost:.4f}  {rec['sec']}s")
    return rec


def _save(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _load_results():
    if not os.path.exists(RESULTS):
        return []
    out = []
    for ln in open(RESULTS):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def _last(model, thinking=None, mode=None):
    """Most recent logged run for a model (optionally specific thinking/mode)."""
    rows = [r for r in _load_results() if r.get("model") == model]
    if thinking:
        rows = [r for r in rows if r.get("thinking") == thinking]
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]
    return rows[-1] if rows else None


def _day_cost(rec):
    """Project a run's token cost to an estimated real $/day."""
    if not rec:
        return None
    prompt = DAILY_VOLUME["prompt"] / 1e6
    comp = DAILY_VOLUME["completion"] / 1e6
    # scale the run's observed mix onto the daily volume
    tot_tok = (rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)
    if tot_tok <= 0:
        return None
    in_frac = (rec.get("prompt_tokens") or 0) / tot_tok
    out_frac = (rec.get("completion_tokens") or 0) / tot_tok
    price = _price_for(rec)
    return round(prompt * in_frac * price["in"] + comp * out_frac * price["out"], 2)


def _price_for(rec):
    m = P.PROVIDERS[rec.get("provider", "deepseek")]["models"].get(
        rec.get("model", ""), {"price_in": 0, "price_out": 0})
    return {"in": m["price_in"], "out": m["price_out"]}


def _decision(cand):
    """Recommend switch vs stay. cand = a full-run record."""
    base = _last(DEFAULT_MODEL, DEFAULT_THINKING, mode="full") or _last(DEFAULT_MODEL, DEFAULT_THINKING)
    if not base:
        return ("?", "no baseline for default %s %s yet — run it first"
                % (DEFAULT_MODEL, DEFAULT_THINKING))
    cb, cc = base["chained_pct"], cand["chained_pct"]
    wb, wc = base.get("composite", 0), cand.get("composite", 0)
    db, dc = _day_cost(base), _day_cost(cand)
    if cc < cb - CHAINED_EPS:
        return ("NO", f"regresses chained reasoning: {cb}% -> {cc}% (refuse to lose "
                      f">{CHAINED_EPS}pt on the true discriminator)")
    if wc >= wb and dc is not None and db is not None and dc <= 1.5 * db:
        return ("YES", f"workload composite {wb} -> {wc} at similar cost "
                       f"(${db}/day -> ${dc}/day)")
    if wc >= wb:
        return ("MAYBE", f"gains {wc - wb:+g}pt composite but cost ${db}/day -> "
                         f"${dc}/day (+{round(100*(dc/db-1))}%); switch only if "
                         f"correctness is worth that price")
    return ("NO", f"worse on workload composite ({wb} -> {wc}) — not an upgrade")


def _line(r):
    d = _day_cost(r)
    cost = f"${d}/day" if d is not None else "-"
    return (f"  {r['ts']}  {r.get('model','?'):22s} {r.get('thinking','?'):4s} "
            f"atomic {r.get('atomic_pct',0)}% chained {r.get('chained_pct',0)}% "
            f"comp={r.get('composite',0)}  {cost}  {r.get('sec',0)}s")


# ----------------------------------------------------------------------------

def cmd_providers():
    print(f"{'provider':12s} {'model':22s} {'$/1M in':>8s} {'$/1M out':>9s}  default-thinking")
    for pname, cfg in P.PROVIDERS.items():
        for mname, m in cfg["models"].items():
            print(f"{pname:12s} {mname:22s} {m['price_in']:8.2f} {m['price_out']:9.2f}  "
                  f"{m['default_thinking']}")


def cmd_smoke(provider, model):
    t0 = time.time()
    try:
        ans, reas, pt, ct = P.ask(provider, model, "Compute 7 + 5. Answer with only the number.", "off")
        print(f"  {model} OK ({time.time()-t0:.1f}s) reasoning={reas} tok={pt}+{ct} -> {ans!r}")
    except Exception as e:
        print(f"  {model} FAIL -> {str(e)[:140]}")


def cmd_screen(provider, model, n):
    th = P.default_thinking(provider, model)
    print(f"== screen {provider}/{model} x {th} (first {n} atomic + {n} chained) ==")
    rec = _run_bank(provider, model, th, n=n)
    _save(rec)


def cmd_full(provider, model):
    th = P.default_thinking(provider, model)
    print(f"== full {provider}/{model} x {th} (all 40 screen-v1 tasks) ==")
    rec = _run_bank(provider, model, th)
    _save(rec)
    print("\ndecision sheet:")
    base = _last(DEFAULT_MODEL, DEFAULT_THINKING, mode="full") or _last(DEFAULT_MODEL, DEFAULT_THINKING)
    if base:
        print(f"  baseline {DEFAULT_MODEL}/{DEFAULT_THINKING}: "
              f"chained {base['chained_pct']}% comp={base.get('composite',0)} "
              f"${_day_cost(base)}/day")
    print(f"  candidate {rec['model']}: chained {rec['chained_pct']}% "
          f"comp={rec.get('composite',0)} ${_day_cost(rec)}/day")
    v, why = _decision(rec)
    print(f"  VERDICT: {v} — {why}")
    print("\n  next: if VERDICT is YES/MAYBE, run it as a shadow default on real")
    print("  traffic for a day before committing — the bank predicts, reality decides.")


def cmd_report():
    rows = _load_results()
    if not rows:
        print("no runs logged yet")
        return
    print("logged runs (last N per model):")
    for r in rows:
        print(_line(r))


def cmd_compare(model_a, model_b):
    a, b = _last(model_a), _last(model_b)
    if not a or not b:
        print(f"need logged runs for both: {model_a}={bool(a)} {model_b}={bool(b)}")
        return
    print(f"  {_line(a)}")
    print(f"  {_line(b)}")
    print("\ndecision sheet (candidate = " + model_b + "):")
    v, why = _decision(b)
    print(f"  VERDICT: {v} — {why}")


def main():
    ap = argparse.ArgumentParser(description="onboard a new model against frozen banks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("providers")
    for c in ("smoke", "screen", "full"):
        p = sub.add_parser(c)
        p.add_argument("provider")
        p.add_argument("model")
        p.add_argument("--n", type=int, default=5)
    sub.add_parser("report")
    cmp = sub.add_parser("compare"); cmp.add_argument("model_a"); cmp.add_argument("model_b")
    a = ap.parse_args()

    if a.cmd == "providers":
        cmd_providers()
    elif a.cmd == "smoke":
        cmd_smoke(a.provider, a.model)
    elif a.cmd == "screen":
        cmd_screen(a.provider, a.model, a.n)
    elif a.cmd == "full":
        cmd_full(a.provider, a.model)
    elif a.cmd == "report":
        cmd_report()
    elif a.cmd == "compare":
        cmd_compare(a.model_a, a.model_b)


if __name__ == "__main__":
    main()
