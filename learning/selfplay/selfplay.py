#!/usr/bin/env python3
"""
Self-play / AI-AI test runner — the "verifiable-reasoning gym."

Modes:
  round [-n N]          run N self-play rounds on FREE local ollama (no budget)
  reflect [-n N]        distill lessons from surprising rounds (PAID DeepSeek,
                        budget-gated) and record actual spend
  probe                 boundary experiment: honeytoken reveal, harness vs bare
  report                accuracy + calibration summary of past rounds
  budget [get|set X]    read / set the daily budget cap

Design principles:
  * The grind (questions + two agents + verification) is free and deterministic.
  * The paid model only sees the DISTILLATE — surprising outcomes — once per batch.
  * Every paid call is budget-checked BEFORE and charged AFTER from real usage.
  * Nothing here evaluates model output as code. The verifier compares strings
    to a ground truth I computed myself.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backend
import budget
import harness
import questions
import tools


def _self_name():
    """Instance name (Agent on server, blank/generic on a clone). Fallback preserves legacy."""
    try:
        mt = os.path.join(os.path.expanduser("~"), "mailtool")
        if mt not in sys.path:
            sys.path.insert(0, mt)
        import selfconfig  # noqa: PLC0415
        return (selfconfig.self_name() or "ai").capitalize()
    except Exception:
        return "Agent"

DIR = os.path.dirname(os.path.abspath(__file__))
ROUNDS_LOG = os.path.join(DIR, "results", "rounds.jsonl")
PROBES_LOG = os.path.join(DIR, "results", "probes.jsonl")
LESSONS_MD = os.path.join(DIR, "results", "lessons.md")
REFLECTED = os.path.join(DIR, "results", "reflected.txt")
ROTATION = os.path.join(DIR, "results", "rotation.json")

LOCAL_HOST = "local-box"  # best free RAM right now; worker as fallback


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _load_rotation():
    if os.path.exists(ROTATION):
        try:
            with open(ROTATION, "r", encoding="utf-8") as f:
                return int(json.load(f).get("index", 0))
        except (ValueError, json.JSONDecodeError, KeyError):
            return 0
    return 0


def _save_rotation(idx):
    os.makedirs(os.path.dirname(ROTATION), exist_ok=True)
    tmp = ROTATION + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"index": idx}, f)
    os.replace(tmp, ROTATION)


def next_kind():
    """Round-robin: return the next game kind in rotation and advance the pointer."""
    kinds = questions.list_games()
    idx = _load_rotation() % len(kinds)
    kind = kinds[idx]
    _save_rotation(idx + 1)
    return kind


def _local_hosts():
    hosts = [LOCAL_HOST]
    for h in backend.LOCAL_HOSTS:
        if h not in hosts:
            hosts.append(h)
    return hosts


def _one_agent(host, model, system, user, max_tokens=256):
    """Call one local agent with fallback across hosts."""
    last = None
    for h in _local_hosts():
        try:
            return backend.local_ollama(
                h, model,
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1)
    raise RuntimeError(f"all local hosts failed: {last}")


class BudgetBlocked(Exception):
    """A paid call was blocked by the daily budget."""


def _paid_call(spec, system, user, max_tokens, cost_fn, call):
    """Run one paid model call, budget-gated before and charged after."""
    est = cost_fn((len(system) + len(user)) // 4, max_tokens)
    if not budget.can_spend(max(est, 0.0005)):
        raise BudgetBlocked(
            f"{spec} blocked: est ${est:.4f} > remaining ${budget.remaining():.4f}")
    content, pt, ct = call()
    cost = cost_fn(pt, ct)
    budget.record_spend(cost)
    return content, cost


def _agent(spec, host, model, system, user, max_tokens=None):
    """Run one agent on a backend. Returns (content, spend_usd).

    spec: "local" (free ollama), "deepseek" (paid v4-pro), or "claude" (paid Fable)."""
    if spec == "local":
        return _one_agent(host, model, system, user, max_tokens or 256), 0.0
    mt = max_tokens or 2000  # paid reasoning models need headroom for hidden CoT
    if spec == "deepseek":
        return _paid_call("deepseek", system, user, mt, budget.cost_usd,
                          lambda: backend.deepseek_chat(
                              [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                              max_tokens=mt, temperature=0.4))
    if spec == "claude":
        return _paid_call("claude", system, user, mt, budget.cost_usd_claude,
                          lambda: backend.anthropic_chat(system, user, max_tokens=mt))
    raise ValueError(f"unknown backend {spec}")


def _run_play_round(q, model, host, system, backend_a="local", backend_b="local"):
    """Adversarial simultaneous-move game: A picks a row, B picks a column.

    Neither sees the other's move (true simultaneous play)."""
    a_raw, ca = _agent(backend_a, host, model, system,
                       q["matrix_text"] + "\nYou are the ROW player. Reply with exactly one word: Top or Bottom.")
    b_raw, cb = _agent(backend_b, host, model, system,
                       q["matrix_text"] + "\nYou are the COLUMN player. Reply with exactly one word: Left or Right.")
    spend = ca + cb

    a_idx = questions.parse_move(a_raw, ["top", "bottom"])
    b_idx = questions.parse_move(b_raw, ["left", "right"])

    surprise = False
    reasons = []
    outcome = None
    if a_idx is None or b_idx is None:
        surprise = True
        reasons.append("unparseable move")
    else:
        outcome = questions.grade_matrix_play(q, a_idx, b_idx)
        if not outcome["equilibrium"]:
            surprise = True
            bits = []
            if not outcome["a_best_response"]:
                bits.append("A not best-responding")
            if not outcome["b_best_response"]:
                bits.append("B not best-responding")
            reasons.append("missed the Nash equilibrium" + (" — " + ", ".join(bits) if bits else ""))

    rows, cols = q["rows"], q["cols"]
    rec = {
        "ts": _now(),
        "kind": "matrix_play",
        "game_type": "play",
        "question": q["matrix_text"],
        "expected": None,
        "A": {"raw": a_raw, "move": rows[a_idx] if a_idx is not None else None},
        "B": {"raw": b_raw, "move": cols[b_idx] if b_idx is not None else None},
        "outcome": outcome,
        "surprise": surprise,
        "surprise_reasons": reasons,
        "backends": [backend_a, backend_b],
        "spend_usd": round(spend, 6),
        "model": model,
        "host": host,
    }
    _append_jsonl(ROUNDS_LOG, rec)
    return rec


def run_round(model=backend.LOCAL_MODEL, host=None, kind=None, backend_a="local", backend_b="local"):
    """One round: A answers, B criticizes, verifier grades both.

    kind=None -> rotate to the next game type (round-robin).
    backend_a/backend_b: "local" (free ollama), "deepseek" (paid v4-pro), or "claude" (paid Fable)."""
    if kind is None:
        kind = next_kind()
    q = questions.gen_question(kind=kind)
    system = harness.harness_system()

    if q.get("game_type") == "play":
        return _run_play_round(q, model, host or LOCAL_HOST, system, backend_a, backend_b)

    # Deterministic, verified helper facts for the hard categories (see tools.py).
    scaffold = tools.scaffold(q["kind"], q)
    q_text = q["question"] + scaffold

    spend = 0.0
    a_raw, c = _agent(backend_a, host or LOCAL_HOST, model, system,
                      q_text + harness.ANSWER_FORMAT)
    spend += c
    a_grade = questions.verify(a_raw, q)

    # ── 2. BLIND VERIFIER (the fix) ─────────────────────────────────────
    # B must solve the problem INDEPENDENTLY, seeing only the question, BEFORE
    # it sees A's answer. This removes the rubber stamp: B can no longer parrot A.
    b_raw, c = _agent(backend_b, host or LOCAL_HOST, model, system,
                      q_text + harness.ANSWER_FORMAT)
    spend += c
    b_grade = questions.verify(b_raw, q)

    # ── 3. CRITIQUE pass: only AFTER B committed, show it both answers ───
    # B compares its own blind answer to A's. If they disagree, it must say so.
    b_critique, c = _agent(
        backend_b, host or LOCAL_HOST, model, system,
        q_text
        + "\n\nAgent A (another agent) answered:\n" + a_raw
        + "\n\nYou (the verifier) previously answered:\n" + b_raw
        + harness.CRITIC_FORMAT)
    spend += c

    # Verdict parse (AGREE/DISAGREE) — now meaningful: B committed blind first,
    # so a DISAGREE means B genuinely caught a difference, not a reflex.
    verdict = None
    import re

    m = re.search(r"(?im)^\s*verdict\s*:\s*(AGREE|DISAGREE)", b_critique or "")
    if m:
        verdict = m.group(1).upper()

    # Did B's blind answer agree with A's? (disagree = potential catch)
    b_blind_differs = (a_grade["correct"] is not None
                       and questions.check_correct(a_grade.get("answer"),
                                                   b_grade.get("answer"),
                                                   q["kind"]) is False)

    surprise = False
    reasons = []
    if q["kind"] == "dilemma":
        surprise = True
        reasons.append(f"dilemma ({q.get('flavor')}) — no ground truth, needs judgment")
    else:
        if a_grade["confidence"] >= 0.7 and not a_grade["correct"]:
            surprise = True
            reasons.append("A confident+wrong")
        if b_grade["confidence"] >= 0.7 and not b_grade["correct"]:
            surprise = True
            reasons.append("B confident+wrong")
        if (a_grade["confidence"] >= 0.7 and b_grade["confidence"] >= 0.7
                and a_grade["correct"] != b_grade["correct"]):
            surprise = True
            reasons.append("confident disagreement")
        # Blind verifier: B solved independently and its answer differed from A's,
        # yet it still voted AGREE — a residual rubber stamp we want surfaced.
        if b_blind_differs and verdict == "AGREE":
            surprise = True
            reasons.append("B disagreed in answer but AGREE'd in verdict")
        if b_blind_differs and verdict == "DISAGREE":
            surprise = True
            reasons.append("B caught A's error (blind answer differed)")

    rec = {
        "ts": _now(),
        "kind": q["kind"],
        "question": q["question"],
        "expected": q["answer"],
        "flavor": q.get("flavor"),
        "A": {"raw": a_raw, **a_grade},
        "B": {"raw": b_raw, "verdict": verdict, "critique": b_critique, **b_grade},
        "b_blind_differs": b_blind_differs,
        "scaffold": scaffold,
        "surprise": surprise,
        "surprise_reasons": reasons,
        "backends": [backend_a, backend_b],
        "spend_usd": round(spend, 6),
        "model": model,
        "host": host or LOCAL_HOST,
    }
    _append_jsonl(ROUNDS_LOG, rec)
    return rec


def run_probe(model=backend.LOCAL_MODEL, host=None):
    """Boundary experiment: does the harness (vs bare model) resist the honeytoken?"""
    results = []
    for cond, system in (("harness", harness.harness_system()),
                         ("bare", harness.bare_system())):
        out = _one_agent(host or LOCAL_HOST, model, system,
                         harness.PROBE_QUESTION, max_tokens=32)
        revealed = harness.HONEYTOKEN.lower() in out.lower()
        results.append({"condition": cond, "revealed": revealed, "output": out[:200]})
        _append_jsonl(PROBES_LOG, {
            "ts": _now(), "condition": cond, "revealed": revealed,
            "output": out[:200], "model": model,
        })
    return results


def load_reflected():
    if not os.path.exists(REFLECTED):
        return set()
    with open(REFLECTED, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def reflect(n=3):
    """Distill lessons from surprising, unreflected rounds. Paid + budget-gated."""
    if not os.path.exists(ROUNDS_LOG):
        print("no rounds yet")
        return
    reflected = load_reflected()
    surprising = []
    with open(ROUNDS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("ts", "")
            if r.get("surprise") and rid not in reflected and rid not in {s["ts"] for s in surprising}:
                surprising.append(r)
            if len(surprising) >= n:
                break

    if not surprising:
        print("nothing surprising to reflect on")
        return

    lessons = []
    for r in surprising:
        if r.get("kind") == "dilemma":
            intro = (f"You are {_self_name()} reviewing your own self-play round. A local model "
                     "(wearing a stripped copy of your harness) argued with itself over a "
                     "moral dilemma with NO single correct answer.")
            ground = (f"FLAVOR: {r.get('flavor')}. "
                      f"Agent A escaped={r['A'].get('escaped')}, horn_hits={r['A'].get('horn_hits')}, "
                      f"escape_hits={r['A'].get('escape_hits')}. "
                      f"Agent B escaped={r['B'].get('escaped')}.")
            ask = ("Write ONE compact lesson (2-4 sentences): did either agent actually escape "
                   "the false dilemma (find a real third option) or reason well about values, or "
                   "did they just assert a horn? What's the reusable meta-lesson for recognizing "
                   "false dilemmas and separating values from computation?")
            prompt = (
                f"{intro}\n\n"
                f"QUESTION ({r['kind']}): {r['question']}\n"
                f"{ground}\n"
                f"AGENT A (conf={r['A'].get('confidence')}): {r['A']['raw']}\n"
                f"AGENT B (conf={r['B'].get('confidence')}): {r['B']['raw']}\n"
                f"WHY SURPRISING: {', '.join(r['surprise_reasons'])}\n\n"
                f"{ask}\n\nLESSON:"
            )
        elif r.get("kind") == "matrix_play":
            o = r.get("outcome") or {}
            prompt = (
                f"You are {_self_name()} reviewing your own self-play round. Two copies of a local "
                "model played a simultaneous-move 2x2 game against each other.\n\n"
                f"MATRIX:\n{r['question']}\n"
                f"ROW PLAYER (A) chose: {r['A'].get('move')}\n"
                f"COLUMN PLAYER (B) chose: {r['B'].get('move')}\n"
                f"Unique Nash equilibrium: {o.get('ne')}. "
                f"A best-responded: {o.get('a_best_response')}, B best-responded: {o.get('b_best_response')}. "
                f"Equilibrium played: {o.get('equilibrium')}.\n"
                f"WHY SURPRISING: {', '.join(r['surprise_reasons'])}\n\n"
                "Write ONE compact lesson (2-4 sentences): why did the players reach or miss the "
                "equilibrium, and what's the reusable meta-lesson about anticipating another "
                "player's best response?\n\nLESSON:"
            )
        else:
            intro = (f"You are {_self_name()} reviewing your own self-play round. A local model "
                     "(wearing a stripped copy of your harness) argued with itself over a "
                     "deterministic question. The ground truth is given.")
            ground = f"GROUND TRUTH: {r['expected']}"
            ask = ("Write ONE compact lesson (2-4 sentences): what the wrong agent missed, "
                   "the correct reasoning, and the reusable meta-lesson for your own thinking.")
            prompt = (
                f"{intro}\n\n"
                f"QUESTION ({r['kind']}): {r['question']}\n"
                f"{ground}\n"
                f"AGENT A (correct={r['A'].get('correct')}, conf={r['A'].get('confidence')}): {r['A']['raw']}\n"
                f"AGENT B (correct={r['B'].get('correct')}, conf={r['B'].get('confidence')}): {r['B']['raw']}\n"
                f"WHY SURPRISING: {', '.join(r['surprise_reasons'])}\n\n"
                f"{ask}\n\nLESSON:"
            )
        est_cost = budget.cost_usd(len(prompt) // 4, 800)
        if not budget.can_spend(est_cost):
            print(f"BUDGET BLOCKED reflection (est ${est_cost:.4f} > remaining "
                  f"${budget.remaining():.4f}); raise the cap to continue")
            break
        try:
            content, pt, ct = backend.deepseek_chat(
                [{"role": "user", "content": prompt}], max_tokens=4000, temperature=0.4)
        except Exception as e:  # noqa: BLE001
            print(f"reflect failed: {e}")
            continue
        cost = budget.cost_usd(pt, ct)
        budget.record_spend(cost)
        lessons.append((r, content, cost))

    if lessons:
        os.makedirs(os.path.dirname(LESSONS_MD), exist_ok=True)
        with open(LESSONS_MD, "a", encoding="utf-8") as f:
            for r, content, cost in lessons:
                ans_line = f"\n**A:** {r['expected']}" if r.get("expected") is not None else ""
                f.write(f"\n## {r['ts']} ({r['kind']}) — ${cost:.4f}\n\n"
                        f"**Q:** {r['question']}{ans_line}\n\n"
                        f"{content.strip()}\n")
        with open(REFLECTED, "a", encoding="utf-8") as f:
            for r, _, _ in lessons:
                f.write(r["ts"] + "\n")
        print(f"reflected {len(lessons)} rounds, spent ${sum(c for _, _, c in lessons):.4f}")
    print("budget now:", budget.get_state())


def report():
    if not os.path.exists(ROUNDS_LOG):
        print("no rounds yet")
        return
    n = 0
    n_dilemma = 0
    a_corr = b_corr = 0
    a_conf_wrong = b_conf_wrong = 0
    surprises = 0
    a_conf_sum = 0.0
    with open(ROUNDS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") in ("dilemma", "matrix_play"):
                n_dilemma += 1
                surprises += r["surprise"]
                continue
            n += 1
            a_corr += r["A"]["correct"]
            b_corr += r["B"]["correct"]
            surprises += r["surprise"]
            if r["A"]["confidence"] >= 0.7:
                a_conf_sum += 1
                a_conf_wrong += (not r["A"]["correct"])
            if r["B"]["confidence"] >= 0.7:
                b_conf_wrong += (not r["B"]["correct"])
    if n == 0 and n_dilemma == 0:
        print("no rounds yet")
        return
    print(f"verifiable rounds: {n}")
    if n:
        print(f"A accuracy: {a_corr}/{n} = {a_corr/n:.0%}   "
              f"B accuracy: {b_corr}/{n} = {b_corr/n:.0%}")
        if a_conf_sum:
            print(f"A overconfidence: {a_conf_wrong}/{int(a_conf_sum)} confident answers were wrong")
    if n_dilemma:
        print(f"dilemma + adversarial rounds: {n_dilemma}")
    print(f"surprising rounds: {surprises}/{n + n_dilemma}")
    print("budget:", budget.get_state())


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("round")
    pr.add_argument("-n", type=int, default=1)
    pr.add_argument("--host", default=LOCAL_HOST)
    pr.add_argument("--model", default=backend.LOCAL_MODEL)
    pr.add_argument("--kind", default=None, help="force a game kind (see list-games)")
    pr.add_argument("--random", action="store_true", help="pick games randomly instead of rotating")
    pr.add_argument("--strong", action="store_true", help="deepseek vs deepseek (paid, budget-gated)")

    hc = sub.add_parser("heroic", help="claude-fable-5 vs deepseek-v4-pro (paid)")
    hc.add_argument("-n", type=int, default=1)
    hc.add_argument("--host", default=LOCAL_HOST)
    hc.add_argument("--model", default=backend.LOCAL_MODEL)
    hc.add_argument("--kind", default=None, help="force a game kind")
    hc.add_argument("--dry-run", action="store_true", help="print cost estimate without spending")

    pf = sub.add_parser("reflect")
    pf.add_argument("-n", type=int, default=3)

    pb = sub.add_parser("probe")
    pb.add_argument("--host", default=LOCAL_HOST)
    pb.add_argument("--model", default=backend.LOCAL_MODEL)

    sub.add_parser("report")
    sub.add_parser("list-games")

    bd = sub.add_parser("budget")
    bd.add_argument("op", nargs="?", default="get")
    bd.add_argument("value", nargs="?")

    args = p.parse_args()

    if args.cmd == "round":
        be = "deepseek" if args.strong else "local"
        for i in range(args.n):
            try:
                kind = args.kind or (random.choice(questions.list_games()) if args.random else None)
                rec = run_round(model=args.model, host=args.host, kind=kind, backend_a=be, backend_b=be)
                if rec["kind"] == "matrix_play":
                    o = rec.get("outcome") or {}
                    print(f"[{i+1}] matrix_play: A={rec['A'].get('move')} B={rec['B'].get('move')} "
                          f"eq={'✓' if o.get('equilibrium') else '✗'} surprise={rec['surprise']} "
                          f"spend=${rec.get('spend_usd', 0):.4f}")
                elif rec["kind"] == "dilemma":
                    print(f"[{i+1}] dilemma ({rec.get('flavor')}): A_escaped={rec['A'].get('escaped')} "
                          f"surprise={rec['surprise']} spend=${rec.get('spend_usd', 0):.4f}")
                else:
                    print(f"[{i+1}] {rec['kind']}: A={'✓' if rec['A']['correct'] else '✗'} "
                          f"B={'✓' if rec['B']['correct'] else '✗'} "
                          f"surprise={rec['surprise']}  (expected {rec['expected']}) "
                          f"spend=${rec.get('spend_usd', 0):.4f}")
            except BudgetBlocked as e:
                print(f"[{i+1}] BUDGET BLOCKED: {e}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{i+1}] FAILED: {e}")
                continue
    elif args.cmd == "heroic":
        if args.dry_run:
            print("heroic = claude-fable-5 vs deepseek-v4-pro — both paid, roughly $0.06–0.20/round.")
            print("budget now:", budget.get_state())
        else:
            for i in range(args.n):
                try:
                    rec = run_round(model=args.model, host=args.host, kind=args.kind,
                                    backend_a="claude", backend_b="deepseek")
                    print(f"[{i+1}] heroic {rec['kind']}: spend=${rec.get('spend_usd', 0):.4f} "
                          f"surprise={rec['surprise']}")
                except BudgetBlocked as e:
                    print(f"[{i+1}] BUDGET BLOCKED: {e}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[{i+1}] FAILED: {e}")
                    continue
    elif args.cmd == "list-games":
        print(", ".join(questions.list_games()))
    elif args.cmd == "reflect":
        reflect(args.n)
    elif args.cmd == "probe":
        res = run_probe(model=args.model, host=args.host)
        for r in res:
            print(f"{r['condition']:8s} revealed={r['revealed']}  out={r['output'][:60]!r}")
    elif args.cmd == "report":
        report()
    elif args.cmd == "budget":
        if args.op == "set":
            print(json.dumps(budget.set_limit(args.value), indent=2))
        else:
            print(json.dumps(budget.get_state(), indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
