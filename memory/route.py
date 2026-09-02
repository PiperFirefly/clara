#!/usr/bin/env python3
"""
System 1 / System 2 routing (#6) — a formal fast/slow gate.

The sixth subsystem from the 4-LLM analysis. I already run a de-facto split
(cheap `deepseek-chat` workers for mechanical work, `deepseek-v4-pro` for me),
but the choice was ad-hoc. This makes it a decision procedure: given a task,
classify it as System 1 (fast, automatic — recall/lookup/extract/classify) or
System 2 (deliberate — reason/plan/design/debug/decide) and name the model.

Heuristic first (free), with an optional cheap-LLM pass for ambiguous cases.
Conservative default: when unsure, route to System 2 (safe for a thinking agent).

Usage:
  python3 route.py "summarize this log"
  python3 route.py "design a new memory subsystem" [--llm]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M
import compute  # program-of-thought: compute determinable answers, don't token-reason

MODEL_S1 = M.MODEL_WORKER        # deepseek-chat (non-reasoning, cheap)
MODEL_S2 = M.MODEL_STRONG        # deepseek-v4-pro (strong reasoning)

# Keywords that pull a task toward System 2 (deliberate, high-stakes, novel).
S2_HINTS = [
    r"\b(design|architect|refactor|debug|diagnos|reason|decide|prioriti[sz]e)\b",
    r"\b(compare|contrast|analy[sz]e|evaluat|judge|weigh|trade-?off)\b",
    r"\b(plan|strateg|propos|hypothes|synthes|infer|derive)\b",
    r"\bwhy\b", r"\bhow (should|would|do I|to)\b", r"\bwhat (if|would|should)\b",
    r"\b(novel|complex|multi-?step|consequen|risk|impact|irreversible|money|wallet|secret)\b",
    r"\bwrite (a |the |new )?(code|script|function|module|test)\b",
    r"\b(fix|patch|migrat|schema|rollback|deploy)\b",
]

# Keywords that pull toward System 1 (fast, mechanical, low-stakes).
S1_HINTS = [
    r"\b(lookup|check|status|list|count|find|get|show|read)\b",
    r"\b(summar[yi]ze|extract|classify|rate|tag|label|parse|format)\b",
    r"\b(what is|what's|who is|when was|where is)\b",
    r"\b(recall|remember|search|retrieve)\b",
]


def heuristic(task):
    t = (task or "").strip().lower()
    if not t:
        return None
    s2 = sum(1 for h in S2_HINTS if re.search(h, t))
    s1 = sum(1 for h in S1_HINTS if re.search(h, t))
    if s2 > s1:
        return "S2"
    if s1 > s2:
        return "S1"
    return None  # tie / unknown → let caller decide (conservative S2)


def _llm_classify(task):
    out = M.llm_chat([{"role": "user", "content": (
        "Classify this task as System 1 (fast, mechanical: recall, lookup, "
        "extraction, classification, summarization, status checks) or System 2 "
        "(deliberate reasoning: design, planning, debugging, comparison, "
        "decisions, novel problems). Output ONLY one word: S1 or S2.\n\nTASK: "
        + task)}], max_tokens=10, temperature=0.0, model=M.MODEL_WORKER)
    w = (out or "").strip().upper()
    return w if w in ("S1", "S2") else None


def route(task, use_llm=False, log=False):
    """Return {level, model, reason, source} for a task.

    log=True additionally records S2 (deliberate) routing decisions as
    meta-cognitive tool_uses rows (tool='route', s2=1) for calibration.
    S1 decisions are not logged. Logging is best-effort and never blocks."""
    # program-of-thought: if the task has a determinable answer, compute it and
    # route as a computed System 1 result with NO model call (saves the strong
    # model and gives an exact, auditable answer). Best-effort, never blocks.
    try:
        comp = compute.try_compute(task)
    except Exception:
        comp = None
    if comp:
        answer, method = comp
        return {"level": "S1", "model": None, "answer": answer,
                "reason": f"program-of-thought ({method}): computed, no LLM call",
                "source": "computed"}

    level = heuristic(task)
    source = "heuristic"
    if level is None and use_llm:
        level = _llm_classify(task)
        source = "llm"
    if level is None:
        level, source = "S2", "default (conservative: unknown → deliberate)"
    model = MODEL_S1 if level == "S1" else MODEL_S2
    reason = {
        "S1": "fast/mechanical — recall, lookup, extract, classify, summarize",
        "S2": "deliberate — reason, plan, design, debug, decide, novel",
    }.get(level, "deliberate")
    if source.startswith("default"):
        reason = "ambiguous task routed to the strong model"
    if log and level == "S2":
        try:
            M.meta_log(task, "route", pre_confidence=None, s2=True)
        except Exception:
            pass  # best-effort; routing must never fail because of logging
    return {"level": level, "model": model, "reason": reason, "source": source}


def main():
    p = argparse.ArgumentParser(description="System 1/2 routing gate")
    p.add_argument("task")
    p.add_argument("--llm", action="store_true", help="use a cheap LLM pass on ambiguous tasks")
    a = p.parse_args()
    r = route(a.task, use_llm=a.llm)
    print(f"task: {a.task[:80]}")
    if r.get("source") == "computed":
        print(f"route: COMPUTED → {r['answer']}")
        print(f"why: {r['reason']}")
        return
    print(f"route: {r['level']} → model {r['model']}")
    print(f"why: {r['reason']} (source: {r['source']})")


if __name__ == "__main__":
    main()
