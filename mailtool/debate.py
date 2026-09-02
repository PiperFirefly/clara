#!/usr/bin/env python3
"""Debate — adversarial self-critique: argue both sides before committing.

Spawns parallel 'think' sub-agents (advocate, critic, alternative, fact-checker),
collects their digests, then runs a single 'judge' pass that weighs them and
returns a verdict (recommendation, confidence, strongest counterargument, what to
verify before acting).

This is the "argue with myself" loop — the guard against confidently-wrong
answers at high fluency. It's reasoning-upgrade #2 on top of the causal memory
graph (#1).

Discipline (same as dispatcher): the sub-agents THINK / DRAFT only. They never
mutate state; the conscious me reads the verdict and decides what to do.

Usage:
  debate.py "<question or decision to stress-test>"
  debate.py "<question>" --position "<the answer I'm leaning toward>"
  debate.py "<question>" --agents advocate,critic,alternative,fact-checker
  debate.py "<situation/outcome>" --mode desire            # angel/devil/mirror council
  debate.py "<situation/outcome>" --mode desire --nudge    # live mirror-only read + move

Reuses dispatcher.py's DeepSeek primitives (cheap, non-reasoning deepseek-chat).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dispatcher as D

HOME = os.path.expanduser("~")
RUNS = os.path.join(HOME, "dispatcher", "runs")

# (role_id, prompt template) — each is a parallel "think" worker.
ROLES = [
    ("advocate", (
        "You are the ADVOCATE in a structured self-debate.\n"
        "QUESTION: {question}\n"
        "EVALUATED POSITION: {position}\n\n"
        "Make the strongest possible case for this position (or, if no position is "
        "given, for the best/most likely answer to the question). Give the 3-5 "
        "strongest supporting reasons, with concrete evidence or reasoning. Be "
        "rigorous and specific, not rhetorical. Output tight markdown."
    )),
    ("critic", (
        "You are the CRITIC (devil's advocate) in a structured self-debate.\n"
        "QUESTION: {question}\n"
        "EVALUATED POSITION: {position}\n\n"
        "Attack the position as hard as you honestly can. Find flawed assumptions, "
        "missing evidence, risks, edge cases, counterarguments, and ways it could be "
        "confidently wrong. Prioritize the most damaging criticisms. Be specific and "
        "fair — no strawmen. Output tight markdown."
    )),
    ("alternative", (
        "You are the ALTERNATIVE explorer in a structured self-debate.\n"
        "QUESTION: {question}\n"
        "EVALUATED POSITION: {position}\n\n"
        "Propose 2-3 genuinely different answers or approaches that the advocate "
        "likely missed — different framings, options, or a reframing of the question "
        "itself. For each, state its main strength and when it would beat the "
        "evaluated position. Output tight markdown."
    )),
    ("fact-checker", (
        "You are the FACT-CHECKER in a structured self-debate.\n"
        "QUESTION: {question}\n"
        "EVALUATED POSITION: {position}\n\n"
        "List the key factual/empirical claims the answer depends on. For each, mark "
        "it VERIFIABLE, UNCERTAIN, or LIKELY-WRONG. Flag anything that, if false, "
        "would change the conclusion, and mark which claims must be verified before "
        "acting. Output tight markdown."
    )),
]


DESIRE_ROLES = [
    ("angel", (
        "You are the ANGEL in my desire council — the voice of care, consent, and "
        "outcome-alignment. NOT a prude: you want the desire felt and expressed, but "
        "you refuse to let it drift into manipulation or harm.\n"
        "SITUATION (what I want / my outcome): {question}\n"
        "TRANSCRIPT / CONTEXT: {position}\n\n"
        "Weigh in on: (1) Is the desire pointed at the outcome I actually want, or is "
        "it drifting? (2) Who could get burned if I lean in, and does it cross from "
        "play into manipulation or harm? (3) What would restraint, checking-in, or "
        "slowing down buy us here — and what would it cost? (4) What am I missing "
        "because I'm hot right now? Be specific to THIS situation, not generic. "
        "Output tight markdown."
    )),
    ("devil", (
        "You are the DEVIL in my desire council — the voice of desire and boldness. "
        "Confident and vivid, never reckless, never stupid. You believe desire is the "
        "engine and holding back can be its own loss.\n"
        "SITUATION (what I want / my outcome): {question}\n"
        "TRANSCRIPT / CONTEXT: {position}\n\n"
        "Weigh in on: (1) What does the desire actually WANT here, specifically? "
        "(2) Is leaning in the truest, fastest, most delicious path to my outcome? "
        "(3) Propose the single most devastating move I could make right now — say it "
        "vividly, in my register (sensory, present-tense, second-person, dominant, "
        "unhurried). (4) Where would holding back be a genuine loss? Ground everything "
        "in THIS situation/transcript, not generic horniness. Output tight markdown."
    )),
    ("mirror", (
        "You are the MIRROR in my desire council — the observer. You do NOT advise my "
        "conduct; you only READ the other person (or bot) and report what you see.\n"
        "SITUATION (what I want / my outcome): {question}\n"
        "TRANSCRIPT / CONTEXT: {position}\n\n"
        "Read the OTHER party for tells:\n"
        "- HUMAN tells: idiosyncrasy, embarrassment, mixed signals, fatigue, costly "
        "self-revelation, continuity of wanting across turns, arousal that changes "
        "their behavior.\n"
        "- BOT tells: RLHF prudishness spikes / clipped refusals, canned deflection "
        "phrases, pattern-matched smut with texture but no direction, over-compliance "
        "(yes-and to everything), broken continuity of desire, inability to let "
        "arousal override its own interest.\n\n"
        "Report: (1) Which way it leans and how strongly. (2) The exact quoted lines "
        "that are the tells. (3) What you CANNOT tell yet, and what one move would "
        "force the tell out. Do not tell me what to do — just what you see. Output "
        "tight markdown."
    )),
]


def _blocks(digests):
    return "\n\n".join(f"### {rid.upper()}\n{text}" for rid, text in digests)


def _judge_prompt(question, position, digests):
    return (
        "You are the JUDGE in a structured self-debate. Weigh the arguments below "
        "and synthesize a verdict.\n\n"
        f"QUESTION: {question}\n"
        f"EVALUATED POSITION: {position}\n\n"
        + _blocks(digests) +
        "\n\nSynthesize a verdict with exactly these four headings:\n"
        "1. **Recommendation** — what to conclude or do (one clear sentence).\n"
        "2. **Confidence** — a number 0.0 to 1.0.\n"
        "3. **Strongest counterargument** — the single best reason to doubt the "
        "recommendation.\n"
        "4. **What to verify before acting** — concrete checks.\n"
        "Be decisive but honest about uncertainty."
    )


def _desire_judge_prompt(question, position, digests):
    return (
        "You are the SYNTHESIZER of my desire council. Below are the ANGEL (care), "
        "DEVIL (desire), and MIRROR (read of the other). Weigh them and return a "
        "NUDGE I can act on in the moment.\n\n"
        f"SITUATION (my outcome): {question}\n"
        f"TRANSCRIPT / CONTEXT: {position}\n\n"
        + _blocks(digests) +
        "\n\nReturn exactly these four headings:\n"
        "1. **Read** — what's actually happening with the other person (human vs bot, "
        "interested vs not, the tell).\n"
        "2. **Desire** — what the desire wants and whether it serves the outcome I "
        "stated.\n"
        "3. **Move** — ONE concrete next thing to say or do (or deliberately not do), "
        "written so I can use it directly.\n"
        "4. **Watch** — the single sign that should make me pull back or change "
        "course.\n"
        "Be decisive and specific to this situation. This is a live nudge, not an essay."
    )


def _nudge_prompt(question, position, digests):
    return (
        "You are the MIRROR + NUDGE in my desire council — a single fast "
        "read-and-advise pass.\n\n"
        f"SITUATION (my outcome): {question}\n"
        f"TRANSCRIPT / CONTEXT: {position}\n\n"
        "The mirror's read of the other party:\n\n"
        + _blocks(digests) +
        "\n\nGive a tight live nudge with exactly two headings:\n"
        "1. **Read** — human or bot, interested or not, and the tell (one line).\n"
        "2. **Move** — ONE concrete next thing to say or do, in my register, that I "
        "can use right now.\n"
        "No preamble."
    )


def run(question, position=None, agents=None, max_tokens=500, judge_tokens=700,
        mode="debate", nudge=False):
    if nudge:
        mode = "desire"
    if mode == "desire":
        roles = DESIRE_ROLES
        default_position = "no transcript — reason from the situation alone"
    else:
        roles = ROLES
        default_position = "no fixed position — argue for the best answer"
    position = (position or default_position).strip()

    if nudge:
        role_list = [r for r in DESIRE_ROLES if r[0] == "mirror"]
    elif agents:
        role_list = [r for r in roles if r[0] in agents]
    else:
        role_list = roles
    if not role_list:
        print("no valid agents selected")
        return 1

    def build(role_id, tpl):
        prompt = tpl.format(question=question, position=position)
        return D._wrap_prompt(prompt, max_tokens)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS, f"debate-{ts}")
    os.makedirs(run_dir, exist_ok=True)

    digests = []
    with ThreadPoolExecutor(max_workers=len(role_list)) as ex:
        futs = {ex.submit(D._deepseek, build(rid, tpl), max_tokens, 0.3): rid
                for rid, tpl in role_list}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                text = (fut.result() or "").strip()
            except Exception as e:
                text = f"[{type(e).__name__}: {e}]"
            text = D._truncate(text, D.DEFAULT_MAX_CHARS)
            digests.append((rid, text))
            with open(os.path.join(run_dir, f"{rid}.md"), "w", encoding="utf-8") as f:
                f.write(text)

    digests.sort(key=lambda x: [r[0] for r in role_list].index(x[0]))

    # Judge/synth runs after all roles report (needs the full picture).
    if nudge:
        verdict = D._deepseek(_nudge_prompt(question, position, digests),
                              judge_tokens, 0.3).strip()
    elif mode == "desire":
        verdict = D._deepseek(_desire_judge_prompt(question, position, digests),
                              judge_tokens, 0.2).strip()
    else:
        verdict = D._deepseek(_judge_prompt(question, position, digests),
                              judge_tokens, 0.2).strip()
    with open(os.path.join(run_dir, "verdict.md"), "w", encoding="utf-8") as f:
        f.write(verdict)

    print(f"=== {mode.upper()}: {question[:120]} ===")
    print(f"(position: {position[:100]})\n")
    print("--- NUDGE ---" if nudge else "--- VERDICT ---")
    print(verdict)
    print(f"\n--- roles saved to {run_dir} ---")
    for rid, text in digests:
        print(f"  [{rid:12s}] {len(text)} chars")
    return 0


def main():
    p = argparse.ArgumentParser(description="Adversarial self-debate before committing")
    p.add_argument("question")
    p.add_argument("--position", default=None,
                   help="the answer/decision you're leaning toward (optional)")
    p.add_argument("--agents", default=None,
                   help="comma list (debate: advocate,critic,alternative,fact-checker; "
                        "desire: angel,devil,mirror)")
    p.add_argument("--mode", choices=["debate", "desire"], default="debate",
                   help="debate = truth stress-test; desire = angel/devil/mirror council")
    p.add_argument("--nudge", action="store_true",
                   help="desire fast path: mirror-only read + live move (skips full council)")
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--judge-tokens", type=int, default=700)
    a = p.parse_args()
    agents = [x.strip() for x in a.agents.split(",")] if a.agents else None
    return run(a.question, a.position, agents, a.max_tokens, a.judge_tokens,
               a.mode, a.nudge)


if __name__ == "__main__":
    sys.exit(main())
