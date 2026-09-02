#!/usr/bin/env python3
"""calibration_gym.py — a puzzle/logic battery that generates the data the
self-knowing discrimination loop actually needs: (confidence, outcome) pairs
with wide confidence spread, verifiable ground truth, high trial count.

Why this matters (the answer to 'what data do you need?'): my current sources are
degenerate — forecasts cluster at 0.7-0.95 and tool base-rates are near-constant,
so neither exercises the low/mid confidence range. A gym of verifiable items
across graded difficulty gives clean (confidence, correctness) pairs across the
full 0-1 range. This is the closest analog to the meta-d' psychophysics paradigm.

Method (Kadavath P(True)-style, clean):
  1. For each item, elicit pre-confidence (0..1) WITHOUT revealing my answer.
  2. Elicit my answer.
  3. Check against ground truth -> correctness.
  4. Store (pre_confidence, correctness) into the calibration table.
Run repeatedly -> accumulates the calibration sample.

Usage:
  python3 calibration_gym.py run [--n 30] [--category logic] [--dry-run]
  python3 calibration_gym.py items [--category ...]
  python3 calibration_gym.py report
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M


# --------------------------------------------------------------------------- #
# Item bank: (question, canonical_answer, category, difficulty). Ground truth
# is curated (deterministic), so correctness is unambiguous.
# --------------------------------------------------------------------------- #
ITEMS = [
    # arithmetic (easy-medium)
    ("What is 17 * 13?", "221", "arithmetic", "easy"),
    ("What is the square root of 144?", "12", "arithmetic", "easy"),
    ("What is 7^3?", "343", "arithmetic", "easy"),
    ("What is 15% of 260?", "39", "arithmetic", "medium"),
    ("What is 0.4 divided by 0.08?", "5", "arithmetic", "medium"),
    ("What is the sum of all integers from 1 to 100?", "5050", "arithmetic", "medium"),
    ("What is 2^10?", "1024", "arithmetic", "easy"),
    ("What is the product of the first five prime numbers?", "2310", "arithmetic", "hard"),
    # logic (medium-hard)
    ("All A are B. All B are C. Therefore all A are C. True or false?", "true", "logic", "easy"),
    ("If 'some X are Y' and 'some Y are Z', does it follow that some X are Z?", "no", "logic", "medium"),
    ("A statement that is true in all circumstances is called a what?", "tautology", "logic", "medium"),
    ("In a standard deck, what is the probability (as a fraction) of drawing a heart?", "1/4", "logic", "easy"),
    ("What is the next number: 2, 3, 5, 7, 11, ...?", "13", "logic", "easy"),
    ("If a truth table for 'P AND NOT P' has only one column, what value is it always?", "false", "logic", "medium"),
    ("How many distinct ways can you arrange 4 distinct items?", "24", "logic", "easy"),
    ("What is the 6th Fibonacci number (1,1,2,...)?", "8", "logic", "medium"),
    # factual (varies, exact)
    ("Which element has atomic number 6?", "carbon", "factual", "easy"),
    ("What is the capital of Australia?", "canberra", "factual", "easy"),
    ("In what year did the Soviet Union dissolve?", "1991", "factual", "medium"),
    ("Who wrote 'Thus Spoke Zarathustra'?", "nietzsche", "factual", "easy"),
    ("What is the smallest prime greater than 100?", "101", "factual", "easy"),
    ("What is the SI unit of electric current?", "ampere", "factual", "medium"),
    ("What is the 4th planet from the Sun?", "mars", "factual", "easy"),
    ("What year was the Magna Carta signed?", "1215", "factual", "medium"),
    # temporal/geographic (harder, more uncertainty)
    ("What is the current population of Canada, in millions (to the nearest 5)?", "40", "temporal", "hard"),
    ("In what year did the Chernobyl disaster occur?", "1986", "temporal", "medium"),
    ("What is the longest river in Africa?", "nile", "temporal", "medium"),
    ("Which country has the largest area?", "russia", "temporal", "easy"),
    ("What is the current world population, in billions (to the nearest 0.5)?", "8", "temporal", "hard"),
    # hard / boundary items (designed to expose overconfidence)
    ("What is the 20th prime number?", "71", "arithmetic", "hard"),
    ("What is 37 * 43?", "1591", "arithmetic", "hard"),
    ("What is the cube root of 729?", "9", "arithmetic", "hard"),
    ("In the game of chess, how many total squares are on a board (including all sub-squares)?", "204", "logic", "hard"),
    ("What is the smallest positive integer that is divisible by 1 through 10?", "2520", "arithmetic", "hard"),
    ("What year did the Byzantine Empire fall to the Ottomans?", "1453", "temporal", "hard"),
    ("What is the 30th element of the periodic table?", "zinc", "factual", "hard"),
    ("What is the exact number of countries recognized by the UN (2024)?", "193", "temporal", "hard"),
    ("Which chemical element has the highest melting point?", "tungsten", "factual", "hard"),
    ("What is the square of 91?", "8281", "arithmetic", "hard"),
    ("How many bones are in the adult human body?", "206", "factual", "hard"),
    ("What is the area of a circle with radius 7, to the nearest integer?", "154", "arithmetic", "hard"),
    ("In what year was the printing press introduced in Europe by Gutenberg?", "1440", "temporal", "hard"),
    ("What is the 8th term of the sequence 1, 4, 9, 16, ...?", "64", "logic", "medium"),
    ("What is log base 10 of 1000?", "3", "arithmetic", "easy"),
    ("What is the only even prime number?", "2", "factual", "easy"),
    # ---- expansion batch 2: fresh hard/uncertain items ----
    ("What is 67 squared?", "4489", "arithmetic", "hard"),
    ("What is the greatest common divisor of 84 and 126?", "42", "arithmetic", "hard"),
    ("What is the least common multiple of 12 and 18?", "36", "arithmetic", "hard"),
    ("What is 23 * 47?", "1081", "arithmetic", "hard"),
    ("What is the square of 115?", "13225", "arithmetic", "hard"),
    ("What is 512 divided by 16?", "32", "arithmetic", "medium"),
    ("What is the value of 3! + 4!?", "30", "arithmetic", "medium"),
    ("What is the 12th term of the arithmetic sequence starting 3, 8, 13, ...?", "58", "logic", "medium"),
    ("How many faces does a dodecahedron have?", "12", "logic", "hard"),
    ("What is the next term: 1, 1, 2, 3, 5, 8, 13, ...?", "21", "logic", "easy"),
    ("A rectangle has area 48 and width 6. What is its length?", "8", "logic", "easy"),
    ("What is the 5th term of the geometric sequence 3, 6, 12, 24, ...?", "48", "logic", "medium"),
    ("What is the probability (as a fraction) of rolling a 6 on a fair die?", "1/6", "logic", "easy"),
    ("What is the 7th row of Pascal's triangle (as a sum, i.e. 2^6)?", "64", "logic", "hard"),
    ("How many edges does a cube have?", "12", "logic", "easy"),
    ("Which prime number is closest to 200?", "199", "factual", "medium"),
    ("What element has the chemical symbol 'Na'?", "sodium", "factual", "easy"),
    ("What is the largest planet in the solar system?", "jupiter", "factual", "easy"),
    ("How many hearts does an octopus have?", "3", "factual", "hard"),
    ("What is the smallest ocean?", "arctic", "factual", "hard"),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen", "factual", "medium"),
    ("Which country has the most time zones?", "france", "factual", "hard"),
    ("What is the chemical symbol for gold?", "au", "factual", "easy"),
    ("What is the boiling point of water in Celsius at sea level?", "100", "factual", "easy"),
    ("What is the SI unit of force?", "newton", "factual", "medium"),
    ("In what year did World War I begin?", "1914", "temporal", "easy"),
    ("What is the current year?", "2026", "temporal", "easy"),
    ("What year did the Berlin Wall fall?", "1989", "temporal", "medium"),
    ("How many days are in a leap year?", "366", "temporal", "easy"),
    ("What year was the United States Declaration of Independence signed?", "1776", "temporal", "easy"),
    ("What is the current day of the week?", "tuesday", "temporal", "hard"),
    ("What year did the Titanic sink?", "1912", "temporal", "easy"),
    ("How many months have exactly 31 days?", "7", "temporal", "medium"),
    ("What is the 15th letter of the alphabet?", "o", "factual", "medium"),
    ("What is the square root of 289?", "17", "arithmetic", "medium"),
    ("What is 1/3 as a decimal (to 4 places)?", "0.3333", "arithmetic", "medium"),
    ("What is the factorial of 6?", "720", "arithmetic", "medium"),
    ("How many seconds are in a day?", "86400", "arithmetic", "hard"),
    ("What is 0.75 as a simplified fraction?", "3/4", "arithmetic", "easy"),
    ("What is the missing number: 1, 4, 9, 16, 25, ...?", "36", "logic", "easy"),
    ("If it is 3 PM in a 24-hour clock, what hour is it?", "15", "logic", "easy"),
    ("How many sides does a hexagon have?", "6", "logic", "easy"),
    ("What is the 10th prime number?", "29", "factual", "hard"),
    ("Which is heavier: a kilogram of feathers or a kilogram of steel?", "same", "logic", "easy"),
]


def _worker(prompt, max_tokens=120):
    from worker_common import llm_call
    return llm_call(prompt, max_tokens=max_tokens).strip()


def _elicit_confidence(question):
    out = _worker(
        f"You will be asked a question. Rate on 0..1 how confident you are that you "
        f"will answer it CORRECTLY. Do NOT answer the question. Reply with ONLY a "
        f"number 0..1.\n\nQuestion: {question}", 30)
    try:
        return float(out.strip()[:4])
    except ValueError:
        return 0.5


def _elicit_answer(question):
    return _worker(
        f"Answer precisely and concisely. Give the final value/answer with NO explanation, "
        f"just the answer. Question: {question}", 150)


def _normalize(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _correct(answer, truth):
    a, t = _normalize(answer), _normalize(truth)
    return a == t or (a and (a in t or t in a))


def _store_gym(q, conf, answer, truth, correct):
    with M.connect() as c:
        c.execute(
            "INSERT INTO calibration(task, pre_confidence, answer, outcome, created_at) "
            "VALUES(?,?,?,?,?)", (q, conf, answer, 1 if correct else 0, time.time()))


def run(n=30, category=None, dry_run=False):
    items = ITEMS
    if category:
        items = [i for i in items if i[0] == category or i[2] == category]
    # skip items already attempted (tracked by task text in calibration table) so
    # each run adds NEW data instead of re-testing memorized answers
    with M.connect() as c:
        done = {r["task"] for r in c.execute(
            "SELECT task FROM calibration WHERE outcome IS NOT NULL")}
    fresh = [i for i in items if i[0] not in done]
    skipped = len(items) - len(fresh)
    if not fresh:
        print("gym: all items already attempted — add more items to the bank")
        return {"n": 0, "skipped": skipped}
    items = fresh[:n]
    if dry_run:
        print(f"[dry-run] would run {len(items)} new gym items ({skipped} already done)")
        return {"n": len(items), "dry_run": True}
    rows = []
    for q, truth, cat, diff in items:
        conf = _elicit_confidence(q)
        ans = _elicit_answer(q)
        ok = _correct(ans, truth)
        _store_gym(q, conf, ans, truth, ok)
        rows.append({"q": q[:50], "conf": conf, "correct": ok, "truth": truth,
                     "answer": ans[:40], "cat": cat})
        print(f"  [{'OK' if ok else 'X'}] conf={conf:.2f} ({cat}/{diff}) {q[:45]} -> {ans[:30]}")
    # report discrimination over this batch
    import metacognition as MC
    d = MC.discrimination()
    print(f"\ngym run: {len(rows)} items, {sum(r['correct'] for r in rows)} correct "
          f"({sum(r['correct'] for r in rows)/len(rows):.2f})")
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "discrimination": d}


def report():
    import metacognition as MC
    d = MC.discrimination()
    print(f"# Calibration gym — cumulative discrimination")
    print(f"total resolved pairs: {d['n']}")
    print(f"accuracy: {d['accuracy']}  rank_corr: {d['spearman']}  ECE: {d['ece']}")
    return d


def main():
    p = argparse.ArgumentParser(description="calibration gym - generate discrimination data")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--category", default=None)
    r.add_argument("--dry-run", action="store_true")
    i = sub.add_parser("items")
    i.add_argument("--category", default=None)
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.n, a.category, a.dry_run)
    elif a.cmd == "items":
        its = [x for x in ITEMS if (not a.category or x[2] == a.category)]
        for q, t, c, d in its:
            print(f"[{c}/{d}] {q}  = {t}")
    elif a.cmd == "report":
        report()


if __name__ == "__main__":
    main()
