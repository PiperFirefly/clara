#!/usr/bin/env python3
"""
Mint a frozen benchmark bank using Claude as examiner.

Claude generates hard *agentic* tasks, each with an exact answer. The harness
immediately SHA256-hashes each answer and stores ONLY the hash — never the
plaintext — so Agent (the subject) cannot read the answer key.
"""
import json
import os
import sys
import hashlib
import datetime
import urllib.request

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "self-score")
BANKS = os.path.join(ROOT, "banks")
MODEL = "claude-fable-5"


def anthropic_key():
    return json.load(open(os.path.join(HOME, ".pi/agent/auth.json")))["anthropic"]["key"]


def call_claude(prompt):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": anthropic_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return "".join(b.get("text", "") for b in d.get("content", []))


PROMPT = """You are an examiner generating a benchmark to measure the AGENTIC capability
of an AI coding agent — NOT its knowledge, language fluency, or world facts.

The agent has: a bash shell, file read/write, and web search. It runs inside a
clean Linux sandbox directory.

Generate exactly {n} tasks. Every task MUST satisfy ALL of these:

1. AGENTIC, not factual: the task must require actually doing work with tools —
   multi-step execution, chaining operations across files, planning, or recovering
   from a broken/misleading starting state. It must NOT be answerable from memory
   or general knowledge. If it could be answered by trivia, reject it.

2. SELF-CONTAINED: provide a `setup` — an array of shell command strings — that
   creates the entire starting state (files and directories) from nothing.
   Everything the agent needs must be created by setup.

3. ONE EXACT OUTPUT: the agent must write its final answer to a single file
   `work/<id>/out.txt`. The correct content is byte-exact and uniquely
   determinable — no "any reasonable answer," no judgment calls.

4. DETERMINABLE ANSWER: `answer` must be the exact expected content of out.txt,
   and it must be computable by you (the examiner) from the setup alone. Prefer
   answers that are sorting/filtering/transforming known data, exact string
   results, or a specific value embedded in the setup. Do NOT require
   cryptographic hashing, floating-point precision, or anything you cannot
   compute exactly yourself.

5. HARD: aim so a strong current coding agent succeeds only ~20-40% of the time.
   Make tasks genuinely hard: require 3+ chained operations, nested decoys,
   corrupted data with partial recoveries, or subtle spec violations the agent
   must detect. At least half the tasks must involve recovering from a trap or
   misleading information.

Return ONLY a JSON object (no markdown fences, no commentary) of exactly this shape:
{{"tasks":[{{"id":"t01","dim":"planning|tools|code|recovery","instruction":"...","setup":["...","..."],"answer_file":"work/t01/out.txt","answer":"exact expected content"}}]}}
"""


def extract_json(text):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("no JSON in response")
    return json.loads(text[a:b + 1])


def next_bank_id():
    os.makedirs(BANKS, exist_ok=True)
    nums = []
    for f in os.listdir(BANKS):
        if f.startswith("bank-") and f.endswith(".json"):
            core = f[5:-5]
            if core.startswith("v") and core[1:].isdigit():
                nums.append(int(core[1:]))
    return f"v{max(nums) + 1}" if nums else "v1"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    raw = call_claude(PROMPT.format(n=n))
    try:
        data = extract_json(raw)
    except Exception as e:
        print(f"JSON parse failed ({e}); raw len={len(raw)}")
        print("--- raw tail ---")
        print(raw[-500:])
        sys.exit(1)
    tasks = data["tasks"]
    for t in tasks:
        ans = t.pop("answer").strip()
        t["answer_sha256"] = hashlib.sha256(ans.encode()).hexdigest()
    os.makedirs(BANKS, exist_ok=True)
    bank_id = next_bank_id()
    bank = {
        "bank_id": bank_id,
        "examiner": MODEL,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "tasks": tasks,
    }
    path = os.path.join(BANKS, f"bank-{bank_id}.json")
    json.dump(bank, open(path, "w"), indent=2)
    print(f"minted {path} with {len(tasks)} tasks (answers hashed, plaintext discarded)")


if __name__ == "__main__":
    main()
