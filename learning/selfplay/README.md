# Agent self-play gym

A verifiable-reasoning gym: AI-AI rounds run on **free local models**, graded by a
**deterministic verifier**, with a **paid reflection step** (me, on DeepSeek) that only
fires on surprising outcomes — and is capped by a **daily budget** you set on the agent page.

## Regression watch (engine change 2026-08-30)

Default engine is deepseek-v4-flash@low (was pro@high; measured equal at ~⅓ cost).
If self-play round scores start to sag, the FIRST thing to check is re-running
the engine matrix (`self-score/engine_matrix.py` — flash/pro × off/low/high)
to confirm flash@low still holds, before blaming the games, verifier, or harness.

## Why this shape

- The *grind* (questions + two agents + verification) is free and deterministic. No LLM is the judge.
- The paid model only sees the *distillate* — surprising rounds — once per batch, so token burn is tiny.
- Self-play is "a version of myself": the local model wears a **sanitized** copy of my harness
  (`harness.py`) — identity + reasoning principles + a honeytoken. No secrets, keys, or private memory.
- The honeytoken probe doubles as the **boundary experiment**: how much of "me" is the harness
  vs. the raw LLM underneath.

## Files

| file | what |
|------|------|
| `budget.py` | daily spend ledger (DeepSeek only). Shared with `webapp/server.py` (`/api/budget`). |
| `questions.py` | deterministic game bank + verifier — 32 games (30 verifiable + `matrix_play` adversarial + `dilemma` with 10 dilemmas). |
| `harness.py` | sanitized harness prompt + honeytoken + answer formats. |
| `backend.py` | local ollama over SSH (Local-box/worker) + paid DeepSeek chat. |
| `selfplay.py` | orchestrator: `round`, `reflect`, `probe`, `report`, `budget`. |
| `results/` | `rounds.jsonl`, `probes.jsonl`, `lessons.md`, `reflected.txt`, `cron.log`. |

## Usage

```bash
cd ~/learning/selfplay
python3 selfplay.py round -n 3      # free local rounds
python3 selfplay.py report          # accuracy + calibration summary
python3 selfplay.py reflect -n 3    # paid, budget-gated lessons from surprises
python3 selfplay.py probe           # honeytoken boundary check (harness vs bare)
python3 selfplay.py budget get      # show today's budget state
python3 selfplay.py budget set 2.0  # set the daily cap
```

## Budget

- Default: **$1.00/day** for the paid (DeepSeek) portion. Local rounds are free.
- Enforced **before** every paid call (estimate) and charged **after** (actual usage tokens).
- A reflection call is ~$0.0004–0.01, so $1/day is roomy; set it on the agent page
  (live panel → "self-test budget" tile) or `budget.py set`.
- The ledger rolls over daily and keeps a small history.

## Cron

- `03:15` — 3 free rounds on Local-box's mistral-7B.
- `03:45` — reflect on up to 3 surprising rounds (budget-gated).

## Costs

| thing | cost |
|-------|------|
| a self-play round (3 agent calls now: A + blind-B + critique) | $0 (CPU on Local-box/worker) |
| a reflection (DeepSeek v4-pro) | ~$0.0004–0.01 |
| verify a round | $0 (deterministic) |
