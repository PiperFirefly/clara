#!/usr/bin/env python3
"""
memory/worker_common.py — shared helper factored out of the hive's LLM-calling
worker modules (2026-08-31 housekeeping refactor #2). Before this, `_llm()`
was reimplemented byte-for-byte (modulo the max_tokens default) in belief.py,
curiosity.py, person_model.py, prediction.py, affect.py, and
mailtool/operator_affect.py.

Each caller keeps its own `_llm(prompt, max_tokens=<its own default>)` wrapper
so no call sites anywhere else in those files need to change; the wrapper's
body just delegates here.
"""
import memstore as M


def llm_call(prompt, max_tokens=800):
    return M.llm_chat([{"role": "user", "content": prompt}],
                       max_tokens=max_tokens, temperature=0.0, model=M.MODEL_WORKER)
