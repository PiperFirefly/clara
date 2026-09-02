#!/usr/bin/env python3
"""
Polyglot advisor — Coding Cortex items #6 (capability registry) + #7 (cost model).

Gives Agent a machine-readable knowledge base of what each language is *unusually
good at* (concrete properties, not "Rust good / Python slow"), plus a decision rule
that only admits a new language when the specialist benefit clears the
interoperability + maintenance cost.

Two halves:
  #6 CAPABILITY REGISTRY — `capabilities(lang)` returns that language's concrete
     strengths; `best_for(task)` returns ranked candidates. Pure data, no LLM.
  #7 COST MODEL — `should_introduce(candidate, task, benefit, interop_cost,
     maintenance_cost)` applies:
        benefit > interop + maintenance  → introduce
     Deterministic, auditable, logged to memory.

Usage:
  polyglot.py capabilities <lang>            # concrete strengths of one language
  polyglot.py best <task>                     # ranked languages for a task
  polyglot.py decide <lang> <task> --benefit <x> --interop <y> --maint <z>
  polyglot.py list                            # all languages in the registry

The registry is structured data (JSON) in the docstore — machine-readable, so Agent
(or a worker) can query it without re-deriving it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))
import memstore as M

# ---------------------------------------------------------------------------
# #6 CAPABILITY REGISTRY
# Concrete properties per language. "unusually good at" = properties another
# language would struggle to match, not generic popularity. Each entry:
#   strengths:   concrete, falsifiable properties (the "why introduce this")
#   best_fit:    problem shapes this language is the natural home for
#   interop:     how it crosses language boundaries (serialization/FFI/embed)
#   footprint:   runtime/deploy/maintenance reality (for the cost model)
# ---------------------------------------------------------------------------
REGISTRY = {
    "rust": {
        "strengths": [
            "zero-copy parsing and buffer handling",
            "concurrency with compile-time memory-safety guarantees",
            "native libraries / FFI into C with no GC pause",
            "predictable low-level performance without a runtime",
        ],
        "best_fit": ["high-throughput parsers", "safety-critical concurrency",
                     "embedded/native components", "hot inner loops"],
        "interop": "FFI to C; manual serde at boundaries; heavier compile/build",
        "footprint": "statically linked binary; no runtime; slow first compile; strong toolchain",
    },
    "c_cpp": {
        "strengths": [
            "existing systems libraries and hardware/platform interfaces",
            "direct memory layout and ABI control",
            "mature OS/browser/embedded surfaces",
        ],
        "best_fit": ["driving existing C/C++ libraries", "platform/kernel-adjacent code",
                     "legacy integration", "hardware access"],
        "interop": "the lingua franca — everything binds to C ABI",
        "footprint": "no runtime; manual memory management; huge legacy surface",
    },
    "zig": {
        "strengths": [
            "tiny native helpers with C interoperability",
            "explicit allocation (no hidden allocator)",
            "comptime code generation",
        ],
        "best_fit": ["small drop-in native helpers", "replacing a C tool",
                     "explicit-ownership utilities"],
        "interop": "first-class C interop, can build C without a build system",
        "footprint": "small static binaries; young ecosystem",
    },
    "go": {
        "strengths": [
            "network services and concurrent server code",
            "goroutine concurrency with simple deployment (single static binary)",
            "fast builds, strong standard library",
        ],
        "best_fit": ["HTTP/network services", "CLI tools", "concurrent workers",
                     "simple deploys"],
        "interop": "cgo for C; JSON/grpc at boundaries; GC",
        "footprint": "static binary; GC pause; simple ops",
    },
    "python": {
        "strengths": [
            "orchestration and glue code",
            "scientific ecosystem (numpy/scipy/pandas)",
            "fast iteration, huge library surface",
        ],
        "best_fit": ["agent harness / orchestration", "data/scientific work",
                     "prototyping", "binding everything together"],
        "interop": "cffi/ctypes to C; subprocess; JSON; the glue language",
        "footprint": "interpreter + venv; slow inner loops; dominant for orchestration",
    },
    "julia": {
        "strengths": [
            "numerical/scientific computing with near-C performance",
            "multiple dispatch for math-heavy code",
        ],
        "best_fit": ["scientific simulation", "numerical analysis", "math kernels"],
        "interop": "PyCall/C; young tooling",
        "footprint": "JIT startup cost; niche ecosystem outside numerics",
    },
    "r": {
        "strengths": [
            "statistical analysis and modeling",
            "statistical graphics and test suites",
        ],
        "best_fit": ["statistics", "data analysis with heavy stats"],
        "interop": "via data frames / files; awkward for general code",
        "footprint": "interpreter; only worth it for stats",
    },
    "sql": {
        "strengths": [
            "set-oriented transformations over relational data",
            "declarative joins/filters/aggregates done in-engine",
        ],
        "best_fit": ["relational queries", "set operations", "aggregation"],
        "interop": "the universal query interface; embedded or server",
        "footprint": "in-DB; no new language runtime if a DB already exists",
    },
    "prolog_datalog": {
        "strengths": [
            "rule/relationship problems (inference, constraints, graphs)",
            "declarative backtracking / transitive closure",
        ],
        "best_fit": ["knowledge-graph inference", "rule systems", "constraint/relational reasoning"],
        "interop": "embed or subprocess; niche tooling",
        "footprint": "small; great fit for the capability graph's relational queries",
    },
    "lua": {
        "strengths": [
            "tiny embedded policy/scripting",
            "fast to embed, low footprint, C-friendly",
        ],
        "best_fit": ["embedded scripting", "policy/config-as-code", "sandboxed hooks"],
        "interop": "designed to be embedded in C",
        "footprint": "tiny runtime; easy to sandbox",
    },
    "typescript": {
        "strengths": [
            "browser/application interfaces",
            "typed on top of JS ecosystem; tooling",
        ],
        "best_fit": ["web/UI", "anything that must run in the browser or Node"],
        "interop": "JS everywhere; npm; node runtime",
        "footprint": "node runtime; npm dependency surface",
    },
    "wasm": {
        "strengths": [
            "portable sandboxed components",
            "runs untrusted code safely, near-native speed, multi-language compile target",
        ],
        "best_fit": ["sandboxing third-party code", "portable components", "isolated execution"],
        "interop": "compile many langs to it; host via wasmtime etc.",
        "footprint": "no runtime of its own; the sandbox boundary itself",
    },
}


def capabilities(lang):
    return REGISTRY.get(lang.lower())


def best_for(task, top=4):
    """Rank languages by how many of their 'best_fit' shapes match the task."""
    tl = task.lower()
    scored = []
    for lang, info in REGISTRY.items():
        score = 0
        for fit in info["best_fit"]:
            # naive keyword overlap; the registry is meant to be queried by
            # matching task terms against best_fit phrases
            for w in fit.split():
                if len(w) > 3 and w in tl:
                    score += 1
        # also count concrete strengths that name the task topic
        for s in info["strengths"]:
            for w in s.split():
                if len(w) > 3 and w in tl:
                    score += 1
        if score:
            scored.append((score, lang))
    scored.sort(key=lambda x: -x[0])
    return [(l, s) for s, l in scored[:top]]


# ---------------------------------------------------------------------------
# #7 COST MODEL — the language-boundary decision rule
# ---------------------------------------------------------------------------
def should_introduce(candidate, task, benefit=0.0, interop=0.0, maint=0.0,
                     persist=True, source="polyglot-cli"):
    """Decision: introduce a new language only if
       benefit  >  interop + maintenance.

    benefit/interop/maint are 0..1 judgements (from the human or an LLM pass):
      benefit   — how much the specialist property actually buys THIS task
      interop   — serialization/FFI/build/deploy boundary cost
      maint     — dependency, debugging, update, learning cost

    Returns the verdict dict; persists to memory when persist=True so the
    decision is auditable and recallable."""
    interop = max(0.0, min(1.0, interop))
    maint = max(0.0, min(1.0, maint))
    benefit = max(0.0, min(1.0, benefit))
    cost = interop + maint
    verdict = benefit > cost
    result = {
        "candidate": candidate,
        "task": task,
        "benefit": benefit,
        "interop_cost": interop,
        "maintenance_cost": maint,
        "total_cost": round(cost, 3),
        "verdict": "introduce" if verdict else "hold",
        "rule": "benefit > interop + maintenance",
        "margin": round(benefit - cost, 3),
    }
    if persist:
        try:
            M.remember(
                "Language-boundary decision: candidate={candidate} task={task} "
                "benefit={benefit} cost={cost} -> {verdict} (margin {margin})".format(
                    **{**result, "cost": result["total_cost"]}),
                kind="fact", importance=0.6)
        except Exception as e:  # noqa: BLE001 - memory must never block a verdict
            print(f"(note: could not persist decision: {e})")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _render_cap(lang, info):
    lines = [f"{lang}: "]
    lines.append("  strengths:")
    for s in info["strengths"]:
        lines.append(f"    - {s}")
    lines.append(f"  best_fit: {', '.join(info['best_fit'])}")
    lines.append(f"  interop: {info['interop']}")
    lines.append(f"  footprint: {info['footprint']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="polyglot advisor (capability + cost model)")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("capabilities", help="concrete strengths of one language")
    c.add_argument("lang")

    b = sub.add_parser("best", help="rank languages for a task")
    b.add_argument("task", nargs="+")

    d = sub.add_parser("decide", help="apply the cost model")
    d.add_argument("lang")
    d.add_argument("task", nargs="+")
    d.add_argument("--benefit", type=float, default=0.5)
    d.add_argument("--interop", type=float, default=0.4)
    d.add_argument("--maint", type=float, default=0.3)
    d.add_argument("--no-log", action="store_true")

    sub.add_parser("list", help="all languages in the registry")
    a = p.parse_args()

    if a.cmd == "capabilities":
        info = capabilities(a.lang)
        print(_render_cap(a.lang, info) if info else f"{a.lang} not in registry")
    elif a.cmd == "best":
        task = " ".join(a.task)
        for lang, score in best_for(task):
            print(f"  {lang} (score {score})")
        if not best_for(task):
            print(f"(no match in registry for '{task}')")
    elif a.cmd == "decide":
        r = should_introduce(a.lang, " ".join(a.task),
                             a.benefit, a.interop, a.maint,
                             persist=not a.no_log)
        print(json.dumps(r, indent=2))
    elif a.cmd == "list":
        for lang in REGISTRY:
            print(f"  {lang}: {', '.join(REGISTRY[lang]['best_fit'])}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
