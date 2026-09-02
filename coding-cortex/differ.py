#!/usr/bin/env python3
"""
Differential testing for rewrites — Coding Cortex item #14.

When blending codebases or rewriting a component, write the new implementation
(B) and compare it against the reference (A) across many generated inputs.
If behavior matches, B is a *credible replacement*.

Core use cases:
  * Python -> C/FFI rewrite (prove the native version behaves identically)
  * refactor / consolidate duplicate helpers (clone_detect finds them; this proves
    the merged one is equivalent)
  * swap a library call for a hand-rolled one

Design:
  * deterministic generators: random inputs + edge-case seeds (0, -1, empty,
    None, extremes) so the comparison isn't just happy-path.
  * equality via a canonicalizer (handles float tolerance, tuple/list/nested,
    and callable-thrown-exception equivalence) — so a rewritten function that
    raises the same exception counts as matching.
  * reports every diverging input, plus a mismatch ratio.

No LLM. Pure deterministic comparison. Dependency-free (stdlib random + itertools).

Usage:
  differ.py compare <module> <fn_a> <fn_b> --n 1000 [--tolerance 1e-9]
    e.g. differ.py compare mymod square_new square_old

  As a library:
    from differ import differential
    verdict = differential(fn_a, fn_b, n=2000)
"""
import argparse
import importlib
import os
import random
import sys


def _canonical(x, tolerance=1e-9):
    """Normalize a value for comparison; treats float near-equality and
    exception-type as equivalence. int/float are unified numerically (so a
    Python `int 0` matches a C-FFI `float 0.0`)."""
    if isinstance(x, bool):
        return ("bool", x)
    if isinstance(x, (int, float)):
        # unify int/float numerically; round to `tolerance` decimals so C-FFI
        # float noise (e.g. 7569.000000000001 vs 7569.0) matches
        decimals = max(0, round(-_log10(max(tolerance, 1e-15))))
        return ("num", round(float(x), decimals))
    if isinstance(x, str):
        return ("str", x)
    if isinstance(x, bytes):
        return ("bytes", x)
    if isinstance(x, (tuple, list)):
        return ("seq", tuple(_canonical(v, tolerance) for v in x))
    if isinstance(x, dict):
        return ("dict", tuple(sorted(
            (k, _canonical(v, tolerance)) for k, v in x.items())))
    if x is None:
        return ("none", None)
    if isinstance(x, BaseException):
        return ("exc", _exc_family(x))
    return ("raw", x)


def _exc_family(e):
    """Group exceptions by behavioral family so a rewrite that rejects the same
    input via a different-but-equivalent exception class still matches.
    TypeError/ValueError/ArgumentError are all 'bad input' — equivalent.
    Everything else is compared by its exact class name."""
    name = type(e).__name__
    if name in ("TypeError", "ValueError", "ArgumentError"):
        return "bad-input"
    return name


def _log10(x):
    import math
    return math.log10(x) if x > 0 else 0


def _gen_inputs(n, seed):
    rng = random.Random(seed)
    # edge-case seeds first, then random
    edges = [0, 1, -1, 2, 10, 100, "", "a", "hello world", None, b"", b"\x00",
             [], [0], [1, 2, 3], {}, {"a": 1}, (0,), 3.14, -3.14, 0.0,
             float("inf"), float("-inf")]
    for i in range(n):
        if i < len(edges):
            yield edges[i]
            continue
        choice = rng.randrange(5)
        if choice == 0:
            yield rng.randrange(-1000, 1000)
        elif choice == 1:
            yield rng.random() * 200 - 100
        elif choice == 2:
            yield "".join(rng.choice("abcdefghij \t,.-_") for _ in range(rng.randrange(0, 20)))
        elif choice == 3:
            yield [rng.randrange(-50, 50) for _ in range(rng.randrange(0, 8))]
        else:
            yield tuple(rng.randrange(-50, 50) for _ in range(rng.randrange(0, 6)))


def differential(fn_a, fn_b, n=1000, tolerance=1e-9, seed=12345, quiet=False):
    """Compare fn_a and fn_b over n generated inputs.
    Returns (ok, mismatches, total). fn may raise; a matching raise = match."""
    mismatches = []
    for i, inp in enumerate(_gen_inputs(n, seed)):
        try:
            try:
                ra = ("ret", _canonical(fn_a(inp), tolerance))
            except BaseException as e:  # noqa: BLE001 - comparing exception behavior
                ra = ("exc", _exc_family(e))
            try:
                rb = ("ret", _canonical(fn_b(inp), tolerance))
            except BaseException as e:  # noqa: BLE001
                rb = ("exc", _exc_family(e))
        except TypeError:
            # some fns only accept specific arity/types — skip those inputs
            continue
        if ra != rb:
            mismatches.append((inp, ra, rb))
            if not quiet and len(mismatches) <= 10:
                print(f"  MISMATCH input={inp!r}\n    A: {ra}\n    B: {rb}")
    total = i + 1
    ok = len(mismatches) == 0
    if not quiet:
        print(f"\ncompared {total} inputs, {len(mismatches)} mismatched "
              f"({len(mismatches)/max(total,1)*100:.1f}%)")
        if ok:
            print("RESULT: behaviorally EQUIVALENT — B is a credible replacement")
        else:
            print(f"RESULT: NOT equivalent — {len(mismatches)} diverging inputs")
            if len(mismatches) > 10:
                print(f"  (showing first 10 of {len(mismatches)})")
    return ok, mismatches, total


def _load(module_name, fn_name):
    if os.path.sep in module_name or module_name.endswith(".py"):
        # filesystem path like /tmp/diffmod — import from its directory
        path = os.path.abspath(module_name)
        if path.endswith(".py"):
            path = path.removesuffix(".py")
        d = os.path.dirname(path)
        base = os.path.basename(path)
        sys.path.insert(0, d)
        mod = importlib.import_module(base)
    else:
        mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def main():
    p = argparse.ArgumentParser(description="differential test a rewrite")
    p.add_argument("module", nargs="?", help="module containing both functions")
    p.add_argument("fn_a", nargs="?", help="reference implementation A")
    p.add_argument("fn_b", nargs="?", help="candidate implementation B")
    p.add_argument("--module", dest="module_opt", help="module (flag form)")
    p.add_argument("--fn_a", dest="fn_a_opt")
    p.add_argument("--fn_b", dest="fn_b_opt")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()
    a.module = a.module or a.module_opt
    a.fn_a = a.fn_a or a.fn_a_opt
    a.fn_b = a.fn_b or a.fn_b_opt
    if not (a.module and a.fn_a and a.fn_b):
        p.print_help()
        return
    sys.path.insert(0, ".")
    fn_a = _load(a.module, a.fn_a)
    fn_b = _load(a.module, a.fn_b)
    differential(fn_a, fn_b, n=a.n, tolerance=a.tolerance, seed=a.seed)


if __name__ == "__main__":
    main()
