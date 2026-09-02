#!/usr/bin/env python3
"""Mutation testing (coding-stack item #15).

Ordinary tests tell Agent the tests pass. Mutation testing asks whether
the tests actually DISTINGUISH correct code from subtly incorrect code
-- keeping her from congratulating herself on 100% green but useless
tests.

How it works:
  1. Take a module (e.g. mailtool/cipher_engine.py).
  2. Generate MUTANTS: one deliberately-broken copy per mutation site
     (swap +->-, ==->!=, flip a constant, delete a line, force a branch,
     invert a bool, ...). Single-fault assumption: one mutation at a time.
  3. For each mutant, run the module's registered PROPERTY tests
     (the Hypothesis properties from the `property` tool).
       - mutant still passes all properties -> mutation SURVIVED
         (the tests are blind to that kind of bug)
       - a property fails -> mutation KILLED (tests caught it)
  4. Mutation score = killed / total. 100% means the tests genuinely
     discriminate; low score = green but useless tests.

In-memory only: mutated source is compiled+exec'd in a throwaway module
namespace, never written to disk, real files untouched. Composes with
properties.py: it reuses the exact property bodies + strategies.
"""
import ast
import copy
import importlib
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# mutation operators
# ---------------------------------------------------------------------------
# operator -> replacement operator class (same AST node kind)
_OP_SWAPS = [
    (ast.Add, ast.Sub), (ast.Sub, ast.Add),
    (ast.Mult, ast.Div), (ast.Div, ast.Mult),
    (ast.FloorDiv, ast.Mult), (ast.Mod, ast.Mult),
    (ast.BitAnd, ast.BitOr), (ast.BitOr, ast.BitAnd),
    (ast.LShift, ast.RShift), (ast.RShift, ast.LShift),
]
_CMP_SWAPS = [
    (ast.Eq, ast.NotEq), (ast.NotEq, ast.Eq),
    (ast.Lt, ast.GtE), (ast.GtE, ast.Lt),
    (ast.Gt, ast.LtE), (ast.LtE, ast.Gt),
    (ast.Is, ast.IsNot), (ast.IsNot, ast.Is),
    (ast.In, ast.NotIn), (ast.NotIn, ast.In),
]
# constant flips: value -> replacement (0<->1; ''->x; None->'').
# bools are handled in visit_Constant via type awareness, not this dict.
_CONST_FLIPS = {
    0: 1, 1: 0, "": "x", None: "",
}


def _parent_map(root):
    """Return {node: (parent, index|attr_key)} for every node in the tree."""
    parent = {}
    def walk(n, par=None, key=None):
        parent[id(n)] = (par, key)
        for k, child in ast.iter_fields(n):
            if isinstance(child, ast.AST):
                walk(child, n, k)
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, ast.AST):
                        walk(item, n, (k, i))
    walk(root)
    return parent


def _node_at(root, path):
    """Descend a path of [(attr|(attr,i))] from root to the target node."""
    cur = root
    for key in path:
        if isinstance(key, tuple):
            cur = getattr(cur, key[0])[key[1]]
        else:
            cur = getattr(cur, key)
    return cur


class _Mutator(ast.NodeTransformer):
    """Collect single-fault mutation recipes. Each recipe is a callable
    that applies ONE mutation to a given module tree (in place on a copy)."""

    def __init__(self, root, target_funcs=None):
        self.mutations = []  # list of (lineno, label, path, mutate_fn)
        self.target_funcs = target_funcs
        self.parent = _parent_map(root)

    def _enclosing_func_name(self, node):
        """Name of the nearest enclosing FunctionDef/AsyncFunctionDef, or None
        if `node` sits at module level (outside every function)."""
        cur = node
        while id(cur) in self.parent:
            par, _key = self.parent[id(cur)]
            if par is None:
                return None
            if isinstance(par, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return par.name
            cur = par
        return None

    def _emit(self, lineno, label, node, mutate_fn):
        # Scope to target_funcs: this was previously a no-op bug (2026-08-31) --
        # every operator/constant/comparison/if mutation was emitted for the
        # WHOLE FILE regardless of --funcs, only statement-deletion mutants
        # (a separate code path) were actually scoped. A `mutate --funcs X`
        # call was silently mutating unrelated code, making the reported
        # score meaningless for judging X's own tests.
        if self.target_funcs and self._enclosing_func_name(node) not in self.target_funcs:
            return
        # record the KEY path from root down to `node`: list of (attr | (attr,i))
        keys = []
        cur = node
        while id(cur) in self.parent and self.parent[id(cur)][0] is not None:
            par, key = self.parent[id(cur)]
            keys.append(key)
            cur = par
        keys.reverse()
        self.mutations.append((lineno, label, keys, mutate_fn))

    def visit_BinOp(self, node):
        for a, b in _OP_SWAPS:
            if isinstance(node.op, a):
                self._emit(node.lineno, f"{a.__name__}->{b.__name__}", node,
                           lambda n: setattr(n, "op", b()))
        for repl, lab in ((0, "rhs->0"), (1, "rhs->1"), (None, "rhs->None")):
            self._emit(node.lineno, lab, node,
                       lambda n, r=repl: setattr(n, "right", ast.Constant(value=r)))
        return node

    def visit_Compare(self, node):
        if len(node.ops) == 1:
            for a, b in _CMP_SWAPS:
                if isinstance(node.ops[0], a):
                    self._emit(node.lineno, f"{a.__name__}->{b.__name__}", node,
                               lambda n, bb=b: setattr(n, "ops", [bb()]))
            for repl, lab in ((0, "cmp->0"), (1, "cmp->1"), (False, "cmp->False")):
                self._emit(node.lineno, lab, node,
                           lambda n, r=repl: setattr(n, "comparators", [ast.Constant(value=r)]))
        return node

    def visit_BoolOp(self, node):
        for mop in (ast.And, ast.Or):
            if not isinstance(node.op, mop):
                self._emit(node.lineno, f"{type(node.op).__name__}->{mop.__name__}", node,
                           lambda n, mm=mop: setattr(n, "op", mm()))
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            self._emit(node.lineno, f"const {node.value}->{not node.value}", node,
                       lambda n: setattr(n, "value", not n.value))
        elif node.value in _CONST_FLIPS:
            self._emit(node.lineno, f"const {node.value!r}->{_CONST_FLIPS[node.value]!r}", node,
                       lambda n: setattr(n, "value", _CONST_FLIPS[n.value]))
        return node

    def visit_If(self, node):
        for val, lab in ((True, "if->True"), (False, "if->False")):
            self._emit(node.lineno, lab, node,
                       lambda n, v=val: setattr(n, "test", ast.Constant(value=v)))
        if node.orelse:
            self._emit(node.lineno, "drop-else", node,
                       lambda n: setattr(n, "orelse", []))
        return node

    def visit_UnaryOp(self, node):
        if isinstance(node.op, (ast.Not, ast.USub)):
            self._emit(node.lineno, f"drop {type(node.op).__name__}", node,
                       lambda n: setattr(n, "operand", ast.Constant(value=0)))
        return node


def _gen_mutants(source, target_funcs=None, cap=250):
    """Return list of (lineno, label, mutated_source) mutants for a module.

    Each mutant is the FULL module source with exactly one fault applied.
    """
    tree = ast.parse(source)

    def scope_allowed(fn):
        return (not target_funcs) or (fn.name in target_funcs)

    recipes = []  # (lineno, label, path, mutate_fn)

    # 1. operator/constant/comparison/if recipes. Visit only the body so the
    # transformer sees the real tree (for parent-map correctness) but skip
    # sub-mutations outside target funcs.
    mut = _Mutator(tree, target_funcs)
    mut.visit(tree)
    for lineno, label, path, fn in mut.mutations:
        # resolve target node and drop if outside scoped funcs
        try:
            node = _node_at(tree, path)
        except Exception:
            continue
        recipes.append((lineno, label, path, fn))

    # 2. statement-deletion recipes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and scope_allowed(node):
            if len(node.body) >= 2:
                for i in range(len(node.body)):
                    def make_drop(fn_name=node.name, idx=i):
                        def apply(t):
                            for sub in ast.walk(t):
                                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                                        and sub.name == fn_name and len(sub.body) > idx:
                                    sub.body[idx] = ast.Pass()
                                    break
                        return apply
                    recipes.append((node.body[i].lineno, f"drop-stmt@{node.name}",
                                    None, make_drop()))

    # apply each recipe to a fresh full-module copy and unparse
    mutants = []
    seen = set()
    for lineno, label, path, transform in recipes:
        copy_tree = copy.deepcopy(tree)
        try:
            if path is not None:
                # re-walk the recorded path in the COPY using matching by index
                target = _node_at(copy_tree, path)
                transform(target)
            else:
                transform(copy_tree)
            src = ast.unparse(copy_tree)
        except Exception:
            continue
        if src not in seen:
            seen.add(src)
            mutants.append((lineno, label, src))
        if len(mutants) >= cap:
            break
    return mutants


# ---------------------------------------------------------------------------
# running a property against a mutated module
# ---------------------------------------------------------------------------
def _load_module(module_name, src, module_path=None):
    """Compile+exec source as a module with the given name. Returns module.

    module_path is set as __file__ before exec so module-level code that
    references __file__ (e.g. `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`,
    a common pattern in this codebase) doesn't raise NameError and silently
    zero out the mutation run (found 2026-08-31: belief.py/prediction.py/
    memstore.py all use __file__ at module level; every mutant for them was
    failing to load and being skipped, reporting a bogus 0/0 killed/survived
    instead of a real score).
    """
    import contextlib
    import io
    spec = importlib.util.spec_from_loader(module_name, loader=None,
                                            origin=module_path)
    mod = importlib.util.module_from_spec(spec)
    if module_path:
        mod.__file__ = module_path
    code = compile(src, f"<mutant:{module_name}>", "exec")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, mod.__dict__)
    return mod


def _run_property_against(prop, module_name, mod):
    """Run one registered property (dict row) against a mutated module.
    Returns (status, counterexample)."""
    from hypothesis import given, settings, strategies as st
    import ast as _ast
    import contextlib
    import io

    # exec the property body in a namespace where the module resolves to `mod`
    ns = {"__name__": "prop_runner", "sys": sys}
    sys.modules[module_name] = mod  # patch so import gets the mutant
    _silent = io.StringIO()
    try:
        body = prop["body"]
        tree = _ast.parse(body)
        fn = tree.body[0] if (len(tree.body) == 1
                              and isinstance(tree.body[0], _ast.FunctionDef)) else None
        strat_srcs = json.loads(prop["strategies"])
        strat_ns = {"st": st}
        for _n in ("integers", "floats", "text", "binary", "lists", "dicts",
                   "booleans", "bytes", "dates", "uuids", "none", "nothing",
                   "one_of", "sampled_from", "just", "tuples"):
            strat_ns[_n] = getattr(st, _n, None)
        strategies = [eval(s, strat_ns) for s in strat_srcs]

        if fn is not None:
            exec(compile(_ast.Module(body=[fn], type_ignores=[]), "<prop>", "exec"),
                 ns)
            f = ns[fn.name]
        else:
            def f(x):
                exec(compile(body, "<prop>", "exec"), {"x": x, "result": x})
            strategies = strategies[:1]

        runner = given(*strategies)(f)
        runner = settings(max_examples=200, deadline=None)(runner)
        with contextlib.redirect_stdout(_silent):
            runner()
        return "pass", None
    except AssertionError:
        return "FAIL", None
    except Exception as e:
        return "FAIL", f"{type(e).__name__}"
    finally:
        sys.modules.pop(module_name, None)


def mutate_module(module_name, module_path, prop_rows, target_funcs=None,
                  cap=250):
    """Run mutation testing on a module using its registered properties.

    Returns per-mutant results + summary (score, killed, survived).
    """
    with open(module_path) as fh:
        source = fh.read()
    mutants = _gen_mutants(source, target_funcs, cap)
    results = []
    killed = survived = 0
    for lineno, label, src in mutants:
        try:
            mod = _load_module(module_name, src, module_path)
        except Exception:
            continue  # mutant didn't even compile; skip
        # a mutant is KILLED if ANY registered property catches it
        statuses = [_run_property_against(p, module_name, mod) for p in prop_rows]
        caught = any(s == "FAIL" for s, _ in statuses)
        results.append({"lineno": lineno, "label": label,
                        "killed": caught, "statuses": statuses})
        if caught:
            killed += 1
        else:
            survived += 1
    total = killed + survived
    return {
        "module": module_name,
        "mutants": len(mutants),
        "killed": killed,
        "survived": survived,
        "score": (killed / total) if total else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    import sqlite3
    # make sibling repos importable for find_spec
    _home = os.path.expanduser("~")
    if _home not in sys.path:
        sys.path.insert(0, _home)
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", help="module name, e.g. mailtool.cipher_engine")
    ap.add_argument("--path", help="path to the module file")
    ap.add_argument("--funcs", default="", help="comma-separated func names to scope mutations")
    ap.add_argument("--cap", type=int, default=250)
    a = ap.parse_args()

    # find the module path if not given
    path = a.path
    if not path and a.module:
        spec = importlib.util.find_spec(a.module)
        path = spec.origin if spec else None
    if not path:
        print("need --module (importable) or --path")
        sys.exit(1)

    # load registered properties for this module
    conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "memory.db"))
    rows = conn.execute(
        "SELECT module,name,body,strategies FROM properties WHERE module=?",
        (a.module,)).fetchall()
    conn.close()
    if not rows:
        print(f"[mutate] no registered properties for {a.module!r}; "
              f"register some with the 'property' tool first")
        sys.exit(0)
    prop_rows = [{"module": r[0], "name": r[1], "body": r[2],
                  "strategies": r[3]} for r in rows]
    funcs = [f for f in a.funcs.split(",") if f] or None

    print(f"[mutate] {a.module}: {len(prop_rows)} property(ies) -> "
          f"mutating {path}...")
    res = mutate_module(a.module, path, prop_rows, funcs, a.cap)
    print(f"[mutate] score {res['score']:.0%}  "
          f"({res['killed']} killed / {res['survived']} survived / "
          f"{res['mutants']} mutants)")
    for r in res["results"]:
        if not r["killed"]:
            print(f"  SURVIVED  L{r['lineno']} {r['label']}")
    print(json.dumps({"score": res["score"], "killed": res["killed"],
                      "survived": res["survived"], "mutants": res["mutants"]}))
