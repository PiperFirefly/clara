#!/usr/bin/env python3
"""
compute.py — program-of-thought: "when the answer is computable, compute it."

The cheapest cognition upgrade left (roadmap item #4 / my #1-RI ranking): intercept
tasks whose answer is determinable and compute it in Python instead of token-space
chain-of-thought. Cuts load on the expensive model (which rambles CoT) and gives an
exact, auditable answer for arithmetic / dates / simple logic.

Zero new dependencies: `ast` (safe expression evaluation), `decimal` (exact
arithmetic), `datetime` (date arithmetic), `math`. No packages to install.

Safety contract (CRITICAL):
  * NEVER evaluate untrusted input as arbitrary code. `_Eval` walks a whitelisted
    AST — numeric literals, the 4 operators, %, **, //, unary +/- , a comparison,
    and a fixed set of functions (abs/round/min/max/pow/sqrt/floor/ceil). Any node
    outside the whitelist raises -> the expression is NOT computed.
  * Fail safe toward the LLM: if we can't parse a *pure* deterministic expression,
    or the answer wouldn't be trustworthy, return None and let the model handle it.
    Over-eager or wrong computation is worse than no computation.

Usage:
  python3 compute.py "what is 1234 * 5678"
  python3 compute.py "what day is it in 45 days"
  import compute; compute.try_compute("12 + 7")   # -> ("19", "arithmetic") | None
"""
import ast
import datetime as _dt
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Safe arithmetic via a whitelisted AST walker
# ---------------------------------------------------------------------------
_ALLOWED_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
}

def _to_number(node):
    """Safely coerce a whitelisted AST value into a Decimal/float with no eval."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        try:
            return Decimal(str(node.value))
        except InvalidOperation:
            return float(node.value)
    raise ValueError("non-numeric literal")


def _eval_expr(node):
    """Evaluate a whitelisted arithmetic AST node. Raises on anything unknown."""
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _to_number(node)
    if isinstance(node, ast.BinOp):
        l, r = _eval_expr(node.left), _eval_expr(node.right)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r
        if isinstance(node.op, ast.FloorDiv):
            return l // r
        if isinstance(node.op, ast.Mod):
            return l % r
        if isinstance(node.op, ast.Pow):
            # Decimal ** with non-integer exponent is unsupported; use float.
            return float(l) ** float(r)
        raise ValueError("unsupported operator")
    if isinstance(node, ast.UnaryOp):
        v = _eval_expr(node.operand)
        return -v if isinstance(node.op, ast.USub) else (v if isinstance(node.op, ast.UAdd) else None)  # noqa: E501
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("disallowed function")
        args = [_eval_expr(a) for a in node.args]
        fn = _ALLOWED_FUNCS[node.func.id]
        return fn(*args)
    if isinstance(node, ast.Compare):
        # single comparison of two numeric operands
        left = _eval_expr(node.left)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("only simple comparisons")
        right = _eval_expr(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Lt):   return left < right
        if isinstance(op, ast.Gt):   return left > right
        if isinstance(op, ast.LtE):  return left <= right
        if isinstance(op, ast.GtE):  return left >= right
        if isinstance(op, ast.Eq):   return left == right
        if isinstance(op, ast.NotEq):return left != right
        raise ValueError("unsupported comparison")
    if isinstance(node, ast.BoolOp):
        vals = [_eval_expr(v) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_expr(node.operand)
    raise ValueError("disallowed node")


def _fmt(x):
    """Render a numeric result tersely (strip trailing .0, keep precision)."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if math.isclose(x, round(x), abs_tol=1e-9):
            return str(int(round(x)))
        return f"{x:g}"
    if isinstance(x, Decimal):
        if x == x.to_integral_value():
            return str(int(x))
        return str(x.quantize(Decimal("1e-6"), rounding=ROUND_HALF_UP)).rstrip("0").rstrip(".")
    return str(x)


# arithmetic expression detection: digits/operators/parens/dots, no identifiers
_ARITH_RE = re.compile(r"(?<![\w)])([-+]?\s*\d[\d\s.,+\-*/%()^]*\d)(?!\w)")

def _arith(t):
    """Try to evaluate a purely-numeric expression; return answer string or None."""
    s = t.strip()
    if not re.fullmatch(r"[-+*/%()\d\s.,^]*", s):
        return None
    s = s.replace("^", "**")
    try:
        tree = ast.parse(s, mode="eval")
        val = _eval_expr(tree.body)
    except Exception:
        return None
    if not isinstance(val, (int, float, Decimal, bool)):
        return None
    return _fmt(val)


def _arith_word(t):
    """Word-form arithmetic: 'X plus Y', 'X times Y', 'X divided by Y'."""
    m = re.fullmatch(
        r"\s*([\d.]+)\s+(plus|minus|times|multiplied by|divided by|divided into|"
        r"over|taken away from|subtract(?:ed)? from)\s+([\d.]+)\s*", t.strip(), re.I)
    if not m:
        return None
    a = Decimal(m.group(1)); b = Decimal(m.group(3))
    op = m.group(2).lower()
    try:
        if op in ("plus", "add", "added to"):
            return _fmt(a + b)
        if op in ("minus", "subtracted from", "take away"):
            return _fmt(a - b)
        if op in ("times", "multiplied by"):
            return _fmt(a * b)
        if op in ("divided by", "over"):
            return _fmt(a / b)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Date arithmetic (no external libs)
# ---------------------------------------------------------------------------
def _parse_date(s):
    s = s.strip().lower()
    today = _dt.date.today()
    if s in ("today", "now"):
        return today
    if s == "tomorrow":
        return today + _dt.timedelta(days=1)
    if s == "yesterday":
        return today - _dt.timedelta(days=1)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d %B %Y", "%d %b %Y",
                "%B %d, %Y", "%b %d, %Y", "%B %d", "%b %d"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # month name + day without year -> current year
    m = re.fullmatch(r"([a-z]+)\s+(\d{1,2})", s)
    if m:
        for fmt in ("%B", "%b"):
            try:
                mo = _dt.datetime.strptime(m.group(1), fmt).month
                return _dt.date(today.year, mo, int(m.group(2)))
            except ValueError:
                continue
    return None


_DATE_RE = re.compile(
    r"(?i)(?:what(?: is|'s)?(?: the)? (?:day|date)|which day|when)\b.*?"
    r"\b(\d+)\s+(day|week|month|year)s?\s*"
    r"(?:from\s+)?(today|now|tonight|tomorrow|this week)?\b")

def _add_months(d, n):
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return _dt.date(y, m, day)

def _date_arith(t):
    """'what day is it in N days', 'what is the date in 3 weeks', etc."""
    m = _DATE_RE.search(t)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    base = _dt.date.today()
    if unit == "day":
        res = base + _dt.timedelta(days=n)
    elif unit == "week":
        res = base + _dt.timedelta(weeks=n)
    elif unit == "month":
        res = _add_months(base, n)
    else:
        res = _dt.date(base.year + n, base.month, base.day)
    return res.strftime("%A, %B %d, %Y")


def _days_between(t):
    """'how many days between <a> and <b>' / 'how many days from <a> to <b>'."""
    m = re.search(
        r"(?i)how many (days|weeks|months|years) (between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)[?.\s]*$", t)
    if not m:
        return None
    a, b = _parse_date(m.group(3)), _parse_date(m.group(4))
    if a is None or b is None:
        return None
    unit = m.group(1)
    delta = abs((b - a).days)
    if unit == "days":
        return str(delta)
    if unit == "weeks":
        return _fmt(Decimal(delta) / 7)
    if unit == "months":
        return str(round(delta / 30.44))
    return str(round(delta / 365.25))


def _day_of_week(t):
    """'what day of the week is <date>' / 'what day is <date>'."""
    m = re.search(r"(?i)what day(?: of the week)? (?:is|was|will)\s+(.+?)[?.\s]*$", t)
    if not m or "in " in m.group(1) and _DATE_RE.search(t):
        # avoid stealing 'what day is it in N days' (handled by _date_arith)
        pass
    if not m:
        return None
    d = _parse_date(m.group(1))
    if d is None:
        return None
    return d.strftime("%A")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def try_compute(text):
    """Return (answer_string, method) if `text` has a determinable answer, else None.

    Methods: 'arithmetic', 'arith-word', 'date', 'days-between', 'day-of-week'.
    """
    if not text:
        return None
    t = text.strip().strip("?.")
    # strip a leading 'what is' / 'compute' / 'calculate' framing
    t2 = re.sub(r"(?i)^\s*(what\s+is|what's|compute|calculate|how much is|"
                r"what\s+does|solve)\s*:?\s*", "", t).strip()

    # arithmetic — try the whole (reframed) string first, then embedded candidates
    r = _arith(t2)
    if r:
        return (r, "arithmetic")
    # embedded: 'what is <expr>' (only a real question framing, not a bare '=')
    m = re.search(r"(?i)(?:what (?:is|'s)|compute|calculate|how much is)\s+([-+*/%()\d\s.,^]+)\s*[?]?\s*$", t)
    if m:
        r = _arith(m.group(1))
        if r:
            return (r, "arithmetic")
    r = _arith_word(t2)
    if r:
        return (r, "arith-word")
    r = _date_arith(t)
    if r:
        return (r, "date")
    r = _days_between(t)
    if r:
        return (r, "days-between")
    r = _day_of_week(t)
    if r:
        return (r, "day-of-week")
    return None


def main():
    p = __import__("argparse").ArgumentParser(description="program-of-thought compute")
    p.add_argument("text")
    a = p.parse_args()
    res = try_compute(a.text)
    if res:
        print(f"{a.text}\n  = {res[0]}   [{res[1]}]")
    else:
        print(f"{a.text}\n  not computable -> defer to LLM")


if __name__ == "__main__":
    main()
