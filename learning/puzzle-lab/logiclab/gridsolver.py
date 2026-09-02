"""logiclab.gridsolver — constraint solvers over Z3 (SMT).

Every grid/logic puzzle is a constraint-satisfaction problem; Z3 is the general
engine. This module gives ready-made solvers for the most common hunt puzzle
families plus a template for writing your own.

    sudoku(puzzle)                      -> solved 9x9 grid
    latin_square(n, givens)             -> n x n Latin square
    skyscrapers(size, top,bottom,left,right) -> solved grid
    regex_crossword(rows, cols)         -> grid of letters (Z3 string theory)

A puzzle with 0 in `sudoku` means "empty cell". Solver returns None if
unsatisfiable (your givens contradict), or the filled grid.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import z3

# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def sudoku(puzzle: List[List[int]]) -> Optional[List[List[int]]]:
    """Solve a 9x9 sudoku. `puzzle` uses 0 for empty cells."""
    assert len(puzzle) == 9 and all(len(r) == 9 for r in puzzle)
    X = [[z3.Int(f"x_{r}_{c}") for c in range(9)] for r in range(9)]
    s = z3.Solver()
    for r in range(9):
        for c in range(9):
            s.add(X[r][c] >= 1, X[r][c] <= 9)
            if puzzle[r][c] != 0:
                s.add(X[r][c] == puzzle[r][c])
    for r in range(9):
        s.add(z3.Distinct(*X[r]))
    for c in range(9):
        s.add(z3.Distinct(*[X[r][c] for r in range(9)]))
    for br in range(3):
        for bc in range(3):
            s.add(z3.Distinct(*[X[br*3+i][bc*3+j] for i in range(3) for j in range(3)]))
    if s.check() != z3.sat:
        return None
    m = s.model()
    return [[m.evaluate(X[r][c]).as_long() for c in range(9)] for r in range(9)]


def latin_square(n: int, givens: Dict[Tuple[int, int], int] = None) -> Optional[List[List[int]]]:
    """n x n Latin square (each symbol once per row and column)."""
    givens = givens or {}
    X = [[z3.Int(f"x_{r}_{c}") for c in range(n)] for r in range(n)]
    s = z3.Solver()
    for r in range(n):
        for c in range(n):
            s.add(X[r][c] >= 1, X[r][c] <= n)
    for (r, c), v in givens.items():
        s.add(X[r][c] == v)
    for r in range(n):
        s.add(z3.Distinct(*X[r]))
    for c in range(n):
        s.add(z3.Distinct(*[X[r][c] for r in range(n)]))
    if s.check() != z3.sat:
        return None
    m = s.model()
    return [[m.evaluate(X[r][c]).as_long() for c in range(n)] for r in range(n)]


# ---------------------------------------------------------------------------
# Skyscrapers
# ---------------------------------------------------------------------------

def _visible(s: z3.Solver, seq: List[z3.ArithRef], n: int) -> z3.ArithRef:
    """How many buildings in `seq` are visible (taller than all before them)."""
    # visible[i] = seq[i] > max(seq[0..i-1]) ? 1 : 0
    vis = []
    for i in range(len(seq)):
        if i == 0:
            vis.append(1)
        else:
            mx = seq[0]
            for j in range(1, i):
                mx = z3.If(seq[j] > mx, seq[j], mx)
            vis.append(z3.If(seq[i] > mx, 1, 0))
    return z3.Sum(vis)


def skyscrapers(size: int, top: Sequence[int], bottom: Sequence[int],
                left: Sequence[int], right: Sequence[int]) -> Optional[List[List[int]]]:
    """Solve an N x N skyscrapers grid. Each side list gives visible counts,
    `0` = no clue."""
    n = size
    X = [[z3.Int(f"x_{r}_{c}") for c in range(n)] for r in range(n)]
    s = z3.Solver()
    for r in range(n):
        for c in range(n):
            s.add(X[r][c] >= 1, X[r][c] <= n)
    for r in range(n):
        s.add(z3.Distinct(*X[r]))
    for c in range(n):
        s.add(z3.Distinct(*[X[r][c] for r in range(n)]))
    # visibility from each side
    for c in range(n):
        col_top_bottom = [X[r][c] for r in range(n)]
        col_bottom_top = [X[n-1-r][c] for r in range(n)]
        if top[c]:
            s.add(_visible(s, col_top_bottom, n) == top[c])
        if bottom[c]:
            s.add(_visible(s, col_bottom_top, n) == bottom[c])
    for r in range(n):
        row_lr = [X[r][c] for c in range(n)]
        row_rl = [X[r][n-1-c] for c in range(n)]
        if left[r]:
            s.add(_visible(s, row_lr, n) == left[r])
        if right[r]:
            s.add(_visible(s, row_rl, n) == right[r])
    if s.check() != z3.sat:
        return None
    m = s.model()
    return [[m.evaluate(X[r][c]).as_long() for c in range(n)] for r in range(n)]


# ---------------------------------------------------------------------------
# Regex crossword (Z3 string theory)
# ---------------------------------------------------------------------------

_RSORT = None


def _regex_sort():
    global _RSORT
    if _RSORT is None:
        _RSORT = z3.Re("x").sort()
    return _RSORT


def _parse_re(pattern: str):
    """Parse a regex string into a Z3 regex AST.

    Supports literals, `.` (any char), `[...]` classes with `a-z` ranges,
    `* + ?`, `|`, and `(...)` groups. No lookarounds/backrefs/anchors (Z3
    regexes are full-match already, so `^$` are dropped)."""
    rs = _regex_sort()
    p = pattern.replace("^", "").replace("$", "")
    pos = [0]  # mutable counter — avoids nonlocal scoping headaches

    def char_lit(c: str):
        return z3.Re(c)

    def parse_class():
        pos[0] += 1  # consume '['
        parts = []
        chars = []
        while pos[0] < len(p) and p[pos[0]] != "]":
            if pos[0] + 2 < len(p) and p[pos[0]+1] == "-" and p[pos[0]+2] != "]":
                parts.append(z3.Range(p[pos[0]], p[pos[0]+2]))
                pos[0] += 3
            else:
                chars.append(p[pos[0]])
                pos[0] += 1
        pos[0] += 1  # consume ']'
        for c in chars:
            parts.append(z3.Range(c, c))
        out = parts[0]
        for prt in parts[1:]:
            out = z3.Union(out, prt)
        return out

    def parse_atom():
        c = p[pos[0]]
        if c == "(":
            pos[0] += 1
            inner = parse_alternation()
            pos[0] += 1  # consume ')'
            return inner
        if c == ".":
            pos[0] += 1
            return z3.AllChar(rs)
        if c == "[":
            return parse_class()
        if c == "\\" and pos[0] + 1 < len(p):
            pos[0] += 1
            c = p[pos[0]]
            pos[0] += 1
            return char_lit(c)
        pos[0] += 1
        return char_lit(c)

    def parse_repeat():
        atom = parse_atom()
        if pos[0] < len(p) and p[pos[0]] in "*+?":
            op = p[pos[0]]
            pos[0] += 1
            if op == "*":
                return z3.Star(atom)
            if op == "+":
                return z3.Plus(atom)
            return z3.Option(atom)
        return atom

    def parse_concat():
        items = []
        while pos[0] < len(p) and p[pos[0]] not in "|)":
            items.append(parse_repeat())
        if not items:
            return z3.Re("")
        if len(items) == 1:
            return items[0]
        return z3.Concat(*items)

    def parse_alternation():
        left = parse_concat()
        while pos[0] < len(p) and p[pos[0]] == "|":
            pos[0] += 1
            right = parse_concat()
            left = z3.Union(left, right)
        return left

    return parse_alternation()


def regex_crossword(rows: Sequence[str], cols: Sequence[str],
                    alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> Optional[List[str]]:
    """Solve a regex crossword: grid height = len(rows), width = len(cols).
    Row i must match rows[i], column j must match cols[j]. Returns list of row
    strings, or None. Z3 regexes lack lookarounds/backrefs but cover .*+?[]|()."""
    h, w = len(rows), len(cols)
    base = ord("A")
    cells = [[z3.Int(f"c_{r}_{c}") for c in range(w)] for r in range(h)]

    def _str(codes):
        codes = list(codes)
        if len(codes) == 1:
            return z3.StrFromCode(codes[0])
        return z3.Concat(*[z3.StrFromCode(c) for c in codes])

    s = z3.Solver()
    for r in range(h):
        for c in range(w):
            s.add(cells[r][c] >= 0, cells[r][c] < len(alphabet))
    for r in range(h):
        s.add(z3.InRe(_str(cells[r][c] + base for c in range(w)), _parse_re(rows[r])))
    for c in range(w):
        s.add(z3.InRe(_str(cells[r][c] + base for r in range(h)), _parse_re(cols[c])))
    if s.check() != z3.sat:
        return None
    m = s.model()
    out = []
    for r in range(h):
        out.append("".join(alphabet[m.evaluate(cells[r][c]).as_long()] for c in range(w)))
    return out
