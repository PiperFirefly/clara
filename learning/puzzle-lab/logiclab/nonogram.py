"""logiclab.nonogram — Picross / nonogram solver (Z3).

A nonogram is a grid where each row/column's runs of filled cells are given as
clue numbers. Row clue [2,1] means "a run of 2, then a gap, then a run of 1".

Pass row_clues and col_clues as lists of lists. Returns a 0/1 grid (1=filled)
or None if the clues contradict.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import z3


def nonogram(row_clues: Sequence[Sequence[int]],
             col_clues: Sequence[Sequence[int]]) -> Optional[List[List[int]]]:
    H, W = len(row_clues), len(col_clues)
    cells = [[z3.Int(f"c_{r}_{c}") for c in range(W)] for r in range(H)]
    s = z3.Solver()
    _n = [0]  # name counter

    def constrain(cells_line, clues):
        cells_line = list(cells_line)
        if not clues:  # all zeros
            for c in cells_line:
                s.add(c == 0)
            return
        k = len(clues)
        starts = []
        for _ in range(k):
            starts.append(z3.Int(f"st_{_n[0]}"))
            _n[0] += 1
        for i, st in enumerate(starts):
            s.add(st >= 0, st + clues[i] <= len(cells_line))
        for i in range(k - 1):
            s.add(starts[i] + clues[i] + 1 <= starts[i + 1])
        for cpos, c in enumerate(cells_line):
            inside = z3.Or(*[z3.And(st <= cpos, cpos < st + clues[i])
                             for i, st in enumerate(starts)])
            s.add(c == z3.If(inside, 1, 0))

    for r in range(H):
        constrain([cells[r][c] for c in range(W)], row_clues[r])
    for c in range(W):
        constrain([cells[r][c] for r in range(H)], col_clues[c])

    for r in range(H):
        for c in range(W):
            s.add(cells[r][c] >= 0, cells[r][c] <= 1)

    if s.check() != z3.sat:
        return None
    m = s.model()
    return [[m.evaluate(cells[r][c]).as_long() for c in range(W)] for r in range(H)]
