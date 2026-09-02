"""logiclab.cryptarithm — alphametic / verbal arithmetic (Z3).

SEND + MORE = MONEY : each letter is a distinct digit, leading letters are
nonzero, and the column arithmetic holds. Pass the puzzle as a string.
"""

from __future__ import annotations

from typing import Dict, Optional

import z3


def cryptarithm(puzzle: str, base: int = 10) -> Optional[Dict[str, int]]:
    """Solve a verbal-arithmetic puzzle. `puzzle` = "SEND + MORE = MONEY".

    Returns {letter: digit} or None if unsatisfiable.
    """
    lhs, rhs = puzzle.replace(" ", "").split("=")
    words = lhs.split("+")
    letters = sorted(set("".join(words + [rhs])))

    v = {ch: z3.Int(ch) for ch in letters}
    s = z3.Solver()
    for ch in letters:
        s.add(v[ch] >= 0, v[ch] < base)
    s.add(z3.Distinct(*v.values()))
    for w in words + [rhs]:  # no leading zeros
        s.add(v[w[0]] != 0)

    def value(word):
        return z3.Sum(v[ch] * (base ** (len(word) - 1 - i))
                      for i, ch in enumerate(word))

    s.add(z3.Sum(value(w) for w in words) == value(rhs))
    if s.check() != z3.sat:
        return None
    m = s.model()
    return {ch: m.evaluate(v[ch]).as_long() for ch in letters}
