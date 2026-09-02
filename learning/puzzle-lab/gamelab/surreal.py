"""gamelab.surreal — Conway's combinatorial game theory, hand-rolled.

Two things live here:
  1. Surreal numbers / games — the `Game` class with the recursive
     {L | R} construction, comparison, addition, negation. This is the
     *actual* math underneath impartial games (nim, dots-and-boxes, etc).
  2. Nimbers (Sprague-Grundy) — mex, nim-sum, and a Nim solver.

No deps beyond the stdlib. These are small enough (and instructive enough)
that hand-rolling beats importing the niche <10-star CGT libraries.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from typing import FrozenSet, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Surreal numbers / games
# ---------------------------------------------------------------------------

# canonical hashable form: (frozenset(left forms), frozenset(right forms))
_Form = Tuple[FrozenSet, FrozenSet]


class Game:
    """A combinatorial game G = {L | R} (Conway). L/R are sets of Games."""

    __slots__ = ("left", "right")

    def __init__(self, left: Set["Game"] = frozenset(), right: Set["Game"] = frozenset()):
        self.left = frozenset(left)
        self.right = frozenset(right)

    # -- canonical hashable representation --------------------------------
    def _form(self) -> _Form:
        return (frozenset(g._form() for g in self.left),
                frozenset(g._form() for g in self.right))

    def __hash__(self) -> int:
        return hash(self._form())

    # -- comparison (Conway's rule, memoized) -----------------------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _le(a: _Form, b: _Form) -> bool:
        # a <= b iff no aL in a.left has (b <= aL) and no bR in b.right has (bR <= a)
        aL, aR = a
        bL, bR = b
        for gl in aL:
            if Game._le(b, gl):
                return False
        for hr in bR:
            if Game._le(hr, a):
                return False
        return True

    def __le__(self, other: "Game") -> bool:
        return Game._le(self._form(), other._form())

    def __ge__(self, other: "Game") -> bool:
        return other <= self

    def __eq__(self, other) -> bool:
        if not isinstance(other, Game):
            return NotImplemented
        return (self <= other) and (other <= self)

    def __lt__(self, other: "Game") -> bool:
        return (self <= other) and not (other <= self)

    def __gt__(self, other: "Game") -> bool:
        return (other <= self) and not (self <= other)

    # -- arithmetic --------------------------------------------------------
    def __neg__(self) -> "Game":
        return Game({-g for g in self.right}, {-g for g in self.left})

    def __add__(self, other: "Game") -> "Game":
        left = {g + other for g in self.left} | {self + g for g in other.left}
        right = {g + other for g in self.right} | {self + g for g in other.right}
        return Game(left, right)

    def __sub__(self, other: "Game") -> "Game":
        return self + (-other)

    def __repr__(self) -> str:
        if self == ZERO:
            return "0"
        if self == ONE:
            return "1"
        if self == NEG_ONE:
            return "-1"
        return "{" + ",".join(map(repr, self.left)) + "|" + ",".join(map(repr, self.right)) + "}"


ZERO = Game()
ONE = Game({ZERO})
NEG_ONE = Game(frozenset(), {ZERO})


def from_int(n: int) -> Game:
    """The surreal number for integer n."""
    if n == 0:
        return ZERO
    if n > 0:
        return Game({from_int(n - 1)})
    return -from_int(-n)


def from_dyadic(p: int, q: int = 1) -> Game:
    """Surreal for a dyadic rational p/2^q. (q = power of 2 in the denominator.)"""
    # simplest surreal equal to p/2^q; via the standard recursive cut
    f = Fraction(p, 2 ** q)
    return _dyadic(f)


def _dyadic(f) -> Game:
    if f.denominator == 1:
        return from_int(f.numerator)
    # f = m / 2^k  ->  { f - 1/2^k | f + 1/2^k }
    k = f.denominator.bit_length() - 1
    step = Fraction(1, 2 ** k)
    return Game({_dyadic(f - step)}, {_dyadic(f + step)})


# ---------------------------------------------------------------------------
# Nimbers (Sprague-Grundy)
# ---------------------------------------------------------------------------

def mex(values) -> int:
    """Smallest non-negative integer not in `values`."""
    s = set(values)
    m = 0
    while m in s:
        m += 1
    return m


@lru_cache(maxsize=None)
def nimber(n: int) -> Game:
    """The nimber *n = { *0,...,*n-1 | *0,...,*n-1 }."""
    if n == 0:
        return ZERO
    opts = {nimber(i) for i in range(n)}
    return Game(opts, opts)


def nim_sum(*heaps: int) -> int:
    """Xor of heap sizes = Grundy value of a Nim position."""
    x = 0
    for h in heaps:
        x ^= h
    return x


def nim_winning_move(heaps) -> Optional[Tuple[int, int]]:
    """If the Nim position is a win, return (heap_index, new_size); else None."""
    x = nim_sum(*heaps)
    if x == 0:
        return None
    for i, h in enumerate(heaps):
        target = h ^ x
        if target < h:
            return (i, target)
    return None


def subtraction_game(k: int, n: int) -> bool:
    """Subtraction game: from pile of `n`, remove 1..k; losing positions are
    n ≡ 0 (mod k+1). Returns True if `n` is a winning position."""
    return n % (k + 1) != 0
