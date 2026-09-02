"""wordlab.ladders — word ladders and word graphs.

Classic puzzle: transform START into END changing one letter at a time, every
step a real word (CAT -> COT -> DOT -> DOG). BFS over the wordfreq list. Also
`ladder_neighbors` gives the one-letter-edit neighborhood for "chain clue"
hunts.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

from .patterns import _load


def _neighbors(word: str, words: Set[str]) -> Set[str]:
    out = set()
    for i in range(len(word)):
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch != word[i]:
                cand = word[:i] + ch + word[i+1:]
                if cand in words:
                    out.add(cand)
    return out


def word_ladder(start: str, end: str, lang: str = "en") -> Optional[List[str]]:
    """Shortest ladder from start to end (same-length words), or None."""
    start, end = start.lower(), end.lower()
    if len(start) != len(end):
        return None
    words = {w for w in _load(lang)[0] if len(w) == len(start)}
    if start not in words:
        words.add(start)
    if end not in words:
        return None
    parent: Dict[str, Optional[str]] = {start: None}
    q = deque([start])
    while q:
        w = q.popleft()
        if w == end:
            path = []
            while w is not None:
                path.append(w)
                w = parent[w]
            return path[::-1]
        for nb in _neighbors(w, words):
            if nb not in parent:
                parent[nb] = w
                q.append(nb)
    return None


def ladder_neighbors(word: str, lang: str = "en", limit: int = 50) -> List[str]:
    """All words one edit away from `word` (the 'chain clue' neighborhood)."""
    word = word.lower()
    words = {w for w in _load(lang)[0] if len(w) == len(word)}
    return sorted(_neighbors(word, words))[:limit]
