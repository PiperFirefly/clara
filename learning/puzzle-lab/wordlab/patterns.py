"""wordlab.patterns — pattern search + anagrams over a local wordlist.

Pattern syntax (puzzle-hunt friendly, case-insensitive):
    .  or  _  -> exactly one unknown letter
    [abc]     -> one of a, b, c
    [^abc]    -> anything but a, b, c
    {n} {n,m} -> repeat the previous token (regex-style)
    ^ $       -> anchors (optional; patterns are full-match by default)
Everything else is literal.

The wordlist is wordfreq's 'large' list (~321k words), already sorted by
frequency, so matches rank naturally by commonness.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

from wordfreq import iter_wordlist, zipf_frequency

_WORDS: Optional[List[str]] = None
_BY_ALPHA: Optional[dict] = None  # alphagram -> list of words (for anagrams)
_CACHE: dict = {}  # lang -> (words, by_alpha)


def _load(lang: str = "en") -> Tuple[List[str], dict]:
    global _WORDS, _BY_ALPHA
    if lang not in _CACHE:
        words = list(iter_wordlist(lang, "large"))
        by_alpha = defaultdict(list)
        for w in words:
            if 2 <= len(w) <= 20:  # anagram space: skip 1-letter and absurdly long
                by_alpha["".join(sorted(w))].append(w)
        _CACHE[lang] = (words, dict(by_alpha))
    if lang == "en":
        _WORDS, _BY_ALPHA = _CACHE[lang]
    return _CACHE[lang]


def _compile(pattern: str) -> re.Pattern:
    """Translate friendly pattern syntax into a full-match regex."""
    # replace `_` with `.` (both mean "one unknown letter")
    pat = pattern.replace("_", ".")
    pat = re.sub(r"(?<!\\)\*", ".*", pat)          # * -> any run (greedy)
    try:
        return re.compile("^" + pat + "$", re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"bad pattern {pattern!r}: {e}") from e


def search(pattern: str, limit: int = 30, min_freq: float = 0.0,
           lang: str = "en") -> List[Tuple[str, float]]:
    """Return (word, zipf_frequency) matches for a pattern, best-first."""
    words, _ = _load(lang)
    rx = _compile(pattern)
    out = []
    for w in words:
        if rx.match(w):
            z = zipf_frequency(w, lang)
            if z >= min_freq:
                out.append((w, z))
        if len(out) >= limit:
            break
    return out


def alphagram(word: str) -> str:
    """Letters of `word` in sorted order (the anagram key)."""
    return "".join(sorted(word.lower()))


def anagram(letters: str, limit: int = 30, lang: str = "en") -> List[Tuple[str, float]]:
    """Single-word anagrams of `letters`, best-first."""
    _, by_alpha = _load(lang)
    cands = by_alpha.get(alphagram(letters), [])
    out = sorted(
        ((w, zipf_frequency(w, lang)) for w in cands),
        key=lambda x: -x[1],
    )
    return out[:limit]


def multi_anagram(letters: str, max_words: int = 3, limit: int = 30,
                  lang: str = "en") -> List[str]:
    """Anagram `letters` into up to `max_words` dictionary words.

    Exact-cover partition search over alphagrams, exploring most-common words
    first so natural splits surface fast. Returns phrases like
    ('the', 'phoenix', 'flew')."""
    words, by_alpha = _load(lang)
    target = alphagram(letters)
    if len(target) > 20:
        return []

    def is_submultiset(a: str, b: str) -> bool:
        it = iter(b)
        return all(ch in it for ch in a)

    # best (most common) word per distinct alphagram that fits in `target`
    alpha_best: dict = {}
    for a, ws in by_alpha.items():
        if 2 <= len(a) <= len(target) and is_submultiset(a, target):
            alpha_best[a] = max(ws, key=lambda w: zipf_frequency(w, lang))
    cand_alphas = sorted(alpha_best, key=lambda a: -zipf_frequency(alpha_best[a], lang))

    results: List[Tuple[str, ...]] = []
    seen = set()

    def rec(remaining: str, chosen: List[str], depth: int):
        if len(results) >= limit * 6:
            return
        words_left = max_words - depth
        for a in cand_alphas:
            if len(a) > len(remaining):
                continue
            if a == remaining:                      # finishes exactly
                phrase = tuple(sorted(chosen + [alpha_best[a]]))
                if phrase not in seen:
                    seen.add(phrase)
                    results.append(phrase)
            elif words_left > 1 and len(a) <= len(remaining) - 2 \
                    and is_submultiset(a, remaining):
                rec(_subtract(remaining, a), chosen + [alpha_best[a]], depth + 1)

    rec(target, [], 0)
    # rank: fewer words first, then more common
    results.sort(key=lambda p: (len(p), -sum(zipf_frequency(w, lang) for w in p)))
    return [" ".join(p) for p in results[:limit]]


def _subtract(super: str, sub: str) -> str:
    """Remove the letters of `sub` (as a multiset) from `super`."""
    lst = list(super)
    for ch in sub:
        lst.remove(ch)
    return "".join(lst)


def segment(text: str) -> List[str]:
    """Split concatenated words with no spaces ('thephoenixflew' -> words)."""
    from wordninja import split
    return split(text)


class WordIndex:
    """Opaque handle to the loaded wordlist (avoids reloading)."""

    def __init__(self):
        _load()

    @property
    def size(self) -> int:
        return len(_load()[0])

    def search(self, pattern: str, limit: int = 30, lang: str = "en") -> List[Tuple[str, float]]:
        return search(pattern, limit, lang=lang)

    def anagram(self, letters: str, limit: int = 30, lang: str = "en") -> List[Tuple[str, float]]:
        return anagram(letters, limit, lang=lang)
