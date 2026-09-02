"""wordlab.ciphers — classical cipher encode/decode/auto-solve.

Caesar (auto via wordlist scoring), Atbash, affine, Vigenere (keyed),
monoalphabetic substitution (hill-climbing auto-solver), and a letter
frequency table. Scoring uses wordfreq zipf frequencies over wordninja
segmentation, so it prefers plaintext that looks like real English.
"""

from __future__ import annotations

import random
import re
import string
from typing import List, Tuple

from wordfreq import zipf_frequency
from wordninja import split

ALPHABET = string.ascii_uppercase

# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _score(text: str) -> float:
    """Higher = more English-like. Sum of zipf frequencies of segmentable words."""
    if not text:
        return -1e9
    words = split(text.lower())
    words = [w for w in words if w.isalpha()]
    if not words:
        return -1e9
    return sum(zipf_frequency(w, "en") for w in words) / len(words)


# ---------------------------------------------------------------------------
# Caesar
# ---------------------------------------------------------------------------

def caesar_decode(text: str, shift: int) -> str:
    return "".join(
        chr((ord(c) - 65 - shift) % 26 + 65) if c.isalpha() else c
        for c in text.upper()
    )


def caesar_solve(text: str, top: int = 3) -> List[Tuple[int, str, float]]:
    """Try all 26 shifts, return (shift, plaintext, score) best-first."""
    out = []
    for s in range(26):
        p = caesar_decode(text, s)
        out.append((s, p, _score(p)))
    out.sort(key=lambda x: -x[2])
    return out[:top]


# ---------------------------------------------------------------------------
# Atbash, affine
# ---------------------------------------------------------------------------

def atbash(text: str) -> str:
    table = str.maketrans(ALPHABET, ALPHABET[::-1])
    return text.upper().translate(table)


def affine_decode(text: str, a: int, b: int) -> str:
    """Decode with y = (a*x + b) mod 26; `a` must be coprime to 26."""
    from math import gcd
    if gcd(a, 26) != 1:
        raise ValueError("a must be coprime to 26")
    a_inv = pow(a, -1, 26)
    out = []
    for c in text.upper():
        if c.isalpha():
            x = (a_inv * (ord(c) - 65 - b)) % 26
            out.append(chr(x + 65))
        else:
            out.append(c)
    return "".join(out)


# ---------------------------------------------------------------------------
# Vigenere (keyed)
# ---------------------------------------------------------------------------

def vigenere(text: str, key: str, decrypt: bool = True) -> str:
    key = re.sub(r"[^A-Z]", "", key.upper())
    if not key:
        raise ValueError("key must contain letters")
    out, ki = [], 0
    for c in text.upper():
        if c.isalpha():
            k = ord(key[ki % len(key)]) - 65
            if decrypt:
                k = -k
            out.append(chr((ord(c) - 65 + k) % 26 + 65))
            ki += 1
        else:
            out.append(c)
    return "".join(out)


# ---------------------------------------------------------------------------
# Monoalphabetic substitution (hill-climb auto-solver)
# ---------------------------------------------------------------------------

def _apply_map(text: str, mapping: dict) -> str:
    """Decode ciphertext -> plaintext: mapping[cipher_letter] = plain_letter."""
    return "".join(mapping.get(c, c) for c in text.upper())


def substitution_solve(text: str, iters: int = 20000, top: int = 3) -> List[Tuple[str, float]]:
    """Simulated-annealing monoalphabetic substitution solver.

    Returns list of (plaintext, score) best-first. Works well on patristocrats
    (spaceless) and full-word cryptograms when the text is long enough.
    """
    text = re.sub(r"[^A-Z]", "", text.upper())
    if not text:
        return []

    # frequency-aligned starting key
    from collections import Counter
    cfreq = [c for c, _ in Counter(text).most_common()]
    # English letter frequency order (etaoin shrdlu...)
    efreq = list("ETAOINSHRDLCUMWFGYPBVKJXQZ")
    mapping = dict(zip(cfreq, efreq))
    # fill any unused cipher letters with remaining plaintext letters
    used_plain = set(mapping.values())
    leftover = [c for c in ALPHABET if c not in used_plain]
    for c in ALPHABET:
        if c not in mapping:
            mapping[c] = leftover.pop() if leftover else "A"

    def score_map(m):
        return _score(_apply_map(text, m))

    best_map = dict(mapping)
    best_score = score_map(best_map)
    temp = 10.0
    rng = random.Random(0)

    for i in range(iters):
        a, b = rng.sample(ALPHABET, 2)
        cand = dict(mapping)
        cand[a], cand[b] = cand[b], cand[a]
        s = score_map(cand)
        delta = s - best_score
        if delta > 0 or rng.random() < _accept(delta, temp):
            mapping = cand
            if s > best_score:
                best_map, best_score = dict(cand), s
        temp *= 0.9995

    plain = _apply_map(text, best_map)
    return [(plain, best_score)]


def _accept(delta: float, temp: float) -> float:
    import math
    return math.exp(delta / max(temp, 1e-6))


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------

def frequency_analysis(text: str) -> List[Tuple[str, float]]:
    """Letter frequencies (percent), most common first."""
    from collections import Counter
    text = re.sub(r"[^A-Za-z]", "", text.upper())
    n = len(text) or 1
    c = Counter(text)
    return [(ch, round(100.0 * cnt / n, 2)) for ch, cnt in c.most_common()]
