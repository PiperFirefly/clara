#!/usr/bin/env python3
"""Artemis cipher engine — classical + book ciphers for treasure-hunt cryptanalysis.

Self-contained. Word list from /usr/share/dict/words if present, else a small
built-in set. Solvers are statistical (frequency / chi-squared / index of
coincidence / word-pattern), Bletchley-lite rather than exhaustive, but the
fast paths are exact.

CLI:
  cipher_engine.py solve "<ciphertext>"          # auto-detect + attack
  cipher_engine.py morse --decode "..."
  cipher_engine.py caesar "<text>"               # solve caesar
  cipher_engine.py vigenere "<text>"             # solve vigenere
  cipher_engine.py substitution "<text>"         # solve substitution
  cipher_engine.py beale "1 2 3 ..." --key "<keytext>"
"""
import re, math, sys, json
from collections import Counter

# --------------------------------------------------------------------------
# language model
# --------------------------------------------------------------------------
ENGLISH_FREQ = {
    'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702, 'f': 2.228,
    'g': 2.015, 'h': 6.094, 'i': 6.966, 'j': 0.153, 'k': 0.772, 'l': 4.025,
    'm': 2.406, 'n': 6.749, 'o': 7.507, 'p': 1.929, 'q': 0.095, 'r': 5.987,
    's': 6.327, 't': 9.056, 'u': 2.758, 'v': 0.978, 'w': 2.360, 'x': 0.150,
    'y': 1.974, 'z': 0.074,
}

_WORDS = set()
def _load_words():
    global _WORDS
    if _WORDS:
        return _WORDS
    try:
        with open("/usr/share/dict/words") as f:
            _WORDS = {w.strip().lower() for w in f if 2 <= len(w.strip()) <= 20 and w.strip().isalpha()}
    except Exception:
        _WORDS = set("the and for not you are but all any can her was one our out day get has him his how man new now old see two way who did its let put say she too use".split())
    return _WORDS


def _letters(text):
    return re.sub(r"[^a-z]", "", text.lower())


_BIGRAM_LOGP = None
def _bigram_logp():
    """Lazy log-likelihood bigram model from the system dictionary (for scoring)."""
    global _BIGRAM_LOGP
    if _BIGRAM_LOGP is not None:
        return _BIGRAM_LOGP
    import math as _m
    counts = Counter()
    total = 0
    try:
        with open("/usr/share/dict/words") as f:
            for w in f:
                w = ' ' + w.strip().lower() + ' '
                if not w.strip().isalpha():
                    continue
                for a, b in zip(w, w[1:]):
                    counts[a + b] += 1
                    total += 1
    except Exception:
        pass
    if total == 0:
        _BIGRAM_LOGP = lambda bg: -10.0
        return _BIGRAM_LOGP
    def lp(bg):
        c = counts.get(bg, 0)
        return _m.log((c + 1) / (total + 26 * 26))
    _BIGRAM_LOGP = lp
    return _BIGRAM_LOGP


def bigram_score(text):
    """Sum of bigram log-probs (higher = more English-like)."""
    lp = _bigram_logp()
    t = ' ' + _letters(text) + ' '
    return sum(lp(t[i] + t[i + 1]) for i in range(len(t) - 1))


_QUAD = None
def _quad_model():
    """Lazy quadgram log-likelihood model (159M chars of English, built from the library)."""
    global _QUAD
    if _QUAD is not None:
        return _QUAD
    import os as _os
    p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "english_ngrams.json")
    try:
        d = json.load(open(p))
        counts = d["quadgram"]
    except Exception:
        counts = {}
    total = sum(counts.values()) or 1
    V = 26 ** 4
    import math as _m
    _QUAD = lambda q: _m.log((counts.get(q, 0) + 1) / (total + V))
    return _QUAD


def quadgram_score(text):
    """Quadgram log-likelihood of a plaintext candidate (the standard substitution
    cryptanalysis objective). Higher = more English-like."""
    q = _quad_model()
    t = _letters(text)
    if len(t) < 4:
        return 0.0
    return sum(q(t[i:i + 4]) for i in range(len(t) - 3))


def chi_squared(text):
    """Lower = closer to English letter distribution."""
    t = _letters(text)
    n = len(t)
    if n == 0:
        return float("inf")
    counts = Counter(t)
    score = 0.0
    for c, exp in ENGLISH_FREQ.items():
        obs = counts.get(c, 0) / n * 100.0
        score += (obs - exp) ** 2 / exp
    return score


def index_of_coincidence(text):
    """~0.067 for English, ~0.038 for random. Used to key-length Vigenere."""
    t = _letters(text)
    n = len(t)
    if n < 2:
        return 0.0
    counts = Counter(t)
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def word_score(text):
    """Fraction of whitespace tokens that are dictionary words. 1.0 = all real words."""
    words = set(_load_words())
    toks = re.findall(r"[a-z]+", text.lower())
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in words)
    return hits / len(toks)


def combined_score(text):
    return 100.0 * word_score(text) - 0.05 * chi_squared(text)


# --------------------------------------------------------------------------
# morse
# --------------------------------------------------------------------------
MORSE = {
    'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.',
    'g': '--.', 'h': '....', 'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
    'm': '--', 'n': '-.', 'o': '---', 'p': '.--.', 'q': '--.-', 'r': '.-.',
    's': '...', 't': '-', 'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-',
    'y': '-.--', 'z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
}
MORSE_REV = {v: k for k, v in MORSE.items()}


def morse_encode(text):
    out = []
    for ch in text.lower():
        if ch == ' ':
            out.append('/')
        elif ch in MORSE:
            out.append(MORSE[ch])
    return ' '.join(out)


def morse_decode(code, letter_sep=' ', word_sep='/'):
    words = code.strip().split(word_sep)
    res = []
    for w in words:
        res.append(''.join(MORSE_REV.get(s, '?') for s in w.split(letter_sep) if s))
    return ' '.join(res)


# --------------------------------------------------------------------------
# caesar + affine + atbash + rot13
# --------------------------------------------------------------------------
def _shift(c, k):
    return chr((ord(c) - 97 + k) % 26 + 97)


def caesar_encode(text, k):
    return ''.join(_shift(c, k) if c.isalpha() else c for c in _letters(text))


def caesar_decode(text, k):
    return caesar_encode(text, -k)


def caesar_solve(text):
    best, bestk = None, None
    for k in range(26):
        cand = caesar_decode(text, k)
        s = combined_score(cand)
        if best is None or s > best:
            best, bestk = s, k
    return caesar_decode(text, bestk), bestk


def atbash(text):
    return ''.join(chr(219 - ord(c)) if c.isalpha() else c for c in _letters(text))


def rot13(text):
    return caesar_encode(text, 13)


def affine_encode(text, a, b):
    assert math.gcd(a, 26) == 1, "a must be coprime with 26"
    return ''.join(chr((a * (ord(c) - 97) + b) % 26 + 97) for c in _letters(text))


def affine_decode(text, a, b):
    ainv = pow(a, -1, 26)
    return ''.join(chr((ainv * (ord(c) - 97 - b)) % 26 + 97) for c in _letters(text))


# --------------------------------------------------------------------------
# vigenere
# --------------------------------------------------------------------------
def vigenere_encode(text, key):
    key = _letters(key)
    t = _letters(text)
    return ''.join(_shift(c, ord(key[i % len(key)]) - 97) for i, c in enumerate(t))


def vigenere_decode(text, key):
    key = _letters(key)
    t = _letters(text)
    return ''.join(_shift(c, -(ord(key[i % len(key)]) - 97)) for i, c in enumerate(t))


def _kasiski(text, max_len=20):
    """Repeated-substring distances -> candidate key lengths via GCD clustering."""
    t = _letters(text)
    seen = {}
    distances = []
    for i in range(len(t) - 3):
        gram = t[i:i + 3]
        if gram in seen:
            d = i - seen[gram]
            if d >= 2:
                distances.append(d)
        seen[gram] = i
    if not distances:
        return {}
    votes = Counter()
    for d in distances:
        for kl in range(2, max_len + 1):
            if d % kl == 0:
                votes[kl] += 1
    return votes


def _likely_vigenere_keylen(text, max_len=20):
    """Combine Kasiski repeated-substring votes with IC. Kasiski wins when present."""
    t = _letters(text)
    kasiski = _kasiski(t, max_len)
    if kasiski:
        best_kl, best_votes = max(kasiski.items(), key=lambda kv: kv[1])
        # require a reasonable number of votes, else fall back to IC
        if best_votes >= 1 and len(t) > 60:
            return best_kl
    best_len, best_score = 1, -1.0
    for kl in range(1, max_len + 1):
        ics = []
        for i in range(kl):
            col = t[i::kl]
            if len(col) >= 2:
                ics.append(index_of_coincidence(col))
        if ics:
            avg = sum(ics) / len(ics)
            score = abs(avg - 0.067)
            if score < best_score or best_score < 0:
                best_len, best_score = kl, score
    return best_len


def vigenere_solve(text, max_len=20):
    t = _letters(text)
    kl = _likely_vigenere_keylen(t, max_len)
    key = []
    for i in range(kl):
        col = t[i::kl]
        # each column is a caesar shift of English; brute the shift by chi-squared
        best_k, best_cs = 0, float("inf")
        for k in range(26):
            cs = chi_squared(''.join(_shift(c, -k) for c in col))
            if cs < best_cs:
                best_k, best_cs = k, cs
        key.append(chr(best_k + 97))
    return vigenere_decode(t, ''.join(key)), ''.join(key), kl


# --------------------------------------------------------------------------
# monoalphabetic substitution (hill climbing)
# --------------------------------------------------------------------------
def _substitution_seed(t):
    """Frequency-seeded starting key (ciphertext letter -> guessed plaintext
    letter), shared by both the Rust-accelerated and pure-Python solvers."""
    freq_order = "etaoinshrdlcumwfgypbvkjxqz"
    observed = [c for c, _ in Counter(t).most_common() if c.isalpha()]
    seed = {}
    for i, c in enumerate(observed):
        if i < 26:
            seed[c] = freq_order[i]
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c not in seed:
            free = [x for x in "abcdefghijklmnopqrstuvwxyz" if x not in seed.values()]
            seed[c] = free[0] if free else 'e'
    return seed


def _substitution_solve_python(text, rounds=20000, restarts=4):
    """Greedy hill-climb on a monoalphabetic substitution, scored by quadgram
    log-likelihood, seeded from letter-frequency and restarted from the best.
    Pure-Python reference implementation (~6.6s/solve at defaults) -- kept as
    the fallback for substitution_solve() below, and as the reference for
    quality-comparison against the Rust-accelerated path."""
    t = _letters(text)
    import random

    seed = _substitution_seed(t)

    def decrypt(k):
        return ''.join(k.get(c, c) for c in t)

    def score(k):
        return quadgram_score(decrypt(k))

    keys = list("abcdefghijklmnopqrstuvwxyz")
    cur = dict(seed)
    cur_score = score(cur)
    for _ in range(restarts):
        # gentle perturb of the current best to escape local minima
        k2 = dict(cur)
        for _ in range(random.randint(2, 6)):
            a, b = random.sample(keys, 2)
            k2[a], k2[b] = k2[b], k2[a]
        k2_score = score(k2)
        for _ in range(rounds):
            a, b = random.sample(keys, 2)
            k3 = dict(k2)
            k3[a], k3[b] = k3[b], k3[a]
            s = score(k3)
            if s > k2_score:
                k2, k2_score = k3, s
        if k2_score > cur_score:
            cur, cur_score = k2, k2_score
    return decrypt(cur), cur


def substitution_solve(text, rounds=20000, restarts=4):
    """Greedy hill-climb on a monoalphabetic substitution, scored by quadgram
    log-likelihood, seeded from letter-frequency and restarted from the best.

    Tries the Rust-accelerated inner loop first (2026-08-31 polyglot eval:
    the pure-Python version measured 6.6s/solve on ciphersleuth's auto-detect
    critical path); falls back to the pure-Python implementation if the
    compiled extension isn't available for any reason. Same signature, same
    return shape, either way -- callers never need to know which ran."""
    t = _letters(text)
    if len(t) < 4:
        return _substitution_solve_python(text, rounds, restarts)
    try:
        import cipher_hillclimb_rust as hc
        if hc.available():
            seed = _substitution_seed(t)
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            seed_bytes = bytearray(ord(seed[c]) - 97 for c in alphabet)
            best_bytes, _score = hc.hillclimb(t, seed_bytes, rounds, restarts)
            key = {alphabet[i]: alphabet[best_bytes[i]] for i in range(26)}
            decrypted = ''.join(key.get(c, c) for c in t)
            return decrypted, key
    except Exception:
        pass  # any failure in the Rust path -> pure-Python fallback below
    return _substitution_solve_python(text, rounds, restarts)


# --------------------------------------------------------------------------
# rail fence + columnar transposition
# --------------------------------------------------------------------------
def rail_fence_encode(text, rails):
    t = _letters(text)
    rails = max(2, rails)
    rows = [[] for _ in range(rails)]
    r, d = 0, 1
    for c in t:
        rows[r].append(c)
        if r == 0:
            d = 1
        elif r == rails - 1:
            d = -1
        r += d
    return ''.join(''.join(row) for row in rows)


def rail_fence_decode(text, rails):
    t = _letters(text)
    rails = max(2, rails)
    n = len(t)
    pattern = []
    r, d = 0, 1
    for _ in range(n):
        pattern.append(r)
        if r == 0:
            d = 1
        elif r == rails - 1:
            d = -1
        r += d
    counts = [pattern.count(i) for i in range(rails)]
    rows, pos = [], 0
    for c in counts:
        rows.append(list(t[pos:pos + c]))
        pos += c
    out, idx = [], 0
    for r in pattern:
        out.append(rows[r].pop(0))
    return ''.join(out)


def rail_fence_solve(text, max_rails=12):
    best, bestr = None, None
    for r in range(2, max_rails + 1):
        cand = rail_fence_decode(text, r)
        s = combined_score(cand)
        if best is None or s > best:
            best, bestr = s, r
    return rail_fence_decode(text, bestr), bestr


def columnar_transposition_decode(text, ncols):
    t = _letters(text)
    nrows = -(-len(t) // ncols)
    # fill grid row-major from ciphertext (simple rectangular)
    grid = [list(t[i * ncols:(i + 1) * ncols]) for i in range(nrows)]
    out = []
    for c in range(ncols):
        for r in range(nrows):
            if c < len(grid[r]):
                out.append(grid[r][c])
    return ''.join(out)


# --------------------------------------------------------------------------
# book cipher / beale
# --------------------------------------------------------------------------
def book_cipher_decode(numbers, key_text, mode="first-letter"):
    """Decode a Beale-style book cipher. numbers are 1-indexed word positions in
    key_text. mode: 'first-letter' (default Beale) or 'whole-word'."""
    words = re.findall(r"[A-Za-z]+", key_text)
    out = []
    for num in numbers:
        if 1 <= num <= len(words):
            w = words[num - 1]
            out.append(w[0].lower() if mode == "first-letter" else w.lower())
        else:
            out.append("?")
    return ''.join(out) if mode == "first-letter" else ' '.join(out)


def beale_decode(numbers, key_text):
    return book_cipher_decode(numbers, key_text, mode="first-letter")


# --------------------------------------------------------------------------
# misc encodings common in treasure hunts
# --------------------------------------------------------------------------
def b64_decode(s):
    import base64
    try:
        return base64.b64decode(s).decode("utf-8", "replace")
    except Exception:
        return None


def hex_decode(s):
    try:
        return bytes.fromhex(s).decode("utf-8", "replace")
    except Exception:
        return None


def binary_decode(s):
    try:
        return ''.join(chr(int(b, 2)) for b in re.findall(r"[01]{1,8}", s.replace(' ', '')))
    except Exception:
        return None


# --------------------------------------------------------------------------
# detection + auto-solve
# --------------------------------------------------------------------------
def detect(text):
    t = _letters(text)
    ic = index_of_coincidence(t)
    report = {"ic": round(ic, 4), "length": len(t)}
    if all(c in ".-/ " for c in text.strip()):
        report["cipher"] = "morse"
    elif ic < 0.05 and len(t) > 40:
        report["cipher"] = "vigenere-or-polyalphabetic"
    elif ic < 0.055:
        report["cipher"] = "possibly-polyalphabetic"
    else:
        report["cipher"] = "monoalphabetic-or-transposition"
    return report


def solve(text):
    """Auto-detect and attack. Returns a list of candidate decodings."""
    results = []
    d = detect(text)

    # morse
    if d.get("cipher") == "morse":
        return [("morse", morse_decode(text))]

    # caesar (always cheap + often right)
    plain, k = caesar_solve(text)
    results.append((f"caesar (shift {k})", plain))

    # vigenere
    try:
        plain, key, kl = vigenere_solve(text)
        results.append((f"vigenere (key={key}, len={kl})", plain))
    except Exception:
        pass

    # substitution
    try:
        plain, _ = substitution_solve(text)
        results.append(("substitution", plain))
    except Exception:
        pass

    # rail fence
    try:
        plain, r = rail_fence_solve(text)
        results.append((f"rail-fence (rails={r})", plain))
    except Exception:
        pass

    # rank by word score
    results.sort(key=lambda x: -word_score(x[1]))
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    rest = args[1:]

    if cmd == "morse":
        text = rest[-1] if rest else ""
        if "--encode" in rest:
            print(morse_encode(text))
        else:
            print(morse_decode(text))
    elif cmd == "caesar":
        print(caesar_solve(rest[-1])[0])
    elif cmd == "vigenere":
        print(vigenere_solve(rest[-1])[0])
    elif cmd == "substitution":
        print(substitution_solve(rest[-1])[0])
    elif cmd == "beale":
        nums = [int(x) for x in re.findall(r"\d+", rest[0])] if rest else []
        key = ""
        if "--key" in rest:
            key = rest[rest.index("--key") + 1]
        print(beale_decode(nums, key))
    elif cmd == "solve":
        for name, plain in solve(rest[-1]):
            print(f"[{name}]\n{plain}\n")
    else:
        print(__doc__)
