#!/usr/bin/env python3
"""cipher_hillclimb_rust.py — Python/ctypes bridge to the Rust-accelerated
substitution-cipher hill-climbing solver (2026-08-31, coding-stack polyglot
eval item #1). See doc `agent/eval-refactor-polyglot-2026-08-31` for why:
this is the one measured CPU-bound pure-Python hot loop in the whole
codebase (6.6s/solve), on ciphersleuth's auto-detect critical path.

Design: Rust does ONLY the ~80,000-iteration numeric inner loop (see
cipher_hillclimb.rs, source in ~/projects/cipher-hillclimb/, no external
crates — just rustc + stdlib). Python still does everything else: loading
the quadgram corpus, building the frequency-seeded starting key, and
presenting the same dict-based API cipher_engine.py already had. Call
sites in cipher_engine.py never change; substitution_solve() there tries
this module first and falls back to a pure-Python implementation if
anything here is unavailable (missing .so, ctypes failure, etc.) — the
polyglot boundary should never be a single point of failure.

The logp table (456976 float64 = ~3.5MB) is derived from
mailtool/english_ngrams.json once and cached as a flat binary file
(cheap to regenerate; not committed to recovery — it's derived data).
"""
import ctypes
import json
import math
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(BASE, "..", "projects", "cipher-hillclimb", "libcipher_hillclimb.so")
SO_PATH = os.path.abspath(SO_PATH)
NGRAMS_JSON = os.path.join(BASE, "english_ngrams.json")
LOGP_CACHE = os.path.join(os.path.dirname(SO_PATH), "quadgram_logp.f64.bin")

V = 26 ** 4  # 456976 possible 4-letter combinations

_lib = None
_logp_buf = None  # keeps the ctypes array alive for the process lifetime


def _quad_index(q):
    """4-letter string ('aaaa'..'zzzz') -> flat table index, base-26."""
    idx = 0
    for ch in q:
        idx = idx * 26 + (ord(ch) - 97)
    return idx


def _build_logp_table():
    """Dense 26^4-entry Laplace-smoothed log-likelihood table, same formula
    as cipher_engine._quad_model(): log((count+1) / (total+V))."""
    try:
        counts = json.load(open(NGRAMS_JSON))["quadgram"]
    except Exception:
        counts = {}
    total = sum(counts.values()) or 1
    table = [math.log(1.0 / (total + V))] * V  # default: unseen quadgram
    default = table[0]
    for q, c in counts.items():
        if len(q) == 4 and q.isalpha() and q.islower():
            table[_quad_index(q)] = math.log((c + 1) / (total + V))
    return table, default


def _load_logp_table():
    """Load the cached flat binary table, rebuilding it if missing/stale."""
    global _logp_buf
    if _logp_buf is not None:
        return _logp_buf
    need_rebuild = True
    if os.path.exists(LOGP_CACHE) and os.path.exists(NGRAMS_JSON):
        if os.path.getmtime(LOGP_CACHE) >= os.path.getmtime(NGRAMS_JSON) and \
           os.path.getsize(LOGP_CACHE) == V * 8:
            need_rebuild = False
    if need_rebuild:
        table, _default = _build_logp_table()
        arr = (ctypes.c_double * V)(*table)
        with open(LOGP_CACHE, "wb") as f:
            f.write(bytes(arr))
    else:
        arr = (ctypes.c_double * V)()
        with open(LOGP_CACHE, "rb") as f:
            f.readinto(arr)
    _logp_buf = arr
    return _logp_buf


def available():
    """True if the compiled Rust extension can actually be loaded right now."""
    global _lib
    if _lib is not None:
        return True
    if not os.path.exists(SO_PATH):
        return False
    try:
        lib = ctypes.CDLL(SO_PATH)
        lib.hillclimb_solve.restype = ctypes.c_double
        lib.hillclimb_solve.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        _lib = lib
        return True
    except Exception:
        return False


def hillclimb(cipher_letters, seed_key_bytes, rounds, restarts):
    """Run the Rust hill-climber.

    cipher_letters: str of lowercase a-z only (the ciphertext, already
        stripped of non-letters — same preprocessing cipher_engine.py does).
    seed_key_bytes: bytearray[26], seed_key_bytes[c] = guessed plain-letter
        index (0-25) for cipher-letter index c (0-25 = 'a'-'z').
    Returns (best_key_bytes: bytearray[26], best_score: float).
    """
    if not available():
        raise RuntimeError("Rust hillclimb extension not available")
    logp = _load_logp_table()
    cipher_idx = (ctypes.c_uint8 * len(cipher_letters))(
        *[ord(c) - 97 for c in cipher_letters]
    )
    key = (ctypes.c_uint8 * 26)(*seed_key_bytes)
    seed = int.from_bytes(os.urandom(8), "little")
    score = _lib.hillclimb_solve(
        cipher_idx, len(cipher_letters),
        logp, V,
        rounds, restarts,
        seed,
        key,
    )
    return bytearray(key), score
