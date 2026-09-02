"""wordlab.encodings — data transforms + a "try everything" decoder (CyberChef-lite).

Clue fragments in hunts are frequently just a known encoding: base64, hex,
binary, octal, morse, rot13, atbash, braille, reversed. This module decodes
each and `try_all` runs the whole battery so you can eyeball which one is
human-readable.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import string
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Morse
# ---------------------------------------------------------------------------

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}
_MORSE_REV = {v: k for k, v in MORSE.items()}


def morse_decode(s: str) -> str:
    """'.... . .-.. .-.. --- / .-- --- .-. .-.. -..' -> 'HELLO WORLD'."""
    out = []
    for word in s.split("/"):
        letters = []
        for code in word.split():
            letters.append(_MORSE_REV.get(code, "?"))
        out.append("".join(letters))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Braille (6-dot, letters a-z)
# ---------------------------------------------------------------------------

_BRAILLE = {
    "a": 1, "b": 3, "c": 9, "d": 25, "e": 17, "f": 11, "g": 27, "h": 19,
    "i": 10, "j": 26, "k": 5, "l": 7, "m": 13, "n": 29, "o": 21, "p": 15,
    "q": 31, "r": 23, "s": 14, "t": 30, "u": 37, "v": 39, "w": 58, "x": 45,
    "y": 61, "z": 53,
}
_BRAILLE_REV = {v: k for k, v in _BRAILLE.items()}


def braille_decode(s: str) -> str:
    """Unicode braille (U+2800) or dot-number lists ('145 15') -> letters."""
    if re.match(r"^[\u2800-\u28ff]", s):
        out = []
        for ch in s:
            v = ord(ch) - 0x2800
            if v == 0:
                out.append(" ")
            else:
                # decode 6-dot bitmask to dots, map to letter
                dots = tuple(i for i in range(6) if v & (1 << i))
                key = sum(1 << (d) for d in dots)  # already the mask
                out.append(_BRAILLE_REV.get(key, "?"))
        return "".join(out)
    out = []
    for tok in s.split():
        if tok.isdigit():
            out.append(_BRAILLE_REV.get(int(tok), "?"))
        else:
            out.append(" ")
    return "".join(out)


# ---------------------------------------------------------------------------
# Simple decodings
# ---------------------------------------------------------------------------

def _b64(s: str) -> Optional[str]:
    try:
        return base64.b64decode(s, validate=True).decode("utf-8", "replace")
    except Exception:
        return None


def _b32(s: str) -> Optional[str]:
    try:
        return base64.b32decode(s.upper() + "=" * (-len(s) % 8)).decode("utf-8", "replace")
    except Exception:
        return None


def _hex(s: str) -> Optional[str]:
    try:
        return bytes.fromhex(s.replace(" ", "")).decode("utf-8", "replace")
    except Exception:
        return None


def _bin(s: str) -> Optional[str]:
    try:
        bits = s.replace(" ", "")
        if not re.fullmatch(r"[01]+", bits) or len(bits) % 8:
            return None
        return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8)).decode("utf-8", "replace")
    except Exception:
        return None


def _oct(s: str) -> Optional[str]:
    try:
        toks = s.split()
        if not all(re.fullmatch(r"[0-7]{3}", t) for t in toks):
            return None
        return bytes(int(t, 8) for t in toks).decode("utf-8", "replace")
    except Exception:
        return None


def rot13(s: str) -> str:
    return codecs.decode(s, "rot_13")


def reverse(s: str) -> str:
    return s[::-1]


def atbash(s: str) -> str:
    table = str.maketrans(string.ascii_uppercase, string.ascii_uppercase[::-1])
    return s.upper().translate(table)


# ---------------------------------------------------------------------------
# try_all
# ---------------------------------------------------------------------------

def _printable(s: Optional[str]) -> bool:
    if not s:
        return False
    alnum = sum(c.isalnum() or c in " .,!?'-/:;()" for c in s)
    return alnum / max(len(s), 1) > 0.7


def try_all(s: str) -> List[Tuple[str, str]]:
    """Run every decoder and return [(name, result)] for plausible (printable)
    non-identity outputs, best guesses first."""
    cands = []
    for name, fn in [
        ("base64", _b64), ("base32", _b32), ("hex", _hex),
        ("binary", _bin), ("octal", _oct), ("morse", lambda x: morse_decode(x) if "." in x or "-" in x else None),
        ("braille", lambda x: braille_decode(x) if "\u2800" <= (x[:1] or " ") <= "\u28ff" else None),
        ("rot13", lambda x: rot13(x)),
        ("atbash", lambda x: atbash(x)),
        ("reverse", reverse),
    ]:
        try:
            r = fn(s)
        except Exception:
            continue
        if r is not None and r.strip() and r.strip() != s.strip() and _printable(r):
            cands.append((name, r))
    # heuristic: prefer outputs that look like English words/sentences
    return sorted(cands, key=lambda x: -(sum(c.isalpha() or c == " " for c in x[1]) / len(x[1])))
