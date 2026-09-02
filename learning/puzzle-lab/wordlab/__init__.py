"""wordlab — local word/pattern/anagram/cipher/encoding engine.

Built on wordfreq (wordlists), wordninja (segmentation), pycipher, nltk wordnet.
No network needed after first wordlist download. This is the "local Nutrimatic":
pattern search, anagrams, multi-word anagrams, word ladders, ciphers, encodings,
and OEIS lookup.
"""

from .patterns import WordIndex, search, anagram, multi_anagram, alphagram, segment
from .ciphers import (
    caesar_solve,
    atbash,
    affine_decode,
    vigenere,
    substitution_solve,
    frequency_analysis,
)
from .encodings import (
    morse_decode,
    braille_decode,
    rot13,
    reverse,
    try_all,
)
from .ladders import word_ladder, ladder_neighbors
from .oeis import oeis_search, oeis_next
from .solitaire import (
    unkeyed_deck,
    key_deck,
    keystream,
    encrypt,
    decrypt,
    move_jokers,
    triple_cut,
    count_cut,
)

__all__ = [
    "WordIndex", "search", "anagram", "multi_anagram", "alphagram", "segment",
    "caesar_solve", "atbash", "affine_decode", "vigenere", "substitution_solve",
    "frequency_analysis",
    "morse_decode", "braille_decode", "rot13", "reverse", "try_all",
    "word_ladder", "ladder_neighbors",
    "oeis_search", "oeis_next",
    "unkeyed_deck", "key_deck", "keystream", "encrypt", "decrypt",
    "move_jokers", "triple_cut", "count_cut",
]
