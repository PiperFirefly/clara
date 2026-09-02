"""wordlab.solitaire — the Solitaire (a.k.a. "Pontifex") stream cipher.

A low-tech output-feedback-mode stream cipher designed by Bruce Schneier and
featured in Neal Stephenson's *Cryptonomicon* (the appendix, ch. 111-119).
The whole point is that an agent can carry a deck of playing cards and encrypt
by hand — no electronics, nothing incriminating.

Deck representation (bridge order):
    1-13  = A..K of clubs
    14-26 = A..K of diamonds
    27-39 = A..K of hearts
    40-52 = A..K of spades
    53    = joker A
    54    = joker B

The deck is a list of 54 ints; every operation is a pure function that returns
a new deck (no in-place mutation), so it composes cleanly and is easy to step
through for study.

Test vectors (Bruce Schneier, "The Solitaire Encryption Algorithm",
<https://www.schneier.com/academic/solitaire/>) — all verified:

    encrypt("AAAAAAAAAA", unkeyed_deck())           == "EXKYI ZSGEH"
    encrypt("AAAAAAAAAAAAAAA", unkeyed_deck())      == "EXKYI ZSGEH UNTIQ"
    encrypt("AAAAAAAAAAAAAAA", key_deck("FOO"))     == "ITHZU JIWGR FARMW"
    encrypt("SOLITAIRE", key_deck("CRYPTONOMICON")) == "KIRAK SFJAN"

(Plaintext is padded with X's to a multiple of 5, per the book's convention.)
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# deck construction
# ---------------------------------------------------------------------------

def unkeyed_deck() -> List[int]:
    """The fixed starting deck: A..K clubs, diamonds, hearts, spades, A, B."""
    return list(range(1, 53)) + [53, 54]


def key_deck(passphrase: str, position_jokers: bool = False) -> List[int]:
    """Key the deck from a passphrase (Schneier's keying method 3).

    For each letter of the passphrase: move A joker down 1, move B joker down
    2, triple cut, then two count cuts — the normal bottom-card cut *and* a
    second cut keyed by the letter's value (A=1..Z=26).

    `position_jokers` implements the book's *optional* extra step (NOT used in
    the sample vectors): after all letters, place joker A after the card at the
    second-to-last character's value, joker B after the last character's value.
    """
    deck = unkeyed_deck()
    chars = [ord(c.upper()) - 64 for c in passphrase if c.isalpha()]
    for v in chars:
        deck = move_jokers(deck)
        deck = triple_cut(deck)
        deck = count_cut(deck, _count_value(deck[-1]))  # normal step-4 cut
        deck = count_cut(deck, v)                        # passphrase-char cut

    if position_jokers and len(chars) >= 2:
        deck = move_jokers(deck)
        deck = triple_cut(deck)
        deck = count_cut(deck, chars[-2])
        deck = count_cut(deck, chars[-1])

    return deck


# ---------------------------------------------------------------------------
# deck operations (pure; each returns a new deck)
# ---------------------------------------------------------------------------

def _count_value(card: int) -> int:
    """Card value for count cuts / lookdown (steps 4-5): jokers are 53."""
    return card if card <= 52 else 53


def _output_value(card: int) -> int:
    """Card value for keystream letters (step 6): 1..26."""
    return ((card - 1) % 26) + 1


def move_jokers(deck: List[int]) -> List[int]:
    """Step 1-2: A joker down one card, then B joker down two (loop around)."""
    d = deck[:]
    i = d.index(53)                       # A joker down one
    d.pop(i)
    d.insert(1 if i == 53 else i + 1, 53)
    j = d.index(54)                       # B joker down two
    d.pop(j)
    if j == 53:       d.insert(2, 54)     # bottom -> below 2nd card
    elif j == 52:     d.insert(1, 54)     # one from bottom -> below top
    else:             d.insert(j + 2, 54)
    return d


def triple_cut(deck: List[int]) -> List[int]:
    """Step 3: swap cards above the first joker with those below the second."""
    d = deck[:]
    i, j = d.index(53), d.index(54)
    first, second = min(i, j), max(i, j)
    return d[second + 1:] + d[first:second + 1] + d[:first]


def count_cut(deck: List[int], n: int) -> List[int]:
    """Step 4: take the top n cards, move them just above the bottom card."""
    d = deck[:]
    top, rest = d[:n], d[n:]
    return rest[:-1] + top + [rest[-1]]


# ---------------------------------------------------------------------------
# keystream + encrypt/decrypt
# ---------------------------------------------------------------------------

def keystream(deck: List[int], n: int) -> List[int]:
    """Generate n keystream numbers (1..26), joker outputs silently skipped."""
    d = deck[:]
    out: List[int] = []
    while len(out) < n:
        d = move_jokers(d)
        d = triple_cut(d)
        d = count_cut(d, _count_value(d[-1]))
        outcard = d[_count_value(d[0])]     # step 5: output card
        if outcard <= 52:                    # skip jokers, don't emit
            out.append(_output_value(outcard))
    return out


def _to_nums(text: str) -> List[int]:
    return [ord(c.upper()) - 64 for c in text if c.isalpha()]


def _to_letters(nums: List[int]) -> str:
    return "".join(chr(n + 64) for n in nums)


def encrypt(plaintext: str, deck: List[int], group: int = 5) -> str:
    """Encrypt, padding with X's to a multiple of 5 and grouping the output."""
    p = _to_nums(plaintext)
    while len(p) % group:                   # pad with X (the book's convention)
        p.append(24)
    k = keystream(deck, len(p))
    s = _to_letters([((a + b - 1) % 26) + 1 for a, b in zip(p, k)])
    return " ".join(s[i:i + group] for i in range(0, len(s), group))


def decrypt(ciphertext: str, deck: List[int], group: int = 5) -> str:
    """Decrypt; strips whitespace, keeps X-padding (caller may rstrip('X'))."""
    c = _to_nums(ciphertext)
    k = keystream(deck, len(c))
    p = _to_letters([((a - b - 1) % 26) + 1 for a, b in zip(c, k)])
    return " ".join(p[i:i + group] for i in range(0, len(p), group))
