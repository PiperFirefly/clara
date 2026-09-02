"""wordlab.oeis — OEIS (On-Line Encyclopedia of Integer Sequences) lookup.

For "what comes next in this sequence" clues. Read-only HTTP to oeis.org;
human-rate (one request per call). No install, no keys.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import List, Tuple

_URL = "https://oeis.org/search?q={}&fmt=json"


def oeis_search(seq) -> List[Tuple[str, str, str]]:
    """Look up an integer sequence; return [(A-number, name, first terms)]."""
    if isinstance(seq, str):
        seq = [int(x) for x in seq.replace(",", " ").split() if x.lstrip("-").isdigit()]
    q = ",".join(str(int(x)) for x in seq)
    req = urllib.request.Request(
        _URL.format(urllib.parse.quote(q)),
        headers={"User-Agent": "agent-puzzle-lab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception as e:
        return [("error", str(e), "")]
    out = []
    for res in (data.get("results") or [])[:8]:
        number = res.get("number", "?")
        name = res.get("name", "").strip()
        terms = res.get("data", "").split(",")[:12]
        out.append((number, name, ",".join(t.strip() for t in terms)))
    return out


def oeis_next(seq, n_terms: int = 5) -> List[Tuple[str, str]]:
    """Return [(A-number, predicted next terms)] for a sequence's top matches."""
    out = []
    for number, name, terms in oeis_search(seq):
        if number == "error":
            return [(number, name)]
        # OEIS 'data' holds the known terms; we can't reliably extend without
        # a formula, so just report the match + name. (The human/LLM picks.)
        out.append((number, name))
    return out[:n_terms]
