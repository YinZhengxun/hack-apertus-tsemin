"""Locate a quoted phrase in a tokenised document and return the token span it covers.

The model is asked to copy the differing words exactly. Matching is exact on the characters, after
two harmless normalisations: all whitespace is ignored (the source texts have spaces around
punctuation, "Switzerland ' s", which models tend to drop) and typographic variants of quotes,
apostrophes and dashes are unified. There is no fuzzy matching: a quote that cannot be found is
reported as not found and contributes nothing to the prediction.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_TYPOGRAPHY = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‑": "-", "‐": "-",
})

OK = "ok"                      # exactly one match inside the named block
AMBIGUOUS = "ambiguous"        # several matches inside the named block; the first one is used
ELSEWHERE = "elsewhere"        # not in the named block, but exactly one match in the rest of that document
NOT_FOUND = "not_found"        # no usable match
EMPTY = "empty"                # nothing to look for

COUNTED = (OK, AMBIGUOUS, ELSEWHERE)   # statuses that put labels on tokens


@dataclass
class Location:
    status: str
    start: Optional[int] = None    # token index in the document (inclusive)
    end: Optional[int] = None      # token index in the document (exclusive)
    n_matches: int = 0
    how: str = ""                  # "exact", "casefold", "inside_token"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_TYPOGRAPHY)
    return "".join(text.split())


def _index(tokens: Sequence[str], start: int, end: int, fold: bool):
    """Concatenate the normalised tokens of [start, end) and remember which token owns each character."""
    pieces: List[str] = []
    owner: List[int] = []
    first_char = set()
    last_char = set()
    pos = 0
    for i in range(start, end):
        t = normalize(tokens[i])
        if fold:
            t = t.casefold()
        if not t:
            continue
        first_char.add(pos)
        pieces.append(t)
        owner.extend([i] * len(t))
        pos += len(t)
        last_char.add(pos - 1)
    return "".join(pieces), owner, first_char, last_char


def _matches(needle: str, tokens: Sequence[str], start: int, end: int, fold: bool,
             aligned: bool) -> List[Tuple[int, int]]:
    hay, owner, first_char, last_char = _index(tokens, start, end, fold)
    if fold:
        needle = needle.casefold()
    found: List[Tuple[int, int]] = []
    if not needle:
        return found
    p = hay.find(needle)
    while p != -1:
        q = p + len(needle) - 1
        if not aligned or (p in first_char and q in last_char):
            span = (owner[p], owner[q] + 1)
            if span not in found:
                found.append(span)
        p = hay.find(needle, p + 1)
    return found


def _search(needle: str, tokens: Sequence[str], start: int, end: int):
    """Try the strictest reading first: whole tokens and exact case, then case-insensitive,
    then matches that begin or end inside a token."""
    for how, fold, aligned in (("exact", False, True), ("casefold", True, True), ("inside_token", True, False)):
        found = _matches(needle, tokens, start, end, fold, aligned)
        if found:
            return found, how
    return [], ""


def locate(quote: str, tokens: Sequence[str], block: Optional[Tuple[int, int]] = None) -> Location:
    """Find `quote` in `tokens`; `block` is the (start, end) token span the model pointed at."""
    needle = normalize(quote or "")
    if not needle:
        return Location(EMPTY)
    if block is not None:
        found, how = _search(needle, tokens, block[0], block[1])
        if len(found) == 1:
            return Location(OK, found[0][0], found[0][1], 1, how)
        if len(found) > 1:
            return Location(AMBIGUOUS, found[0][0], found[0][1], len(found), how)
    # whole document: accepted only when there is exactly one place it can be
    for how, fold in (("exact", False), ("casefold", True)):
        found = _matches(needle, tokens, 0, len(tokens), fold, True)
        if len(found) == 1:
            status = ELSEWHERE if block is not None else OK
            return Location(status, found[0][0], found[0][1], 1, how)
        if len(found) > 1:
            # several possible places and no valid block to choose between them: do not guess
            return Location(NOT_FOUND, None, None, len(found), how)
    return Location(NOT_FOUND)


def apply(labels: List[float], location: Location, value: float = 1.0) -> None:
    """Mark the located tokens; keeps the larger value where spans overlap."""
    if location.status in COUNTED and location.start is not None and location.end is not None:
        for i in range(location.start, location.end):
            if value > labels[i]:
                labels[i] = value
