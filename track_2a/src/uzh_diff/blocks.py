"""Cut a tokenised document into short blocks and remember where each block sits.

The source texts are one long line without paragraph marks. Blocks give the model something to
point at ("a007") and keep quotes short enough to be located without ambiguity. Every token belongs
to exactly one block, so a block-local position maps back to exactly one document position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

SENTENCE_END = {".", "!", "?", "…"}
SOFT_BREAK = {";", ":", ","}
OPENERS = {"(", "[", "«", "\u201c", "\u201e", '"', "'", "\u2018", "-", "\u2013"}


def _is_number(tok: str) -> bool:
    return bool(tok) and all(ch.isdigit() or ch in ".,'%" for ch in tok) and any(ch.isdigit() for ch in tok)


def _sentence_boundary(tokens: Sequence[str], i: int) -> bool:
    """Is the token at i the end of a sentence? Only for ".", "!", "?" followed by something that looks like a
    sentence start; a period after a number ("22 . 12 . 2017", "1 . 5 MB") or after a single capital letter
    ("A . Müller") does not count."""
    tok = tokens[i]
    if tok not in SENTENCE_END:
        return False
    if i + 1 >= len(tokens):
        return True
    nxt = tokens[i + 1]
    if tok == ".":
        prev = tokens[i - 1] if i > 0 else ""
        if _is_number(prev) and _is_number(nxt):      # "22 . 12 . 2017", "1 . 5 MB"
            return False
        if len(prev) == 1 and prev.isupper():         # "A . Müller", "U . S ."
            return False
    if nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in OPENERS:
        return True
    return False


@dataclass
class Block:
    id: str          # "a000", "b012", ...
    start: int       # index of the first token in the document
    end: int         # index after the last token
    tokens: List[str]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def _split_long(start: int, end: int, tokens: Sequence[str], max_len: int) -> List[Tuple[int, int]]:
    """Split [start, end) at the soft break closest to the middle until every piece fits."""
    if end - start <= max_len:
        return [(start, end)]
    middle = (start + end) // 2
    # only cuts that leave at least five tokens on each side
    candidates = [i + 1 for i in range(start + 4, end - 5) if tokens[i] in SOFT_BREAK]
    cut = min(candidates, key=lambda i: abs(i - middle)) if candidates else middle
    if cut <= start or cut >= end:
        cut = middle
    return _split_long(start, cut, tokens, max_len) + _split_long(cut, end, tokens, max_len)


def segment(tokens: Sequence[str], max_len: int = 60, min_len: int = 4) -> List[Tuple[int, int]]:
    """Return (start, end) spans that partition the token list."""
    n = len(tokens)
    if n == 0:
        return []
    # 1) cut after sentence-final punctuation (but not inside dates, numbers or abbreviations)
    spans, start = [], 0
    for i in range(n):
        if _sentence_boundary(tokens, i):
            spans.append((start, i + 1))
            start = i + 1
    if start < n:
        spans.append((start, n))
    # 2) glue very short pieces (headings, stray fragments) to the following piece
    merged: List[Tuple[int, int]] = []
    carry = None
    for s, e in spans:
        if carry is not None:
            s = carry
            carry = None
        if e - s < min_len:
            carry = s
            continue
        merged.append((s, e))
    if carry is not None:
        if merged:
            merged[-1] = (merged[-1][0], n)
        else:
            merged.append((carry, n))
    # 3) split pieces that are too long
    out: List[Tuple[int, int]] = []
    for s, e in merged:
        out.extend(_split_long(s, e, tokens, max_len))
    return out


def make_blocks(tokens: Sequence[str], side: str, max_len: int = 60, min_len: int = 4) -> List[Block]:
    """side is "a" or "b"; block ids are side + three digits."""
    return [Block("%s%03d" % (side, k), s, e, list(tokens[s:e]))
            for k, (s, e) in enumerate(segment(tokens, max_len, min_len))]


def render(blocks: Sequence[Block]) -> str:
    return "\n".join("[%s] %s" % (b.id, b.text) for b in blocks)
