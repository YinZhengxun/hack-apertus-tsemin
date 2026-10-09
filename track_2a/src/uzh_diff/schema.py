"""Read the model's JSON answer without eval() and turn it into a list of nominated differences.

Expected answer (prompt m1):

    {"decision": "differences_found" | "no_difference" | "uncertain",
     "differences": [
        {"a": {"block_id": "a003", "quote": "..."} | null,
         "b": {"block_id": "b004", "quote": "..."} | null,
         "category": "...", "explanation": "..."}]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

DECISIONS = ("differences_found", "no_difference", "uncertain")


class InvalidAnswer(ValueError):
    pass


@dataclass
class Nomination:
    side: str                 # "a" or "b"
    block_id: str
    quote: str
    category: str = ""
    explanation: str = ""
    item: int = -1            # index of the difference this quote belongs to


@dataclass
class ParsedAnswer:
    decision: str
    nominations: List[Nomination] = field(default_factory=list)
    n_items: int = 0
    problems: List[str] = field(default_factory=list)   # item-level format problems (kept visible)
    salvaged: bool = False                               # answer was cut off and repaired


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _balanced_object(text: str) -> Optional[str]:
    """The first complete {...} in text, honouring strings and escapes; None if it never closes."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _salvage(text: str, max_tries: int = 200) -> Optional[Any]:
    """An answer cut off in the middle of the list: keep the complete items, drop the broken tail."""
    start = text.find("{")
    if start == -1:
        return None
    body = text[start:]
    positions = [m.start() for m in re.finditer(r"\}", body)]
    for pos in reversed(positions[-max_tries:]):
        candidate = body[:pos + 1]
        for closing in ("]}", "}", ""):
            try:
                obj = json.loads(candidate + closing)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def extract_json(text: str) -> Tuple[Any, bool]:
    """Return (object, salvaged). Raises InvalidAnswer when nothing usable is there."""
    if text is None:
        raise InvalidAnswer("empty answer")
    cleaned = strip_fences(text)
    candidate = _balanced_object(cleaned)
    if candidate is not None:
        try:
            return json.loads(candidate), False
        except ValueError:
            pass
    obj = _salvage(cleaned)
    if obj is not None:
        return obj, True
    raise InvalidAnswer("no JSON object found")


def _side_entries(value: Any) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def parse_m1(text: str) -> ParsedAnswer:
    obj, salvaged = extract_json(text)
    if not isinstance(obj, dict):
        raise InvalidAnswer("top level is not an object")
    items = obj.get("differences")
    if items is None:
        items = []
    if not isinstance(items, list):
        raise InvalidAnswer('"differences" is not a list')
    decision = obj.get("decision")
    if decision not in DECISIONS:
        decision = "differences_found" if items else "no_difference"
    out = ParsedAnswer(decision=decision, n_items=len(items), salvaged=salvaged)
    for k, item in enumerate(items):
        if not isinstance(item, dict):
            out.problems.append("item %d is not an object" % k)
            continue
        category = str(item.get("category") or "")
        explanation = str(item.get("explanation") or "")
        got = 0
        # flat form: {"side": "a", "block_id": "a003", "quote": "..."}
        if "quote" in item and ("side" in item or "block_id" in item):
            side = str(item.get("side") or str(item.get("block_id") or "")[:1]).lower()
            if side in ("a", "b") and isinstance(item.get("quote"), str):
                out.nominations.append(Nomination(side, str(item.get("block_id") or ""), item["quote"], category, explanation, k))
                got += 1
        for side in ("a", "b"):
            for entry in _side_entries(item.get(side)):
                quote = entry.get("quote")
                if not isinstance(quote, str) or not quote.strip():
                    continue
                out.nominations.append(Nomination(side, str(entry.get("block_id") or ""), quote, category, explanation, k))
                got += 1
        if got == 0:
            out.problems.append("item %d has no usable quote" % k)
    return out
