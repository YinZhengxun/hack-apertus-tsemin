"""Output layer: combine the block judgements of m2 with the read-back surprisal of d4.

Why this exists. On the dev60 development documents (20 pages × 3 languages) the single methods are
weak on their own: m2's "whole block has no counterpart" judgements are precise in German (gold rate
0.53 inside such blocks) but not in French (0.27); m2's quotes are at chance level; d4's surprisal is a
useful but blurry continuous signal. Two steps fix both problems:

1. Verification. A block that m2 calls absent is kept only if its words are still surprising after the
   model has read the other document — block mean of the d4 signal ≥ a per-document quantile.
   Blocks the other document explains well were mis-judged by m2 and are dropped.
2. Ordering. The official metric is a Spearman correlation over all tokens of a language, so ties among
   the ~90 % unlabelled tokens waste information. The verified absent blocks are ranked on top; every
   other token is ordered by the d4 signal.

The d4 signal of a word is  s_c − coef · s_u  (surprisal given the other document, minus part of the
surprisal without it: words that are surprising anyway are discounted), smoothed with a moving mean over
±window words because differences come in spans.

All numbers below are chosen on dev60 and then confirmed on dev/val; nothing here reads gold labels.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .blocks import make_blocks
from .data import DocPair

Labels = Tuple[List[float], List[float]]

RULES = ("absent", "verified", "verified+d4", "d4")


@dataclass
class FuseParams:
    rule: str = "verified+d4"
    quantile: float = 0.7        # verification: block mean signal must reach this per-document quantile
    verify_coef: float = 0.5     # s_u coefficient in the verification signal (unsmoothed)
    rank_coef: float = 0.75      # s_u coefficient in the ordering signal
    window: int = 4              # ±words of the moving mean for the ordering signal
    scale: float = 3.0           # squash temperature; the floor at 0 ties all non-positive signals (part of the rule),
                                 # the scale itself does not change the order of the positive ones

    def to_dict(self) -> dict:
        return asdict(self)


def moving_mean(values: Sequence[float], radius: int) -> List[float]:
    if radius <= 0:
        return list(values)
    return [statistics.mean(values[max(0, i - radius): i + radius + 1]) for i in range(len(values))]


def d4_signal(details: dict, side: str, coef: float, window: int = 0) -> Optional[List[float]]:
    """s_c − coef·s_u for one side ("a"/"b"), or None when that side is missing from the d4 result."""
    s_c = details.get("s_c_" + side)
    s_u = details.get("s_u_" + side)
    if s_c is None:
        return None
    if coef and s_u is None:
        return None
    v = [c - coef * u for c, u in zip(s_c, s_u)] if coef else list(s_c)
    return moving_mean(v, window)


def squash(v: float, scale: float) -> float:
    """Monotone map of a surprisal difference into [0, 1): negative values all become 0."""
    return 1.0 - math.exp(-max(v, 0.0) / scale)


def block_statuses(doc: DocPair, m2_details: dict) -> Dict[str, List[Optional[str]]]:
    """Per-token m2 status ("absent" / "partly" / "covered" / other) for both sides."""
    status = {b["id"]: b["status"] for b in m2_details.get("blocks", [])}
    out: Dict[str, List[Optional[str]]] = {}
    for side, tokens in (("a", doc.tokens_a), ("b", doc.tokens_b)):
        lab: List[Optional[str]] = [None] * len(tokens)
        for b in make_blocks(tokens, side):
            s = status.get(b.id)
            for i in range(b.start, b.end):
                lab[i] = s
        out[side] = lab
    return out


def verified_absent(doc: DocPair, m2_details: dict, d4_details: Optional[dict], p: FuseParams) -> Dict[str, List[bool]]:
    """Which tokens are in an absent block that survives verification (block by block).
    Without a d4 result for a side, that side's absent blocks are all kept."""
    status = {b["id"]: b["status"] for b in m2_details.get("blocks", [])}
    sig = {s: (d4_signal(d4_details, s, p.verify_coef) if d4_details else None) for s in ("a", "b")}
    pool = [x for s in ("a", "b") if sig[s] is not None for x in sig[s]]
    thr = sorted(pool)[int(p.quantile * (len(pool) - 1))] if pool else None
    out: Dict[str, List[bool]] = {}
    for side, tokens in (("a", doc.tokens_a), ("b", doc.tokens_b)):
        keep = [False] * len(tokens)
        v = sig[side]
        for b in make_blocks(tokens, side):
            if status.get(b.id) != "absent":
                continue
            if v is None or thr is None or statistics.mean(v[b.start:b.end]) >= thr:
                for i in range(b.start, b.end):
                    keep[i] = True
        out[side] = keep
    return out


def fuse(doc: DocPair, m2_details: Optional[dict], d4_details: Optional[dict], p: FuseParams = FuseParams()) -> Tuple[Labels, dict]:
    """Return ((labels_a, labels_b), info). Values lie in [0, 1]; only their order matters to the metric.

    rule "absent":      m2 absent blocks = 1, everything else 0 (no d4 needed)
    rule "verified":    verified absent blocks = 1, everything else 0
    rule "verified+d4": verified absent blocks in [0.5, 1), everything else in [0, 0.49), both ordered by d4
    rule "d4":          d4 ordering signal only, in [0, 1)
    """
    if p.rule not in RULES:
        raise ValueError("unknown rule %r (choose from %s)" % (p.rule, ", ".join(RULES)))
    info = {"rule": p.rule, "d4_sides": [], "m2": m2_details is not None}
    out: List[List[float]] = []
    if p.rule == "d4":
        for side, tokens in (("a", doc.tokens_a), ("b", doc.tokens_b)):
            v = d4_signal(d4_details, side, p.rank_coef, p.window) if d4_details else None
            if v is None:
                out.append([0.0] * len(tokens))
            else:
                info["d4_sides"].append(side)
                out.append([squash(x, p.scale) for x in v])
        return (out[0], out[1]), info

    if m2_details is None:
        raise ValueError("%s: rule %s needs an m2 result" % (doc.id, p.rule))
    if p.rule == "absent":
        st = block_statuses(doc, m2_details)
        keep = {s: [x == "absent" for x in st[s]] for s in ("a", "b")}
    else:
        keep = verified_absent(doc, m2_details, d4_details, p)
    for side, tokens in (("a", doc.tokens_a), ("b", doc.tokens_b)):
        k = keep[side]
        if p.rule == "verified+d4":
            v = d4_signal(d4_details, side, p.rank_coef, p.window) if d4_details else None
            if v is not None:
                info["d4_sides"].append(side)
                out.append([(0.5 + 0.5 * squash(x, p.scale)) if flag else 0.49 * squash(x, p.scale) for flag, x in zip(k, v)])
                continue
        out.append([1.0 if flag else 0.0 for flag in k])
    info["n_absent_tokens"] = sum(sum(1 for x in keep[s] if x) for s in ("a", "b"))
    return (out[0], out[1]), info
