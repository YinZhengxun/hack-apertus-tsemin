"""Spearman rank correlation with average ranks for ties (no third-party dependency).

Same definition as scipy.stats.spearmanr: Pearson correlation of the average ranks.
Returns NaN when either input is constant.
"""
from __future__ import annotations

import math
from typing import List, Sequence


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n != len(y):
        raise ValueError("length mismatch: %d vs %d" % (n, len(y)))
    if n < 2:
        return float("nan")
    mean_x, mean_y = math.fsum(x) / n, math.fsum(y) / n
    sxy = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    sxx = math.fsum((a - mean_x) ** 2 for a in x)
    syy = math.fsum((b - mean_y) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("length mismatch: %d vs %d" % (len(x), len(y)))
    return pearson(average_ranks(x), average_ranks(y))
