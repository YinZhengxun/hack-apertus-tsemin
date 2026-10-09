"""Paired, page-grouped bootstrap for the official metric.

The unit of resampling is the English page: position k of the de/fr/it gold files is the same English
page (checked: identical English text at every position on dev and test), so one draw of page k brings
its three document pairs and all their words into every language's pool at once. For each resample the
official statistic is recomputed exactly as the evaluation script does it (one pooled Spearman per
language over all tokens with gold != -1, then the mean over languages), for both systems on the same
resample, and the difference is recorded. Duplicated pages keep their multiplicity.

Reported: the original point estimates (the scores of the actual files, which the report must quote),
the bootstrap means (only to show they are close), percentile intervals, and the plain count of
resamples in which the difference was not positive. That count is an empirical tail proportion, not a
hypothesis-test p-value.

numpy is used when available (the pure-Python rank computation is ~100x slower); results are identical.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .data import LANGS, DocPair

try:  # optional accelerator
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

Labels = Tuple[Sequence[float], Sequence[float]]


def _ranks(values: Sequence[float]) -> List[float]:
    if _np is not None:
        x = _np.asarray(values, dtype=float)
        _, inv, counts = _np.unique(x, return_inverse=True, return_counts=True)
        before = _np.cumsum(counts) - counts
        return (before + (counts + 1) / 2.0)[inv].tolist()
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pooled_spearman(gold: Sequence[float], pred: Sequence[float]) -> float:
    if _np is not None:
        rg, rp = _np.asarray(_ranks(gold)), _np.asarray(_ranks(pred))
        if rg.std() == 0 or rp.std() == 0:
            return float("nan")
        return float(_np.corrcoef(rg, rp)[0, 1])
    rg, rp = _ranks(gold), _ranks(pred)
    n = len(rg)
    mg, mp = sum(rg) / n, sum(rp) / n
    cov = sum((a - mg) * (b - mp) for a, b in zip(rg, rp))
    vg = math.sqrt(sum((a - mg) ** 2 for a in rg))
    vp = math.sqrt(sum((b - mp) ** 2 for b in rp))
    return float("nan") if vg == 0 or vp == 0 else cov / (vg * vp)


def _per_page(docs_by_lang: Dict[str, Sequence[DocPair]], preds: Dict[str, Dict[str, Labels]]):
    """gold and prediction token lists per (language, page position), punctuation removed."""
    out = {}
    for lang, docs in docs_by_lang.items():
        for k, d in enumerate(docs):
            pa, pb = preds[lang][d.id]
            g, p = [], []
            for labs, pr in ((d.labels_a, pa), (d.labels_b, pb)):
                for x, y in zip(labs, pr):
                    if x != -1:
                        g.append(float(x))
                        p.append(float(y))
            out[(lang, k)] = (g, p)
    return out


def macro(docs_by_lang: Dict[str, Sequence[DocPair]], preds: Dict[str, Dict[str, Labels]]) -> Dict[str, float]:
    per = _per_page(docs_by_lang, preds)
    res = {}
    for lang, docs in docs_by_lang.items():
        g = [x for k in range(len(docs)) for x in per[(lang, k)][0]]
        p = [x for k in range(len(docs)) for x in per[(lang, k)][1]]
        res[lang] = pooled_spearman(g, p)
    res["mean"] = sum(res[l] for l in docs_by_lang) / len(docs_by_lang)
    return res


def paired_bootstrap(docs_by_lang: Dict[str, Sequence[DocPair]], preds_a: Dict[str, Dict[str, Labels]],
                     preds_b: Dict[str, Dict[str, Labels]], resamples: int = 1000, seed: int = 0) -> dict:
    langs = list(docs_by_lang)
    n_pages = len(docs_by_lang[langs[0]])
    for l in langs:
        if len(docs_by_lang[l]) != n_pages:
            raise ValueError("languages must have the same number of documents (one per page)")
    per_a, per_b = _per_page(docs_by_lang, preds_a), _per_page(docs_by_lang, preds_b)
    point_a, point_b = macro(docs_by_lang, preds_a), macro(docs_by_lang, preds_b)
    rng = random.Random(seed)
    means_a, means_b, diffs = [], [], []
    undefined = 0
    for _ in range(resamples):
        idx = [rng.randrange(n_pages) for _ in range(n_pages)]      # with replacement, multiplicity kept
        ma = mb = 0.0
        ok = True
        for l in langs:
            g = [x for k in idx for x in per_a[(l, k)][0]]
            pa = [x for k in idx for x in per_a[(l, k)][1]]
            pb = [x for k in idx for x in per_b[(l, k)][1]]
            sa, sb = pooled_spearman(g, pa), pooled_spearman(g, pb)
            if math.isnan(sa) or math.isnan(sb):
                ok = False
                break
            ma += sa / len(langs)
            mb += sb / len(langs)
        if not ok:
            undefined += 1
            continue
        means_a.append(ma)
        means_b.append(mb)
        diffs.append(ma - mb)

    def pct(v: List[float], q: float) -> float:
        s = sorted(v)
        if not s:
            return float("nan")
        pos = q * (len(s) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    def summary(v: List[float]) -> dict:
        return {"bootstrap_mean": sum(v) / len(v) if v else float("nan"), "ci95": [pct(v, 0.025), pct(v, 0.975)]}

    return {
        "unit": "English page (its de/fr/it document pairs drawn together)", "pages": n_pages,
        "resamples_requested": resamples, "resamples_valid": len(diffs), "resamples_undefined": undefined, "seed": seed,
        "statistic": "mean over languages of the pooled token-level Spearman (official definition)",
        "interval": "percentile, 2.5–97.5",
        "a": {"point": point_a, **summary(means_a)}, "b": {"point": point_b, **summary(means_b)},
        "diff": {"point": point_a["mean"] - point_b["mean"], **summary(diffs),
                 "resamples_diff_not_positive": sum(1 for d in diffs if d <= 0)},
    }


def format_bootstrap(name_a: str, name_b: str, r: dict) -> str:
    a, b, d = r["a"], r["b"], r["diff"]
    lines = [
        "配对 bootstrap：%s 对 %s；重采样单位 = 英文页面（%d 页，三语种同抽），%d 次（有效 %d，未定义 %d），种子 %d，95%% 百分位区间"
        % (name_a, name_b, r["pages"], r["resamples_requested"], r["resamples_valid"], r["resamples_undefined"], r["seed"]),
        "  %-14s 原始点估计 %.3f（de %.3f / fr %.3f / it %.3f），重采样均值 %.3f，区间 [%.3f, %.3f]"
        % (name_a, a["point"]["mean"], a["point"]["de"], a["point"]["fr"], a["point"]["it"], a["bootstrap_mean"], *a["ci95"]),
        "  %-14s 原始点估计 %.3f（de %.3f / fr %.3f / it %.3f），重采样均值 %.3f，区间 [%.3f, %.3f]"
        % (name_b, b["point"]["mean"], b["point"]["de"], b["point"]["fr"], b["point"]["it"], b["bootstrap_mean"], *b["ci95"]),
        "  差值           原始 %+.3f，重采样均值 %+.3f，区间 [%+.3f, %+.3f]；%d 次重采样里差值不为正的有 %d 次"
        % (d["point"], d["bootstrap_mean"], *d["ci95"], r["resamples_valid"], d["resamples_diff_not_positive"]),
    ]
    return "\n".join(lines)
