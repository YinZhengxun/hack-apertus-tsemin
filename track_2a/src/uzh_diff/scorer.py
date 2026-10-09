"""Token-level scoring, same definition as the official script
(ZurichNLP/SwissGov-RSD, scripts/evaluate_predictions_admin.py):

  per language pair: concatenate the tokens of side a and side b of all documents,
  drop the positions whose gold label is -1 (punctuation), Spearman(gold, prediction);
  the headline number is the mean over EN-DE, EN-FR, EN-IT.

Differences from the official script, on purpose:
  * predictions are joined to gold by document id, not by line number;
  * a missing document or a length mismatch is an error (the official script silently pads with
    zeros or truncates), unless the caller explicitly marks a document as failed, in which case it
    is scored as "no difference predicted" and counted in the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .data import DocPair
from .spearman import spearman

Labels = Tuple[Sequence[float], Sequence[float]]   # (labels_a, labels_b)


@dataclass
class LangScore:
    lang: str
    n_docs: int
    n_scored_tokens: int
    spearman: float
    precision: float     # of tokens predicted > 0, share with gold > 0
    recall: float        # of tokens with gold > 0, share predicted > 0
    f1: float
    gold_positive_rate: float
    pred_positive_rate: float
    failed_docs: List[str] = field(default_factory=list)


@dataclass
class Score:
    per_lang: Dict[str, LangScore]
    macro_spearman: float   # NaN if any language is undefined

    def table(self) -> str:
        lines = ["语种  篇数  计分词数  Spearman   准确率   召回率    F1   金标差异率  预测差异率  失败篇数"]
        for lang, s in self.per_lang.items():
            lines.append("%-4s %5d %9d %9.3f %8.3f %8.3f %6.3f %10.3f %10.3f %8d" % (
                lang, s.n_docs, s.n_scored_tokens, s.spearman, s.precision, s.recall, s.f1,
                s.gold_positive_rate, s.pred_positive_rate, len(s.failed_docs)))
        lines.append("三个语种平均 Spearman: %.3f" % self.macro_spearman)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def clean(x):
            return None if isinstance(x, float) and math.isnan(x) else x
        return {
            "macro_spearman": clean(self.macro_spearman),
            "per_lang": {l: {k: clean(v) for k, v in s.__dict__.items()} for l, s in self.per_lang.items()},
        }


def score_lang(lang: str, docs: Sequence[DocPair], preds: Dict[str, Labels],
               failed: Optional[Sequence[str]] = None) -> LangScore:
    failed = list(failed or [])
    gold_all: List[float] = []
    pred_all: List[float] = []
    for d in docs:
        if d.labels_a is None or d.labels_b is None:
            raise ValueError("%s has no gold labels" % d.id)
        if d.id in preds:
            pa, pb = preds[d.id]
        elif d.id in failed:
            pa, pb = [0.0] * len(d.tokens_a), [0.0] * len(d.tokens_b)
        else:
            raise KeyError("no prediction for %s (and it is not marked as failed)" % d.id)
        if len(pa) != len(d.labels_a) or len(pb) != len(d.labels_b):
            raise ValueError("%s: prediction length (%d, %d) != gold length (%d, %d)" % (
                d.id, len(pa), len(pb), len(d.labels_a), len(d.labels_b)))
        for g, p in list(zip(d.labels_a, pa)) + list(zip(d.labels_b, pb)):
            if g == -1:
                continue
            gold_all.append(float(g))
            pred_all.append(float(p))
    n = len(gold_all)
    tp = sum(1 for g, p in zip(gold_all, pred_all) if g > 0 and p > 0)
    gold_pos = sum(1 for g in gold_all if g > 0)
    pred_pos = sum(1 for p in pred_all if p > 0)
    precision = tp / pred_pos if pred_pos else float("nan")
    recall = tp / gold_pos if gold_pos else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if pred_pos and gold_pos and (precision + recall) > 0 else float("nan")
    return LangScore(
        lang=lang, n_docs=len(docs), n_scored_tokens=n,
        spearman=spearman(gold_all, pred_all) if n else float("nan"),
        precision=precision, recall=recall, f1=f1,
        gold_positive_rate=gold_pos / n if n else float("nan"),
        pred_positive_rate=pred_pos / n if n else float("nan"),
        failed_docs=[d.id for d in docs if d.id in failed and d.id not in preds],
    )


def score(docs_by_lang: Dict[str, Sequence[DocPair]], preds_by_lang: Dict[str, Dict[str, Labels]],
          failed_by_lang: Optional[Dict[str, Sequence[str]]] = None) -> Score:
    per_lang = {}
    for lang, docs in docs_by_lang.items():
        per_lang[lang] = score_lang(lang, docs, preds_by_lang.get(lang, {}), (failed_by_lang or {}).get(lang))
    values = [s.spearman for s in per_lang.values()]
    macro = float("nan") if (not values or any(math.isnan(v) for v in values)) else sum(values) / len(values)
    return Score(per_lang=per_lang, macro_spearman=macro)
