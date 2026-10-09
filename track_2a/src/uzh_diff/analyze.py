"""Output-layer analysis on finished runs: how the block judgements of m2 and the read-back surprisal of d4
should be turned into the final scores. Everything here needs gold labels, so it is for development
documents only; the rule that comes out is frozen in fuse.py and applied to the final run.

Variants are deliberately simple (one or two numbers per language), because 60–170 documents cannot
support more without over-fitting. The table is computed on the documents that have the results a
variant needs; the n column says how many that was.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .blocks import make_blocks
from .data import LANGS, DocPair
from .fuse import FuseParams, block_statuses, d4_signal, fuse
from .scorer import score

Labels = Tuple[List[float], List[float]]


def load_results(run_dir: Path, method: str) -> Dict[str, dict]:
    """results_<method>.jsonl of a run directory (or its gzipped copy, as kept in data/frozen_runs)."""
    import gzip
    path = run_dir / ("results_%s.jsonl" % method)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    elif path.with_suffix(".jsonl.gz").exists():
        with gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        return {}
    out = {}
    for line in text.splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["doc_id"]] = r
    return out


def _tiers(r: dict) -> Tuple[List[int], List[int]]:
    return r["details"]["tiers_a"], r["details"]["tiers_b"]


def _has_d4(r: Optional[dict]) -> bool:
    return bool(r) and "s_c_a" in r["details"] and "s_c_b" in r["details"] and "s_u_a" in r["details"] and "s_u_b" in r["details"]


def block_gold_rates(docs_by_lang: Dict[str, Sequence[DocPair]], m2: Dict[str, dict]) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
    """Per language and m2 block status: (blocks, scored words, gold-difference words).
    Tells whether a judgement carries information: a status whose gold rate equals the overall rate says nothing."""
    out: Dict[str, Dict[str, Tuple[int, int, int]]] = {}
    for lang, docs in docs_by_lang.items():
        agg: Dict[str, List[int]] = {}
        for d in docs:
            if d.id not in m2:
                continue
            status = {b["id"]: b["status"] for b in m2[d.id]["details"].get("blocks", [])}
            for side, tokens, labs in (("a", d.tokens_a, d.labels_a), ("b", d.tokens_b, d.labels_b)):
                for b in make_blocks(tokens, side):
                    a = agg.setdefault(status.get(b.id, "unjudged"), [0, 0, 0])
                    a[0] += 1
                    for g in labs[b.start:b.end]:
                        if g != -1:
                            a[1] += 1
                            a[2] += 1 if g > 0 else 0
        out[lang] = {k: (v[0], v[1], v[2]) for k, v in agg.items()}
    return out


def analyze(docs_by_lang: Dict[str, Sequence[DocPair]], m2: Dict[str, dict], d4: Dict[str, dict]) -> List[Tuple[str, dict]]:
    """Return [(variant name, score dict)] for every variant that the available results allow."""
    rows: List[Tuple[str, dict]] = []

    def add(name: str, make: Callable[[str, DocPair], Optional[Labels]]) -> None:
        docs_scored, preds = {}, {}
        for lang, docs in docs_by_lang.items():
            docs_scored[lang], preds[lang] = [], {}
            for d in docs:
                lab = make(lang, d)
                if lab is not None:
                    docs_scored[lang].append(d)
                    preds[lang][d.id] = lab
        if all(docs_scored[l] for l in docs_scored):
            sc = score(docs_scored, preds)
            rows.append((name, {"de": sc.per_lang["de"].spearman, "fr": sc.per_lang["fr"].spearman,
                                "it": sc.per_lang["it"].spearman, "mean": sc.macro_spearman,
                                "q": {l: sc.per_lang[l].pred_positive_rate for l in LANGS},
                                "n": {l: len(docs_scored[l]) for l in LANGS}}))

    def m2_of(d: DocPair) -> Optional[dict]:
        return m2.get(d.id)

    def d4_of(d: DocPair) -> Optional[dict]:
        r = d4.get(d.id)
        return r if _has_d4(r) else None

    have_m2 = any(d.id in m2 for docs in docs_by_lang.values() for d in docs)
    have_d4 = any(d4_of(d) is not None for docs in docs_by_lang.values() for d in docs)

    if have_m2:
        add("m2 原样：整块无对应 + 引用，0/1",
            lambda l, d: None if m2_of(d) is None else tuple([1.0 if t else 0.0 for t in side] for side in _tiers(m2_of(d))))
        add("m2 只用引用（0/1）",
            lambda l, d: None if m2_of(d) is None else tuple([1.0 if t == 2 else 0.0 for t in side] for side in _tiers(m2_of(d))))
        add("m2 只用整块无对应（0/1）  [rule absent]",
            lambda l, d: None if m2_of(d) is None else fuse(d, m2_of(d)["details"], None, FuseParams(rule="absent"))[0])

    if have_d4:
        add("d4 看过对面后的意外程度 s_c，±1 词取最大（连续）",
            lambda l, d: None if d4_of(d) is None else tuple(
                [max(v[max(0, i - 1): i + 2]) for i in range(len(v))] for v in (d4_of(d)["details"]["s_c_a"], d4_of(d)["details"]["s_c_b"])))
        add("d4 s_c − 0.5·s_u，不平滑（连续）",
            lambda l, d: None if d4_of(d) is None else tuple(d4_signal(d4_of(d)["details"], s, 0.5, 0) for s in ("a", "b")))
        for coef, win in ((0.5, 4), (0.75, 4), (1.0, 4), (0.75, 2), (0.75, 8)):
            add("d4 s_c − %.2f·s_u，±%d 词滑动平均（连续）%s" % (coef, win, "  [rule d4]" if (coef, win) == (0.75, 4) else ""),
                lambda l, d, coef=coef, win=win: None if d4_of(d) is None else tuple(d4_signal(d4_of(d)["details"], s, coef, win) for s in ("a", "b")))

    if have_m2 and have_d4:
        def both(d: DocPair) -> bool:
            return m2_of(d) is not None and d4_of(d) is not None

        for q in (0.6, 0.7, 0.75, 0.8):
            add("整块无对应经 d4 核实（块均 s_c−0.5·s_u ≥ 本篇 %.0f%% 分位），0/1%s" % (q * 100, "  [rule verified]" if q == 0.7 else ""),
                lambda l, d, q=q: None if not both(d) else fuse(d, m2_of(d)["details"], d4_of(d)["details"], FuseParams(rule="verified", quantile=q))[0])
        for coef in (0.5, 0.75, 1.0):
            add("核实后的整块无对应排最前，其余按 d4（s_c−%.2f·s_u, ±4）排序%s" % (coef, "  [rule verified+d4]" if coef == 0.75 else ""),
                lambda l, d, coef=coef: None if not both(d) else fuse(d, m2_of(d)["details"], d4_of(d)["details"], FuseParams(rule="verified+d4", rank_coef=coef))[0])

        def partly_tier(l: str, d: DocPair, frac: float = 0.1) -> Optional[Labels]:
            if not both(d):
                return None
            base = fuse(d, m2_of(d)["details"], d4_of(d)["details"], FuseParams(rule="verified"))[0]
            st = block_statuses(d, m2_of(d)["details"])
            out = []
            for side, lab in zip(("a", "b"), base):
                v = d4_signal(d4_of(d)["details"], side, 0.5, 4)
                lab = list(lab)
                elig = [i for i, s in enumerate(st[side]) if s == "partly" and lab[i] == 0.0]
                if v is not None and elig:
                    k = max(1, int(round(frac * len(elig))))
                    cut = sorted((v[i] for i in elig), reverse=True)[k - 1]
                    for i in elig:
                        if v[i] >= cut:
                            lab[i] = 0.5
                out.append(lab)
            return tuple(out)
        add("核实后的整块无对应=1 + partly 块里 d4 最高的 10% 词=0.5，其余 0", partly_tier)
    return rows


def format_rows(rows: List[Tuple[str, dict]], n_total: Optional[Dict[str, int]] = None) -> str:
    lines = ["%-74s %7s %7s %7s %7s   标出比例 de/fr/it   篇数" % ("输出规则", "de", "fr", "it", "平均")]
    for name, s in rows:
        n = "/".join(str(s["n"][l]) for l in LANGS)
        lines.append("%-74s %7.3f %7.3f %7.3f %7.3f   %s   %s" % (
            name, s["de"], s["fr"], s["it"], s["mean"], " ".join("%.2f" % s["q"][l] for l in LANGS), n))
    return "\n".join(lines)


def format_block_gold_rates(rates: Dict[str, Dict[str, Tuple[int, int, int]]]) -> str:
    lines = ["m2 块判定里金标差异词的比例（等于整体比例的判定没有信息量）:"]
    for lang, agg in rates.items():
        tot = sum(v[1] for v in agg.values())
        pos = sum(v[2] for v in agg.values())
        parts = ["整体 %.3f" % (pos / tot if tot else float("nan"))]
        for key in ("absent", "partly", "covered"):
            if key in agg:
                b, n, p = agg[key]
                parts.append("%s %.3f（%d 块 %d 词）" % (key, p / n if n else float("nan"), b, n))
        lines.append("  %s: " % lang + "; ".join(parts))
    return "\n".join(lines)
