"""Acceptance audit of a fuse run: is every document, block and word accounted for?

A status of "ok" in a summary only says that the calls came back and parsed. This audit checks what the
scores are actually made of: every expected document present exactly once; every block that m2 was asked
about judged (and how many kept the default "no difference" because the model did not answer); every d4
score array present with one value per word and an exact alignment; how many sides fell back to m2 only;
and the prediction files' row counts, label lengths and hashes. It never changes any prediction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .analyze import load_results
from .blocks import make_blocks
from .data import LANGS, DocPair

UNJUDGED = ("missing_in_answer", "call_failed", "unreadable")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(docs_by_lang: Dict[str, Sequence[DocPair]], fuse_dir: Path, m2_dir: Optional[Path], d4_dir: Optional[Path],
          system: str = "fuse") -> dict:
    report: Dict[str, object] = {"fuse_run": str(fuse_dir), "m2_run": str(m2_dir) if m2_dir else None,
                                 "d4_run": str(d4_dir) if d4_dir else None, "languages": {}, "problems": [], "warnings": []}
    # "problems" are fatal for a delivery (the predictions must not be handed over); "warnings" are declared
    # limitations that do not change the predictions (a few m2 blocks without a judgement, fuzzy d4 alignment)
    problems: List[str] = report["problems"]  # type: ignore[assignment]
    warnings: List[str] = report["warnings"]  # type: ignore[assignment]
    m2 = load_results(m2_dir, "m2") if m2_dir else {}
    d4 = load_results(d4_dir, "d4") if d4_dir else {}
    fu = load_results(fuse_dir, "fuse")

    for lang, docs in docs_by_lang.items():
        ids = [d.id for d in docs]
        L: Dict[str, object] = {"documents_expected": len(ids)}
        if len(set(ids)) != len(ids):
            problems.append("%s: duplicate document ids in the gold file" % lang)
        for name, res in (("m2", m2), ("d4", d4), ("fuse", fu)):
            if not res and name != "fuse" and (m2_dir if name == "m2" else d4_dir) is None:
                continue
            missing = [i for i in ids if i not in res]
            extra = [i for i in res if i not in set(ids) and res[i].get("lang") == lang]
            L["%s_missing_documents" % name] = missing
            if missing:
                problems.append("%s: %d documents without a %s result: %s" % (lang, len(missing), name, ", ".join(missing[:5])))
            if extra:
                L["%s_unexpected_documents" % name] = extra

        # ---- m2: blocks asked vs judged
        if m2:
            asked = judged = unjudged = 0
            docs_with_unjudged = []
            judged_status: Dict[str, int] = {}
            for d in docs:
                r = m2.get(d.id)
                if r is None:
                    continue
                n_blocks = len(make_blocks(d.tokens_a, "a")) + len(make_blocks(d.tokens_b, "b"))
                asked += n_blocks
                st = [b["status"] for b in r["details"].get("blocks", []) if b.get("id", "-") != "-"]
                j = sum(1 for s in st if s in ("absent", "partly", "covered"))
                u = sum(1 for s in st if s in UNJUDGED)
                for s in st:
                    if s in ("absent", "partly", "covered"):
                        judged_status[s] = judged_status.get(s, 0) + 1
                judged += j
                unjudged += u
                missing_from_list = n_blocks - j - u
                if missing_from_list > 0:
                    unjudged += missing_from_list
                    u += missing_from_list
                if u:
                    docs_with_unjudged.append({"id": d.id, "unjudged_blocks": u, "recorded_status": r.get("status")})
            L["m2_blocks_asked"] = asked
            L["m2_blocks_judged"] = judged
            L["m2_blocks_unjudged"] = unjudged
            L["m2_judged_status_counts"] = judged_status
            L["m2_documents_with_unjudged_blocks"] = docs_with_unjudged
            if unjudged:
                warnings.append("%s: %d of %d m2 blocks were not judged (kept the default 'no difference') in %d documents, %d of them recorded as ok" % (
                    lang, unjudged, asked, len(docs_with_unjudged), sum(1 for x in docs_with_unjudged if x["recorded_status"] == "ok")))

        # ---- d4: arrays present, right length, exact alignment
        if d4:
            sides_expected = 2 * len(docs)
            sides_ok = 0
            length_mismatch, missing_sides, align = [], [], {}
            for d in docs:
                r = d4.get(d.id)
                if r is None:
                    missing_sides.extend(["%s/a" % d.id, "%s/b" % d.id])
                    continue
                det = r["details"]
                for side, toks in (("a", d.tokens_a), ("b", d.tokens_b)):
                    c, u = det.get("s_c_" + side), det.get("s_u_" + side)
                    if c is None or u is None:
                        missing_sides.append("%s/%s" % (d.id, side))
                        continue
                    if len(c) != len(toks) or len(u) != len(toks):
                        length_mismatch.append("%s/%s" % (d.id, side))
                        continue
                    sides_ok += 1
                for k, v in (det.get("alignment") or {}).items():
                    align[v] = align.get(v, 0) + 1
            L["d4_sides_expected"] = sides_expected
            L["d4_sides_complete"] = sides_ok
            L["d4_sides_missing"] = missing_sides
            L["d4_length_mismatch"] = length_mismatch
            L["d4_alignment_counts"] = align
            if missing_sides:
                problems.append("%s: %d document sides without complete d4 scores (fell back to m2 only): %s" % (lang, len(missing_sides), ", ".join(missing_sides[:6])))
            if length_mismatch:
                problems.append("%s: d4 score length != word count for %s" % (lang, ", ".join(length_mismatch[:6])))
            if any(k not in ("exact", "fuzzy") for k in align):
                problems.append("%s: failed d4 alignments: %s" % (lang, {k: v for k, v in align.items() if k not in ("exact", "fuzzy")}))
            if align.get("fuzzy"):
                warnings.append("%s: %d d4 sides aligned fuzzily (template text differed from the prefilled text)" % (lang, align["fuzzy"]))

        # ---- fuse: statuses and fallbacks
        if fu:
            st: Dict[str, int] = {}
            fallbacks = 0
            for d in docs:
                r = fu.get(d.id)
                if r is None:
                    continue
                st[r["status"]] = st.get(r["status"], 0) + 1
                fallbacks += 2 - len((r.get("details") or {}).get("d4_sides", []))
            L["fuse_status_counts"] = st
            L["fuse_sides_without_d4"] = fallbacks
            if fallbacks:
                problems.append("%s: %d document sides were fused without d4 (m2-only fallback)" % (lang, fallbacks))
            salvaged = sum(v for k, v in st.items() if k != "ok")
            if salvaged:
                warnings.append("%s: %d documents with a non-ok fuse status (%s)" % (lang, salvaged, {k: v for k, v in st.items() if k != "ok"}))

        # ---- prediction file
        path = fuse_dir / "predictions" / "fuse" / ("%s_admin_%s.jsonl.jsonl" % (system, lang))
        if path.exists():
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            bad_len = [r["id"] for r, d in zip(rows, docs) if len(r["labels_a"]) != len(d.tokens_a) or len(r["labels_b"]) != len(d.tokens_b)]
            order_ok = [r["id"] for r in rows] == ids
            vals = [v for r in rows for v in r["labels_a"] + r["labels_b"]]
            L["prediction_file"] = {"path": str(path), "rows": len(rows), "order_matches_gold": order_ok,
                                    "label_length_mismatch": bad_len, "min": min(vals) if vals else None,
                                    "max": max(vals) if vals else None, "sha256": sha256(path)}
            if len(rows) != len(ids) or not order_ok:
                problems.append("%s: prediction file has %d rows / order matches gold: %s" % (lang, len(rows), order_ok))
            if bad_len:
                problems.append("%s: label length mismatch in prediction file for %s" % (lang, ", ".join(bad_len[:5])))
        else:
            problems.append("%s: prediction file missing: %s" % (lang, path))
        report["languages"][lang] = L  # type: ignore[index]
    return report


def format_audit(rep: dict) -> str:
    lines = ["验收摘要（只检查，不改任何预测）", "fuse: %s" % rep["fuse_run"], "m2: %s" % rep["m2_run"], "d4: %s" % rep["d4_run"], ""]
    for lang, L in rep["languages"].items():
        lines.append("[%s] 文档 %d" % (lang, L["documents_expected"]))
        for name in ("m2", "d4", "fuse"):
            m = L.get("%s_missing_documents" % name)
            if m is not None:
                lines.append("  %s 结果缺的文档: %d" % (name, len(m)))
        if "m2_blocks_asked" in L:
            lines.append("  m2 块: 问了 %d，拿到判断 %d，没拿到 %d（按'无差异'记）；判断分布 %s；受影响文档 %d" % (
                L["m2_blocks_asked"], L["m2_blocks_judged"], L["m2_blocks_unjudged"], L["m2_judged_status_counts"], len(L["m2_documents_with_unjudged_blocks"])))
        if "d4_sides_expected" in L:
            lines.append("  d4 侧: 应有 %d，完整 %d，缺 %d，长度不符 %d；对齐 %s" % (
                L["d4_sides_expected"], L["d4_sides_complete"], len(L["d4_sides_missing"]), len(L["d4_length_mismatch"]), L["d4_alignment_counts"]))
        if "fuse_status_counts" in L:
            lines.append("  fuse 状态 %s；没有 d4、退回 m2 的侧 %d" % (L["fuse_status_counts"], L["fuse_sides_without_d4"]))
        pf = L.get("prediction_file")
        if pf:
            lines.append("  预测文件: %d 行，顺序与金标一致 %s，长度不符 %d，取值 [%.3f, %.3f]，sha256 %s" % (
                pf["rows"], pf["order_matches_gold"], len(pf["label_length_mismatch"]), pf["min"], pf["max"], pf["sha256"][:16]))
    lines.append("")
    if rep.get("warnings"):
        lines.append("已声明的限制（不改预测，报告里写明）:")
        lines.extend("  - " + p for p in rep["warnings"])
    if rep["problems"]:
        lines.append("致命问题（预测不能交付）:")
        lines.extend("  - " + p for p in rep["problems"])
    elif not rep.get("warnings"):
        lines.append("没有发现问题。")
    else:
        lines.append("没有致命问题。")
    return "\n".join(lines)
