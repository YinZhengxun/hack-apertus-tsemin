"""Run a method over a list of documents, keep everything on disk, export and score.

Layout of one run:

    runs/<run_id>/
        manifest.json                 what was asked for (method, model, documents, prompt version)
        calls.jsonl                   every request and raw response
        results_<method>.jsonl        one line per document: status, labels, quotes, token counts
        predictions/<method>/<system>_admin_<lang>.jsonl.jsonl    five-field files for the official script
        summary_<method>.json / .txt  scores, failures, cost
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config
from .client import ChatClient
from .data import DocPair
from .methods import FAILED, DocResult, get_method
from .scorer import Score, score, score_lang


def new_run_id(label: str) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + label


def run_method(method: str, docs: Sequence[DocPair], client: ChatClient, workers: int = 2,
               progress: bool = True, **method_kwargs) -> List[DocResult]:
    fn = get_method(method)
    results: Dict[str, DocResult] = {}
    started = time.time()

    def one(doc: DocPair) -> DocResult:
        try:
            return fn(doc, client, **method_kwargs)
        except Exception as e:   # a bug in one document must not lose the others
            la, lb = [0.0] * len(doc.tokens_a), [0.0] * len(doc.tokens_b)
            return DocResult(doc.id, doc.lang, method, "", client.cfg.name, "invalid_answer", la, lb,
                             error="internal error: %s: %s" % (type(e).__name__, e))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, d): d for d in docs}
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results[r.doc_id] = r
            if progress:
                extra = "" if r.status == "ok" else "  [%s]" % r.status
                print("  %s %3d/%d  %-13s %6.1fs  %2d 次调用  进 %6d / 出 %5d token%s" % (
                    method, n, len(docs), r.doc_id, r.latency_s, r.n_calls, r.prompt_tokens, r.completion_tokens, extra), flush=True)
    if progress:
        print("  %s 完成，用时 %.0f 秒" % (method, time.time() - started), flush=True)
    return [results[d.id] for d in docs]


def save_results(run_dir: Path, method: str, results: Sequence[DocResult]) -> Path:
    path = run_dir / ("results_%s.jsonl" % method)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    return path


def export_predictions(out_dir: Path, system: str, docs_by_lang: Dict[str, Sequence[DocPair]],
                       results_by_id: Dict[str, DocResult]) -> List[Path]:
    """Five-field JSONL per language, documents in the order given (use the official gold order).

    File names follow the official evaluation script: <prefix><lang>.jsonl.jsonl. The script switches
    to a different code path when the path contains the letters "llm", so that is refused here.
    """
    if "llm" in system.lower() or "llm" in str(out_dir).lower().replace(str(config.ROOT).lower(), ""):
        raise ValueError('prediction paths must not contain "llm" (the official script treats them differently)')
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for lang, docs in docs_by_lang.items():
        path = out_dir / ("%s_admin_%s.jsonl.jsonl" % (system, lang))
        with open(path, "w", encoding="utf-8") as f:
            for d in docs:
                r = results_by_id[d.id]
                if len(r.labels_a) != len(d.tokens_a) or len(r.labels_b) != len(d.tokens_b):
                    raise ValueError("%s: label length does not match token count" % d.id)
                f.write(json.dumps({"id": d.id, "text_a": d.text_a, "text_b": d.text_b,
                                    "labels_a": r.labels_a, "labels_b": r.labels_b}, ensure_ascii=False) + "\n")
        paths.append(path)
    return paths


def _diagnostics(docs_by_lang: Dict[str, Sequence[DocPair]], by_id: Dict[str, DocResult]) -> dict:
    """Numbers that explain a score: recall by position in the document, quote lengths, tier precision,
    and the score under alternative tier orderings (all need gold labels)."""
    out: Dict[str, dict] = {}
    for lang, docs in docs_by_lang.items():
        dec_hit = [0] * 10
        dec_gold = [0] * 10
        tier_tp = {1: 0, 2: 0}
        tier_n = {1: 0, 2: 0}
        q_lengths: List[int] = []
        tier_preds = {"absent_first": {}, "quote_first": {}}
        for d in docs:
            r = by_id[d.id]
            if d.labels_a is None:
                continue
            tiers = (r.details.get("tiers_a"), r.details.get("tiers_b"))
            for side, (gold, pred, tier) in enumerate(((d.labels_a, r.labels_a, tiers[0]), (d.labels_b, r.labels_b, tiers[1]))):
                n = len(gold)
                for i, g in enumerate(gold):
                    if g == -1:
                        continue
                    if g > 0:
                        k = min(9, 10 * i // max(1, n))
                        dec_gold[k] += 1
                        if pred[i] > 0:
                            dec_hit[k] += 1
                    if tier and tier[i] in tier_n:
                        tier_n[tier[i]] += 1
                        if g > 0:
                            tier_tp[tier[i]] += 1
            for q in r.details.get("quotes") or []:
                if q.get("start") is not None and q.get("end") is not None:
                    q_lengths.append(q["end"] - q["start"])
            if tiers[0] and tiers[1]:
                a1 = [1.0 if t == 1 else (0.7 if t == 2 else 0.0) for t in tiers[0]]
                b1 = [1.0 if t == 1 else (0.7 if t == 2 else 0.0) for t in tiers[1]]
                a2 = [0.7 if t == 1 else (1.0 if t == 2 else 0.0) for t in tiers[0]]
                b2 = [0.7 if t == 1 else (1.0 if t == 2 else 0.0) for t in tiers[1]]
                tier_preds["absent_first"][d.id] = (a1, b1)
                tier_preds["quote_first"][d.id] = (a2, b2)
        entry = {
            "recall_by_position_decile": [round(dec_hit[k] / dec_gold[k], 3) if dec_gold[k] else None for k in range(10)],
            "gold_positive_by_decile": dec_gold,
            "quote_length_median": (sorted(q_lengths)[len(q_lengths) // 2] if q_lengths else None),
            "quote_length_p90": (sorted(q_lengths)[int(0.9 * (len(q_lengths) - 1))] if q_lengths else None),
            "n_quotes_located": len(q_lengths),
            "tier_precision": {("absent_block" if t == 1 else "quote"): (round(tier_tp[t] / tier_n[t], 3) if tier_n[t] else None)
                               for t in (1, 2)},
            "tier_tokens": {("absent_block" if t == 1 else "quote"): tier_n[t] for t in (1, 2)},
        }
        for name, preds in tier_preds.items():
            if preds and len(preds) == len(docs):
                try:
                    entry["spearman_" + name] = round(score_lang(lang, docs, preds).spearman, 4)
                except (ValueError, KeyError):
                    pass
        out[lang] = entry
    return out


def summarize(method: str, docs_by_lang: Dict[str, Sequence[DocPair]], results: Sequence[DocResult]) -> dict:
    by_id = {r.doc_id: r for r in results}
    preds = {lang: {d.id: (by_id[d.id].labels_a, by_id[d.id].labels_b) for d in docs} for lang, docs in docs_by_lang.items()}
    has_gold = all(d.labels_a is not None and d.labels_b is not None for docs in docs_by_lang.values() for d in docs)
    sc: Optional[Score] = score(docs_by_lang, preds) if has_gold else None
    diagnostics: dict
    if has_gold:
        try:
            diagnostics = _diagnostics(docs_by_lang, by_id)
        except Exception as e:   # diagnostics must never block the summary
            diagnostics = {"error": "%s: %s" % (type(e).__name__, e)}
    else:
        diagnostics = {}
    statuses: Dict[str, int] = {}
    locations: Dict[str, int] = {}
    block_statuses: Dict[str, int] = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
        for k, v in (r.details.get("location_counts") or {}).items():
            locations[k] = locations.get(k, 0) + v
        for k, v in (r.details.get("block_status_counts") or {}).items():
            block_statuses[k] = block_statuses.get(k, 0) + v
    n = max(1, len(results))
    real = [r for r in results if r.n_calls > r.n_cached]
    cost = {
        "documents": len(results),
        "calls": sum(r.n_calls for r in results),
        "calls_from_cache": sum(r.n_cached for r in results),
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "completion_tokens": sum(r.completion_tokens for r in results),
        "prompt_tokens_per_doc": round(sum(r.prompt_tokens for r in results) / n, 1),
        "completion_tokens_per_doc": round(sum(r.completion_tokens for r in results) / n, 1),
        "seconds_per_doc_uncached": round(sum(r.latency_s for r in real) / len(real), 2) if real else None,
    }
    failures = [{"id": r.doc_id, "status": r.status, "error": r.error, "finish_reason": r.finish_reason}
                for r in results if r.status != "ok"]
    return {"method": method, "model": results[0].model if results else None,
            "prompt_version": sorted({r.prompt_version for r in results}),
            "score": sc.to_dict() if sc else None,
            "score_table": sc.table() if sc else "（输入文件没有金标，只生成预测，不打分）",
            "statuses": statuses,
            "quote_locations": locations, "block_statuses": block_statuses, "cost": cost, "not_ok": failures,
            "diagnostics": diagnostics,
            "failed_documents": [r.doc_id for r in results if r.status in FAILED]}


def summary_text(s: dict) -> str:
    lines = ["方法 %s | 模型 %s | 提示词 %s" % (s["method"], s["model"], ", ".join(s["prompt_version"])),
             s["score_table"],
             "文档状态: " + ", ".join("%s=%d" % kv for kv in sorted(s["statuses"].items()))]
    if s.get("block_statuses"):
        lines.append("块判定: " + ", ".join("%s=%d" % kv for kv in sorted(s["block_statuses"].items())))
    if s["quote_locations"]:
        lines.append("引用定位: " + ", ".join("%s=%d" % kv for kv in sorted(s["quote_locations"].items())))
    diag = s.get("diagnostics") or {}
    for lang, dg in diag.items():
        if not isinstance(dg, dict):
            continue
        parts = []
        rec = dg.get("recall_by_position_decile")
        if rec and any(v is not None for v in rec):
            parts.append("按文档位置十等分的召回: " + " ".join("-" if v is None else "%.2f" % v for v in rec))
        if dg.get("quote_length_median") is not None:
            parts.append("引用长度中位数 %d / p90 %d 词 (%d 条)" % (dg["quote_length_median"], dg["quote_length_p90"], dg["n_quotes_located"]))
        tp = dg.get("tier_precision") or {}
        if any(v is not None for v in tp.values()):
            parts.append("分档精度: 整块无对应 %s (%d 词), 引用 %s (%d 词)" % (
                tp.get("absent_block"), dg["tier_tokens"]["absent_block"], tp.get("quote"), dg["tier_tokens"]["quote"]))
        if "spearman_absent_first" in dg:
            parts.append("分档 Spearman: 整块=1.0/引用=0.7 → %.3f, 反过来 → %.3f" % (dg["spearman_absent_first"], dg["spearman_quote_first"]))
        if parts:
            lines.append("  %s: " % lang + "; ".join(parts))
    c = s["cost"]
    secs = c["seconds_per_doc_uncached"]
    lines.append("开销: %d 篇, %d 次调用 (其中缓存 %d), 每篇进 %.0f / 出 %.0f token, 每篇 %s" % (
        c["documents"], c["calls"], c["calls_from_cache"], c["prompt_tokens_per_doc"],
        c["completion_tokens_per_doc"], ("%.1f 秒" % secs) if secs is not None else "用时未计（全部来自缓存）"))
    for f in s["not_ok"]:
        lines.append("  未正常完成: %s  %s  %s" % (f["id"], f["status"], (f["error"] or "")[:160]))
    return "\n".join(lines)


def transient_failures(results: Sequence[DocResult]) -> List[str]:
    """Documents whose result is incomplete because a request failed (endpoint errors after the client's retries).
    Re-running the same command re-sends only these requests (successful ones are cached), so the caller may retry.
    Not included: answers that came back but were unusable (a block the model skipped, a truncated b0s chunk) -
    the same request would give the same answer again."""
    out = []
    for r in results:
        det = r.details or {}
        if r.status == "api_error":
            out.append(r.doc_id)
        elif r.method == "d4" and any("alignment" not in (f.get("error") or "") for f in det.get("failed", [])):
            out.append(r.doc_id)
        elif r.method == "m2" and any(b.get("status") == "call_failed" for b in det.get("blocks", [])):
            out.append(r.doc_id)
        elif r.method == "b0s" and det.get("chunk_statuses", {}).get("api_error"):
            out.append(r.doc_id)
    return out


def run_and_report(method: str, docs_by_lang: Dict[str, Sequence[DocPair]], client: ChatClient,
                   run_dir: Path, workers: int = 2, system: Optional[str] = None, **method_kwargs) -> dict:
    docs = [d for lang in docs_by_lang for d in docs_by_lang[lang]]
    print("\n=== %s：%d 篇，模型 %s ===" % (method, len(docs), client.cfg.name), flush=True)
    results = run_method(method, docs, client, workers=workers, **method_kwargs)
    save_results(run_dir, method, results)
    by_id = {r.doc_id: r for r in results}
    export_predictions(run_dir / "predictions" / method, system or method, docs_by_lang, by_id)
    s = summarize(method, docs_by_lang, results)
    s["transient_failures"] = transient_failures(results)
    (run_dir / ("summary_%s.json" % method)).write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    text = summary_text(s)
    if s["transient_failures"]:
        text += "\n请求失败的文档 %d 篇（%s）：原样重跑同一条命令只会重发这些失败的请求，其余走缓存。" % (
            len(s["transient_failures"]), ", ".join(s["transient_failures"][:6]) + ("…" if len(s["transient_failures"]) > 6 else ""))
    (run_dir / ("summary_%s.txt" % method)).write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return s
