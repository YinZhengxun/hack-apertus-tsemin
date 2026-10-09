"""Command line.

    python run.py selftest            offline tests, no network
    python run.py check               is the endpoint reachable, which models, JSON mode, speed
    python run.py probe               which vLLM extras the endpoint honours (prompt_logprobs, n, structured outputs, ...)
    python run.py smoke               12 development documents end to end (default: m2), scored
    python run.py run --method m1 --split dev/train --limit 20
    python run.py compare --run runs/<id>  our methods and the reference systems on the same documents
    python run.py analyze --run runs/<id> --d4-run runs/<id2>  output-layer variants (m2 tiers, d4 surprisal, combinations)
    python run.py fuse --m2-run runs/<id> --d4-run runs/<id2>  final predictions with the frozen output rule
    python run.py score --pred-dir runs/<id>/predictions/m1 --system m1 --split dev
    python run.py viewer --allow-test             offline HTML viewer of data/predictions/full next to the gold labels
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Sequence

from . import config
from .client import ChatClient
from .data import LANGS, DocPair, load_docs, load_manifest, read_jsonl, select
from .runner import new_run_id, run_and_report
from .scorer import score

MODEL_ALIASES = {"8b": config.DEFAULT_MODEL, "70b": config.MODEL_70B}


def _model(name):
    return MODEL_ALIASES.get((name or "").lower(), name)


def _write_manifest(run_dir: Path, **info) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- selftest
def cmd_selftest(args) -> int:
    suite = unittest.defaultTestLoader.discover(str(config.ROOT / "tests"), pattern="test_*.py",
                                                top_level_dir=str(config.ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print("自检%s：%d 项测试，失败 %d，出错 %d" % ("通过" if result.wasSuccessful() else "没通过",
                                           result.testsRun, len(result.failures), len(result.errors)))
    return 0 if result.wasSuccessful() else 1


# ----------------------------------------------------------------------------- check
def cmd_check(args) -> int:
    cfg = config.get_llm_config()
    run_dir = config.RUNS / new_run_id("check")
    client = ChatClient(cfg, run_dir, use_cache=False, max_retries=1, timeout=180)
    report: Dict[str, object] = {"base_url": cfg.base_url, "probes": []}
    print("接口: %s" % cfg.base_url)

    def note(name: str, ok: bool, detail: str) -> None:
        report["probes"].append({"name": name, "ok": ok, "detail": detail})
        print("[%s] %s: %s" % ("通过" if ok else "失败", name, detail), flush=True)

    try:
        models = client.list_models()
        report["models"] = models
        note("模型列表", True, "%d 个: %s" % (len(models), ", ".join(models)))
    except Exception as e:
        models = []
        note("模型列表", False, "%s: %s" % (type(e).__name__, str(e)[:300]))

    hello = [{"role": "user", "content": "Reply with the single word OK."}]
    for model in (config.DEFAULT_MODEL, config.MODEL_70B):
        r = client.chat(hello, tag="check:hello", model=model, temperature=0, max_tokens=10)
        note("最短请求 " + model, r.ok,
             ("回答 %r, %.1f 秒, 进 %s / 出 %s token, 限流相关响应头 %s" % (
                 (r.content or "").strip()[:40], r.latency_s, r.prompt_tokens, r.completion_tokens, r.headers or "无"))
             if r.ok else str(r.error))

    r = client.chat([{"role": "system", "content": "Whatever the user says, answer with the single word PONG."},
                     {"role": "user", "content": "ping"}], tag="check:system", temperature=0, max_tokens=10)
    note("system 角色", r.ok and "PONG" in (r.content or "").upper(),
         "回答 %r" % (r.content or "").strip()[:40] if r.ok else str(r.error))

    r = client.chat([{"role": "user", "content": 'Return a JSON object with the key "answer" and the value "ok".'}],
                    tag="check:json", temperature=0, max_tokens=50, response_format={"type": "json_object"})
    json_ok = False
    if r.ok:
        try:
            json_ok = isinstance(json.loads(r.content or ""), dict)
        except ValueError:
            json_ok = False
    note("强制 JSON 输出 (response_format)", r.ok and json_ok and not r.adjustments,
         ("回答 %r%s" % ((r.content or "").strip()[:60], ("；服务器不接受该字段，已自动去掉: %s" % r.adjustments) if r.adjustments else ""))
         if r.ok else str(r.error))

    r = client.chat(hello, tag="check:seed", temperature=0, max_tokens=10, seed=0)
    note("固定随机种子 (seed)", r.ok and not r.adjustments,
         ("接受" if not r.adjustments else "服务器不接受该字段，已自动去掉") if r.ok else str(r.error))

    r = client.chat([{"role": "user", "content": "Write the integers from 1 to 300 separated by single spaces. Output only the numbers."}],
                    tag="check:speed", temperature=0, max_tokens=1500)
    if r.ok:
        out = r.completion_tokens or 0
        speed = ("约 %.0f token/秒" % (out / r.latency_s)) if out and r.latency_s > 0 else "速度算不出（服务器没返回 token 数）"
        note("生成速度 (8B)", True, "出 %d token 用 %.1f 秒，%s，结束原因 %s" % (out, r.latency_s, speed, r.finish_reason))
    else:
        note("生成速度 (8B)", False, str(r.error))

    (run_dir / "check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = [p for p in report["probes"] if not p["ok"]]
    print("\n结果已存到 %s" % (run_dir / "check.json"))
    print("全部通过。" if not failed else "有 %d 项没通过（见上面的[失败]行）。" % len(failed))
    return 0


# ----------------------------------------------------------------------------- probe
def cmd_probe(args) -> int:
    from .probes import run_probes
    cfg = config.get_llm_config(_model(args.model))
    run_dir = config.RUNS / new_run_id("probe")
    client = ChatClient(cfg, run_dir, use_cache=False, max_retries=0, timeout=180)
    print("接口能力测试: %s, 模型 %s（每项发 1–4 个请求，结果看效果不看状态码）" % (cfg.base_url, cfg.name))
    results = run_probes(client, cfg.name, which=[w.strip() for w in args.only.split(",")] if args.only else None)
    for r in results:
        print("[%s] %s %s: %s" % ("通过" if r["ok"] else "失败", r["id"], r["name"], r["detail"]), flush=True)
    fps = sorted({r.get("fingerprint") for r in results if r.get("fingerprint")})
    print("system_fingerprint: %s" % (", ".join(fps) if fps else "（响应里没有）"))
    (run_dir / "probe.json").write_text(json.dumps({"base_url": cfg.base_url, "model": cfg.name, "fingerprints": fps,
                                                     "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    verdict = {r["id"]: r["ok"] for r in results}
    print("\n结论：")
    if verdict.get("T2b"):
        print("  - D4（让模型照着念、读每个词的意外程度）可以直接在接口上做。")
    elif verdict.get("T1"):
        print("  - prompt_logprobs 可用，但预填没生效：D4 要改用不预填的写法。")
    else:
        print("  - 接口不返回 prompt 的概率：D4 只能靠本地权重和 GPU。")
    print("  - 投票（D3）：%s" % ("一次请求可以要 3 份回答，便宜。" if verdict.get("T3") else "n 不生效，要发 3 次请求。"))
    print("  - 强制格式（D2）：%s" % ("structured_outputs 可用。" if verdict.get("T4") else ("json_schema 可用。" if verdict.get("T6") else "都不可用，靠代码检查和重试。")))
    print("  - 前缀缓存：%s" % ("看得到加速，逐块提问的输入成本不高。" if verdict.get("T5") else "看不到加速，按完整输入 token 计成本。"))
    print("结果已存到 %s" % (run_dir / "probe.json"))
    required = [x.strip() for x in (args.require or "").split(",") if x.strip()]
    failed = [x for x in required if not verdict.get(x)]
    if failed:
        print("REQUIRED CAPABILITY MISSING: %s. The endpoint does not support what d4 needs (prompt_logprobs on a "
              "prefilled assistant message); the pipeline stops here instead of producing degraded predictions." % ", ".join(failed))
        return 4
    return 0


# ----------------------------------------------------------------------------- smoke / run
def _docs_for(split: str, langs: Sequence[str], limit: int = 0, allow_test: bool = False) -> Dict[str, List[DocPair]]:
    out = {}
    for lang in langs:
        docs = load_docs(lang, split, allow_test=allow_test)
        out[lang] = docs[:limit] if limit else docs
    return out


TEST_REFUSED = ("test 集只在冻结后的最终一次运行里读。确认方法和参数都已经定下、dev 的结果已经写进报告后，"
                "把数据集仓库 data/evaluation/gold_labels/test/gold_admin_{de,fr,it}.jsonl 复制到 data/gold/test/，再加 --allow-test。")


def _guard_test(split: str, allow_test: bool) -> None:
    if split in ("test", "full") and not allow_test:
        raise SystemExit(TEST_REFUSED)


def cmd_smoke(args) -> int:
    cfg = config.get_llm_config(_model(args.model))
    ids = load_manifest("smoke12")
    docs_by_lang = {lang: select(load_docs(lang, "dev/train"), ids[lang]) for lang in LANGS}
    run_dir = config.RUNS / new_run_id("smoke")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    _write_manifest(run_dir, kind="smoke", model=cfg.name, base_url=cfg.base_url, methods=methods,
                    split="dev/train", ids=ids, workers=args.workers)
    client = ChatClient(cfg, run_dir, use_cache=not args.no_cache)
    print("试跑 12 篇（dev/train，每个语种 4 篇）。结果目录: %s" % run_dir)
    for method in methods:
        run_and_report(method, docs_by_lang, client, run_dir, workers=args.workers, **_method_kwargs(method, args))
    print("\n跑完了。把上面的输出贴给 Claude，或者只说一句“跑完了”。结果在 %s" % run_dir)
    return 0


def _method_kwargs(method: str, args) -> dict:
    if method == "m2":
        return {"blocks_per_call": args.blocks_per_call, "max_ref_words": args.max_ref_words}
    return {}


def cmd_run(args) -> int:
    _guard_test(args.split, args.allow_test)
    cfg = config.get_llm_config(_model(args.model))
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    if args.manifest:
        ids = load_manifest(args.manifest)
        docs_by_lang = {lang: select(load_docs(lang, args.split, allow_test=args.allow_test), ids[lang]) for lang in langs}
    else:
        docs_by_lang = _docs_for(args.split, langs, args.limit, allow_test=args.allow_test)
    run_dir = config.RUNS / new_run_id(args.label or ("%s-%s" % (args.method, args.manifest or args.split.replace("/", "_"))))
    _write_manifest(run_dir, kind="run", model=cfg.name, base_url=cfg.base_url, methods=[args.method],
                    split=args.split, limit=args.limit, langs=langs, workers=args.workers,
                    ids={l: [d.id for d in ds] for l, ds in docs_by_lang.items()})
    client = ChatClient(cfg, run_dir, use_cache=not args.no_cache)
    s = run_and_report(args.method, docs_by_lang, client, run_dir, workers=args.workers,
                       system=args.system, **_method_kwargs(args.method, args))
    print("\n结果在 %s" % run_dir)
    if s.get("transient_failures"):
        print("INCOMPLETE: %d documents with failed requests (exit code 5); re-run the same command to re-send only those."
              % len(s["transient_failures"]))
        return 5
    return 0


# ----------------------------------------------------------------------------- compare
REFERENCE_SYSTEMS = {
    "ctfalign_qwen3emb4b_l20": "CTFAlign, Qwen3-Emb-4B layer 20 (organisers' follow-up; not Apertus)",
    "mmbert_simcse_parallel_all": "mmBERT-SimCSE DiffAlign (benchmark repo; not Apertus)",
}


def _reference_preds(system: str, lang: str) -> Dict[str, tuple]:
    rows = read_jsonl(config.DATA / "reference" / ("%s_admin_%s.jsonl.jsonl" % (system, lang)))
    return {r["id"]: (r["labels_a"], r["labels_b"]) for r in rows}


def cmd_compare(args) -> int:
    """Our run(s) and the reference systems on exactly the same documents."""
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    split = manifest.get("split", "dev")
    _guard_test(split, getattr(args, "allow_test", False))
    ids = manifest["ids"]
    docs_by_lang = {lang: select(load_docs(lang, split, allow_test=getattr(args, "allow_test", False)), ids[lang]) for lang in LANGS if lang in ids}
    rows = []
    for path in sorted(run_dir.glob("results_*.jsonl")):
        method = path.stem[len("results_"):]
        res = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                res[r["doc_id"]] = (r["labels_a"], r["labels_b"])
        preds = {lang: {d.id: res[d.id] for d in docs if d.id in res} for lang, docs in docs_by_lang.items()}
        docs_scored = {lang: [d for d in docs if d.id in res] for lang, docs in docs_by_lang.items()}
        sc = score(docs_scored, preds)
        rows.append(("%s (本项目, %s)" % (method, manifest.get("model", "")), sc))
    for system, label in REFERENCE_SYSTEMS.items():
        try:
            preds = {lang: _reference_preds(system, lang) for lang in docs_by_lang}
        except FileNotFoundError:
            continue
        preds = {lang: {d.id: p[d.id] for d in docs if d.id in p} for lang, (docs, p) in
                 ((lang, (docs_by_lang[lang], preds[lang])) for lang in docs_by_lang)}
        docs_scored = {lang: [d for d in docs if d.id in preds[lang]] for lang, docs in docs_by_lang.items()}
        sc = score(docs_scored, preds)
        rows.append((label, sc))
    n = {lang: len(docs) for lang, docs in docs_by_lang.items()}
    print("同一批文档上的对比（%s）:" % ", ".join("%s %d 篇" % kv for kv in n.items()))
    print("%-62s %7s %7s %7s %7s" % ("系统", "de", "fr", "it", "平均"))
    for label, sc in rows:
        print("%-62s %7.3f %7.3f %7.3f %7.3f" % (label, sc.per_lang["de"].spearman, sc.per_lang["fr"].spearman,
                                                sc.per_lang["it"].spearman, sc.macro_spearman))
    print("参考系统的分数来自它们公开的 dev 预测文件（data/reference/README.md），不是本项目重跑的。")
    return 0


# ----------------------------------------------------------------------------- assemble
def cmd_assemble(args) -> int:
    """Merge the fuse results of several runs (e.g. the dev run and the test run) into prediction files in the
    organisers' full-set order (all 224 documents per language, numeric id order). Later runs win on duplicates."""
    import hashlib
    from .analyze import load_results
    from .data import numeric_id
    merged: Dict[str, dict] = {}
    for run in args.runs:
        res = load_results(Path(run), "fuse")
        if not res:
            print("%s 里没有 results_fuse.jsonl" % run)
            return 2
        merged.update(res)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_scored = {}
    preds = {}
    for lang in LANGS:
        docs = load_docs(lang, "full", allow_test=True)
        missing = [d.id for d in docs if d.id not in merged]
        if missing:
            print("%s: 缺 %d 篇的预测（%s …），全集文件必须 224 篇齐全" % (lang, len(missing), ", ".join(missing[:5])))
            return 3
        path = out_dir / ("%s_admin_%s.jsonl.jsonl" % (args.system, lang))
        with open(path, "w", encoding="utf-8") as f:
            for d in docs:
                r = merged[d.id]
                if len(r["labels_a"]) != len(d.tokens_a) or len(r["labels_b"]) != len(d.tokens_b):
                    raise SystemExit("%s: 预测长度与词数不符" % d.id)
                f.write(json.dumps({"id": d.id, "text_a": d.text_a, "text_b": d.text_b,
                                    "labels_a": r["labels_a"], "labels_b": r["labels_b"]}, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print("%s: %d 篇 → %s  sha256 %s" % (lang, len(docs), path, digest[:16]))
        docs_scored[lang] = docs
        preds[lang] = {d.id: (merged[d.id]["labels_a"], merged[d.id]["labels_b"]) for d in docs}
    # scores of the dev and test subsets, as the official script computes them from the full-order file
    for split in ("dev", "test"):
        sub = {lang: [d for d in docs_scored[lang] if d.id in {x.id for x in load_docs(lang, split, allow_test=True)}] for lang in LANGS}
        sc = score(sub, {lang: {d.id: preds[lang][d.id] for d in sub[lang]} for lang in LANGS})
        print("  %s 子集（%s 篇/语种）: %s | %.3f" % (split, len(sub["de"]), " ".join("%.3f" % sc.per_lang[l].spearman for l in LANGS), sc.macro_spearman))
    return 0


# ----------------------------------------------------------------------------- audit
def _locate_run(path_str, near: Path):
    """A run directory named in a manifest: as written, or copied to runs/, data/frozen_runs/ or next to `near`."""
    if not path_str:
        return None
    d = Path(path_str)
    if d.exists():
        return d
    for alt in (config.RUNS / d.name, config.DATA / "frozen_runs" / d.name, near.parent / d.name):
        if alt.exists():
            return alt
    return d


def cmd_audit(args) -> int:
    from .audit import audit, format_audit
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    split = manifest.get("split", "dev")
    _guard_test(split, args.allow_test)
    ids = manifest["ids"]
    docs_by_lang = {lang: select(load_docs(lang, split, allow_test=args.allow_test), ids[lang]) for lang in LANGS if lang in ids}
    m2_dir = _locate_run(manifest.get("m2_run"), run_dir)
    d4_dir = _locate_run(manifest.get("d4_run"), run_dir)
    for name, d in (("m2", m2_dir), ("d4", d4_dir)):
        if d is not None and not d.exists():
            print("%s 运行目录不存在: %s" % (name, d))
            return 1
    system = None
    for f in (run_dir / "predictions" / "fuse").glob("*_admin_de.jsonl.jsonl"):
        system = f.name[: -len("_admin_de.jsonl.jsonl")]
    rep = audit(docs_by_lang, run_dir, m2_dir, d4_dir, system=system or "fuse")
    text = format_audit(rep)
    print(text)
    (run_dir / "audit.txt").write_text(text + "\n", encoding="utf-8")
    (run_dir / "audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if not rep["problems"] else 3


# ----------------------------------------------------------------------------- stats
def cmd_stats(args) -> int:
    """Paired page-grouped bootstrap of a fuse run against a comparison system."""
    from .analyze import load_results
    from .fuse import FuseParams, fuse
    from .stats import format_bootstrap, paired_bootstrap
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    split = manifest.get("split", "dev")
    _guard_test(split, args.allow_test)
    ids = manifest["ids"]
    docs_by_lang = {lang: select(load_docs(lang, split, allow_test=args.allow_test), ids[lang]) for lang in LANGS if lang in ids}
    ours = load_results(run_dir, "fuse")
    preds_a = {lang: {d.id: (ours[d.id]["labels_a"], ours[d.id]["labels_b"]) for d in docs} for lang, docs in docs_by_lang.items()}
    name_a = "fuse(%s)" % (manifest.get("params", {}).get("rule", "?"))

    if args.vs == "ctfalign":
        sub = "test" if split == "test" else ""
        preds_b = {}
        for lang, docs in docs_by_lang.items():
            path = config.DATA / "reference" / sub / ("ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl" % lang) if sub else \
                config.DATA / "reference" / ("ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl" % lang)
            rows = read_jsonl(path)
            if "id" in rows[0]:
                by = {r["id"]: (r["labels_a"], r["labels_b"]) for r in rows}
            else:  # test files carry no ids: pair by order, as the official script does
                full = load_docs(lang, split, allow_test=args.allow_test)
                by = {d.id: (r["labels_a"], r["labels_b"]) for d, r in zip(full, rows)}
            preds_b[lang] = {d.id: by[d.id] for d in docs}
        name_b = "CTFAlign"
    elif args.vs in ("d4", "absent", "verified"):
        d4_dir = _locate_run(manifest.get("d4_run"), run_dir)
        m2_dir = _locate_run(manifest.get("m2_run"), run_dir)
        d4 = load_results(d4_dir, "d4") if d4_dir else {}
        m2 = load_results(m2_dir, "m2") if m2_dir else {}
        p = FuseParams(**{**manifest.get("params", {}), "rule": args.vs})
        preds_b = {lang: {d.id: fuse(d, m2.get(d.id, {}).get("details"), d4.get(d.id, {}).get("details"), p)[0] for d in docs}
                   for lang, docs in docs_by_lang.items()}
        name_b = "rule " + args.vs
    else:
        other = Path(args.vs)
        res = {}
        for path in other.glob("results_*.jsonl"):
            res.update(load_results(other, path.stem[len("results_"):]))
        preds_b = {lang: {d.id: (res[d.id]["labels_a"], res[d.id]["labels_b"]) for d in docs} for lang, docs in docs_by_lang.items()}
        name_b = other.name
    r = paired_bootstrap(docs_by_lang, preds_a, preds_b, resamples=args.resamples, seed=args.seed)
    text = format_bootstrap(name_a, name_b, r)
    print(text)
    tag = args.vs.replace("/", "_").replace("\\", "_")
    (run_dir / ("stats_vs_%s.json" % tag)).write_text(json.dumps({"a": name_a, "b": name_b, **r}, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / ("stats_vs_%s.txt" % tag)).write_text(text + "\n", encoding="utf-8")
    return 0


# ----------------------------------------------------------------------------- analyze
def cmd_analyze(args) -> int:
    from .analyze import analyze, block_gold_rates, format_block_gold_rates, format_rows, load_results
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["ids"]
    docs_by_lang = {lang: select(load_docs(lang, manifest.get("split", "dev")), ids[lang]) for lang in LANGS if lang in ids}
    m2 = load_results(run_dir, "m2")
    d4 = load_results(run_dir, "d4")
    if args.d4_run:
        d4 = load_results(Path(args.d4_run), "d4") or d4
    if args.m2_run:
        m2 = load_results(Path(args.m2_run), "m2") or m2
    rows = analyze(docs_by_lang, m2, d4)
    if not rows:
        print("这个运行目录里没有 m2 或 d4 的结果（results_m2.jsonl / results_d4.jsonl）。")
        return 1
    print("输出层对比（%s；分数在同一批文档上算，篇数列是该规则实际用到的文档数）:" % ", ".join("%s %d 篇" % (l, len(docs_by_lang[l])) for l in docs_by_lang))
    text = format_rows(rows)
    if m2:
        text += "\n\n" + format_block_gold_rates(block_gold_rates(docs_by_lang, m2))
    print(text)
    (run_dir / "analysis.txt").write_text(text + "\n", encoding="utf-8")
    (run_dir / "analysis.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


# ----------------------------------------------------------------------------- fuse
def cmd_fuse(args) -> int:
    """Combine an m2 run and a d4 run into the final predictions with the frozen output rule."""
    from .analyze import load_results
    from .fuse import FuseParams, RULES, fuse
    from .methods import DocResult
    from .runner import export_predictions, save_results, summarize, summary_text

    p = FuseParams(rule=args.rule, quantile=args.quantile, verify_coef=args.verify_coef,
                   rank_coef=args.rank_coef, window=args.window)
    m2_dir = Path(args.m2_run) if args.m2_run else None
    d4_dir = Path(args.d4_run) if args.d4_run else None
    if p.rule != "d4" and m2_dir is None:
        print("规则 %s 需要 --m2-run" % p.rule)
        return 2
    if p.rule != "absent" and d4_dir is None:
        print("规则 %s 需要 --d4-run" % p.rule)
        return 2
    lead = m2_dir or d4_dir
    manifest = json.loads((lead / "manifest.json").read_text(encoding="utf-8"))
    split = manifest.get("split", "dev")
    _guard_test(split, args.allow_test)
    ids = manifest["ids"]
    m2 = load_results(m2_dir, "m2") if m2_dir else {}
    d4 = load_results(d4_dir, "d4") if d4_dir else {}
    docs_by_lang = {lang: select(load_docs(lang, split, allow_test=args.allow_test), ids[lang]) for lang in LANGS if lang in ids}
    # the input runs must cover exactly the documents of the manifest: a run that did not finish must not
    # silently turn into a smaller evaluation set
    expected = {d.id for docs in docs_by_lang.values() for d in docs}
    if p.rule != "d4":
        lacking = sorted(expected - set(m2))
        if lacking and not args.allow_missing:
            raise SystemExit("m2 结果缺 %d 篇（%s …）。先把 m2 跑全，或明确加 --allow-missing 让这些篇按'无差异'记。" % (len(lacking), ", ".join(lacking[:5])))
    if p.rule != "absent":
        lacking = sorted(expected - set(d4))
        if lacking and not args.allow_missing:
            raise SystemExit("d4 结果缺 %d 篇（%s …）。先重跑同一条 d4 命令补齐（缓存会跳过已成功的），或明确加 --allow-missing 让这些篇退回 m2。" % (len(lacking), ", ".join(lacking[:5])))
    results: List[DocResult] = []
    missing_d4 = []
    for lang, docs in docs_by_lang.items():
        for d in docs:
            m2r = m2.get(d.id)
            d4r = d4.get(d.id)
            if p.rule != "d4" and m2r is None:
                m2r = {"details": {"blocks": []}, "n_calls": 0, "n_cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
                print("%s 没有 m2 结果，按'无差异'记（--allow-missing）" % d.id)
            if d4r is None and p.rule != "absent":
                missing_d4.append(d.id)
            (la, lb), info = fuse(d, m2r["details"] if m2r else None, d4r["details"] if d4r else None, p)
            d4_complete = d4r is not None and all(k in d4r["details"] for k in ("s_c_a", "s_c_b", "s_u_a", "s_u_b"))
            status = "ok" if (p.rule == "absent" or d4_complete) else "salvaged"
            results.append(DocResult(d.id, lang, "fuse", "fuse:" + p.rule, manifest.get("model", ""), status, la, lb,
                                     n_calls=(m2r or {}).get("n_calls", 0) + (d4r or {}).get("n_calls", 0),
                                     n_cached=(m2r or {}).get("n_cached", 0) + (d4r or {}).get("n_cached", 0),
                                     prompt_tokens=(m2r or {}).get("prompt_tokens", 0) + (d4r or {}).get("prompt_tokens", 0),
                                     completion_tokens=(m2r or {}).get("completion_tokens", 0) + (d4r or {}).get("completion_tokens", 0),
                                     latency_s=(m2r or {}).get("latency_s", 0.0) + (d4r or {}).get("latency_s", 0.0),
                                     error=None if status == "ok" else "d4 result missing for a side; that side falls back to m2 only",
                                     details=info))
    by_id = {r.doc_id: r for r in results}
    assert set(by_id) == expected, "internal error: fused documents != manifest documents"
    run_dir = config.RUNS / new_run_id(args.label or ("fuse-%s-%s" % (p.rule.replace("+", "_"), manifest.get("split", "dev").replace("/", "_"))))
    _write_manifest(run_dir, kind="fuse", model=manifest.get("model"), split=split, ids=ids, methods=["fuse"],
                    params=p.to_dict(), m2_run=str(m2_dir) if m2_dir else None, d4_run=str(d4_dir) if d4_dir else None)
    save_results(run_dir, "fuse", results)
    export_predictions(run_dir / "predictions" / "fuse", args.system or "fuse", docs_by_lang, by_id)
    s = summarize("fuse", docs_by_lang, results)
    s["params"] = p.to_dict()
    (run_dir / "summary_fuse.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    text = "输出规则 %s  参数 %s\n" % (p.rule, json.dumps(p.to_dict())) + summary_text(s)
    if missing_d4:
        text += "\n没有 d4 结果、只用 m2 的文档: %s" % ", ".join(missing_d4)
    (run_dir / "summary_fuse.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print("\n结果在 %s" % run_dir)
    return 0


# ----------------------------------------------------------------------------- score
def cmd_viewer(args) -> int:
    """One self-contained HTML file: documents side by side, words shaded by the predicted score, gold marked."""
    from .viewer import build_viewer
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    info = build_viewer(Path(args.pred_dir), args.system, Path(args.out), allow_test=args.allow_test,
                        system_label=args.label, langs=langs, title=args.title or "")
    print("查看器写到 %s（%.1f MB），文档数 %s，系统 %s" % (
        info["out"], info["bytes"] / 1e6, ", ".join("%s %d" % kv for kv in info["docs"].items()), " / ".join(info["systems"])))
    for lang, summary in info["summary"].items():
        for grp, row in summary.items():
            print("  %s %-4s n=%-3d %s" % (lang, grp, row["n"], "  ".join(
                "%s %s" % (k, "—" if v is None else "%.3f" % v) for k, v in row.items() if k != "n")))
    print("没有金标的 test 文档不会出现在查看器里；预测文件里多出的文档被忽略。" if not args.allow_test else
          "包含 test 文档（已传 --allow-test）。")
    return 0


def cmd_score(args) -> int:
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    pred_dir = Path(args.pred_dir)
    docs_by_lang, preds = {}, {}
    for lang in langs:
        rows = read_jsonl(pred_dir / ("%s_admin_%s.jsonl.jsonl" % (args.system, lang)))
        preds[lang] = {r["id"]: (r["labels_a"], r["labels_b"]) for r in rows}
        docs = load_docs(lang, args.split)
        if args.only_predicted:
            docs = [d for d in docs if d.id in preds[lang]]
        docs_by_lang[lang] = docs
    print(score(docs_by_lang, preds).table())
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="python run.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    sub.add_parser("check").set_defaults(fn=cmd_check)

    p = sub.add_parser("probe", help="capability probes: prompt_logprobs, prefill, n, structured outputs, json schema, logprobs, prefix cache")
    p.add_argument("--model", default=None)
    p.add_argument("--only", default=None, help="comma list among prompt_logprobs,prefill,prefill_logprobs,n,structured_choice,json_schema,logprobs,prefix_cache")
    p.add_argument("--require", default=None, help="comma list of probe ids (e.g. T2b) that must pass; exit code 4 otherwise")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("smoke")
    p.add_argument("--methods", default="m2,d4,b0s")
    p.add_argument("--model", default=None, help="8b (default), 70b, or a full model name")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--blocks-per-call", type=int, default=3, help="m2: blocks per request (0 = a whole side at once)")
    p.add_argument("--max-ref-words", type=int, default=1500, help="m2: longer reference sides are windowed to this many tokens")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("run")
    p.add_argument("--method", required=True, choices=["m1", "b0", "b0s", "m2", "d4"])
    p.add_argument("--manifest", default=None, help="name of a data/manifests/<name>.json id list (overrides --limit)")
    p.add_argument("--blocks-per-call", type=int, default=3, help="m2: blocks per request (0 = a whole side at once)")
    p.add_argument("--max-ref-words", type=int, default=1500, help="m2: longer reference sides are windowed to this many tokens")
    p.add_argument("--system", default=None, help="name used in the prediction files (default: the method)")
    p.add_argument("--split", default="dev/train", choices=["dev", "dev/train", "dev/val", "test", "full"])
    p.add_argument("--allow-test", action="store_true", help="only for the single frozen final run on the held-out test split (also needed for full)")
    p.add_argument("--langs", default="de,fr,it")
    p.add_argument("--limit", type=int, default=0, help="first N documents per language (0 = all)")
    p.add_argument("--model", default=None)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--label", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("analyze", help="output-layer variants from m2 block judgements and d4 surprisal (needs gold)")
    p.add_argument("--run", required=True, help="a runs/<id> directory with results_m2.jsonl and/or results_d4.jsonl")
    p.add_argument("--d4-run", default=None, help="take results_d4.jsonl from another run directory")
    p.add_argument("--m2-run", default=None, help="take results_m2.jsonl from another run directory")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("fuse", help="final predictions from an m2 run and a d4 run with the frozen output rule")
    p.add_argument("--m2-run", default=None, help="runs/<id> directory with results_m2.jsonl")
    p.add_argument("--d4-run", default=None, help="runs/<id> directory with results_d4.jsonl")
    p.add_argument("--rule", default="verified+d4", choices=["absent", "verified", "verified+d4", "d4"])
    p.add_argument("--quantile", type=float, default=0.7, help="verification: block mean signal must reach this per-document quantile")
    p.add_argument("--verify-coef", type=float, default=0.5, help="s_u coefficient in the verification signal")
    p.add_argument("--rank-coef", type=float, default=0.75, help="s_u coefficient in the ordering signal")
    p.add_argument("--window", type=int, default=4, help="±words of the moving mean for the ordering signal")
    p.add_argument("--system", default=None, help="name used in the prediction files (default: fuse)")
    p.add_argument("--label", default=None)
    p.add_argument("--allow-test", action="store_true", help="only for the single frozen final run on the held-out test split")
    p.add_argument("--allow-missing", action="store_true", help="documents without an m2/d4 result are scored as 'no difference' / fall back instead of aborting")
    p.set_defaults(fn=cmd_fuse)

    p = sub.add_parser("stats", help="paired page-grouped bootstrap of a fuse run against CTFAlign, one of its own simpler rules, or another run")
    p.add_argument("--run", required=True, help="a runs/<id> fuse directory")
    p.add_argument("--vs", default="ctfalign", help="ctfalign | d4 | verified | absent | path to another run directory")
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-test", action="store_true")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("assemble", help="merge fuse runs (dev + test) into full-set-order prediction files (224 documents per language)")
    p.add_argument("--runs", nargs="+", required=True, help="fuse run directories; later ones win on duplicate documents")
    p.add_argument("--system", default="tsemin")
    p.add_argument("--out", default=str(config.DATA / "predictions" / "full"))
    p.set_defaults(fn=cmd_assemble)

    p = sub.add_parser("audit", help="acceptance audit of a fuse run: documents, blocks, d4 arrays, fallbacks, prediction files")
    p.add_argument("--run", required=True, help="a runs/<id> fuse directory (its manifest names the m2 and d4 runs)")
    p.add_argument("--allow-test", action="store_true")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("compare", help="our run and the reference systems on the same documents")
    p.add_argument("--run", required=True, help="a runs/<id> directory")
    p.add_argument("--allow-test", action="store_true")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("viewer", help="offline HTML viewer of prediction files next to the gold labels (one self-contained file)")
    p.add_argument("--pred-dir", default=str(config.DATA / "predictions" / "full"))
    p.add_argument("--system", default="tsemin", help="file name prefix: <system>_admin_<lang>.jsonl.jsonl")
    p.add_argument("--label", default="Fuse", help="display name of the system in the viewer")
    p.add_argument("--out", default=str(config.ROOT / "demo" / "viewer.html"))
    p.add_argument("--langs", default="de,fr,it")
    p.add_argument("--title", default=None)
    p.add_argument("--allow-test", action="store_true", help="also show the test documents (only after the frozen final run)")
    p.set_defaults(fn=cmd_viewer)

    p = sub.add_parser("score")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--split", default="dev", choices=["dev", "dev/train", "dev/val"])
    p.add_argument("--langs", default="de,fr,it")
    p.add_argument("--only-predicted", action="store_true", help="score only the documents present in the files")
    p.set_defaults(fn=cmd_score)

    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
