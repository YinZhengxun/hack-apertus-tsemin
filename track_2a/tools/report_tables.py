"""Recompute every number in technical_report.md from the run directories (no model calls).

    python tools/report_tables.py > docs/report_tables.txt

Reads the frozen run results shipped in data/frozen_runs/ (gzipped per-document results of the runs behind the
submission), or the original runs/ directories when present.
Scores use the official definition (per language one pooled Spearman over all scored tokens, then the mean
over de/fr/it); data/predictions/README.md shows that the official script gives the same numbers.
"""
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uzh_diff.analyze import load_results  # noqa: E402
from uzh_diff.data import LANGS, load_docs, load_manifest, read_jsonl, select  # noqa: E402
from uzh_diff.fuse import FuseParams, d4_signal, fuse, verified_absent  # noqa: E402
from uzh_diff.scorer import score  # noqa: E402

RUNS = ROOT / "data" / "frozen_runs"        # gzipped copies shipped with the repository
if not (RUNS / "20261007-112326-d4-dev").exists():
    RUNS = ROOT / "runs"                     # the original run directories
M2 = {"dev": "20261006-214833-m2-dev", "test": "20261007-114305-m2-test"}
D4 = {"dev": "20261007-112326-d4-dev", "test": "20261007-122232-d4-test"}
B0S = {"dev60": "20261006-210552-b0s-dev60", "test": "20261008-140115-b0s-test"}


def load_pred_file(path):
    return {r["id"]: (r["labels_a"], r["labels_b"]) for r in read_jsonl(path)}


def sigmoid(x, scale=3.0):
    return 1.0 / (1.0 + math.exp(-x / scale))


def fuse_keep_negatives(doc, m2d, d4d, p):
    """Ablation: same as the frozen rule but negative signals keep their order (monotone sigmoid, no floor)."""
    keep = verified_absent(doc, m2d, d4d, p) if p.rule != "d4" else None
    out = []
    for side, toks in (("a", doc.tokens_a), ("b", doc.tokens_b)):
        v = d4_signal(d4d, side, p.rank_coef, p.window)
        if p.rule == "d4":
            out.append([sigmoid(x) for x in v])
        else:
            out.append([(0.5 + 0.5 * sigmoid(x)) if k else 0.49 * sigmoid(x) for k, x in zip(keep[side], v)])
    return out[0], out[1]


def fuse_unverified(doc, m2d, d4d, p):
    """Ablation: every m2 absent block on top (no d4 verification), the rest ordered by d4."""
    p_abs = FuseParams(rule="absent")
    (ka, kb), _ = fuse(doc, m2d, None, p_abs)
    d4only, _ = fuse(doc, None, d4d, FuseParams(rule="d4", rank_coef=p.rank_coef, window=p.window))
    out = []
    for k, v in ((ka, d4only[0]), (kb, d4only[1])):
        out.append([(0.5 + 0.5 * x) if f else 0.49 * x for f, x in zip(k, v)])
    return out[0], out[1]


def row(name, docs, preds):
    s = score(docs, preds)
    return "  %-48s %s | %.3f" % (name, " ".join("%.3f" % s.per_lang[l].spearman for l in LANGS), s.macro_spearman), s.macro_spearman


def main():
    allow = True
    docs_dev = {l: load_docs(l, "dev") for l in LANGS}
    docs_test = {l: load_docs(l, "test", allow_test=allow) for l in LANGS}
    subsets = {
        "dev (168/lang)": docs_dev,
        "dev/train (134/lang)": {l: load_docs(l, "dev/train") for l in LANGS},
        "dev/val (34/lang)": {l: load_docs(l, "dev/val") for l in LANGS},
        "dev60 (20/lang, parameter selection)": {l: select(docs_dev[l], load_manifest("dev60")[l]) for l in LANGS},
        "test (56/lang)": docs_test,
    }
    res = {}
    for split in ("dev", "test"):
        res[("m2", split)] = load_results(RUNS / M2[split], "m2")
        res[("d4", split)] = load_results(RUNS / D4[split], "d4")
    ref = {}
    for l in LANGS:
        ref.setdefault("CTFAlign", {}).update(load_pred_file(ROOT / "data/reference" / ("ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl" % l)))
        ref["CTFAlign"].update(load_pred_file(ROOT / "data/reference/test" / ("ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl" % l)))
        ref.setdefault("mmBERT-SimCSE (dev only)", {}).update(load_pred_file(ROOT / "data/reference" / ("mmbert_simcse_parallel_all_admin_%s.jsonl.jsonl" % l)))
    b0s = {}
    for k, run in B0S.items():
        r = load_results(RUNS / run, "b0s")
        b0s.update({i: (x["labels_a"], x["labels_b"]) for i, x in r.items()})

    P = FuseParams()

    def split_of(doc_id, test_ids):
        return "test" if doc_id in test_ids else "dev"

    test_ids = {d.id for l in LANGS for d in docs_test[l]}

    def systems_for(docs):
        out = {}
        flat = [d for l in LANGS for d in docs[l]]
        def m2(d): return res[("m2", split_of(d.id, test_ids))][d.id]["details"]
        def d4(d): return res[("d4", split_of(d.id, test_ids))][d.id]["details"]
        out["Fuse (submitted)"] = {d.id: fuse(d, m2(d), d4(d), P)[0] for d in flat}
        out["D4-contrastive"] = {d.id: fuse(d, None, d4(d), FuseParams(rule="d4"))[0] for d in flat}
        out["D4-raw (s_c, max over +-1 word)"] = {d.id: (res[("d4", split_of(d.id, test_ids))][d.id]["labels_a"],
                                                    res[("d4", split_of(d.id, test_ids))][d.id]["labels_b"]) for d in flat}
        out["M2-verified (0/1)"] = {d.id: fuse(d, m2(d), d4(d), FuseParams(rule="verified"))[0] for d in flat}
        out["M2-absent (0/1)"] = {d.id: fuse(d, m2(d), None, FuseParams(rule="absent"))[0] for d in flat}
        for name, pr in ref.items():
            if all(d.id in pr for d in flat):
                out[name] = pr
        if all(d.id in b0s for d in flat):
            out["b0s (paper prompt, Apertus-8B)"] = b0s
        return out

    def by_lang(preds, docs):
        return {l: {d.id: preds[d.id] for d in docs[l]} for l in LANGS}

    print("== Main table: Spearman de fr it | mean ==")
    for name, docs in subsets.items():
        print(name)
        for sysname, pr in systems_for(docs).items():
            print(row(sysname, docs, by_lang(pr, docs))[0])
        print()

    print("== Ablations of the output layer (full dev and test; parameters were chosen on dev60 only) ==")
    for name in ("dev (168/lang)", "test (56/lang)"):
        docs = subsets[name]
        flat = [d for l in LANGS for d in docs[l]]
        def m2(d): return res[("m2", split_of(d.id, test_ids))][d.id]["details"]
        def d4(d): return res[("d4", split_of(d.id, test_ids))][d.id]["details"]
        variants = [
            ("Fuse (frozen)", lambda d: fuse(d, m2(d), d4(d), P)[0]),
            ("  without verification (all M2-absent on top)", lambda d: fuse_unverified(d, m2(d), d4(d), P)),
            ("  negatives keep their order (no floor)", lambda d: fuse_keep_negatives(d, m2(d), d4(d), P)),
            ("D4-contrastive (frozen)", lambda d: fuse(d, None, d4(d), FuseParams(rule="d4"))[0]),
            ("  no s_u discount (coef 0)", lambda d: fuse(d, None, d4(d), FuseParams(rule="d4", rank_coef=0.0))[0]),
            ("  no smoothing (window 0)", lambda d: fuse(d, None, d4(d), FuseParams(rule="d4", window=0))[0]),
            ("  negatives keep their order (no floor)", lambda d: fuse_keep_negatives(d, None, d4(d), FuseParams(rule="d4"))),
            ("  s_u only (no other document)", None),
        ]
        print(name)
        for vname, fn in variants:
            if fn is None:
                pr = {}
                for d in flat:
                    det = d4(d)
                    pr[d.id] = tuple([1 - math.exp(-max(x, 0) / 3.0) for x in
                                      d4_signal({"s_c_" + s: det["s_u_" + s], "s_u_" + s: det["s_u_" + s]}, s, 0.0, 4)]
                                     for s in ("a", "b"))
            else:
                pr = {d.id: fn(d) for d in flat}
            print(row(vname, docs, by_lang(pr, docs))[0])
        print()

    print("== Share of words in verified absent blocks (score >= 0.5) and gold-difference rate inside them ==")
    for name in ("dev (168/lang)", "test (56/lang)"):
        docs = subsets[name]
        for l in LANGS:
            n = flagged = gold_in = gold_all = 0
            for d in docs[l]:
                pa, pb = fuse(d, res[("m2", split_of(d.id, test_ids))][d.id]["details"],
                              res[("d4", split_of(d.id, test_ids))][d.id]["details"], P)[0]
                for g, p in list(zip(d.labels_a, pa)) + list(zip(d.labels_b, pb)):
                    if g == -1:
                        continue
                    n += 1
                    gold_all += g > 0
                    if p >= 0.5:
                        flagged += 1
                        gold_in += g > 0
            print("  %-16s %s: flagged %.1f%% of scored words, gold rate inside %.2f (overall %.3f)" % (
                name, l, 100 * flagged / n, gold_in / max(1, flagged), gold_all / n))
    print()

    print("== Cost per document pair (from the run logs; 2 documents in parallel, cached calls excluded from latency) ==")
    for split in ("dev", "test"):
        for method in ("m2", "d4"):
            r = list(res[(method, split)].values())
            n = len(r)
            print("  %-5s %-3s docs %3d | calls %.1f | prompt tok %6.0f | completion tok %5.0f | latency %.1f s" % (
                split, method, n, sum(x["n_calls"] for x in r) / n, sum(x["prompt_tokens"] for x in r) / n,
                sum(x["completion_tokens"] for x in r) / n, sum(x["latency_s"] for x in r) / n))
    r = list(load_results(RUNS / B0S["test"], "b0s").values())
    n = len(r)
    print("  test  b0s docs %3d | calls %.1f | prompt tok %6.0f | completion tok %5.0f | latency %.1f s" % (
        n, sum(x["n_calls"] for x in r) / n, sum(x["prompt_tokens"] for x in r) / n,
        sum(x["completion_tokens"] for x in r) / n, sum(x["latency_s"] for x in r) / n))

    print("\n== m2 block judgements: share of gold-difference words per status (full dev) ==")
    from uzh_diff.fuse import block_statuses
    for l in LANGS:
        tot = {}
        for d in docs_dev[l]:
            st = block_statuses(d, res[("m2", "dev")][d.id]["details"])
            for side, labs in (("a", d.labels_a), ("b", d.labels_b)):
                for g, s in zip(labs, st[side]):
                    if g == -1:
                        continue
                    t = tot.setdefault(s or "none", [0, 0])
                    t[0] += 1
                    t[1] += g > 0
        allw = sum(v[0] for v in tot.values())
        print("  %s: " % l + "  ".join("%s %.0f%% of words, gold rate %.2f" % (k, 100 * v[0] / allw, v[1] / v[0])
                                       for k, v in sorted(tot.items(), key=lambda kv: -kv[1][0])))


if __name__ == "__main__":
    main()
