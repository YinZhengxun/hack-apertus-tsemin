"""Additional numbers for the report, all offline from the frozen runs and the reference files (no model calls):

  1. paired page bootstrap D4-contrastive vs CTFAlign (dev and test)
  2. cost per document pair on dev60, where m2, d4 and b0s were run on the same 60 pairs; latency from live
     (non-cached) requests of the first runs
  3. gold-difference spans by length (dev): where do the predicted scores rank them
  4. example selection for the qualitative figure (dev/val, fixed rule)

    python tools/report_extra.py
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uzh_diff.analyze import load_results  # noqa: E402
from uzh_diff.data import LANGS, load_docs, load_manifest, read_jsonl, select  # noqa: E402
from uzh_diff.fuse import FuseParams, fuse, block_statuses, verified_absent  # noqa: E402
from uzh_diff.stats import paired_bootstrap, format_bootstrap, _ranks  # noqa: E402

FR = ROOT / "data" / "frozen_runs"
M2 = {"dev": FR / "20261006-214833-m2-dev", "test": FR / "20261007-114305-m2-test"}
D4 = {"dev": FR / "20261007-112326-d4-dev", "test": FR / "20261007-122232-d4-test"}


def ctf(lang, split, docs):
    sub = "test/" if split == "test" else ""
    rows = read_jsonl(ROOT / "data/reference" / (sub + "ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl" % lang))
    if "id" in rows[0]:
        by = {r["id"]: (r["labels_a"], r["labels_b"]) for r in rows}
    else:
        by = {d.id: (r["labels_a"], r["labels_b"]) for d, r in zip(docs, rows)}
    return {d.id: by[d.id] for d in docs}


def main():
    print("== 1. paired bootstrap, D4-contrastive vs CTFAlign (1000 resamples, seed 0)")
    for split in ("dev", "test"):
        docs = {l: load_docs(l, split, allow_test=True) for l in LANGS}
        d4 = load_results(D4[split], "d4")
        pa = {l: {d.id: fuse(d, None, d4[d.id]["details"], FuseParams(rule="d4"))[0] for d in docs[l]} for l in LANGS}
        pb = {l: ctf(l, split, docs[l]) for l in LANGS}
        r = paired_bootstrap(docs, pa, pb, resamples=1000, seed=0)
        print(split); print(format_bootstrap("D4-contrastive", "CTFAlign", r))
        (FR / ("stats_d4_vs_ctfalign_%s.json" % split)).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n== 2. cost per document pair on dev60 (same 60 pairs for m2, d4, b0s)")
    man = load_manifest("dev60")
    runs = {"m2": FR / "20261006-210552-m2-dev60", "d4": FR / "20261006-210552-d4-dev60", "b0s": FR / "20261006-210552-b0s-dev60"}
    for name, run in runs.items():
        res = load_results(run, name)
        rows = [res[i] for l in LANGS for i in man[l] if i in res]
        n = len(rows)
        calls = sum(r["n_calls"] for r in rows) / n
        cached = sum(r.get("n_cached", 0) for r in rows) / n
        live = sum(r["n_calls"] - r.get("n_cached", 0) for r in rows)
        lat_live = sum(r["latency_s"] for r in rows) / max(1, live)       # cached calls add ~0 latency
        print("  %-4s docs %d | requests/pair %.1f (cached %.1f) | prompt tok %6.0f | generated tok %5.0f | latency per live request %.2f s -> per pair %.1f s" % (
            name, n, calls, cached, sum(r["prompt_tokens"] for r in rows) / n, sum(r["completion_tokens"] for r in rows) / n,
            lat_live, lat_live * calls))

    print("\n== 3. gold-difference spans by length (dev, both sides): percentile rank of the span's mean score within its language")
    docs = {l: load_docs(l, "dev") for l in LANGS}
    m2, d4 = load_results(M2["dev"], "m2"), load_results(D4["dev"], "d4")
    for sysname, mk in (("Fuse", lambda d: fuse(d, m2[d.id]["details"], d4[d.id]["details"], FuseParams())[0]),
                        ("D4-contrastive", lambda d: fuse(d, None, d4[d.id]["details"], FuseParams(rule="d4"))[0]),
                        ("CTFAlign", None)):
        out = {}
        for l in LANGS:
            preds = ctf(l, "dev", docs[l]) if mk is None else {d.id: mk(d) for d in docs[l]}
            # percentile rank of every scored token within the language
            allp = [float(y) for d in docs[l] for side, p in (("a", preds[d.id][0]), ("b", preds[d.id][1])) for x, y in zip(getattr(d, "labels_" + side), p) if x != -1]
            rk = _ranks(allp); n = len(allp)
            pos = {}
            k = 0
            for d in docs[l]:
                for side, p in (("a", preds[d.id][0]), ("b", preds[d.id][1])):
                    labs = getattr(d, "labels_" + side)
                    for x in labs:
                        pass
                    idx = [i for i, x in enumerate(labs) if x != -1]
                    ranks_here = rk[k:k + len(idx)]; k += len(idx)
                    pr = {i: r / n for i, r in zip(idx, ranks_here)}
                    # spans of consecutive gold-positive scored tokens
                    span = []
                    for i in idx + [None]:
                        if i is not None and labs[i] > 0:
                            span.append(i)
                        else:
                            if span:
                                L = len(span)
                                b = "1" if L == 1 else ("2-4" if L <= 4 else "5+")
                                pos.setdefault(b, []).append(statistics.mean(pr[j] for j in span))
                            span = []
            out[l] = {b: (len(v), statistics.median(v)) for b, v in pos.items()}
        print("  %-15s " % sysname + " | ".join("%s: " % l + " ".join("%s n=%d med=%.2f" % (b, out[l][b][0], out[l][b][1]) for b in ("1", "2-4", "5+")) for l in LANGS))

    print("\n== 4. example candidates (dev/val): verified absent blocks with the highest/lowest gold rate")
    for l in LANGS:
        val = select(docs[l], load_manifest("dev60")[l]) if False else load_docs(l, "dev/val")
        cands = []
        for d in val:
            keep = verified_absent(d, m2[d.id]["details"], d4[d.id]["details"], FuseParams())
            st = block_statuses(d, m2[d.id]["details"])
            from uzh_diff.blocks import make_blocks
            for side, toks in (("a", d.tokens_a), ("b", d.tokens_b)):
                labs = getattr(d, "labels_" + side)
                for b in make_blocks(toks, side):
                    if not keep[side][b.start]:
                        continue
                    g = [labs[i] for i in range(b.start, b.end) if labs[i] != -1]
                    if len(g) < 4:
                        continue
                    rate = sum(1 for x in g if x > 0) / len(g)
                    cands.append((rate, d.id, side, b.id, " ".join(toks[b.start:b.end])[:140]))
        cands.sort()
        print("  %s: %d verified absent blocks with >=4 scored words; gold rate mean %.2f" % (l, len(cands), statistics.mean(c[0] for c in cands) if cands else 0))
        for c in cands[:2] + cands[-2:]:
            print("     rate %.2f %s/%s %s: %s" % c)


if __name__ == "__main__":
    main()
