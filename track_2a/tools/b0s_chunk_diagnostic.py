"""Diagnostic of the paper-prompt baseline (b0s) on the test split: score as run, chunk-level failure profile
(truncation, repetition loops, unreadable answers), and the score restricted to the chunks that did answer, with
Fuse / D4-contrastive / CTFAlign recomputed on exactly the same tokens so the subset comparison is fair.

Per-document results are read from data/frozen_runs/; the chunk-level part needs the request log
runs/20261008-140115-b0s-test/calls.jsonl, which is not part of the repository (the output is kept in
data/baselines/b0s/test/diagnostic.txt). Run from track_2a:  python tools/b0s_chunk_diagnostic.py
Output of 2026-10-08 is in data/baselines/b0s/test/diagnostic.txt."""
import json, sys, re
from collections import Counter
from pathlib import Path
sys.path.insert(0, "src")
from uzh_diff.data import LANGS, load_docs, read_jsonl
from uzh_diff.methods import paper_chunks, _b0_load
from uzh_diff.stats import pooled_spearman
from uzh_diff.scorer import score

RUNS = Path("data/frozen_runs") if Path("data/frozen_runs/20261008-140115-b0s-test").exists() else Path("runs")
RUN = RUNS / "20261008-140115-b0s-test"
FUSE = RUNS / "20261007-122413-fuse-verified_d4-test"
CALLS = Path("runs/20261008-140115-b0s-test/calls.jsonl")   # the request log is not part of the repository
docs = {l: load_docs(l, "test", allow_test=True) for l in LANGS}
by_id = {d.id: d for l in LANGS for d in docs[l]}

# predictions
def load_pred(path):
    return {r["id"]: (r["labels_a"], r["labels_b"]) for r in read_jsonl(path)}
b0s = {l: load_pred(Path("data/baselines/b0s/test") / ("b0s_admin_%s.jsonl.jsonl" % l)) for l in LANGS}
fuse = {l: load_pred(FUSE / "predictions/fuse" / ("tsemin_admin_%s.jsonl.jsonl" % l)) for l in LANGS}
ctf = {l: load_pred(Path("data/reference/test") / ("ctfalign_qwen3emb4b_l20_admin_%s.jsonl.jsonl" % l)) for l in LANGS}
# D4-contrastive on test: fuse --rule d4 run? recompute from the d4 run with fuse()
from uzh_diff.analyze import load_results
from uzh_diff.fuse import FuseParams, fuse as fuse_fn
m2res = load_results(RUNS / "20261007-114305-m2-test", "m2")
d4res = load_results(RUNS / "20261007-122232-d4-test", "d4")
d4c = {l: {d.id: fuse_fn(d, m2res[d.id]["details"], d4res[d.id]["details"], FuseParams(rule="d4"))[0] for d in docs[l]} for l in LANGS}

print("=== as run (official pooling), test 56/lang")
for name, p in (("b0s", b0s), ("D4-contrastive", d4c), ("Fuse", fuse), ("CTFAlign", ctf)):
    s = score(docs, p)
    print("  %-15s %s | %.3f" % (name, " ".join("%.3f" % s.per_lang[l].spearman for l in LANGS), s.macro_spearman))

# chunk-level statuses from the call log
calls = [json.loads(l) for l in open(CALLS)]
status = {}   # (doc, k) -> ok / truncated / invalid / api_error
loops = 0
for c in calls:
    _, doc_id, k = c["tag"].split(":")
    res = c["result"]
    if not res.get("ok"):
        status[(doc_id, int(k))] = "api_error"; continue
    content = res.get("content") or ""
    fr = res.get("finish_reason")
    data = _b0_load(content)
    if fr == "length":
        st = "truncated"
        tail = content[-3000:]
        # repetition loop: some 30-char piece repeated many times in the tail
        best = 0
        for i in range(0, max(1, len(tail) - 30), 50):
            piece = tail[i:i + 30]
            if piece.strip():
                best = max(best, tail.count(piece))
        if best >= 8:
            loops += 1
    else:
        d = by_id[doc_id]
        sa, ea, sb, eb = paper_chunks(len(d.tokens_a), len(d.tokens_b))[int(k)]
        need = 1 if (ea == sa or eb == sb) else 2
        got = (isinstance(data, dict) and ("sentence1" in data) and ea > sa) + (isinstance(data, dict) and ("sentence2" in data) and eb > sb)
        st = "ok" if got >= need else "invalid"
    status[(doc_id, int(k))] = st
cnt = Counter(status.values())
print("\n=== chunk calls: %d  %s ; truncated with a repetition loop in the tail: %d" % (len(status), dict(cnt), loops))
per_lang_cnt = {l: Counter(v for (d, k), v in status.items() if d.split("_")[1] == l) for l in LANGS}
for l in LANGS:
    print("  %s: %s" % (l, dict(per_lang_cnt[l])))

# answered-chunks-only subset: tokens (gold != -1) inside chunks with status ok
def subset_scores(preds):
    out = {}
    for l in LANGS:
        g, p = [], []
        for d in docs[l]:
            pa, pb = preds[l][d.id]
            for k, (sa, ea, sb, eb) in enumerate(paper_chunks(len(d.tokens_a), len(d.tokens_b))):
                if status.get((d.id, k)) != "ok":
                    continue
                for x, y in zip(d.labels_a[sa:ea], pa[sa:ea]):
                    if x != -1: g.append(float(x)); p.append(float(y))
                for x, y in zip(d.labels_b[sb:eb], pb[sb:eb]):
                    if x != -1: g.append(float(x)); p.append(float(y))
        out[l] = (pooled_spearman(g, p), len(g))
    return out
tot = {l: sum(1 for d in docs[l] for x in d.labels_a + d.labels_b if x != -1) for l in LANGS}
print("\n=== only tokens inside chunks that answered (status ok)")
for name, p in (("b0s", b0s), ("D4-contrastive", d4c), ("Fuse", fuse), ("CTFAlign", ctf)):
    r = subset_scores(p)
    print("  %-15s %s | %.3f   (tokens %s)" % (name, " ".join("%.3f" % r[l][0] for l in LANGS), sum(r[l][0] for l in LANGS) / 3,
          " ".join("%d/%d=%.0f%%" % (r[l][1], tot[l], 100 * r[l][1] / tot[l]) for l in LANGS)))

# rates
print("\n=== b0s predicted-difference rate (score>0) vs gold rate, and doc statuses")
rows = [json.loads(l) for l in open(RUN / "results_b0s.jsonl")]
print("  doc statuses:", dict(Counter(r["status"] for r in rows)))
print("  calls %d, prompt tokens/doc %.0f, completion tokens/doc %.0f, latency/doc %.1fs, total wall ~ %.0f min at 2 workers" % (
    sum(r["n_calls"] for r in rows), sum(r["prompt_tokens"] for r in rows) / len(rows), sum(r["completion_tokens"] for r in rows) / len(rows),
    sum(r["latency_s"] for r in rows) / len(rows), sum(r["latency_s"] for r in rows) / 2 / 60))

# quality of answered chunks: how many of their tokens got a label from the answer (vs. parser fallback)
from uzh_diff.methods import b0_token_labels
matched = Counter(); total = Counter(); lab = Counter()
for c in calls:
    _, doc_id, k = c["tag"].split(":")
    if status.get((doc_id, int(k))) != "ok": continue
    d = by_id[doc_id]; sa, ea, sb, eb = paper_chunks(len(d.tokens_a), len(d.tokens_b))[int(k)]
    data = _b0_load(c["result"]["content"])
    for key, toks in (("sentence1", d.tokens_a[sa:ea]), ("sentence2", d.tokens_b[sb:eb])):
        if key in data and toks:
            sims = b0_token_labels(toks, data[key], fallback=-99)
            total[d.lang] += len(toks); matched[d.lang] += sum(1 for v in sims if v != -99)
            lab[d.lang] += sum(1 for v in sims if v != -99 and v < 5)
print("\n=== inside answered chunks: tokens that received a label from the answer, and of those labelled 'different' (<5)")
for l in LANGS:
    print("  %s: matched %d/%d = %.0f%%, of matched labelled different %.0f%%" % (l, matched[l], total[l], 100*matched[l]/total[l], 100*lab[l]/max(1,matched[l])))
