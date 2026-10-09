# Submitted predictions

The organisers score the **test** split from prediction files in the **full-set order** (all 224 documents per
language, `data/evaluation/gold_labels/full` order = numeric id order; their script filters the test documents out
of such a file). So the files to submit are:

`full/tsemin_admin_{de,fr,it}.jsonl.jsonl` — 224 rows per language, five-field JSONL, values in [0, 1] (only their
order matters to the metric). Assembled with `python run.py assemble` from the frozen system's dev run
(`runs/20261007-113912-fuse-verified_d4-dev`) and test run (`runs/20261007-122413-fuse-verified_d4-test`):
rule verified+d4, quantile 0.7, verify_coef 0.5, rank_coef 0.75, window 4, model swiss-ai/Apertus-v1.5-8B.
Official script on these files: `--split test` → Spearman de 0.412 / fr 0.189 / it 0.367 (mean 0.323);
`--split dev` → 0.404 / 0.227 / 0.389 (mean 0.340). sha256 (first 16 hex): de `b7e48e339f202c9a`,
fr `d3a8b96df69a3c59`, it `514cedff5b8b20f6`.

`test/` — the same test predictions as a 56-row file per language (the output of the test run itself), with the
acceptance audit (`audit.txt`, hashes inside) and the run summary.

`make run` regenerates the test predictions from scratch on the endpoint (`make run SPLIT=full` all 224 pages). The
hosted model is served with temperature 0, but batching on the server can move a few scores; the ranking metric is
robust to that.
