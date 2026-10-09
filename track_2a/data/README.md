# Data

## What is here

| Path | Content |
|---|---|
| `gold/dev/gold_admin_{de,fr,it}.jsonl` | SwissGov-RSD development split, 168 document pairs per language, with gold token labels. Unchanged copies of `data/evaluation/gold_labels/dev/gold_admin_{de,fr,it}.jsonl`. |
| `splits/dev_train_ids.json`, `splits/dev_val_ids.json` | Document ids of the two development sub-splits (134 and 34 per language), taken from the file lists under `data/evaluation/gold_labels/dev/train` and `dev/val`. Only the ids are used. |
| `gold/test/gold_admin_{de,fr,it}.jsonl` | The held-out test split, 56 pairs per language, unchanged copies of `data/evaluation/gold_labels/test/`. Copied into the project only after the method was frozen (7 Oct 2026); present so that `make run` (default `SPLIT=test`) works on a clean checkout. |
| `manifests/smoke12.json`, `manifests/dev60.json` | The 12 development documents of the end-to-end trial run; the 60 development documents (20 pages × 3 languages, from dev/train) on which the output-layer parameters were chosen. |
| `predictions/full/` | The submitted prediction files (224 pairs per language, full-set order); `predictions/README.md` has hashes and scores. |
| `reference/` | Published predictions of other systems (CTFAlign, mmBERT-SimCSE) on the same documents, comparison rows only. |
| `baselines/b0s/test/` | The paper's prompting baseline re-run with Apertus-8B on the test split. |
| `frozen_runs/` | Per-document results of the frozen runs behind the submission and the report (m2, d4, fuse on dev and test; b0s on dev60 and test), gzipped, with manifests, summaries, audits and bootstrap outputs. `tools/report_tables.py`, `run.py stats` and `run.py audit` read them, so every number in the report can be recomputed without the endpoint. Raw request logs (`calls.jsonl`) are not included (size). |

Source: <https://github.com/ZurichNLP/SwissGov-RSD>, branch `master`, commit `1807a42100e742ed03d337c54c4b9ea86995f565` (2026-07-30).
Dataset licence: CC-BY-4.0 (Hugging Face dataset card `ZurichNLP/SwissGov-RSD`).
Citation: Wastl, Vamvas, Sennrich. SwissGov-RSD. ACL 2026. <https://aclanthology.org/2026.acl-long.1437/>

## The test split

The challenge rules forbid using the held-out test split during development in any form. During development it was
not in the project; `uzh_diff.data.load_docs(..., "test")` still refuses to read it unless `allow_test=True` is passed
(`--allow-test` on the command line; the Docker entry point passes it for `SPLIT=test|full`). The files were added
after the configuration was frozen, for the single final run and for the judges' reproduction.

## Why the sub-splits are stored as id lists

In the source repository at the commit above, the German files under `dev/train` and `dev/val`
still carry older labels: 89 of the 168 German documents (8,508 tokens) differ from
`dev/gold_admin_de.jsonl`, which is identical to `full/gold_admin_de.jsonl` and is what the official
evaluation script reads for `--split dev`. French and Italian are identical everywhere.
To stay on the labels the official script uses, this project reads labels only from
`gold/dev/gold_admin_*.jsonl` and uses the sub-split files only to know which ids belong where.

## Format

One JSON object per line: `id`, `text_a` (English), `text_b` (German / French / Italian),
`labels_a`, `labels_b`, plus `page_en`, `page_other`, `subset`. Texts are whitespace-tokenised single
lines; a label is 0 (no difference), 0.2 to 1.0 (degree of difference) or -1 (punctuation, not scored).
