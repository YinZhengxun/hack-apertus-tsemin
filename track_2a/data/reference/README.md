# Reference predictions (not produced by this project)

Published development-split predictions of other systems, scored here on the same documents as our
runs so that comparisons are paired. None of these systems is part of our pipeline; none uses Apertus.

| Files | System | Source | dev (168/lang) Spearman de / fr / it, mean |
|---|---|---|---|
| `ctfalign_qwen3emb4b_l20_admin_{de,fr,it}.jsonl.jsonl` | DiffAlign + CTFAlign (Argmax), Qwen3-Embedding-4B layer 20 — the organisers' follow-up method, the strongest published result | `ZurichNLP/document-level-word-alignment`, commit `60ad0a32`, `swissgov_rsd/data/evaluation/encoder_predictions/dev/DiffAlignCTFAlign(model=Qwen_Qwen3-Embedding-4B, layer=20, width=8)_admin_*.jsonl.jsonl` (Wastl, Vamvas, Sennrich, arXiv 2608.21023) | 0.346 / 0.208 / 0.365, 0.306 |
| `mmbert_simcse_parallel_all_admin_{de,fr,it}.jsonl.jsonl` | DiffAlign with a SimCSE-trained mmBERT-base (`parallel_all_1epoch`), the best of the 20 systems in the benchmark repository | `ZurichNLP/SwissGov-RSD`, commit `1807a42`, `data/evaluation/encoder_predictions/dev/` | 0.244 / 0.154 / 0.266, 0.221 |

Scores above were recomputed with the official evaluation script (`--split dev`) and with this project's
scorer; both agree. Only development documents are included (the source directories hold no test data).
Texts are the CC-BY-4.0 SwissGov-RSD texts; the label columns are the respective systems' outputs.

## Test-split reference (added 2026-10-07, after our configuration was frozen and our single test run was done)

`test/ctfalign_qwen3emb4b_l20_admin_{de,fr,it}.jsonl.jsonl` — the same CTFAlign system's published **test** predictions
(`swissgov_rsd/data/evaluation/encoder_predictions/test/DiffAlignCTFAlign(model=Qwen_Qwen3-Embedding-4B, layer=20, width=8)_admin_*.jsonl.jsonl`,
same commit). Scored with this project's scorer on the test split: 0.330 / 0.189 / 0.353, mean 0.291 — identical to the
number reported in the paper. They are used only for the paired comparison in the report; they were fetched after the
test run, never during development.
