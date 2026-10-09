# Baselines run by this project (not submitted)

## `b0s/test/` — the paper's prompting baseline, reproduced with Apertus-v1.5-8B on the test split

What it is: the SwissGov-RSD paper's prompt (`prompt_templates/template_{lang}_admin.txt` of the dataset
repository, unchanged), the repository's 250-word chunking (`create_test_data_admin.py`), `response_format=json_object`
as in `predict_openai.py`; temperature 0, seed 0; one call per chunk pair; our client, parser and retries
(`src/uzh_diff/methods.py`, `run_b0s`). One run after the method freeze, 2026-10-08 14:01–15:45 CEST, run id
`20261008-140115-b0s-test` (the call log stays in `runs/`, 11 MB).

| | de | fr | it | mean |
|---|---|---|---|---|
| Spearman, official script (`--split test`) | −0.019 | 0.004 | −0.030 | **−0.015** |
| … restricted to the chunks that produced a readable answer (42 / 47 / 56 % of the scored tokens) | −0.033 | 0.004 | −0.005 | −0.011 |
| Fuse on exactly those tokens | 0.390 | 0.235 | 0.344 | 0.323 |

Failure profile (691 chunk calls): 322 (47 %) hit the output limit (`finish_reason = length`), 206 of them with a
repetition loop in the tail; 11 returned no readable JSON; 358 answered. Inside the answered chunks only 45–48 % of
the tokens received a label at all (the model skips or garbles tokens), and 78–83 % of the labelled tokens were
marked "different" — the output does not separate the 9–16 % gold-difference words from the rest. Cost: 6,994 prompt
and 8,913 completion tokens per document, 54 s per document (2 parallel documents, ~76 min for the split).

Files: `b0s_admin_{de,fr,it}.jsonl.jsonl` (official five-field format, 56 rows each, test-split order),
`summary_b0s.txt` (the run's score table and per-document status), `manifest.json`, `diagnostic.txt`
(output of `tools/b0s_chunk_diagnostic.py`).

On the 60-document development subset (dev60, 2026-10-06) the same baseline scored −0.011 as run and 0.056 on the
answered chunks, so the picture is the same on both splits.
