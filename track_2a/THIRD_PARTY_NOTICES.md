# Third-party material

The code in this repository is licensed under the Apache License 2.0 (`LICENSE` at the repository root); the
technical report (`technical_report.md`, `Tse-min_Report.pdf`) and the method description (`method_description.md`,
`Tse-min_Method.pdf`) under Creative Commons Attribution 4.0. The material below was not created by this project and
keeps its own terms. It is included unchanged, only to run and evaluate the system and to compare it with published
results.

| Path | What it is | Source | Terms |
|---|---|---|---|
| `data/gold/` (dev, test); the same texts inside `data/reference/`, `data/predictions/`, `data/baselines/`, `data/frozen_runs/` and `demo/viewer.html` | SwissGov-RSD: English, German, French and Italian pages of Swiss federal websites with word-level difference labels | Wastl, Vamvas and Sennrich, *SwissGov-RSD*, ACL 2026; <https://github.com/ZurichNLP/SwissGov-RSD>, commit `1807a42100e742ed03d337c54c4b9ea86995f565`; <https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD> | Creative Commons Attribution 4.0 (dataset card). Files under `data/gold/` are unchanged copies; the other files pair the same texts with system scores. |
| `data/reference/ctfalign_*.jsonl.jsonl`, `data/reference/test/ctfalign_*.jsonl.jsonl`; the CTFAlign scores shown in `demo/viewer.html` | Published predictions of DiffAlign + CTFAlign (Qwen3-Embedding-4B, layer 20) | Wastl, Vamvas and Sennrich, *Scaling Unsupervised Word Alignment to Documents via Structural Constraints*, arXiv:2608.21023; <https://github.com/ZurichNLP/document-level-word-alignment>, commit `60ad0a32` | MIT License, Copyright (c) 2026 Michelle Wastl; full text below. |
| `data/reference/mmbert_simcse_parallel_all_admin_*.jsonl.jsonl` | Published development-split predictions of the benchmark's best encoder (mmBERT trained with SimCSE) | SwissGov-RSD repository above, `data/evaluation/encoder_predictions/dev/` | The source repository does not state a licence. Included unchanged as a comparison row; not covered by this repository's licence; all rights remain with the authors. Removed on request. |
| `prompts/b0_paper/template_{de,fr,it}_admin.txt` | The few-shot prompt of the SwissGov-RSD paper | SwissGov-RSD repository above, `prompt_templates/` | The source repository does not state a licence. Included unchanged only to reproduce the paper's baseline with Apertus, as the challenge description asks; not covered by this repository's licence; all rights remain with the authors. Removed on request. |

The Apertus model (`swiss-ai/Apertus-v1.5-8B`) is not distributed here; it is called through the CSCS inference
endpoint. The model outputs stored in `data/` were produced by this project.

## MIT License (document-level-word-alignment)

```
MIT License

Copyright (c) 2026 Michelle Wastl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
