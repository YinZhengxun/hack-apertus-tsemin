# Offline viewer

`viewer.html` is one self-contained page (no network, no external scripts): every document pair of the full set
(224 per language) side by side, words shaded by the submitted Fuse score, the annotators' gold differences
underlined, the organisers' CTFAlign predictions as a second shading for comparison, Spearman per document and
pooled per split. Open it in any browser.

Controls: language, dev / test filter, sorting (by id, best or worst per-document ρ, most gold differences),
which system to shade, view mode (prediction + gold, prediction only, gold only, agreement at a threshold),
←/→ to move between documents, hover a word for its scores and gold label.

Colours: amber = score below 0.5, ordered by the probability signal (D4-contrastive); red = words of a block that
m2 judged "no counterpart" and d4 verified (score ≥ 0.5, the top tier of the submission); blue underline = gold
difference (darker = more annotators); grey = punctuation, not scored (gold −1).

Regenerate from the prediction files (nothing is computed by a model):

```bash
python run.py viewer --allow-test                       # data/predictions/full -> demo/viewer.html
python run.py viewer --pred-dir runs/<fuse>/predictions/fuse --system tsemin --out demo/dev.html   # a dev-only run
```

Per-document Spearman is descriptive only: the official metric pools all words of a language.
