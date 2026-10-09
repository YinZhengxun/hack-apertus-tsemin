# Tse-min — Cross-lingual semantic differences in Swiss government pages with Apertus (Track 2A / UZH)

**What it does.** For an English page and its German / French / Italian version, score every word by how likely
its meaning is missing or different on the other side (SwissGov-RSD). Two Apertus-8B signals, no training:
*m2* asks the model block by block whether a block has a counterpart (generative judgement); *d4* makes the model
read one document while looking at the other and takes each word's remaining surprisal from `prompt_logprobs`
(no generation beyond one token). `fuse` keeps m2's "no counterpart" blocks only when d4 agrees and orders every
other word by d4. Test split: Spearman **0.323** (de .412 / fr .189 / it .367) with the official script, vs 0.291 for
the organisers' CTFAlign on the same documents and −0.015 for the paper's prompting baseline run with the same
Apertus-8B (`data/baselines/`). Details: `technical_report.md` (PDF: `Tse-min_Report.pdf`), `docs/USAGE.md`.

**Run it** (Docker; the three Hack Apertus endpoint variables; `track_2a/` is the project root as defined by the template, a `Makefile` at the repository root forwards to it):

```bash
cd track_2a
export LLM_API_KEY=...                      # LLM_BASE_URL and LLM_NAME default to the CSCS endpoint and Apertus-v1.5-8B
make run                                    # test split: ~2,750 requests, 15-25 min at 2 parallel documents -> output/test/
make run SPLIT=full                         # all 224 pages per language in the organisers' full-set order (~1.5 h)
make run SPLIT=smoke                        # 12 development documents, ~3 min, scored
make selftest                               # 95 offline tests, no endpoint
```

The run stops at the first failing step and copies nothing in that case; step 2 is a capability gate (the endpoint must return
`prompt_logprobs` for a prefilled assistant message, which d4 needs; otherwise exit code 4); an inference step that ends with failed
requests (exit code 5: the endpoint answered 5xx for some documents even after retries) is repeated up to three times, re-sending only those. Everything the run produces (raw requests, cache,
per-document results) stays in `runs/`; the prediction files, the run summary and the acceptance audit are copied to `output/<split>/`. The submitted predictions are in
`data/predictions/full/` (224 documents per language in the organisers' full-set order, from which their script
scores the test split; `data/predictions/README.md` has the hashes and the scores). Requirements: Docker; access to the Apertus endpoint; no GPU, no model weights, no packages
beyond the Python standard library.

**Where things are** (`src/uzh_diff/`):

| Component | File | Role |
|---|---|---|
| Data | `data.py` | reads the official files; refuses the test split unless `--allow-test` is passed |
| D4 | `surprisal.py` | four read-back requests per pair (`prompt_logprobs` on a prefilled assistant message); maps tokens back to whitespace words |
| M2 | `blocks.py`, `methods.py` | cuts each side into sentence-like blocks (≤ 60 words) and asks for one judgement per block |
| Fuse | `fuse.py` | the output layer: D4 ordering, optional verified-absent tier |
| Client | `client.py` | retries (back-off on 429/5xx, a 504 only once), on-disk cache keyed by the request, every request and response logged to `calls.jsonl` |
| Evaluation | `scorer.py`, `stats.py`, `audit.py` | official scoring definition, paired page bootstrap, acceptance audit (fatal problems vs declared warnings) |
| Delivery | `cli.py`, `viewer.py`, `src/entrypoint.sh` | `run`, `fuse`, `assemble` (full-set file order), `audit`, `stats`, `viewer`; the `make run` entry point |
| Report | `tools/report_tables.py`, `tools/report_extra.py`, `tools/build_report_pdf.py` | recompute every number of the report from `data/frozen_runs/`; build `Tse-min_Report.pdf` |

**Look at the predictions:** `demo/viewer.html` (one offline file) shows every document pair side by side with the words shaded
by the submitted score, the gold differences underlined and CTFAlign as a switchable comparison; `python run.py viewer --allow-test`
rebuilds it from `data/predictions/full/`. `demo/README.md` explains the colours.

---

# Academia Challenges

Submissions must use the Apertus model family.
For Track 2 this means that submitted solutions must be built with Apertus. Other open-weights models can be used to support development, e.g. as automatic judges during evaluation. Their role must be clearly described in the submission report.

💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”

## How it works
Pick from 5 academia challenges provided by Swiss academic institutions:

- **FHGR:** AI-Powered Job Interview Coach
- **OpenParlData:** Extracting Parliamentary Affairs from PDFs into One Common Structure
- **OST:** Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **UZH:** Detecting Cross-Lingual Semantic Differences in Swiss Government Websites
- **ZHAW:** See It, Say It, Pick It: Vision-Language Grounding for a Real Robot Arm

The challenges incl. submission and judging criteria are described in our **Getting Started guide**:
https://hackapertus.notion.site/getting-started-guide-onlinehack

## Run it

Keep `track_2a/` as it is: don't rename it or move its files, just delete the
other track directories.

From the root of the project:

```bash
make run
```

Fill in the [Makefile](Makefile) so that it works on a clean checkout. It is
expected to run the project in a Docker container, since that is how the judges
will run it, without relying on anything already installed on your machine.

Requirements: `runtime, hardware, API keys, model weights`

## Data
The `data/` directory must not exceed 100 MB.


## 📦 Submission Requirements & Deliverables
❗️ Submissions are not handled on Devpost. Submit through our website only:
http://hackapertus.ch/online-hack/submissions

Requirements differ by challenge. See the description of the challenge you are entering for the exact deliverables.


## ⚖️ Judging Criteria
Judging criteria also differ by challenge. See the respective challenge description.


## Support

**Licensing requirements**
Please check our Terms & Conditions (6. What you build is open source):
https://hackapertus.ch/terms-and-conditions

## FAQ
💡 https://hackapertus.ch/faq

## Contact
💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”
