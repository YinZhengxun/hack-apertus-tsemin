# Tse-min — Hack Apertus Track 2A (UZH): word-level semantic differences with Apertus read-back probabilities

Team **Tse-min** (Zhengxun Yin, University of Zurich). For an English page of a Swiss federal website and its German,
French or Italian version, the system scores every word by how likely its meaning is missing or different on the other
side (SwissGov-RSD). Apertus-v1.5-8B reads each document with and without the other page in the prompt; words that stay
surprising are the suspects. No training; the core signal needs 4 requests per document pair.

**Test split (official script):** Spearman **0.323** (de .412 / fr .189 / it .367); the organisers' CTFAlign alignment
system 0.291; the paper's prompting baseline with the same Apertus-8B −0.015.

| | |
|---|---|
| Project root (code, data, predictions) | [`track_2a/`](track_2a/) — start with its [README](track_2a/README.md) |
| Method description (2 pages) | [`track_2a/Tse-min_Method.pdf`](track_2a/Tse-min_Method.pdf) |
| Technical report (6 pages) | [`track_2a/Tse-min_Report.pdf`](track_2a/Tse-min_Report.pdf) |
| Demo (opens in the browser) | <https://yinzhengxun.github.io/hack-apertus-tsemin/track_2a/demo/viewer.html> |
| Submitted predictions | [`track_2a/data/predictions/full/`](track_2a/data/predictions/full/) |
| Run | `export LLM_API_KEY=...`, then `make run` here (forwards to `track_2a/`; needs Docker) |

Licences: code Apache-2.0 ([`LICENSE`](LICENSE)); report and method description CC-BY-4.0; third-party data under their
own terms ([`track_2a/THIRD_PARTY_NOTICES.md`](track_2a/THIRD_PARTY_NOTICES.md)).

---

*The rest of this file is the description that came with the Hack Apertus project template.*

# Hack Apertus — project template

Template repository for [Hack Apertus](https://hackapertus.ch/) submissions.
Every project keeps almost the same layout, so organizers and judges find the
same things in the same place.

## Select your track

This repository holds one example project per track:

- `track_1a/`
- `track_1b/`
- `track_2a/`
- `track_2b/`

Keep the directory for the track you are competing in **exactly as it is** —
don't rename it or move its files — and delete the other track directories.
That directory is your project root. Keep the files and directories as shown
below.

## The structure

| Path | What it is |
| --- | --- |
| `README.md` | The challenge description and submission requirements for your track |
| `technical_report.md` | The deeper write-up: architecture, evaluation, limitations |
| `Makefile` | `make run` must spin up your project |
| `src/` | Your code |
| `data/` | Datasets — `track_1a`, `track_2a` and `track_2b` only; max. 100 MB |
| `findings/` | Issue files — `track_1a` only |
| `docs/` | Diagrams, notes, longer write-ups |

## Run it

Judges run `make run` from the root of the project, on a clean checkout:

```bash
make run
```

`make run` is expected to run the project using Docker, since that
is how the judges will run it.
- If you used other local open-weight models, include instructions for running the project in your technical report.
- Use the following environment variables:
```
LLM_NAME — name and version of the model
LLM_BASE_URL — endpoint base URL
LLM_API_KEY — your API key
```

## Getting started

1. Click **Use this template** to create your own repository.
2. Delete the other track directories. Don't rename or restructure yours.
3. Read its `README.md` and fill in `technical_report.md`.
4. Make `make run` work from the root of the project, on a clean checkout.

## License

All Hack Apertus projects are open-sourced. Please check our Terms & Conditions for specific licensing details (6. What you build is open source): https://hackapertus.ch/terms-and-conditions
