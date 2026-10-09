"""Loading SwissGov-RSD document pairs.

Only the development split lives in this repository. The held-out test split must not enter
development in any form (challenge rule), so nothing in this module can read it unless the caller
passes allow_test=True, which only the frozen final run does.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config

LANGS = ("de", "fr", "it")
LANG_NAMES = {"de": "German", "fr": "French", "it": "Italian"}
DEV_SPLITS = ("dev", "dev/train", "dev/val")
FULL_SPLIT = "full"      # dev + test in the order of the official full-set files (numeric id order)


def numeric_id(doc_id: str) -> int:
    return int(doc_id.rsplit("_", 1)[1])


@dataclass
class DocPair:
    id: str
    lang: str
    tokens_a: List[str]          # English
    tokens_b: List[str]          # German / French / Italian
    labels_a: Optional[List[float]] = None   # gold difference labels; -1 = punctuation (not scored)
    labels_b: Optional[List[float]] = None

    @property
    def text_a(self) -> str:
        return " ".join(self.tokens_a)

    @property
    def text_b(self) -> str:
        return " ".join(self.tokens_b)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens_a) + len(self.tokens_b)


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_doc(row: dict, lang: str, with_labels) -> DocPair:
    tokens_a, tokens_b = row["text_a"].split(), row["text_b"].split()
    labels_a = labels_b = None
    if with_labels == "auto":
        with_labels = "labels_a" in row and "labels_b" in row
    if with_labels:
        labels_a, labels_b = list(row["labels_a"]), list(row["labels_b"])
        if len(labels_a) != len(tokens_a) or len(labels_b) != len(tokens_b):
            raise ValueError("%s: label length does not match token count" % row["id"])
    return DocPair(row["id"], lang, tokens_a, tokens_b, labels_a, labels_b)


def split_ids(lang: str, split: str) -> Optional[List[str]]:
    """IDs of a dev sub-split, or None for the whole dev split."""
    if split == "dev":
        return None
    name = {"dev/train": "dev_train_ids.json", "dev/val": "dev_val_ids.json"}[split]
    return json.loads((config.SPLITS / name).read_text(encoding="utf-8"))[lang]


def load_docs(lang: str, split: str = "dev", with_labels="auto",
              allow_test: bool = False, gold_dir: Optional[Path] = None) -> List[DocPair]:
    """Document pairs of one language, in the order of the official file.

    with_labels: True (require gold labels), False (ignore them) or "auto" (take them when the file has them,
    so that inference also works on a file that only carries text_a / text_b)."""
    if lang not in LANGS:
        raise ValueError("unknown language %r" % lang)
    if split == "test":
        if not allow_test:
            raise PermissionError("the test split is held out: it is only read by the frozen final run")
        path = (gold_dir or config.DATA / "gold" / "test") / ("gold_admin_%s.jsonl" % lang)
        rows = read_jsonl(path)
    elif split == FULL_SPLIT:
        # the organisers' full-set order: every dev and test document, by numeric id (checked against the
        # repository's data/evaluation/gold_labels/full files: identical order, texts and labels)
        if not allow_test:
            raise PermissionError("the full split contains the test documents: only for the frozen final run")
        if gold_dir is not None:
            rows = read_jsonl(gold_dir / ("gold_admin_%s.jsonl" % lang))
        else:
            rows = (read_jsonl(config.GOLD_DEV / ("gold_admin_%s.jsonl" % lang))
                    + read_jsonl(config.DATA / "gold" / "test" / ("gold_admin_%s.jsonl" % lang)))
        rows = sorted(rows, key=lambda r: numeric_id(r["id"]))
    elif split in DEV_SPLITS:
        rows = read_jsonl((gold_dir or config.GOLD_DEV) / ("gold_admin_%s.jsonl" % lang))
        wanted = split_ids(lang, split)
        if wanted is not None:
            wanted_set = set(wanted)
            rows = [r for r in rows if r["id"] in wanted_set]
            if len(rows) != len(wanted_set):
                raise ValueError("split %s/%s: expected %d documents, found %d" % (split, lang, len(wanted_set), len(rows)))
    else:
        raise ValueError("unknown split %r" % split)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate document ids in %s/%s" % (split, lang))
    return [_to_doc(r, lang, with_labels) for r in rows]


def load_manifest(name: str) -> Dict[str, List[str]]:
    """A named list of document ids per language (data/manifests/<name>.json)."""
    return json.loads((config.MANIFESTS / (name + ".json")).read_text(encoding="utf-8"))["ids"]


def select(docs: Sequence[DocPair], ids: Sequence[str]) -> List[DocPair]:
    by_id = {d.id: d for d in docs}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError("documents not found: %s" % missing)
    return [by_id[i] for i in ids]
