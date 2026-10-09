"""Offline tests of the HTML viewer (viewer.py): payload, pooled scores, self-contained output."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from uzh_diff.data import LANGS, load_docs  # noqa: E402
from uzh_diff.scorer import score  # noqa: E402
from uzh_diff.viewer import build_payload, build_viewer, doc_rho, page_title, page_url, render_html  # noqa: E402


def write_predictions(tmp: Path, system: str, n_docs: int = 3):
    """Prediction files for the first n dev documents per language: gold-correlated scores with a twist."""
    preds = {}
    for lang in LANGS:
        docs = load_docs(lang, "dev")[:n_docs]
        rows = []
        preds[lang] = {}
        for k, d in enumerate(docs):
            la = [0.0 if x == -1 else min(1.0, 0.3 * float(x) + 0.01 * (i % 7)) for i, x in enumerate(d.labels_a)]
            lb = [0.0 if x == -1 else min(1.0, 0.3 * float(x) + 0.01 * (i % 5)) for i, x in enumerate(d.labels_b)]
            if k == 0:
                la = [0.0] * len(la)   # a document with a constant prediction on one side
            rows.append({"id": d.id, "text_a": d.text_a, "text_b": d.text_b, "labels_a": la, "labels_b": lb})
            preds[lang][d.id] = (la, lb)
        with open(tmp / ("%s_admin_%s.jsonl.jsonl" % (system, lang)), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return preds


class PageNameTest(unittest.TestCase):
    def test_url_and_title(self):
        name = "https___www.bfe.admin.ch_bfe_en_home_policy_international-energy-policy.html"
        self.assertEqual(page_url(name), "https://www.bfe.admin.ch/bfe/en/home/policy/international-energy-policy.html")
        self.assertEqual(page_title(name), "international energy policy")
        self.assertEqual(page_url(""), "")
        self.assertEqual(page_title(""), "")


class ViewerTest(unittest.TestCase):
    def test_payload_scores_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            preds = write_predictions(tmp, "toy")
            payload = build_payload(tmp, "toy", allow_test=False, system_label="Toy")
            # only the predicted documents, dev only, in id order
            for lang in LANGS:
                entries = payload["langs"][lang]["docs"]
                self.assertEqual([e["id"] for e in entries], list(preds[lang]))
                self.assertTrue(all(e["split"] in ("dev/train", "dev/val") for e in entries))
                for e in entries:
                    self.assertEqual(len(e["sys"]["Toy"]["a"]), len(e["ta"].split(" ")))
                    self.assertEqual(len(e["gb"]), len(e["tb"].split(" ")))
                    self.assertTrue(all(-1 <= v <= 100 for v in e["ga"] + e["gb"]))
                    self.assertTrue(all(0 <= v <= 100 for v in e["sys"]["Toy"]["a"] + e["sys"]["Toy"]["b"]))
                # the pooled score of the viewer equals the project scorer on the same documents
                docs = [d for d in load_docs(lang, "dev") if d.id in preds[lang]]
                expected = score({lang: docs}, {lang: preds[lang]}).per_lang[lang].spearman
                self.assertAlmostEqual(payload["summary"][lang]["dev"]["Toy"], expected, places=4)
                self.assertEqual(payload["summary"][lang]["dev"]["n"], 3)
                self.assertNotIn("test", payload["summary"][lang])
                # a side with a constant prediction has no per-side rho but may still have a pooled one
                first = entries[0]["st"]["Toy"]
                self.assertIsNone(first["rho_a"])
            # the reference system is attached when its files exist (dev files ship with the repository)
            self.assertIn("CTFAlign", payload["systems"])
            html = render_html(payload)
            # the embedded JSON survives the round trip (a "</" inside a text cannot close the script element)
            block = html.split('<script id="rsd-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            back = json.loads(block.replace("<\\/", "</"))
            self.assertEqual([e["id"] for e in back["langs"]["de"]["docs"]], list(preds["de"]))
            self.assertIn("admin_de_0", html)
            self.assertNotIn("http://", html.split("<body>")[0])   # no external resources in the head
            self.assertNotIn("<link", html)
            out = tmp / "v.html"
            info = build_viewer(tmp, "toy", out, allow_test=False, system_label="Toy")
            self.assertTrue(out.exists() and out.stat().st_size > 10000)
            self.assertEqual(info["docs"], {"de": 3, "fr": 3, "it": 3})

    def test_doc_rho(self):
        from uzh_diff.data import DocPair
        d = DocPair("x", "de", "a b c d".split(), "e f g".split(), [0, 1, 0, -1], [0, 0, 1])
        r = doc_rho(d, ([0.1, 0.9, 0.2, 0.0], [0.0, 0.1, 0.8]))
        self.assertIsNotNone(r["rho"])
        self.assertAlmostEqual(r["rho"], 0.8402, places=4)   # ranks: four tied zeros vs 0.0 < 0.1 = 0.1 < 0.2 < 0.8 < 0.9
        self.assertAlmostEqual(r["rho_a"], 0.866, places=3)   # tied gold zeros
        self.assertAlmostEqual(r["rho_b"], 0.866, places=3)
        r2 = doc_rho(d, ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
        self.assertIsNone(r2["rho"])

    def test_missing_prediction_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_payload(Path(tmp), "nothing", allow_test=False)


if __name__ == "__main__":
    unittest.main()
