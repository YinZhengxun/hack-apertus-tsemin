"""Offline tests of the output layer (fuse.py) and the `fuse` command."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from uzh_diff import config  # noqa: E402
from uzh_diff.blocks import make_blocks  # noqa: E402
from uzh_diff.data import DocPair, load_docs  # noqa: E402
from uzh_diff.fuse import FuseParams, block_statuses, d4_signal, fuse, moving_mean, verified_absent  # noqa: E402


def toy_doc():
    a = ("The office is open on Monday . Fees are 10 francs per year . This page is only in German . "
         "Contact us by e-mail .").split()
    b = ("Das Büro ist montags geöffnet . Die Gebühr beträgt 12 Franken pro Jahr . Kontakt per E-Mail .").split()
    return DocPair("toy_de_1", "de", a, b, [0] * len(a), [0] * len(b))


def m2_details(doc, statuses):
    """statuses: {block_id: status}; blocks not listed are 'covered'."""
    blocks = []
    for side, toks in (("a", doc.tokens_a), ("b", doc.tokens_b)):
        for b in make_blocks(toks, side):
            blocks.append({"id": b.id, "status": statuses.get(b.id, "covered"), "n_quotes": 0})
    return {"blocks": blocks}


def d4_details(doc, s_c_a, s_c_b, s_u_a=None, s_u_b=None):
    d = {"s_c_a": s_c_a, "s_c_b": s_c_b}
    d["s_u_a"] = s_u_a if s_u_a is not None else [0.0] * len(s_c_a)
    d["s_u_b"] = s_u_b if s_u_b is not None else [0.0] * len(s_c_b)
    return d


class SignalTest(unittest.TestCase):
    def test_moving_mean(self):
        self.assertEqual(moving_mean([1, 2, 3, 4], 0), [1, 2, 3, 4])
        self.assertEqual(moving_mean([0, 0, 3, 0, 0], 1), [0, 1, 1, 1, 0])

    def test_signal_discounts_unconditioned_surprisal(self):
        det = {"s_c_a": [4.0, 1.0], "s_u_a": [4.0, 0.0]}
        self.assertEqual(d4_signal(det, "a", 1.0), [0.0, 1.0])
        self.assertEqual(d4_signal(det, "a", 0.0), [4.0, 1.0])
        self.assertIsNone(d4_signal(det, "b", 0.5))
        self.assertIsNone(d4_signal({"s_c_a": [1.0]}, "a", 0.5))   # needs s_u when coef > 0


class FuseRuleTest(unittest.TestCase):
    def setUp(self):
        self.doc = toy_doc()
        blocks_a = make_blocks(self.doc.tokens_a, "a")
        self.assertGreaterEqual(len(blocks_a), 3)
        # third English sentence ("This page is only in German .") has no counterpart; also mark the
        # first one absent although it is well covered, to see verification drop it
        self.absent_true = blocks_a[2]
        self.absent_false = blocks_a[0]
        self.m2 = m2_details(self.doc, {self.absent_true.id: "absent", self.absent_false.id: "absent", blocks_a[1].id: "partly"})
        n_a, n_b = len(self.doc.tokens_a), len(self.doc.tokens_b)
        s_c_a = [0.1] * n_a
        for i in range(self.absent_true.start, self.absent_true.end):
            s_c_a[i] = 5.0               # surprising even after reading the German text
        s_c_b = [0.2] * n_b
        s_c_b[6] = 6.0                   # "12" differs from "10"
        self.d4 = d4_details(self.doc, s_c_a, s_c_b)

    def test_absent_rule_marks_every_absent_block(self):
        (la, lb), info = fuse(self.doc, self.m2, None, FuseParams(rule="absent"))
        marked = {i for i, v in enumerate(la) if v == 1.0}
        expected = set(range(self.absent_true.start, self.absent_true.end)) | set(range(self.absent_false.start, self.absent_false.end))
        self.assertEqual(marked, expected)
        self.assertEqual(set(lb), {0.0})
        self.assertEqual(info["d4_sides"], [])

    def test_verification_drops_the_well_covered_block(self):
        keep = verified_absent(self.doc, self.m2, self.d4, FuseParams())
        self.assertTrue(all(keep["a"][i] for i in range(self.absent_true.start, self.absent_true.end)))
        self.assertFalse(any(keep["a"][i] for i in range(self.absent_false.start, self.absent_false.end)))
        (la, lb), _ = fuse(self.doc, self.m2, self.d4, FuseParams(rule="verified"))
        self.assertEqual({i for i, v in enumerate(la) if v == 1.0}, set(range(self.absent_true.start, self.absent_true.end)))

    def test_verified_plus_d4_orders_everything(self):
        (la, lb), info = fuse(self.doc, self.m2, self.d4, FuseParams())
        self.assertEqual(info["d4_sides"], ["a", "b"])
        top = [la[i] for i in range(self.absent_true.start, self.absent_true.end)]
        rest = [v for i, v in enumerate(la) if not (self.absent_true.start <= i < self.absent_true.end)]
        self.assertLess(max(rest), min(top))                       # verified absent words rank above all others
        self.assertTrue(all(0.0 <= v <= 1.0 for v in la + lb))
        best = max(range(len(lb)), key=lambda i: lb[i])
        self.assertTrue(2 <= best <= 10)                           # the differing number ("12", index 6) lifts its ±4 neighbourhood
        self.assertGreater(lb[6], lb[14])
        self.assertGreater(len(set(lb)), 2)                        # no longer a 0/1 output

    def test_missing_d4_side_falls_back_to_m2(self):
        d4 = dict(self.d4)
        del d4["s_c_b"], d4["s_u_b"]
        (la, lb), info = fuse(self.doc, self.m2, d4, FuseParams())
        self.assertEqual(info["d4_sides"], ["a"])
        self.assertEqual(set(lb), {0.0})                           # nothing absent on side b
        self.assertGreater(len(set(la)), 2)
        (la2, lb2), info2 = fuse(self.doc, self.m2, None, FuseParams())
        self.assertEqual(info2["d4_sides"], [])
        self.assertEqual(set(la2) | set(lb2), {0.0, 1.0})          # pure m2 fallback keeps all absent blocks

    def test_d4_only_rule_needs_no_m2(self):
        (la, lb), info = fuse(self.doc, None, self.d4, FuseParams(rule="d4"))
        self.assertTrue(2 <= max(range(len(lb)), key=lambda i: lb[i]) <= 10)
        self.assertGreater(lb[6], lb[14])
        self.assertTrue(all(0.0 <= v < 1.0 for v in la + lb))
        with self.assertRaises(ValueError):
            fuse(self.doc, None, self.d4, FuseParams(rule="verified"))
        with self.assertRaises(ValueError):
            fuse(self.doc, self.m2, self.d4, FuseParams(rule="nonsense"))

    def test_block_statuses_cover_every_token(self):
        st = block_statuses(self.doc, self.m2)
        self.assertEqual(len(st["a"]), len(self.doc.tokens_a))
        self.assertEqual(len(st["b"]), len(self.doc.tokens_b))
        self.assertNotIn(None, st["a"] + st["b"])


class FuseCommandTest(unittest.TestCase):
    def test_fuse_command_writes_predictions_and_scores(self):
        from uzh_diff.cli import main
        docs = {lang: load_docs(lang, "dev/train")[:2] for lang in ("de", "fr", "it")}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m2_dir, d4_dir = tmp / "m2run", tmp / "d4run"
            for d in (m2_dir, d4_dir):
                d.mkdir()
                (d / "manifest.json").write_text(json.dumps({"split": "dev/train", "model": "test-model",
                                                             "ids": {l: [x.id for x in docs[l]] for l in docs}}), encoding="utf-8")
            with open(m2_dir / "results_m2.jsonl", "w", encoding="utf-8") as fm, open(d4_dir / "results_d4.jsonl", "w", encoding="utf-8") as fd:
                for lang, ds in docs.items():
                    for k, d in enumerate(ds):
                        blocks_a = make_blocks(d.tokens_a, "a")
                        m2 = m2_details(d, {blocks_a[0].id: "absent"})
                        fm.write(json.dumps({"doc_id": d.id, "lang": lang, "status": "ok", "n_calls": 3, "n_cached": 0,
                                             "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 1.0, "details": m2}) + "\n")
                        if not (lang == "it" and k == 1):      # one document without d4: must be salvaged, not dropped
                            det = d4_details(d, [0.5] * len(d.tokens_a), [0.5] * len(d.tokens_b))
                            fd.write(json.dumps({"doc_id": d.id, "lang": lang, "status": "ok", "n_calls": 4, "n_cached": 0,
                                                 "prompt_tokens": 10, "completion_tokens": 4, "latency_s": 1.0, "details": det}) + "\n")
            old_runs = config.RUNS
            config.RUNS = tmp / "runs"
            try:
                with self.assertRaises(SystemExit):           # one document lacks d4: refused unless allowed
                    main(["fuse", "--m2-run", str(m2_dir), "--d4-run", str(d4_dir), "--label", "t"])
                rc = main(["fuse", "--m2-run", str(m2_dir), "--d4-run", str(d4_dir), "--label", "t", "--allow-missing"])
            finally:
                config.RUNS = old_runs
            self.assertEqual(rc, 0)
            run = next((tmp / "runs").glob("*-t"))
            self.assertTrue((run / "results_fuse.jsonl").exists())
            for lang in ("de", "fr", "it"):
                p = run / "predictions" / "fuse" / ("fuse_admin_%s.jsonl.jsonl" % lang)
                rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["id"] for r in rows], [d.id for d in docs[lang]])
                for r, d in zip(rows, docs[lang]):
                    self.assertEqual(len(r["labels_a"]), len(d.tokens_a))
                    self.assertEqual(len(r["labels_b"]), len(d.tokens_b))
            s = json.loads((run / "summary_fuse.json").read_text(encoding="utf-8"))
            self.assertEqual(s["params"]["rule"], "verified+d4")
            self.assertEqual(s["statuses"], {"ok": 5, "salvaged": 1})
            self.assertIn("没有 d4 结果", (run / "summary_fuse.txt").read_text(encoding="utf-8"))
            # the acceptance audit sees the missing d4 sides and reports them as a problem
            rc = main(["audit", "--run", str(run)])
            self.assertEqual(rc, 3)
            rep = json.loads((run / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["languages"]["it"]["d4_sides_missing"], ["%s/a" % docs["it"][1].id, "%s/b" % docs["it"][1].id])
            self.assertEqual(rep["languages"]["de"]["m2_blocks_unjudged"], 0)
            self.assertEqual(rep["languages"]["de"]["prediction_file"]["rows"], 2)


if __name__ == "__main__":
    unittest.main()


class InferenceWithoutGoldTest(unittest.TestCase):
    """`make run` must produce predictions from text alone: a file with only text_a / text_b, no labels."""

    def test_run_and_report_without_labels(self):
        sys.path.insert(0, HERE)
        from mock_server import MockServer
        from uzh_diff.client import ChatClient
        from uzh_diff.data import load_docs
        from uzh_diff.runner import run_and_report
        with tempfile.TemporaryDirectory() as tmp, MockServer() as server:
            tmp = Path(tmp)
            gold = tmp / "gold"
            gold.mkdir()
            rows = []
            for d in load_docs("de", "dev/train")[:2]:
                rows.append(json.dumps({"id": d.id, "text_a": d.text_a, "text_b": d.text_b}))
            (gold / "gold_admin_de.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            docs = load_docs("de", "dev", gold_dir=gold)
            self.assertIsNone(docs[0].labels_a)
            cfg = config.LLMConfig(name="mock-full", base_url=server.base_url, api_key="k")
            client = ChatClient(cfg, tmp / "run", cache_dir=tmp / "cache", backoff_base=0.01, timeout=10, max_retries=1)
            s = run_and_report("d4", {"de": docs}, client, tmp / "run", workers=1)
            self.assertIsNone(s["score"])
            self.assertIn("没有金标", s["score_table"])
            pred = (tmp / "run" / "predictions" / "d4" / "d4_admin_de.jsonl.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(pred), 2)
            self.assertEqual(len(json.loads(pred[0])["labels_a"]), len(docs[0].tokens_a))


class TransientFailureTest(unittest.TestCase):
    """A run whose requests failed must say so with exit code 5, so that the entry point can repeat it;
    unusable answers (a skipped block) are not transient and must not trigger a repetition."""

    def test_transient_failures_classification(self):
        from uzh_diff.methods import DocResult
        from uzh_diff.runner import transient_failures
        def res(method, status, details):
            return DocResult("x", "de", method, "", "m", status, [0.0], [0.0], details=details)
        self.assertEqual(transient_failures([res("d4", "salvaged", {"failed": [{"side": "a", "kind": "c", "error": "HTTP 500"}]})]), ["x"])
        self.assertEqual(transient_failures([res("d4", "salvaged", {"failed": [{"side": "a", "kind": "c", "error": "alignment failed (fuzzy): 3 of 10 words mapped"}]})]), [])
        self.assertEqual(transient_failures([res("m2", "salvaged", {"blocks": [{"id": "a000", "status": "call_failed"}]})]), ["x"])
        self.assertEqual(transient_failures([res("m2", "salvaged", {"blocks": [{"id": "a000", "status": "missing_in_answer"}]})]), [])
        self.assertEqual(transient_failures([res("m2", "api_error", {"blocks": []})]), ["x"])
        self.assertEqual(transient_failures([res("d4", "ok", {"failed": []})]), [])

    def test_run_command_exit_code_5_on_endpoint_errors(self):
        sys.path.insert(0, HERE)
        from mock_server import MockServer
        import uzh_diff.cli as cli
        orig_client, old_runs, old_env = cli.ChatClient, config.RUNS, dict(os.environ)
        with tempfile.TemporaryDirectory() as tmp, MockServer() as server:
            config.RUNS = Path(tmp) / "runs"
            os.environ.update({"LLM_API_KEY": "test-key", "LLM_BASE_URL": server.base_url})
            cli.ChatClient = lambda cfg, run_dir, **kw: orig_client(cfg, run_dir, backoff_base=0.01, timeout=10, **{k: v for k, v in kw.items()})
            try:
                rc = cli.main(["run", "--method", "d4", "--split", "dev/train", "--limit", "1", "--langs", "de", "--model", "mock-500", "--label", "t5"])
                self.assertEqual(rc, 5)
                rc = cli.main(["run", "--method", "d4", "--split", "dev/train", "--limit", "1", "--langs", "de", "--model", "mock-full", "--label", "t0"])
                self.assertEqual(rc, 0)
            finally:
                cli.ChatClient, config.RUNS = orig_client, old_runs
                os.environ.clear(); os.environ.update(old_env)
