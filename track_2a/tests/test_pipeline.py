"""Offline tests of the client, the two methods and the run/export/score chain against a local mock server,
plus sanity checks of the development data shipped in data/."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from mock_server import MockServer  # noqa: E402

from uzh_diff import config  # noqa: E402
from uzh_diff.client import ChatClient  # noqa: E402
from uzh_diff.data import LANGS, DocPair, load_docs, load_manifest, select  # noqa: E402
from uzh_diff.methods import paper_chunks, reference_window, run_b0, run_b0s, run_m1, run_m2  # noqa: E402
from uzh_diff.probes import run_probes  # noqa: E402
from uzh_diff.runner import export_predictions, run_and_report  # noqa: E402
from uzh_diff.scorer import score  # noqa: E402

HELLO = [{"role": "user", "content": "Reply with the single word OK."}]


def make_client(server, tmp, model="swiss-ai/Apertus-v1.5-8B", key="test-key", **kw):
    cfg = config.LLMConfig(name=model, base_url=server.base_url, api_key=key)
    kw.setdefault("max_retries", 2)
    return ChatClient(cfg, Path(tmp) / "run", cache_dir=Path(tmp) / "cache", backoff_base=0.01, timeout=10, **kw)


def toy_doc(doc_id="toy_de_1", lang="de"):
    a = "Energy policy overview page . The fee is 10 francs per year . Press releases are available in German .".split()
    b = "Überblick zur Energiepolitik . Die Gebühr beträgt 12 Franken pro Jahr . Zum Seitenanfang".split()
    return DocPair(doc_id, lang, a, b, [0] * len(a), [0] * len(b))


class ClientTest(unittest.TestCase):
    def test_success_is_logged_and_cached(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            client = make_client(server, tmp)
            r = client.chat(HELLO, tag="t", temperature=0, max_tokens=10)
            self.assertTrue(r.ok)
            self.assertEqual((r.content, r.attempts, r.cached), ("OK", 1, False))
            self.assertEqual(r.headers.get("x-ratelimit-remaining-requests"), "99")
            again = client.chat(HELLO, tag="t", temperature=0, max_tokens=10)
            self.assertTrue(again.cached)
            self.assertEqual(server.state.counts["swiss-ai/Apertus-v1.5-8B"], 1)
            log = (Path(tmp) / "run" / "calls.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(log.strip().splitlines()), 2)
            self.assertNotIn("test-key", log)

    def test_models(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            self.assertIn("swiss-ai/Apertus-v1.5-70B", make_client(server, tmp).list_models())

    def test_retry_after_429(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = make_client(server, tmp, model="mock-429-once").chat(HELLO, temperature=0)
            self.assertTrue(r.ok)
            self.assertEqual(r.attempts, 2)

    def test_gives_up_after_repeated_500(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = make_client(server, tmp, model="mock-500").chat(HELLO, temperature=0)
            self.assertFalse(r.ok)
            self.assertEqual((r.status, r.attempts), (500, 3))

    def test_wrong_key_is_not_retried(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = make_client(server, tmp, key="wrong").chat(HELLO, temperature=0)
            self.assertFalse(r.ok)
            self.assertEqual((r.status, r.attempts), (401, 1))

    def test_unsupported_field_is_dropped_and_remembered(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            client = make_client(server, tmp, model="mock-no-json-mode")
            r = client.chat(HELLO, temperature=0, response_format={"type": "json_object"})
            self.assertTrue(r.ok)
            self.assertEqual(r.adjustments, ["response_format dropped"])
            client.chat([{"role": "user", "content": "second question"}], temperature=0, response_format={"type": "json_object"})
            self.assertNotIn("response_format", server.state.requests[-1])
            self.assertEqual(server.state.counts["mock-no-json-mode"], 3)

    def test_too_large_max_tokens_is_halved(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = make_client(server, tmp, model="mock-maxtok").chat(HELLO, temperature=0, max_tokens=4000)
            self.assertTrue(r.ok)
            self.assertEqual(r.max_tokens_used, 1000)

    def test_gateway_timeout_is_retried_only_once(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = make_client(server, tmp, model="mock-504", max_retries=5).chat(HELLO, temperature=0)
            self.assertFalse(r.ok)
            self.assertEqual((r.status, r.attempts), (504, 2))

    def test_unreachable_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config.LLMConfig("m", "http://127.0.0.1:9/v1", "k")
            client = ChatClient(cfg, Path(tmp) / "run", cache_dir=Path(tmp) / "cache", backoff_base=0.01, timeout=2, max_retries=1)
            r = client.chat(HELLO)
            self.assertFalse(r.ok)
            self.assertEqual(r.attempts, 2)
            self.assertIsNotNone(r.error)


class MethodsTest(unittest.TestCase):
    def test_m1_quotes_become_token_labels(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            doc = toy_doc()
            r = run_m1(doc, make_client(server, tmp))
            self.assertEqual(r.status, "ok")
            # mock quotes the first two words of block a001 and the last three words of block b000
            self.assertEqual([i for i, v in enumerate(r.labels_a) if v > 0], [5, 6])
            self.assertEqual([doc.tokens_a[i] for i in (5, 6)], ["The", "fee"])
            self.assertEqual([doc.tokens_b[i] for i, v in enumerate(r.labels_b) if v > 0], ["zur", "Energiepolitik", "."])
            self.assertEqual(r.details["location_counts"], {"ok": 2, "not_found": 1})
            self.assertEqual(len(r.labels_a), len(doc.tokens_a))
            self.assertEqual(r.n_calls, 1)
            self.assertGreater(r.prompt_tokens, 0)

    def test_m1_cut_off_answer_is_marked(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = run_m1(toy_doc(), make_client(server, tmp, model="mock-truncate"))
            self.assertIn(r.status, ("salvaged", "invalid_answer"))

    def test_m1_endpoint_failure_is_visible(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = run_m1(toy_doc(), make_client(server, tmp, model="mock-500"))
            self.assertEqual(r.status, "api_error")
            self.assertEqual(sum(r.labels_a) + sum(r.labels_b), 0)

    def test_m2_blocks_become_token_labels(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            doc = toy_doc()   # 3 blocks on side a (a000-a002), 2 on side b (b000, b001)
            r = run_m2(doc, make_client(server, tmp), blocks_per_call=2)
            self.assertEqual(r.status, "ok")
            self.assertEqual(r.n_calls, 3)             # side a: 2 calls (2 + 1 blocks); side b: 1 call
            # in every call the mock marks the first block 'absent' (all tokens) and the second 'partly'
            # (its first two words + one quote that is not in the text)
            self.assertEqual(r.labels_a[:5], [1.0] * 5)                      # a000 "Energy policy overview page ."
            self.assertEqual([doc.tokens_a[i] for i in range(5, 13) if r.labels_a[i] > 0], ["The", "fee"])
            self.assertEqual(r.labels_a[13:], [1.0] * 7)                     # a002 alone in the second call -> absent
            self.assertEqual(r.labels_b[:4], [1.0] * 4)                      # b000 absent
            self.assertEqual([doc.tokens_b[i] for i in range(4, len(doc.tokens_b)) if r.labels_b[i] > 0], ["Die", "Gebühr"])
            self.assertEqual(r.details["block_status_counts"], {"absent": 3, "partly": 2})
            self.assertEqual(r.details["location_counts"], {"ok": 2, "not_found": 2})
            self.assertEqual(len(r.labels_b), len(doc.tokens_b))

    def test_m2_whole_side_in_one_call(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = run_m2(toy_doc(), make_client(server, tmp), blocks_per_call=0)
            self.assertEqual((r.status, r.n_calls), ("ok", 2))

    def test_m2_failed_call_is_visible_and_scored_as_zero(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = run_m2(toy_doc(), make_client(server, tmp, model="mock-504"), blocks_per_call=0)
            self.assertEqual(r.status, "api_error")
            self.assertEqual(sum(r.labels_a) + sum(r.labels_b), 0)
            self.assertEqual(r.details["failed_calls"], 2)

    def test_m2_tiers_are_recorded(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            doc = toy_doc()
            r = run_m2(doc, make_client(server, tmp), blocks_per_call=2)
            ta, tb = r.details["tiers_a"], r.details["tiers_b"]
            self.assertEqual(ta[:5], [1] * 5)                       # absent block -> tier 1
            self.assertEqual([ta[i] for i in (5, 6)], [2, 2])         # quoted words -> tier 2
            self.assertEqual(len(ta), len(doc.tokens_a))
            self.assertEqual(sum(1 for t in tb if t == 1), 4)

    def test_reference_window(self):
        toks = [str(i) for i in range(4000)]
        text, (s, e) = reference_window(toks, 4000, 1900, 1960, 3800, 1500)
        self.assertEqual(e - s, 1500)
        self.assertTrue(s <= 2030 <= e)
        self.assertEqual(reference_window(toks[:900], 900, 0, 60, 800, 1500)[1], (0, 900))

    def test_paper_chunks_match_the_repository_rule(self):
        self.assertEqual(paper_chunks(344, 330), [(0, 125, 0, 125), (125, 250, 125, 250), (250, 344, 250, 330)])
        self.assertEqual(paper_chunks(100, 100), [(0, 100, 0, 100)])
        chunks = paper_chunks(400, 50)
        self.assertEqual(chunks[-1][1], 400)
        self.assertTrue(all(ea - sa <= 125 for sa, ea, _, _ in chunks))

    def test_b0s_scores_chunks_back_into_place(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            a = ("word%d" % i for i in range(300))
            doc = DocPair("toy_de_long", "de", list(a), ["wort%d" % i for i in range(260)], None, None)
            doc.labels_a, doc.labels_b = [0] * 300, [0] * 260
            r = run_b0s(doc, make_client(server, tmp))
            self.assertEqual(r.status, "ok")
            self.assertEqual(r.details["chunks"], 3)
            # the mock gives similarity 0 (difference 1) to the first token of every chunk side
            self.assertEqual([i for i, v in enumerate(r.labels_a) if v > 0], [0, 125, 250])
            self.assertEqual([i for i, v in enumerate(r.labels_b) if v > 0], [0, 125, 250])
            self.assertEqual(r.n_calls, 3)

    def test_b0_reads_token_scores(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            doc = toy_doc()
            r = run_b0(doc, make_client(server, tmp))
            self.assertEqual(r.status, "ok")
            self.assertEqual(r.labels_a[0], 1.0)           # similarity 0 -> difference 1
            self.assertEqual(set(r.labels_a[1:]), {0.0})   # similarity 5 -> difference 0
            self.assertEqual(r.labels_b[0], 1.0)

    def test_b0_cut_off_answer_scores_as_zeros(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = run_b0(toy_doc(), make_client(server, tmp, model="mock-truncate"))
            self.assertEqual(r.status, "truncated")
            self.assertEqual(sum(r.labels_a) + sum(r.labels_b), 0)


class D4Test(unittest.TestCase):
    def test_word_scores_follow_the_token_logprobs(self):
        from uzh_diff.surprisal import run_d4
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            doc = toy_doc()
            r = run_d4(doc, make_client(server, tmp, model="mock-full"))
            self.assertEqual(r.status, "ok")
            self.assertEqual(r.n_calls, 4)
            self.assertEqual(len(r.labels_a), len(doc.tokens_a))
            self.assertEqual(len(r.labels_b), len(doc.tokens_b))
            self.assertEqual(set(r.details["alignment"].values()), {"exact"})
            s_c = r.details["s_c_a"]
            # the mock makes words with digits surprising (5.0) and everything else 1.0
            self.assertEqual(s_c[doc.tokens_a.index("10")], 5.0)
            self.assertEqual(s_c[doc.tokens_a.index("fee")], 1.0)
            # labels are max-smoothed over +-1 word, so the neighbours of "10" rise too
            i = doc.tokens_a.index("10")
            self.assertEqual(r.labels_a[i - 1], 5.0)
            self.assertEqual(r.labels_a[i + 1], 5.0)
            self.assertEqual(r.labels_a[0], 1.0)
            # unconditioned surprisal of digits is lower (3.0): gain is negative there in this mock
            self.assertEqual(r.details["s_u_a"][i], 3.0)

    def test_server_without_prompt_logprobs_is_reported(self):
        from uzh_diff.surprisal import run_d4
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            r = run_d4(toy_doc(), make_client(server, tmp, model="mock-plain"))
            self.assertIn(r.status, ("api_error", "invalid_answer"))
            self.assertEqual(len(r.details["failed"]), 4)


class ProbeTest(unittest.TestCase):
    def test_full_server_passes_all_probes(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            client = make_client(server, tmp, model="mock-full", max_retries=0)
            results = run_probes(client, "mock-full", which=["prompt_logprobs", "prefill", "prefill_logprobs", "n", "structured_choice", "json_schema", "logprobs"])
            self.assertEqual({r["id"]: r["ok"] for r in results},
                             {"T1": True, "T2": True, "T2b": True, "T3": True, "T4": True, "T6": True, "T7": True})

    def test_plain_server_fails_the_probes_without_raising(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            client = make_client(server, tmp, model="mock-plain", max_retries=0)
            results = run_probes(client, "mock-plain", which=["prompt_logprobs", "prefill", "prefill_logprobs", "n", "structured_choice", "json_schema", "logprobs"])
            self.assertEqual({r["id"]: r["ok"] for r in results},
                             {"T1": False, "T2": False, "T2b": False, "T3": False, "T4": False, "T6": False, "T7": False})

    def test_unreachable_server_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config.LLMConfig("m", "http://127.0.0.1:9/v1", "k")
            client = ChatClient(cfg, Path(tmp) / "run", cache_dir=Path(tmp) / "cache", timeout=2, max_retries=0)
            results = run_probes(client, "m", which=["prompt_logprobs", "n"])
            self.assertTrue(all(not r["ok"] for r in results))


class RunTest(unittest.TestCase):
    def test_run_export_and_score(self):
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            docs_by_lang = {}
            for lang in LANGS:
                docs = [toy_doc("toy_%s_%d" % (lang, i), lang) for i in range(2)]
                for d in docs:
                    d.labels_a[5] = 1.0
                    d.labels_b[1] = 0.6
                docs_by_lang[lang] = docs
            run_dir = Path(tmp) / "run"
            client = make_client(server, tmp)
            summary = run_and_report("m1", docs_by_lang, client, run_dir, workers=2)
            self.assertEqual(summary["statuses"], {"ok": 6})
            self.assertEqual(summary["cost"]["calls"], 6)
            self.assertIn("recall_by_position_decile", summary["diagnostics"]["de"])
            summary2 = run_and_report("m2", docs_by_lang, client, run_dir, workers=2, blocks_per_call=2)
            self.assertIn("spearman_absent_first", summary2["diagnostics"]["de"])
            self.assertIsNotNone(summary2["diagnostics"]["de"]["tier_precision"]["absent_block"])
            for lang in LANGS:
                path = run_dir / "predictions" / "m1" / ("m1_admin_%s.jsonl.jsonl" % lang)
                rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["id"] for r in rows], [d.id for d in docs_by_lang[lang]])
                self.assertEqual(sorted(rows[0]), ["id", "labels_a", "labels_b", "text_a", "text_b"])
                self.assertEqual(len(rows[0]["labels_a"]), len(rows[0]["text_a"].split()))
            self.assertTrue((run_dir / "results_m1.jsonl").exists())
            self.assertTrue((run_dir / "summary_m1.txt").exists())

    def test_export_refuses_llm_in_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_predictions(Path(tmp), "my_llm_system", {}, {})


class EnvFileTest(unittest.TestCase):
    def read(self, data: bytes):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_bytes(data)
            saved = os.environ.pop("LLM_API_KEY", None)
            try:
                config.load_env_file(path)
                return os.environ.pop("LLM_API_KEY", None)
            finally:
                if saved is not None:
                    os.environ["LLM_API_KEY"] = saved

    def test_plain_ascii(self):
        self.assertEqual(self.read(b"LLM_API_KEY=abc-123\r\n"), "abc-123")

    def test_utf8_with_byte_order_mark(self):
        self.assertEqual(self.read(b"\xef\xbb\xbfLLM_API_KEY=abc-123\n"), "abc-123")

    def test_utf16_as_written_by_windows_powershell_redirection(self):
        self.assertEqual(self.read("LLM_API_KEY=abc-123\r\n".encode("utf-16")), "abc-123")

    def test_quotes_are_removed(self):
        self.assertEqual(self.read(b'LLM_API_KEY="abc-123"\n'), "abc-123")

    def test_key_clean_up(self):
        self.assertEqual(config._clean_key('  "Bearer abc-123" '), "abc-123")


class DataTest(unittest.TestCase):
    def test_dev_split_sizes(self):
        for lang in LANGS:
            self.assertEqual(len(load_docs(lang, "dev")), 168)
            self.assertEqual(len(load_docs(lang, "dev/train")), 134)
            self.assertEqual(len(load_docs(lang, "dev/val")), 34)

    def test_train_and_val_do_not_overlap(self):
        for lang in LANGS:
            train = {d.id for d in load_docs(lang, "dev/train")}
            val = {d.id for d in load_docs(lang, "dev/val")}
            self.assertEqual(len(train & val), 0)

    def test_test_split_is_locked(self):
        with self.assertRaises(PermissionError):
            load_docs("de", "test")

    def test_smoke_manifest_is_inside_dev_train(self):
        ids = load_manifest("smoke12")
        for lang in LANGS:
            self.assertEqual(len(select(load_docs(lang, "dev/train"), ids[lang])), 4)

    def test_yes_no_oracle_scores_above_099_on_dev(self):
        docs_by_lang = {lang: load_docs(lang, "dev") for lang in LANGS}
        preds = {lang: {d.id: ([1.0 if v > 0 else 0.0 for v in d.labels_a], [1.0 if v > 0 else 0.0 for v in d.labels_b])
                        for d in docs} for lang, docs in docs_by_lang.items()}
        sc = score(docs_by_lang, preds)
        for lang in LANGS:
            self.assertGreater(sc.per_lang[lang].spearman, 0.99)
            self.assertEqual(sc.per_lang[lang].recall, 1.0)


if __name__ == "__main__":
    unittest.main()
