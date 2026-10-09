"""Offline tests of the deterministic parts: ranks, blocks, quote location, answer parsing, scoring."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from uzh_diff import project, schema  # noqa: E402
from uzh_diff.blocks import make_blocks, segment  # noqa: E402
from uzh_diff.data import DocPair  # noqa: E402
from uzh_diff.methods import b0_token_labels, _b0_load, _to_difference  # noqa: E402
from uzh_diff.scorer import score, score_lang  # noqa: E402
from uzh_diff.spearman import average_ranks, spearman  # noqa: E402


class SpearmanTest(unittest.TestCase):
    def test_average_ranks_with_ties(self):
        self.assertEqual(average_ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_perfect_and_inverse(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_known_value_with_ties(self):
        # ranks x: 1.5 1.5 3 4 5 ; ranks y: 1 3 3 3 5  -> Pearson of the ranks = 7 / sqrt(76)
        self.assertAlmostEqual(spearman([0, 0, 1, 2, 3], [0, 1, 1, 1, 2]), 7 / math.sqrt(76), places=12)

    def test_constant_input_is_undefined(self):
        self.assertTrue(math.isnan(spearman([0, 0, 0], [1, 2, 3])))

    def test_matches_scipy_when_available(self):
        try:
            from scipy.stats import spearmanr
        except Exception:
            self.skipTest("scipy not installed")
        import random
        rng = random.Random(0)
        x = [rng.choice([0, 0, 0, 0.2, 0.4, 1]) for _ in range(5000)]
        y = [rng.choice([0, 0, 1]) for _ in range(5000)]
        self.assertAlmostEqual(spearman(x, y), float(spearmanr(x, y).statistic), places=10)


class BlocksTest(unittest.TestCase):
    def check_partition(self, tokens, **kw):
        spans = segment(tokens, **kw)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(tokens))
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            self.assertEqual(e1, s2)
            self.assertLess(s1, e1)
        return spans

    def test_sentences(self):
        tokens = "The fee is 10 francs . It is due in May . Thank you very much !".split()
        spans = self.check_partition(tokens)
        self.assertEqual(len(spans), 3)

    def test_short_heading_is_glued_to_next_sentence(self):
        tokens = "Energy . The fee is 10 francs today .".split()
        spans = self.check_partition(tokens)
        self.assertEqual(spans, [(0, len(tokens))])

    def test_long_sentence_is_split(self):
        tokens = ("word " * 30 + ", " + "word " * 40 + ", " + "word " * 35 + ".").split()
        spans = self.check_partition(tokens, max_len=60)
        self.assertTrue(all(e - s <= 60 for s, e in spans))
        self.assertGreater(len(spans), 1)

    def test_no_punctuation_at_all(self):
        tokens = ("word " * 150).split()
        spans = self.check_partition(tokens, max_len=60)
        self.assertTrue(all(e - s <= 60 for s, e in spans))

    def test_dates_and_numbers_do_not_split(self):
        tokens = "It enters into force on 22 . 12 . 2017 . The file ( PDF , 1 . 5 MB , 22 . 06 . 2020 ) is online . Next".split()
        spans = self.check_partition(tokens)
        texts = [" ".join(tokens[s:e]) for s, e in spans]
        self.assertEqual(texts[0], "It enters into force on 22 . 12 . 2017 .")
        self.assertTrue(texts[1].startswith("The file ( PDF , 1 . 5 MB , 22 . 06 . 2020 ) is online ."))

    def test_lowercase_after_period_does_not_split(self):
        tokens = "See www . admin . ch for details . The second sentence is long enough to stand alone .".split()
        spans = self.check_partition(tokens)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], (0, 9))   # "See www . admin . ch for details ." = 9 tokens

    def test_block_ids_and_offsets(self):
        tokens = "A b c d . E f g h .".split()
        blocks = make_blocks(tokens, "b")
        self.assertEqual([b.id for b in blocks], ["b000", "b001"])
        self.assertEqual(blocks[1].tokens, tokens[blocks[1].start:blocks[1].end])

    def test_empty(self):
        self.assertEqual(segment([]), [])


class ProjectTest(unittest.TestCase):
    def test_unique_match_in_block(self):
        tokens = "The fee is 10 francs . The fee is paid in May .".split()
        loc = project.locate("10 francs", tokens, (0, 6))
        self.assertEqual((loc.status, loc.start, loc.end), (project.OK, 3, 5))

    def test_spacing_around_punctuation_is_ignored(self):
        tokens = "Switzerland ' s energy supply ( in German ) .".split()
        loc = project.locate("Switzerland's energy", tokens, (0, len(tokens)))
        self.assertEqual((loc.status, loc.start, loc.end), (project.OK, 0, 4))
        loc = project.locate("(in German)", tokens, (0, len(tokens)))
        self.assertEqual((loc.status, loc.start, loc.end), (project.OK, 5, 9))

    def test_typographic_apostrophe_and_umlauts(self):
        tokens = "l’ordonnance sur l ' énergie für höhere Berufsbildung".split()
        self.assertEqual(project.locate("l'ordonnance", tokens, (0, len(tokens))).status, project.OK)
        loc = project.locate("höhere Berufsbildung", tokens, (0, len(tokens)))
        self.assertEqual((loc.start, loc.end), (6, 8))

    def test_repeated_phrase_in_block_is_flagged(self):
        tokens = "the fee and the fee .".split()
        loc = project.locate("the fee", tokens, (0, len(tokens)))
        self.assertEqual((loc.status, loc.start, loc.end, loc.n_matches), (project.AMBIGUOUS, 0, 2, 2))

    def test_whole_words_preferred_over_parts_of_words(self):
        tokens = "information in German".split()
        loc = project.locate("in", tokens, (0, len(tokens)))
        self.assertEqual((loc.status, loc.start, loc.end), (project.OK, 1, 2))

    def test_part_of_a_word_covers_the_word(self):
        tokens = "das Energiegesetz gilt".split()
        loc = project.locate("Energie", tokens, (0, len(tokens)))
        self.assertEqual((loc.status, loc.start, loc.end, loc.how), (project.OK, 1, 2, "inside_token"))

    def test_wrong_block_but_unique_in_document(self):
        tokens = "A b c d . The fee is 10 francs .".split()
        loc = project.locate("10 francs", tokens, (0, 5))
        self.assertEqual((loc.status, loc.start, loc.end), (project.ELSEWHERE, 8, 10))

    def test_wrong_block_and_several_places_is_not_guessed(self):
        tokens = "A b c d . The fee . The fee .".split()
        self.assertEqual(project.locate("The fee", tokens, (0, 5)).status, project.NOT_FOUND)

    def test_not_found_and_empty(self):
        tokens = "The fee is 10 francs .".split()
        self.assertEqual(project.locate("12 francs", tokens, (0, 6)).status, project.NOT_FOUND)
        self.assertEqual(project.locate("   ", tokens, (0, 6)).status, project.EMPTY)

    def test_case_difference_is_accepted_as_casefold(self):
        tokens = "Zum Seitenanfang".split()
        loc = project.locate("zum seitenanfang", tokens, (0, 2))
        self.assertEqual((loc.status, loc.how), (project.OK, "casefold"))

    def test_apply_marks_only_located_tokens(self):
        labels = [0.0] * 6
        project.apply(labels, project.Location(project.OK, 3, 5, 1, "exact"))
        project.apply(labels, project.Location(project.NOT_FOUND))
        self.assertEqual(labels, [0, 0, 0, 1.0, 1.0, 0])


class SchemaTest(unittest.TestCase):
    ANSWER = ('{"decision": "differences_found", "differences": ['
              '{"a": {"block_id": "a001", "quote": "10 francs"}, "b": {"block_id": "b001", "quote": "12 Franken"}, '
              '"category": "different_content", "explanation": "amount differs"}, '
              '{"a": null, "b": {"block_id": "b002", "quote": "Zum Seitenanfang"}, "category": "only_in_b", "explanation": "nav"}]}')

    def test_plain(self):
        p = schema.parse_m1(self.ANSWER)
        self.assertEqual(p.decision, "differences_found")
        self.assertEqual([(n.side, n.block_id, n.quote) for n in p.nominations],
                         [("a", "a001", "10 francs"), ("b", "b001", "12 Franken"), ("b", "b002", "Zum Seitenanfang")])
        self.assertFalse(p.salvaged)

    def test_code_fence_and_chatter(self):
        p = schema.parse_m1("Here is the JSON:\n```json\n" + self.ANSWER + "\n```\nHope this helps.")
        self.assertEqual(len(p.nominations), 3)

    def test_braces_inside_strings(self):
        p = schema.parse_m1('{"differences": [{"a": {"block_id": "a000", "quote": "x } y { z"}, "b": null}]}')
        self.assertEqual(p.nominations[0].quote, "x } y { z")

    def test_cut_off_answer_keeps_complete_items(self):
        cut = self.ANSWER[: self.ANSWER.index('{"a": null')] + '{"a": null, "b": {"block_id": "b00'
        p = schema.parse_m1(cut)
        self.assertTrue(p.salvaged)
        self.assertEqual(len(p.nominations), 2)

    def test_no_difference(self):
        p = schema.parse_m1('{"decision": "no_difference", "differences": []}')
        self.assertEqual((p.decision, p.nominations), ("no_difference", []))

    def test_flat_items_are_accepted(self):
        p = schema.parse_m1('{"differences": [{"side": "b", "block_id": "b003", "quote": "en bref"}]}')
        self.assertEqual([(n.side, n.quote) for n in p.nominations], [("b", "en bref")])

    def test_item_without_quote_is_reported(self):
        p = schema.parse_m1('{"differences": [{"a": null, "b": null, "category": "x"}]}')
        self.assertEqual(p.nominations, [])
        self.assertEqual(len(p.problems), 1)

    def test_garbage_raises(self):
        with self.assertRaises(schema.InvalidAnswer):
            schema.parse_m1("I cannot help with that.")
        with self.assertRaises(schema.InvalidAnswer):
            schema.parse_m1('{"differences": "none"}')


class B0ParseTest(unittest.TestCase):
    def test_walk_matches_current_or_next_token(self):
        tokens = ["The", "fee", "is", "10", "francs", "."]
        emitted = [["The", 5], ["is", 4], ["BOGUS", 0], ["10", 0], ["francs", "x"], ["francs", 3], [".", -1]]
        # "fee" skipped by the model -> keeps fallback 5; "BOGUS" ignored; unreadable label does not advance
        self.assertEqual(b0_token_labels(tokens, emitted), [5.0, 5.0, 4.0, 0.0, 3.0, -1.0])

    def test_similarity_to_difference(self):
        self.assertEqual([_to_difference(s) for s in (5, 0, -1)], [0.0, 1.0, -1.0])
        self.assertAlmostEqual(_to_difference(4), 0.2)

    def test_unreadable_answer_gives_empty_object(self):
        self.assertEqual(_b0_load('{"sentence1": [["The", 5], ["fe'), {})
        self.assertEqual(_b0_load('```json\n{"sentence1": [["The", 5]], "sentence2": []}\n```')["sentence1"], [["The", 5]])


class ScorerTest(unittest.TestCase):
    def doc(self, i="d1"):
        return DocPair(i, "de", "The fee is 10 francs .".split(), "Die Gebühr beträgt 12 Franken .".split(),
                       [0, 0, 0, 1, 0, -1], [0, 0, 0, 0.6, 0, -1])

    def test_perfect_prediction(self):
        d = self.doc()
        s = score_lang("de", [d], {"d1": ([0, 0, 0, 1, 0, 0], [0, 0, 0, 0.6, 0, 0])})
        self.assertAlmostEqual(s.spearman, 1.0)
        self.assertEqual((s.n_scored_tokens, s.precision, s.recall), (10, 1.0, 1.0))

    def test_yes_no_prediction_is_almost_perfect(self):
        # gold has two grades (1 and 0.6); a prediction that only says "different" ties them
        s = score_lang("de", [self.doc()], {"d1": ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0])})
        self.assertGreater(s.spearman, 0.99)
        self.assertLess(s.spearman, 1.0)

    def test_punctuation_is_not_scored(self):
        d = self.doc()
        a = score_lang("de", [d], {"d1": ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0])}).spearman
        b = score_lang("de", [d], {"d1": ([0, 0, 0, 1, 0, 9], [0, 0, 0, 1, 0, 9])}).spearman
        self.assertEqual(a, b)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            score_lang("de", [self.doc()], {"d1": ([0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0])})

    def test_missing_document_is_an_error_unless_marked_failed(self):
        with self.assertRaises(KeyError):
            score_lang("de", [self.doc()], {})
        s = score_lang("de", [self.doc("d1"), self.doc("d2")], {"d1": ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0])}, failed=["d2"])
        self.assertEqual(s.failed_docs, ["d2"])
        self.assertLess(s.recall, 1.0)

    def test_macro_is_undefined_if_one_language_is(self):
        d = self.doc()
        sc = score({"de": [d], "fr": [d]}, {"de": {"d1": ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0])},
                                             "fr": {"d1": ([0] * 6, [0] * 6)}})
        self.assertTrue(math.isnan(sc.macro_spearman))


if __name__ == "__main__":
    unittest.main()
