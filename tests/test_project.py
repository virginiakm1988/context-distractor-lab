import unittest, tempfile, json
from pathlib import Path
from project_demo.runtime import parse_object, wilson


class SharedTests(unittest.TestCase):
    def test_no_letter_mining(self):
        with self.assertRaises(ValueError):
            parse_object("An answer cannot be determined")
        self.assertEqual(parse_object('Analysis text. {"answer":"D"}')["answer"], "D")

    def test_confidence_interval(self):
        lo, hi = wilson(0, 3)
        self.assertLessEqual(lo, 1e-10)
        self.assertGreater(hi, 0.4)
        self.assertEqual(wilson(0, 0), [None, None])


from project_demo import core
from project_demo.designer import validate_template


class InkTests(unittest.TestCase):
    def test_duplicate_rounded_conditions_are_rejected(self):
        config = dict(core.DEFAULT_CONFIG, hard_fractions=[0, 0.01])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            core.prepare(config, "fixture", Path("."))

    def test_format_and_answer_are_scored_separately(self):
        config = dict(core.DEFAULT_CONFIG, replicates=1, hard_fractions=[0])
        row = core.prepare(config, "fixture", Path("."))[0]
        prediction = {
            **row,
            "prediction": "",
            "raw": row["answer"],
            "error": "invalid JSON",
        }
        metric = core.evaluate([prediction], config)["curve"][0]
        self.assertEqual(metric["accuracy"], 0)
        self.assertEqual(metric["answer_exact_match"], 1)
        prediction["raw"] = "Maybe " + row["answer"]
        self.assertEqual(
            core.evaluate([prediction], config)["curve"][0]["answer_exact_match"], 0
        )

    def test_controls_are_paired(self):
        rows = core.prepare(dict(core.DEFAULT_CONFIG), "fixture", Path("."))
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {r["question"]["scenario"] for r in rows},
            {"record_retrieval", "two_hop_entity_join", "numeric_aggregation"},
        )
        for replicate in range(3):
            group = [r for r in rows if r["replicate"] == replicate]
            self.assertEqual(
                len(
                    {
                        (
                            r["input_tokens"],
                            r["gold_token_start"],
                            r["gold_token_end"],
                            r["answer"],
                        )
                        for r in group
                    }
                ),
                1,
            )
            self.assertEqual([r["hard_count"] for r in group], [0, 1, 5, 10])

    def test_answer_not_present_in_distractors(self):
        for row in core.prepare(dict(core.DEFAULT_CONFIG), "fixture", Path(".")):
            for doc in row["documents"]:
                if doc["kind"] != "gold":
                    self.assertNotIn(row["answer"], doc["text"])

    def test_invalid_ratios(self):
        with self.assertRaises(ValueError):
            core.prepare(
                dict(core.DEFAULT_CONFIG, hard_fractions=[-1]), "fixture", Path(".")
            )

    def test_no_manufactured_drop(self):
        c = dict(core.DEFAULT_CONFIG)
        rows = core.run(c, core.prepare(c, "fixture", Path(".")), "fixture", Path("."))
        self.assertTrue(
            all(r["accuracy"] == 1 for r in core.evaluate(rows, c)["curve"])
        )

    def test_template_constraints(self):
        self.assertTrue(
            validate_template("Unverified record {ENTITY}: {WRONG}.", "distractor")
        )
        self.assertFalse(
            validate_template("Verified record {ENTITY}: {WRONG}.", "distractor")
        )
