"""Verify which functions the mechanical catalog can ground in unfamiliar briefs."""

from unittest import TestCase

from design_research_workshop.functions_catalog import infer_functions


class MechanicalFunctionCatalogTests(TestCase):
    def test_new_briefs_retain_exact_evidence_and_mechanism_options(self):
        briefs = (
            (
                "Brainstorm a garden downspout attachment to collect rainwater and filter debris.",
                (
                    ("Collect water", "collect rainwater"),
                    ("Filter debris", "filter debris"),
                ),
            ),
            (
                "A drawing-board stand SUPPORTS a panel and adjusts its viewing angle.",
                (
                    ("Support a panel", "SUPPORTS a panel"),
                    ("Adjust tilt", "adjusts its viewing angle"),
                ),
            ),
            (
                "Brainstorm a jig that aligns a blank, then clamps the workpiece.",
                (
                    ("Locate a workpiece", "aligns a blank"),
                    ("Grip a workpiece", "clamps the workpiece"),
                ),
            ),
        )
        for text, expected in briefs:
            with self.subTest(text=text):
                matches = infer_functions(text)
                self.assertEqual([(m.name, m.phrase) for m in matches], list(expected))
                for m in matches:
                    self.assertIn(m.phrase, text)
                    self.assertEqual(len(m.options), len(set(m.options)))
                    self.assertGreaterEqual(len(m.options), 2)

    def test_topic_vocabulary_and_duplicate_mentions_do_not_imply_two_functions(self):
        for text in (
            "Brainstorm an improved garden downspout for rainwater and debris.",
            "Collect the test data and filter out noisy observations.",
            "Collect rainwater and capture roof runoff.",
            "Move the meeting and adjust the schedules.",
            "Build a chip tray that catches chips and encourages calm.",
        ):
            with self.subTest(text=text):
                self.assertEqual(infer_functions(text), ())

    def test_negation_and_alternatives_do_not_create_a_required_function(self):
        for text in (
            "Design a stand to support a panel without adjusting its angle.",
            "Should we collect rainwater or filter debris?",
            "Do not collect rainwater or filter debris.",
            "A diverter should collect rainwater and filter debris. Actually, don’t filter debris.",
        ):
            with self.subTest(text=text):
                self.assertEqual(infer_functions(text), ())
        brief = "Do not filter debris; collect rainwater and route the overflow."
        self.assertEqual(
            [m.name for m in infer_functions(brief)], ["Collect water", "Guide water"]
        )
        both = "The unit should not only collect rainwater but also filter debris."
        self.assertEqual(len(infer_functions(both)), 2)

    def test_honor_chart_ownership_and_handle_explicit_chart_requests(self):
        brief = "Show a chart for a drawing stand that supports a board and adjusts its tilt."
        self.assertEqual(len(infer_functions(brief)), 2)
        self.assertEqual(infer_functions(brief + " I will provide my own chart."), ())
        self.assertEqual(infer_functions(brief + " Please use my own chart."), ())
        self.assertEqual(
            infer_functions(brief + " Do not suggest mechanism options."), ()
        )
        self.assertEqual(
            infer_functions(
                "Collect rainwater: tank | tray\nFilter debris: sieve | mesh"
            ),
            (),
        )
        # Adding an author's chart row supplies no evidence for a second function.
        self.assertEqual(
            infer_functions("Collect rainwater.\nFilter debris: sieve | mesh"), ()
        )
