"""Exercise chart search with the workshop example and a previously unseen brief."""

from unittest import TestCase

from design_research_workshop.brainstorm import (
    DesignBrief,
    JAR_OPENER,
    generate_concepts,
    respond,
)


class BrainstormTests(TestCase):
    def test_prompt_framing_and_independently_counted_coverage(self):
        direct = respond("I need five ideas for opening jars with one hand.")
        decomposed = respond(
            "Suggest five one-handed jar openers, function by function."
        )
        self.assertEqual(direct["strategy"], "direct")
        self.assertEqual(decomposed["strategy"], "function-first")
        self.assertEqual(len(direct["concepts"]), 5)
        self.assertEqual(len(decomposed["concepts"]), 5)
        for result in (direct, decomposed):
            self.assertEqual(result["available"], 27)
            self.assertEqual(len({tuple(c.items()) for c in result["concepts"]}), 5)
            observed = set()
            for concept, step in zip(result["concepts"], result["trace"], strict=True):
                novel = [
                    fn for fn, option in concept.items() if (fn, option) not in observed
                ]
                self.assertEqual(step["new_functions"], novel)
                observed.update(concept.items())

        def options_in_first_three(result):
            return {item for c in result["concepts"][:3] for item in c.items()}

        self.assertLess(
            len(options_in_first_three(direct)), len(options_in_first_three(decomposed))
        )
        self.assertEqual(len(options_in_first_three(decomposed)), 9)

    def test_frozen_mechanism_and_seeded_ties(self):
        prompt = (
            "Give me six function-first jar opener concepts. Keep the lever-lock strap."
        )
        ideas = respond(prompt)
        self.assertEqual(ideas["available"], 9)
        self.assertTrue(
            all(c["Grip the lid"] == "Lever-lock strap" for c in ideas["concepts"])
        )
        self.assertEqual(ideas, respond(prompt + " Seed: 7"))
        self.assertNotEqual(
            ideas["concepts"], respond(prompt + " Seed: 13")["concepts"]
        )
        with self.assertRaisesRegex(ValueError, "one fixed option"):
            respond(prompt + " Keep the spring-loaded jaws.")

    def test_unseen_design_brief_with_mixed_axis_sizes_and_shared_option_names(self):
        book_return = DesignBrief(
            title="After-hours book return",
            statement="Receive and store returned books while the library is closed.",
            functions={
                "Receive": ("Rollers", "Belt"),
                "Store": ("Bin", "Shelf", "Belt", "Cart"),
            },
        )
        result = generate_concepts(book_return, "List four combinations by function.")
        self.assertEqual(result["brief"], "After-hours book return")
        self.assertEqual(result["available"], 8)
        self.assertEqual(
            {idea["Store"] for idea in result["concepts"]},
            set(book_return.functions["Store"]),
        )
        self.assertEqual(
            {idea["Receive"] for idea in result["concepts"]},
            set(book_return.functions["Receive"]),
        )
        single = DesignBrief("One choice", "Place one object.", {"Place": ["Shelf"]})
        self.assertEqual(
            generate_concepts(single, "Offer ideas.")["concepts"], [{"Place": "Shelf"}]
        )

    def test_chart_and_counts_validated_before_expanding_candidates(self):
        bad_charts = ({}, {"Grip": []}, {"Grip": "Claws"}, {"Grip": ["Pin", "pin"]})
        for chart in bad_charts:
            with self.subTest(chart=chart), self.assertRaises(ValueError):
                DesignBrief("Example", "Open a jar.", chart)
        for prompt in ("Suggest zero ideas.", "Concept count: -2", "Give me 25 ideas."):
            with (
                self.subTest(prompt=prompt),
                self.assertRaisesRegex(ValueError, "1 to 24"),
            ):
                respond(prompt)
        oversized = DesignBrief(
            "Large", "Combine choices.", {str(i): ["A", "B"] for i in range(13)}
        )
        with self.assertRaisesRegex(ValueError, "4096"):
            generate_concepts(oversized, "Generate some ideas.")
        with self.assertRaisesRegex(ValueError, "integer seed"):
            generate_concepts(JAR_OPENER, "Suggest concepts.", seed=0.5)

    def test_natural_revisions_and_exclusions(self):
        initial = "Think of five ways to twist off a jar lid with one hand."
        self.assertEqual(len(respond(initial)["concepts"]), 5)
        follow_up = initial + "\nCan I see more variety? Try four instead. Seed 11."
        revised = respond(follow_up)
        self.assertEqual((revised["strategy"], revised["seed"]), ("function-first", 11))
        self.assertEqual(len(revised["concepts"]), 4)
        narrowed = (
            follow_up + "\nKeep the ratcheting arm.\nNow switch to a T handle instead."
        )
        result = respond(narrowed)
        self.assertEqual({c["Turn the lid"] for c in result["concepts"]}, {"T-handle"})
        excluded = respond(
            narrowed + "\nDo not use a T-handle. Could you narrow that to two?"
        )
        self.assertEqual(len(excluded["concepts"]), 2)
        self.assertEqual(excluded["available"], 18)
        self.assertTrue(
            all(c["Turn the lid"] != "T-handle" for c in excluded["concepts"])
        )
        with self.assertRaisesRegex(ValueError, "mechanism from the chart"):
            respond("Please keep the electric drive.")
        with self.assertRaisesRegex(ValueError, "Allow at least one"):
            respond(
                "Avoid the T-handle. Avoid the offset lever. Avoid the ratcheting arm."
            )

    def test_joined_options_and_typographic_apostrophe(self):
        only_offset = respond(
            "Give me three ideas. Don’t use the T-handle. "
            "Avoid the ratcheting arm and the T-handle."
        )
        self.assertEqual(only_offset["available"], 9)
        self.assertEqual(
            {c["Turn the lid"] for c in only_offset["concepts"]}, {"Offset lever"}
        )
        with self.assertRaisesRegex(ValueError, "one fixed option"):
            respond("Keep the T-handle and the ratcheting arm.")

    def test_framing_negation_and_unknown_standalone_option(self):
        prompt = "Give me five one-handed jar opener ideas."
        self.assertEqual(
            respond(prompt + " Do not prioritize coverage.")["strategy"], "direct"
        )
        self.assertEqual(
            respond(prompt + " Do not use a direct prompt.")["strategy"],
            "function-first",
        )
        with self.assertRaisesRegex(ValueError, "mechanism from the chart"):
            respond(prompt + " Use a hydraulic piston.")
        self.assertEqual(
            respond(prompt + " Use a function-first framing.")["strategy"],
            "function-first",
        )

    def test_catalog_ideation_brief_with_instructor_options(self):
        from design_research.problems import get_problem
        from design_research_workshop import WorkshopPyBot, brainstorm_rule

        problem_id = "ideation_hands_free_smartphone_viewing"
        options = {
            "Retain phone": ("Slot", "Edge clips"),
            "Set viewing angle": ("Fixed wedge", "Indexed strut"),
        }
        brief = DesignBrief.from_problem(problem_id, options)
        problem = get_problem(problem_id)
        self.assertEqual(brief.title, problem.metadata.title)
        self.assertEqual(brief.statement, problem.statement_markdown)
        rule = brainstorm_rule(brief, name="phone-viewer", topics=("phone viewer",))
        reply = (
            WorkshopPyBot()
            .with_rules(rule)
            .reply("Suggest three phone viewer ideas by function.")
        )
        self.assertEqual(reply.rule, "phone-viewer")
        self.assertEqual(reply.content["problem_id"], problem_id)
        self.assertEqual(len(reply.content["concepts"]), 3)
        self.assertEqual(set(reply.content["concepts"][0]), set(options))
        standalone = DesignBrief("Local", "Set a viewing angle.", options)
        self.assertNotIn(
            "problem_id", generate_concepts(standalone, "Give me one idea.")
        )
        for invalid in ("ideation_not_in_the_catalog", "planar_truss_span"):
            with self.subTest(problem_id=invalid), self.assertRaises(ValueError):
                DesignBrief.from_problem(invalid, options)
