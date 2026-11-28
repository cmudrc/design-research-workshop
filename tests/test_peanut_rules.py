"""Exercise the critique loop, natural requests, and the reusable provision rubric."""

import json
from dataclasses import asdict, replace
from unittest import TestCase

from design_research_workshop import workshop_client
from design_research_workshop.peanut import (
    CLEANING,
    DEFAULT_CONCEPT,
    FIXED_CLEANING,
    FIXED_FEED,
    FIXED_PROTECTION,
    PROTECTIONS,
    respond_sheller,
    sheller_pattern,
    sheller_task,
)
from design_research_workshop.rubric import (
    ProvisionCriterion,
    apply_revision_goals,
    review_provisions,
)


class ShellerRuleTests(TestCase):
    def test_budget_makes_priority_change_the_second_draft(self):
        with workshop_client() as client:
            for priority, changed_field, unchanged_field in [
                ("cleaning access", "cleaning_access", "kernel_protection"),
                ("kernel protection", "kernel_protection", "cleaning_access"),
            ]:
                with self.subTest(priority=priority):
                    run = sheller_pattern(client, max_iterations=2).run(
                        sheller_task(DEFAULT_CONCEPT, priority, changes_per_revision=1)
                    )
                    self.assertFalse(run.approved)
                    first, second = run.critique_iterations
                    self.assertEqual(
                        json.loads(first["proposal"]), asdict(DEFAULT_CONCEPT)
                    )
                    self.assertEqual(len(first["revision_goals"]), 1)
                    final = json.loads(second["proposal"])
                    self.assertNotEqual(
                        final[changed_field], getattr(DEFAULT_CONCEPT, changed_field)
                    )
                    self.assertEqual(
                        final[unchanged_field],
                        getattr(DEFAULT_CONCEPT, unchanged_field),
                    )

    def test_multiple_rounds_retain_earlier_changes_and_pass_existing_provisions(self):
        seed = replace(DEFAULT_CONCEPT, kernel_protection="adjustable gap")
        with workshop_client() as client:
            run = sheller_pattern(client, max_iterations=4).run(
                sheller_task(
                    seed,
                    "feed control",
                    criteria=("kernel protection", "cleaning access", "feed control"),
                    changes_per_revision=1,
                )
            )
            already_met = sheller_pattern(client).run(
                sheller_task(seed, criteria=("kernel protection",))
            )
        self.assertTrue(run.approved)
        self.assertTrue(already_met.approved)
        self.assertEqual(already_met.iterations, 1)
        self.assertEqual(run.iterations, 3)
        intermediate = json.loads(run.critique_iterations[1]["proposal"])
        final = json.loads(run.critique_iterations[2]["proposal"])
        self.assertEqual(final["kernel_protection"], "adjustable gap")
        self.assertEqual(intermediate["feed_control"], final["feed_control"])
        self.assertNotEqual(intermediate["feed_control"], FIXED_FEED)
        self.assertEqual(intermediate["cleaning_access"], FIXED_CLEANING)
        self.assertEqual(final["cleaning_access"], CLEANING[seed.separate])

    def test_conversational_review_reuses_proposal_and_revision_targets_requested_field(
        self,
    ):
        seed = respond_sheller(
            "Propose a sheller with rub plates and a tilted screen.", role="propose"
        )
        review = respond_sheller(
            "Review it for cleanout only.", role="critic", previous=seed
        )
        self.assertFalse(review["approved"])
        self.assertEqual(review["concept"], seed)
        revised = respond_sheller("Revise it.", role="propose", previous=review)
        self.assertEqual(revised["cleaning_access"], CLEANING["Tilted screen"])
        self.assertEqual(revised["kernel_protection"], FIXED_PROTECTION)
        self.assertEqual(revised["crack"], "Sliding rub plates")
        self.assertTrue(
            respond_sheller("Review cleaning only", role="critic", previous=revised)[
                "approved"
            ]
        )

    def test_critic_distinguishes_an_existing_provision_from_a_request_or_absence(self):
        review = respond_sheller(
            "Review a roller sheller with an adjustable gap and removable sieve and tray.",
            role="critic",
        )
        self.assertTrue(review["approved"])
        for text in (
            "It lacks an adjustable gap and has a removable sieve and tray.",
            "An adjustable gap is missing but it has a removable sieve and tray.",
        ):
            with self.subTest(text=text):
                later = respond_sheller(text, role="critic", previous=review)
                self.assertFalse(later["approved"])
                self.assertEqual(
                    later["concept"]["cleaning_access"], "removable sieve and tray"
                )
                self.assertEqual(
                    later["concept"]["kernel_protection"], FIXED_PROTECTION
                )
        later = respond_sheller(
            "Evaluate a new sheller that needs an adjustable roller gap.", role="critic"
        )
        self.assertFalse(later["approved"])
        self.assertEqual(later["concept"]["kernel_protection"], FIXED_PROTECTION)

    def test_changing_a_mechanism_triggers_a_new_review_of_its_provision(self):
        proposal = respond_sheller(
            "Propose a roller sheller that keeps kernels intact.", role="propose"
        )
        self.assertEqual(
            proposal["kernel_protection"], PROTECTIONS["Hand-cranked rollers"]
        )
        review = respond_sheller(
            "Now use a hinged press; review kernel protection only.",
            role="critic",
            previous=proposal,
        )
        self.assertFalse(review["approved"])
        self.assertEqual(review["concept"]["kernel_protection"], FIXED_PROTECTION)
        revised = respond_sheller("Revise it.", role="propose", previous=review)
        self.assertEqual(
            revised["kernel_protection"], PROTECTIONS["Hinged lever press"]
        )

    def test_excluding_an_unused_mechanism_preserves_the_design(self):
        press = respond_sheller(
            "Propose a peanut sheller with a lever press and a sieve.", role="propose"
        )
        same = respond_sheller(
            "Revise it without rollers.", role="propose", previous=press
        )
        self.assertEqual(same, press)

    def test_an_exclusion_replaces_an_excluded_mechanism_and_resets_its_specific_provision(
        self,
    ):
        rollers = respond_sheller(
            "Propose a roller sheller that protects whole kernels.", role="propose"
        )
        revised = respond_sheller(
            "Revise it without rollers or a sieve.", role="propose", previous=rollers
        )
        self.assertNotEqual(revised["crack"], rollers["crack"])
        self.assertNotEqual(revised["separate"], rollers["separate"])
        self.assertEqual(revised["kernel_protection"], FIXED_PROTECTION)
        self.assertEqual(revised["feed"], rollers["feed"])
        with self.assertRaisesRegex(ValueError, "Which crack mechanism"):
            respond_sheller(
                "Review it without rollers.", role="critic", previous=rollers
            )

    def test_replacement_phrases_use_the_named_substitute(self):
        for request in [
            "Use a lever press instead of rollers.",
            "Instead of rollers, use a lever press.",
            "Replace the rollers with a lever press.",
            "Switch from rollers to a lever press.",
        ]:
            with self.subTest(request=request):
                revised = respond_sheller(request, role="propose")
                self.assertEqual(revised["crack"], "Hinged lever press")
        with self.assertRaisesRegex(ValueError, "Allow at least one crack mechanism"):
            respond_sheller(
                "Exclude rollers, the lever press and rub plates.", role="propose"
            )

    def test_unsupported_criterion_keeps_an_otherwise_qualified_design_unapproved(self):
        seed = respond_sheller(
            "Propose a peanut sheller with an adjustable roller gap and removable sieve and tray.",
            role="propose",
        )
        for prompt, missing in [
            ("Review it for running torque only.", "running torque"),
            ("Review throughput.", "throughput"),
            (
                "Please check my sheller for cleaning access and bearing life.",
                "bearing life",
            ),
        ]:
            with self.subTest(prompt=prompt):
                review = respond_sheller(prompt, role="critic", previous=seed)
                self.assertFalse(review["approved"])
                self.assertIn(missing, review["unassessed"])
                self.assertEqual(review["revision_goals"], [])

    def test_unrecognized_design_detail_does_not_count_as_kernel_protection(self):
        seed = replace(
            DEFAULT_CONCEPT, kernel_protection="Blue rollers with a glossy coating."
        )
        with workshop_client() as client:
            outcome = sheller_pattern(client, max_iterations=1).run(
                sheller_task(seed, criteria=("kernel protection",))
            )
        self.assertFalse(outcome.approved)
        self.assertIn("limit compression", outcome.critique_iterations[0]["feedback"])

    def test_alternate_evidence_phrasing_and_a_contraction_in_a_correction(self):
        review = respond_sheller(
            "Assess my peanut sheller. The roller spacing can be adjusted; the sieve and tray lift out.",
            role="critic",
        )
        self.assertTrue(review["approved"])
        correction = respond_sheller(
            "Correction: it doesn't have an adjustable gap after all.",
            role="critic",
            previous=review,
        )
        self.assertFalse(correction["approved"])
        self.assertEqual(correction["concept"]["kernel_protection"], FIXED_PROTECTION)

    def test_fixed_features_and_dropped_review_criteria_override_previous_turn(self):
        seed = respond_sheller(
            "Propose a sheller with an adjustable gap and removable sieve and tray.",
            role="propose",
        )
        for request, field, default in [
            (
                "Review it. Actually the rollers have a fixed gap.",
                "kernel_protection",
                FIXED_PROTECTION,
            ),
            (
                "Review it. Cleaning access is missing.",
                "cleaning_access",
                FIXED_CLEANING,
            ),
        ]:
            with self.subTest(request=request):
                review = respond_sheller(request, role="critic", previous=seed)
                self.assertFalse(review["approved"])
                self.assertEqual(review["concept"][field], default)
        review = respond_sheller(
            "Review it. Skip kernel protection.", role="critic", previous=seed
        )
        self.assertEqual(review["criteria"], ["cleaning access"])
        review = respond_sheller(
            "Review it. Don't prioritize cleaning access.", role="critic", previous=seed
        )
        self.assertEqual(review["criteria"], ["kernel protection", "cleaning access"])

    def test_bad_controls_report_the_mistake_at_the_call_site(self):
        for options in [
            {"criteria": "feed control"},
            {"criteria": ()},
            {"criteria": ("feed control",), "priority": "cleaning access"},
            {"changes_per_revision": 0},
            {"changes_per_revision": True},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                sheller_task(DEFAULT_CONCEPT, **options)
        with self.assertRaisesRegex(ValueError, "Choose one crack mechanism"):
            respond_sheller("Compare a hinged press and rollers.", role="critic")

    def test_the_same_rubric_can_review_and_revise_another_engineering_task(self):
        restraint = ProvisionCriterion(
            "Cable retention",
            "retention",
            "mount",
            "Specify a feature to retain the cord.",
            {"wall rail": "snap-in clip"},
            {"wall rail": ("spring clip",)},
        )
        fit = ProvisionCriterion(
            "Cable fit",
            "fit",
            "mount",
            "Allow adjustment for different cable diameters.",
            {"wall rail": "indexed width adjuster"},
            {"wall rail": ()},
        )
        criteria = [fit, restraint]
        seed = {
            "mount": "wall rail",
            "retention": "open ledge",
            "fit": "fixed width",
            "color": "blue",
        }
        rejected = review_provisions(seed, criteria, changes_per_revision=1)
        self.assertFalse(rejected["approved"])
        first = apply_revision_goals(seed, criteria, rejected["revision_goals"])
        next_review = review_provisions(first, criteria, changes_per_revision=1)
        revised = apply_revision_goals(seed, criteria, next_review["revision_goals"])
        self.assertTrue(
            review_provisions(revised, criteria, changes_per_revision=1)["approved"]
        )
        self.assertEqual(first["fit"], revised["fit"])
        self.assertEqual(revised["color"], "blue")
        self.assertEqual(seed["retention"], "open ledge")
