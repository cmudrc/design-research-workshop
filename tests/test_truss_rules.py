"""Check the rule-selected actions against the actual truss solver and grammar."""

import json
from unittest import TestCase

from design_research_workshop import workshop_client
from design_research_workshop.truss import (
    ADD,
    COMPARE,
    DEFLECTION,
    INSPECT,
    MASS,
    SAFETY,
    TrussBrief,
    TrussDesign,
    truss_agent,
    truss_reply,
)


class TrussRuleTests(TestCase):
    def test_goals_select_distinct_solver_evaluated_layouts(self):
        snapshot = TrussDesign(center_joint=True).compare({})
        layouts = snapshot["feasible_layouts"]
        self.assertEqual((snapshot["evaluated"], len(layouts)), (7, 6))
        light = truss_reply("Which truss weighs as little as possible?")
        safe = truss_reply("Give me the largest overall safety factor.")
        stiff = truss_reply("Find the lowest peak displacement.")
        self.assertEqual(light["mass_kg"], min(c["mass_kg"] for c in layouts))
        self.assertEqual(
            safe["factor_of_safety"], max(c["factor_of_safety"] for c in layouts)
        )
        self.assertEqual(
            stiff["max_deflection_mm"], min(c["max_deflection_mm"] for c in layouts)
        )
        self.assertLess(light["mass_kg"], safe["mass_kg"])
        self.assertLess(light["factor_of_safety"], safe["factor_of_safety"])

    def test_followups_change_one_bound_and_preserve_the_other(self):
        initial = truss_reply(
            "Find the stiffest truss with a safety factor of at least 2."
        )
        bounded = truss_reply("Now cap it at 60 kg.", previous=initial)
        self.assertEqual(bounded["objective"], DEFLECTION)
        self.assertEqual(bounded["min_fos"], 2.0)
        self.assertLessEqual(bounded["mass_kg"], 60)
        self.assertNotEqual(initial["members"], bounded["members"])
        infeasible = truss_reply(
            "Require a safety factor of at least 2.2.", previous=bounded
        )
        self.assertEqual(infeasible["status"], "no feasible completion")
        self.assertEqual(infeasible["max_mass_kg"], 60)
        unrestricted = truss_reply("Drop the mass limit.", previous=infeasible)
        self.assertIsNone(unrestricted["max_mass_kg"])
        self.assertEqual(unrestricted["min_fos"], 2.2)
        self.assertGreater(unrestricted["mass_kg"], 60)

    def test_unsupported_objectives_units_and_directions_prompt_for_clarification(self):
        prior = truss_reply("The lightest truss with at most 60 kg.")
        for request in (
            "maximize weight",
            "Design the lightest truss to resist a lateral load.",
            "Minimize truss mass while also maximizing fatigue life.",
            "The safest truss using aluminum members.",
            "Maximize the safety factor with an additional support.",
            "Choose the lightest truss with thicker sections.",
            "Design a lightweight symmetric truss.",
            "Design the lightest truss without a bottom chord.",
            "Minimize mass while also maximizing the number of paint colors.",
            "lightest truss with a -60 kg budget",
            "minimize safety factor",
            "minimize cost",
            "I'd like the greatest deflection",
            "at least as stiff",
            "minimum mass while maximizing safety factor",
            "lightest truss under 60 lb",
            "the safest span under -15 kg",
            "now cap it at 60",
            "maximize mass under 65 kg",
        ):
            with self.subTest(request=request):
                answer = truss_reply(request, previous=prior)
                self.assertEqual(answer["status"], "clarification needed")
                self.assertEqual(answer["objective"], MASS)
                self.assertEqual(answer["max_mass_kg"], 60)
                self.assertNotIn("members", answer)
        self.assertEqual(truss_reply("under 60 kg")["status"], "clarification needed")

    def test_limits_exclude_equality_only_when_requested(self):
        old = TrussBrief(SAFETY)
        layout = [
            {
                "members": [],
                "mass_kg": 60.0,
                "factor_of_safety": 2.0,
                "max_deflection_mm": 1.0,
            }
        ]
        for request in ("under 60 kg", "a factor of safety above 2.0"):
            self.assertIsNone(
                TrussBrief.from_text(request, previous=old).select(layout, set())
            )
        for request in ("at most 60 kg", "a factor of safety of at least 2.0"):
            self.assertEqual(
                TrussBrief.from_text(request, previous=old).select(layout, set()),
                layout[0],
            )

    def test_rejected_member_and_revised_plan_use_actual_tool_feedback(self):
        design = TrussDesign(center_joint=True)
        task = {"brief": "Design the safest truss.", "try_first": [0, 99]}
        with workshop_client() as client, truss_agent(client, design) as agent:
            result = agent.run(json.dumps(task))
        self.assertTrue(result.success, result.error)
        inspected, compared, rejected, recovered, *_ = result.tool_results
        self.assertEqual((inspected.tool_name, compared.tool_name), (INSPECT, COMPARE))
        self.assertEqual(rejected.tool_name, ADD)
        self.assertIn("Rejected", rejected.result["status"])
        self.assertEqual(rejected.result["members"], inspected.result["members"])
        self.assertEqual(recovered.result["members"], [[0, 1]])
        checked = design.grammar.evaluate(design.state)
        self.assertTrue(checked.is_feasible)
        self.assertEqual(
            sorted(design.inspect({})["members"]), result.final_output["members"]
        )
        self.assertAlmostEqual(checked.fos, result.final_output["factor_of_safety"])

    def test_conflicting_bounds_stop_before_editing_and_report_best_attainable_values(
        self,
    ):
        design = TrussDesign(center_joint=True)
        task = {
            "brief": "Find the stiffest truss under 60 kg with a minimum safety factor of 2.2.",
            "try_first": [0, 1],
        }
        with workshop_client() as client, truss_agent(client, design) as agent:
            result = agent.run(json.dumps(task))
        self.assertTrue(result.success, result.error)
        self.assertEqual([t.tool_name for t in result.tool_results], [INSPECT, COMPARE])
        self.assertEqual(design.state.members, ())
        report = result.final_output
        self.assertEqual(report["status"], "no feasible completion")
        layouts = result.tool_results[1].result["feasible_layouts"]
        affordable = [c["factor_of_safety"] for c in layouts if c["mass_kg"] < 60]
        strong_enough = [c["mass_kg"] for c in layouts if c["factor_of_safety"] >= 2.2]
        self.assertEqual(report["best_fos_within_mass_limit"], max(affordable))
        self.assertEqual(report["least_mass_kg_meeting_fos_floor"], min(strong_enough))
        self.assertLess(max(affordable), 2.2)
        self.assertGreater(min(strong_enough), 60)
        impossible = truss_reply(
            "Find the lightest truss under 20 kg with a safety factor of at least 3."
        )
        self.assertIsNone(impossible["best_fos_within_mass_limit"])
        self.assertIsNone(impossible["least_mass_kg_meeting_fos_floor"])
        too_light = truss_reply("Find the safest truss under 20 kg.")
        self.assertIsNone(too_light["best_fos_within_mass_limit"])
        self.assertEqual(
            too_light["least_mass_kg_meeting_fos_floor"],
            min(c["mass_kg"] for c in layouts),
        )

    def test_existing_edits_change_the_best_available_completion(self):
        design = TrussDesign(center_joint=True)
        lighter = truss_reply("Choose the lightest truss.")
        for start, end in [[0, 2], [1, 2]]:
            design.add({"start": start, "end": end})
        with workshop_client() as client, truss_agent(client, design) as agent:
            result = agent.run("Minimize the truss mass.")
        self.assertTrue(result.success, result.error)
        self.assertGreater(result.final_output["mass_kg"], lighter["mass_kg"])
        self.assertGreaterEqual(result.final_output["factor_of_safety"], 1)
        self.assertIn([0, 2], result.final_output["members"])
        self.assertIn([1, 2], result.final_output["members"])
