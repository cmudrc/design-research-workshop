"""Check the exercise behaviors beyond the default notebook screenshots."""

import json
from dataclasses import asdict
from unittest import TestCase

from design_research_workshop import workshop_client
from design_research_workshop.peanut import (
    DIRECT,
    FUNCTION_FIRST,
    ShellerConcept,
    run_brainstorm,
    sheller_pattern,
    sheller_task,
)
from design_research_workshop.truss import TrussDesign, truss_agent


class AgentDemoTests(TestCase):
    def test_framing_changes_mechanism_coverage_and_concept_survives_handoff(self):
        with workshop_client() as client:
            direct = run_brainstorm(client, DIRECT)
            framed = run_brainstorm(client, FUNCTION_FIRST)
            self.assertEqual(len(set(direct)), 6)
            self.assertEqual(len(set(framed)), 6)
            self.assertGreater(
                len({concept.separate for concept in framed}),
                len({concept.separate for concept in direct}),
            )
            transferred = ShellerConcept(**json.loads(json.dumps(asdict(framed[1]))))
            result = sheller_pattern(client).run(
                sheller_task(transferred, "cleaning access")
            )
        rejected, approved = result.critique_iterations
        self.assertFalse(rejected["approved"])
        self.assertTrue(rejected["feedback"].startswith("Cleaning access"))
        self.assertTrue(approved["approved"])
        revised = ShellerConcept(**json.loads(approved["proposal"]))
        self.assertEqual(
            (revised.feed, revised.crack, revised.separate),
            (transferred.feed, transferred.crack, transferred.separate),
        )
        self.assertNotEqual(revised.kernel_protection, transferred.kernel_protection)
        self.assertNotEqual(revised.cleaning_access, transferred.cleaning_access)

    def test_react_uses_legal_observations_after_a_rejected_action(self):
        target = [[0, 2], [1, 2]]
        design = TrussDesign()
        with workshop_client() as client, truss_agent(client, design) as agent:
            result = agent.run(
                json.dumps({"target_members": target, "try_first": [0, 99]})
            )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.final_output, {"members": target, "missing": []})
        rejected = result.tool_results[1].result
        self.assertIn("Rejected 0–99", rejected["status"])
        self.assertEqual(rejected["members"], [])
        self.assertEqual(result.tool_results[2].result["members"], [[0, 2]])
