"""Exercise the package through the same client used by the notebooks."""

import json
from unittest import TestCase

from design_research.agents import LLMMessage, LLMRequest

from design_research_workshop import laptop_propose_critic, workshop_client
from design_research_workshop.laptop import LaptopPolicy


class ProxyTests(TestCase):
    def test_price_change_produces_different_catalog_candidate(self) -> None:
        with workshop_client() as client:
            requests = [
                LLMRequest(
                    model=client.default_model(),
                    messages=(LLMMessage(role="user", content=prompt),),
                )
                for prompt in (
                    "Choose a student laptop configuration as JSON with a price ceiling of $1,000.",
                    "Choose a student laptop configuration as JSON under $1,000.",
                )
            ]
            candidates = [
                json.loads(client.generate(request).text) for request in requests
            ]
        self.assertEqual([candidate["z5"] for candidate in candidates], [10.0, 7.5])

    def test_critique_changes_only_one_constrained_attribute(self) -> None:
        task = "Choose a student laptop configuration as JSON with maximum weight of 3 pounds."
        with workshop_client() as client:
            result = laptop_propose_critic(client).run(task)
        self.assertTrue(result.success and result.approved, result.terminated_reason)
        first, last = result.critique_iterations
        self.assertFalse(first["approved"])
        self.assertIn("Weight", first["feedback"])
        first_candidate = json.loads(first["proposal"])
        last_candidate = json.loads(last["proposal"])
        self.assertEqual(last_candidate, {**first_candidate, "z4": 2.5})

    def test_first_proposal_already_satisfies_relaxed_limits(self) -> None:
        task = "Choose a student laptop configuration as JSON with at least 6 hours of battery life."
        with workshop_client() as client:
            result = laptop_propose_critic(client).run(task)
        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 1)

    def test_infeasible_and_out_of_scope_requests_are_rejected(self) -> None:
        policy = LaptopPolicy()
        for prompt in (
            "Choose a wind turbine configuration as JSON.",
            "Choose a student laptop design as JSON with at least 9 hours of battery life.",
        ):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                policy.complete(prompt)
