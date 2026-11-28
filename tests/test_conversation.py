"""Check learner questions against recorded outcomes and preserve subsequent task state."""

import json
import subprocess
import sys
from unittest import TestCase

from design_research.agents import LLMMessage, LLMRequest
from design_research_workshop import Rule, WorkshopPyBot, workshop_client


class ConversationTests(TestCase):
    def test_evidence_uses_recorded_run_and_sessions_are_independent(self):
        calls = []

        def run(turn):
            calls.append(turn.prompt)
            return {"measurement": len(calls)}

        rule = Rule("gage", "Take a gage reading.", ("gage",), run)
        bot = WorkshopPyBot([rule])
        first, other = bot.session(), bot.session()
        self.assertEqual(first.reply("Read the gage.").content["measurement"], 1)
        self.assertEqual(first.reply("Why?").content["evidence"]["measurement"], 1)
        self.assertEqual(
            first.reply("Show me the evidence.").content["evidence"]["measurement"], 1
        )
        self.assertIsNone(other.reply("Why?").rule)
        self.assertEqual(calls, ["Read the gage."])
        first.reset()
        self.assertIsNone(first.reply("Explain that.").rule)

    def test_sheller_why_does_not_erase_the_pending_revision(self):
        session = WorkshopPyBot().session()
        initial = session.reply(
            "Propose a peanut sheller with rubbing plates and a sieve."
        ).content
        report = session.reply("Review it for cleanout only.").content
        explanation = session.reply("Why was it rejected?").content
        self.assertEqual(
            explanation["evidence"]["revision_goals"], report["revision_goals"]
        )
        revised = session.reply("Revise it.").content
        diff = session.reply("What changed?").content["changes"]
        self.assertEqual(
            diff,
            {
                "cleaning_access": {
                    "before": initial["cleaning_access"],
                    "after": revised["cleaning_access"],
                }
            },
        )
        self.assertEqual(revised["crack"], initial["crack"])
        self.assertTrue(session.reply("Review the cleanout only.").content["approved"])

    def test_jar_why_round_trip_through_real_client(self):
        messages = [
            LLMMessage(
                role="user",
                content="Give me three one-handed jar opener ideas, function by function.",
            )
        ]
        with workshop_client() as client:
            result = client.generate(LLMRequest(messages=messages)).text
            messages += [
                LLMMessage(role="assistant", content=result),
                LLMMessage(role="user", content="Why did you choose that?"),
            ]
            explanation = json.loads(
                client.generate(LLMRequest(messages=messages)).text
            )
        self.assertEqual(explanation["evidence"]["mechanisms_in_first_concepts"], 9)
        self.assertEqual(
            explanation["evidence"]["selection_trace"], json.loads(result)["trace"]
        )

    def test_truss_comparison_subtracts_recorded_values_and_preserves_constraint(self):
        chat = WorkshopPyBot().session()
        first = chat.reply("Find the stiffest truss.").content
        second = chat.reply("Now cap it at 60 kg.").content
        reason = chat.reply("Why?").content
        self.assertEqual(reason["evidence"]["members"], second["members"])
        diff = chat.reply("What changed?").content["changes"]
        self.assertAlmostEqual(
            diff["mass_kg"]["delta"], second["mass_kg"] - first["mass_kg"]
        )
        self.assertLessEqual(
            chat.reply("Maximize the safety factor.").content["mass_kg"], 60
        )

    def test_constraint_explanation_and_strict_bound_comparison(self):
        chat = WorkshopPyBot().session()
        chat.reply("Design the lightest truss with at most 60 kg.")
        chat.reply("Use under 60 kg.")
        diff = chat.reply("What changed?").content["changes"]
        self.assertEqual(diff["strict_mass_cap"], {"before": False, "after": True})
        self.assertEqual(
            chat.reply("Minimize peak displacement instead.").content["objective"],
            "minimum deflection",
        )
        chat.reply("Require a safety factor of at least 2.2.")
        why = chat.reply("Why?").content
        self.assertIn("2.012", why["explanation"])
        self.assertIn("70.89 kg", why["explanation"])
        self.assertEqual(
            why["evidence"]["structurally_feasible_before_requested_bounds"], 6
        )

    def test_brief_clarification_accepts_two_functions_in_prose(self):
        chat = WorkshopPyBot().session()
        self.assertIn(
            "question", chat.reply("Brainstorm a downspout attachment.").content
        )
        result = chat.reply("I want to collect rainwater and filter debris.")
        self.assertEqual(result.rule, "morphology")
        self.assertEqual(
            set(result.content["chart"]), {"Collect water", "Filter debris"}
        )

    def test_answering_a_task_choice_keeps_the_original_count(self):
        chat = WorkshopPyBot().session()
        prompt = "Show me two ideas for a jar opener or a peanut sheller."
        self.assertFalse(chat.reply(prompt).matched)
        chosen = chat.reply("The jar opener, please.")
        self.assertEqual(chosen.rule, "jar-opener")
        self.assertEqual(len(chosen.content["concepts"]), 2)
        extra = chat.reply("Show me two more ideas.")
        self.assertFalse(
            any(
                idea in chosen.content["concepts"] for idea in extra.content["concepts"]
            )
        )

    def test_new_ideas_retain_history_across_an_explanation(self):
        chat = WorkshopPyBot().session()
        first = chat.reply(
            "Suggest four jar opener ideas, function by function."
        ).content["concepts"]
        chat.reply("Why?")
        second = chat.reply("Give me two more ideas.").content["concepts"]
        third = chat.reply("Show me three additional ideas.").content["concepts"]
        records = [tuple(row.items()) for row in first + second + third]
        self.assertEqual(len(set(records)), 9)

    def test_terminal_json_is_readable_by_programs(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "design_research_workshop.chat",
                "--json",
                "Give me two jar opener ideas.",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        message = json.loads(result.stdout)
        self.assertEqual(message["rule"], "jar-opener")
        self.assertEqual(len(message["content"]["concepts"]), 2)
