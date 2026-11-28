"""Exercise responder selection, conversation state, and transfers to new tasks."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from unittest import TestCase
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from design_research.agents import LLMMessage, LLMRequest

from design_research_workshop import (
    DesignBrief,
    Rule,
    Turn,
    WorkshopPyBot,
    brainstorm_rule,
    workshop_client,
)


from design_research_workshop.server import LEGACY_MODEL, MODEL, WorkshopServer


def conversation(bot, *prompts):
    history = []
    replies = []
    for prompt in prompts:
        reply = bot.reply(prompt, history=history)
        history.extend(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(reply.content)},
            )
        )
        replies.append(reply)
    return replies


def call(client, prompt):
    response = client.generate(
        LLMRequest(messages=(LLMMessage(role="user", content=prompt),))
    )
    return json.loads(response.text)


class WorkshopPyBotTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = WorkshopPyBot()

    def test_paraphrase_counts_and_topic_change(self):
        first, varied, different = conversation(
            self.bot,
            "Give me four ideas for a jar opener.",
            "Make it three, function by function.",
            "Review a peanut sheller with a lever press and hand-shaken sieve.",
        )
        self.assertEqual(len(first.content["concepts"]), 4)
        self.assertEqual(
            (varied.rule, varied.content["strategy"]), ("jar-opener", "function-first")
        )
        self.assertEqual(len(varied.content["concepts"]), 3)
        self.assertEqual(different.rule, "peanut-sheller")
        self.assertIn("press", different.content["concept"]["crack"].lower())

    def test_sheller_review_and_revision_use_prior_design(self):
        proposed, review, revision = conversation(
            self.bot,
            "Propose a peanut sheller with rubbing plates and a sieve.",
            "Review it, especially the cleanout. Give me one change.",
            "Revise it using the critique.",
        )
        self.assertFalse(review.content["approved"])
        self.assertIn("cleaning access", review.content["revision_goals"][0].lower())
        self.assertEqual(revision.content["crack"], proposed.content["crack"])
        self.assertEqual(
            revision.content["kernel_protection"], proposed.content["kernel_protection"]
        )
        self.assertNotEqual(
            revision.content["cleaning_access"], proposed.content["cleaning_access"]
        )

    def test_structured_envelopes_without_system_and_trailing_schema_instruction(self):
        tree = self.bot.complete(
            [
                {
                    "role": "user",
                    "content": 'Task JSON: {"target_members": [[0, 2]]}\nMemory JSON: []',
                },
                {
                    "role": "system",
                    "content": "Return an object following the supplied schema.",
                },
            ]
        )
        self.assertEqual(tree.rule, "planar-truss")
        self.assertEqual(tree.content["tool_name"], "truss.inspect")

    def test_unsupported_and_competing_topics_yield_clarification(self):
        unknown = self.bot.reply("Read me a sonnet.")
        ambiguous = self.bot.reply("Should we try the peanut sheller or a jar opener?")
        self.assertFalse(unknown.matched)
        self.assertIn("morphology", unknown.content["tasks"])
        self.assertFalse(ambiguous.matched)
        self.assertEqual(set(ambiguous.candidates), {"jar-opener", "peanut-sheller"})
        chosen = self.bot.reply("Give me three ideas.", rule="jar-opener")
        self.assertEqual(
            (chosen.rule, len(chosen.content["concepts"])), ("jar-opener", 3)
        )
        with self.assertRaisesRegex(ValueError, "Unknown workshop rule"):
            self.bot.reply("Hello", rule="unregistered")

    def test_fresh_chart_then_follow_up(self):
        ask, first, second = conversation(
            self.bot,
            "Brainstorm a wet umbrella stand for a library.",
            "Support: hooks | slots\nCollect drips: tray | mat",
            "Give me two ideas, function by function.",
        )
        self.assertIn("question", ask.content)
        self.assertEqual(first.rule, "morphology")
        self.assertEqual(first.content["available"], 4)
        self.assertIn("umbrella", first.content["brief"])
        self.assertEqual(len(second.content["concepts"]), 2)
        self.assertEqual(second.content["strategy"], "function-first")
        for axis in ("Support", "Collect drips"):
            self.assertEqual(len({c[axis] for c in second.content["concepts"]}), 2)

    def test_unseen_brief_can_use_catalog_functions_and_override_one_row(self):
        inferred, edited = conversation(
            self.bot,
            "Brainstorm a garden downspout attachment to collect rainwater and filter debris.",
            "Filter debris: removable basket | coarse screen",
        )
        self.assertEqual(inferred.rule, "morphology")
        original = inferred.content["chart"]
        self.assertEqual(
            original["Collect water"]["matched_phrase"], "collect rainwater"
        )
        self.assertEqual(original["Filter debris"]["source"], "workshop catalog")
        chart = edited.content["chart"]
        self.assertEqual(chart["Collect water"], original["Collect water"])
        self.assertEqual(chart["Filter debris"]["source"], "participant")
        self.assertEqual(edited.content["available"], 6)
        self.assertEqual(
            {idea["Filter debris"] for idea in edited.content["concepts"]},
            {"removable basket", "coarse screen"},
        )

    def test_count_in_a_new_brief_does_not_carry_over_the_previous_chart(self):
        _, switched, switched_again = conversation(
            self.bot,
            "Suggest three one-handed jar opener concepts.",
            "Now brainstorm a downspout attachment that collects rainwater and filters debris. Give me two concepts.",
            "Now brainstorm a fixture that aligns a blank and clamps the workpiece. Give me two concepts.",
        )
        self.assertEqual(switched.rule, "morphology")
        self.assertEqual(switched.content["available"], 9)
        self.assertIn("Collect water", switched.content["chart"])
        self.assertEqual(len(switched.content["concepts"]), 2)
        self.assertEqual(switched_again.rule, "morphology")
        self.assertNotIn("Collect water", switched_again.content["chart"])

    def test_generic_charts_are_isolated_on_switch_and_across_requests(self):
        _, switch = conversation(
            self.bot,
            "Brief: Fixture\nSupport: hooks | pegs",
            "Brief: Cart\nMove: wheels | slides",
        )
        self.assertEqual(
            switch.content["concepts"], [{"Move": "wheels"}, {"Move": "slides"}]
        )
        prompts = [
            "Brief: A\nGrip: clamp | latch",
            "Brief: B\nDrain: tray | hose",
        ] * 3
        with ThreadPoolExecutor(max_workers=3) as pool:
            result = list(pool.map(self.bot.reply, prompts))
        self.assertEqual([r.content["brief"] for r in result], ["A", "B"] * 3)

    def test_register_a_new_chart_and_a_different_kind_of_responder(self):
        cart = brainstorm_rule(
            DesignBrief(
                "Book cart",
                "Move books between shelves.",
                {
                    "Support": ("shelf", "bin"),
                    "Steer": ("handle", "tiller"),
                },
            ),
            name="book-cart",
            topics=("book cart",),
        )

        def count_trials(turn: Turn):
            design = json.loads(turn.prompt)
            return {"trials": len(design["variants"]) * design["replicates"]}

        trial_rule = Rule(
            name="trial-count",
            description="Count trials in a replicated experiment.",
            topics=(),
            respond=count_trials,
            protocol=lambda prompt: prompt.startswith("{") and '"replicates"' in prompt,
        )
        extended = self.bot.with_rules(cart, trial_rule)
        self.assertEqual(
            extended.reply("Brainstorm two book cart ideas.").rule, "book-cart"
        )
        self.assertEqual(
            self.bot.reply("Brainstorm two book cart ideas.").rule, "morphology"
        )
        prompt = '{"variants": ["A", "B", "C"], "replicates": 4}'
        direct = extended.reply(prompt)
        self.assertEqual((direct.rule, direct.content), ("trial-count", {"trials": 12}))
        with workshop_client(bot=extended) as client:
            self.assertEqual(call(client, prompt), direct.content)
            self.assertEqual(
                len(call(client, "Give me two book cart concepts.")["concepts"]), 2
            )
        with self.assertRaisesRegex(ValueError, "unique name"):
            extended.with_rules(cart)

    def test_explicit_instruction_beats_overlapping_topic_and_chart_text(self):
        keep = Rule(
            "prompt-test",
            "Echo a controlled prompt.",
            (),
            lambda t: {"echo": t.prompt},
            systems=("Echo.",),
        )
        bot = self.bot.with_rules(keep)
        reply = bot.complete(
            [
                {"role": "system", "content": "Echo."},
                {
                    "role": "user",
                    "content": "Brainstorm a jar opener.\nHold: clamp | cradle",
                },
            ]
        )
        self.assertEqual(
            (reply.rule, reply.reason), ("prompt-test", "system instruction")
        )

    def test_follow_up_constraints_then_switch_to_a_new_unnamed_task(self):
        first, narrowed, excluded, new, unknown = conversation(
            self.bot,
            "Give me ideas for opening a jar using one hand.",
            "Could you narrow that to two?",
            "Please avoid the ratcheting arm.",
            "Give me three ideas for bookshelves.",
            "What is the capital of Denmark?",
        )
        self.assertEqual(first.rule, "jar-opener")
        self.assertEqual(len(narrowed.content["concepts"]), 2)
        self.assertTrue(
            all(
                idea["Turn the lid"] != "Ratcheting arm"
                for idea in excluded.content["concepts"]
            )
        )
        self.assertEqual(new.rule, "morphology")
        self.assertIn("question", new.content)
        self.assertIsNone(unknown.rule)

    def test_accessory_brief_does_not_trigger_a_catalog_choice(self):
        accessory = self.bot.reply("Brainstorm a laptop stand for a studio.")
        self.assertEqual(accessory.rule, "morphology")
        catalog = self.bot.reply("Recommend a laptop configuration under $1,000.")
        self.assertEqual(catalog.rule, "laptop")
        self.assertLess(catalog.content["z5"], 10)
        prompt = "Pick a laptop design as JSON under $1,000."
        with workshop_client() as client:
            legacy = client.generate(
                LLMRequest(
                    model=LEGACY_MODEL,
                    messages=[LLMMessage(role="user", content=prompt)],
                )
            )
        self.assertEqual(json.loads(legacy.text), self.bot.reply(prompt).content)

    def test_new_truss_constraint_keeps_the_requested_objective(self):
        first, limited = conversation(
            self.bot,
            "Find the stiffest truss.",
            "Now cap it at 60 kg.",
        )
        self.assertEqual(first.content["objective"], "minimum deflection")
        self.assertEqual(limited.content["objective"], first.content["objective"])
        self.assertLessEqual(limited.content["mass_kg"], 60)
        self.assertNotEqual(first.content["members"], limited.content["members"])

    def test_http_model_discovery_legacy_alias_and_malformed_input(self):
        with WorkshopServer(0, bot=self.bot) as server:
            thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/v1"

            def post(messages, model=MODEL):
                body = json.dumps({"model": model, "messages": messages}).encode()
                with urlopen(
                    Request(
                        url + "/chat/completions",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    )
                ) as response:
                    return json.load(response)

            try:
                with urlopen(url + "/models") as response:
                    self.assertEqual(json.load(response)["data"][0]["id"], MODEL)
                reply = post(
                    [{"role": "user", "content": "Choose a student laptop as JSON."}],
                    LEGACY_MODEL,
                )
                self.assertEqual(reply["model"], LEGACY_MODEL)
                self.assertEqual(reply["workshop"]["rule"], "laptop")
                self.assertIn(
                    "z4", json.loads(reply["choices"][0]["message"]["content"])
                )
                unknown = [
                    {"role": "user", "content": "What is the capital of Denmark?"}
                ]
                question = post(unknown)
                self.assertIn(
                    "question", json.loads(question["choices"][0]["message"]["content"])
                )
                self.assertIsNone(question["workshop"]["rule"])
                with self.assertRaises(HTTPError) as legacy:
                    post(unknown, LEGACY_MODEL)
                self.assertEqual(legacy.exception.code, 400)
                legacy.exception.close()
                for messages in (
                    None,
                    [],
                    [{"role": "user", "content": 4}],
                ):
                    with (
                        self.subTest(messages=messages),
                        self.assertRaises(HTTPError) as caught,
                    ):
                        post(messages)
                    self.assertEqual(caught.exception.code, 400)
                    self.assertTrue(json.load(caught.exception)["error"]["message"])
            finally:
                server.shutdown()
                thread.join()

    def test_validation(self):
        for messages in (
            [],
            [{"role": "user", "content": ""}],
            [{"role": "unknown", "content": "Hi"}],
            [{"role": "assistant", "content": "answer"}],
        ):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.bot.complete(messages)
