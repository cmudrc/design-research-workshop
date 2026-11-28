"""Deterministic response selection for workshop conversations."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .evidence import compare_results

Content = dict[str, object]
Message = Mapping[str, str]


def _words(text: str) -> str:
    """Normalize spacing, punctuation, and case for whole-phrase topic matches."""
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


_EXPLAIN = re.compile(
    r"(?:please )?(?:why|why (?:this|that)(?: design| choice| result)?|"
    r"why (?:did (?:you|it|the bot|pybot) (?:choose|pick|select|recommend|reject|approve) "
    r"(?:that|this|it|those)(?: design| layout| concept| option)?|"
    r"was (?:that|this|it|the design|the proposal) (?:chosen|rejected|approved|selected))|"
    r"(?:can you |could you )?(?:explain (?:why|that|this|your choice|the result|the last result|the decision)|"
    r"show (?:me )?the evidence)|what changed(?: since (?:last time|the last (?:run|round)))?)"
)


def _is_explanation(prompt: str) -> bool:
    return bool(_EXPLAIN.fullmatch(_words(prompt)))


def _design_history(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Omit evidence questions and their answers from subsequent design operations."""
    result = []
    explaining = False
    for message in messages:
        if message["role"] == "user":
            explaining = _is_explanation(message["content"])
        if not explaining and message["role"] in ("user", "assistant"):
            result.append(message)
    return tuple(result)


def _last_result(messages: Sequence[Message]) -> Content | None:
    if not messages or messages[-1]["role"] != "assistant":
        return None
    try:
        value = json.loads(messages[-1]["content"])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class Turn:
    """The current prompt plus preceding turns from this task in this conversation."""

    prompt: str
    systems: tuple[str, ...] = ()
    history: tuple[Message, ...] = ()

    @property
    def user_text(self) -> str:
        """Related user instructions, oldest to newest, for cumulative controls."""
        prompts = [m["content"] for m in self.history if m["role"] == "user"]
        return "\n".join((*prompts, self.prompt))


@dataclass(frozen=True)
class Rule:
    """A task responder and the evidence that lets the bot select it.

    A protocol matcher recognizes a structured request. A follow-up matcher
    identifies edits the rule knows how to apply to the previous task.
    """

    name: str
    description: str
    topics: tuple[str, ...]
    respond: Callable[[Turn], Content]
    systems: tuple[str, ...] = ()
    protocol: Callable[[str], bool] | None = None
    follow_up: Callable[[str], bool] | None = None
    intent: Callable[[str], bool] | None = None
    explain: Callable[[Content], Content] | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise ValueError("Use a lowercase identifier for a rule name.")
        if not self.description.strip() or not callable(self.respond):
            raise ValueError("A rule needs a description and a callable responder.")
        if any(not _words(topic) for topic in self.topics):
            raise ValueError("Topic phrases must contain at least one word.")


@dataclass(frozen=True)
class Reply:
    """Task output is separate from the evidence for choosing its rule."""

    rule: str | None
    reason: str
    content: Content
    candidates: tuple[str, ...]

    @property
    def matched(self) -> bool:
        return self.rule is not None


class WorkshopPyBot:
    """A stateless, extensible collection of design-task responders."""

    def __init__(self, rules: Sequence[Rule] | None = None) -> None:
        if rules is None:
            from .responders import workshop_rules

            rules = workshop_rules()
        self.rules = tuple(rules)
        if len({rule.name for rule in self.rules}) != len(self.rules):
            raise ValueError("Each workshop rule needs a unique name.")
        self._topics = {
            rule.name: tuple((topic, f" {_words(topic)} ") for topic in rule.topics)
            for rule in self.rules
        }

    def with_rules(self, *rules: Rule) -> WorkshopPyBot:
        """Return a new bot with additional rules. The source bot is reusable."""
        return WorkshopPyBot((*self.rules, *rules))

    def session(self) -> WorkshopSession:
        """Start a fresh conversation using these rules."""
        return WorkshopSession(self)

    def _matches(self, prompt: str, systems: tuple[str, ...]) -> list[tuple[Rule, str]]:
        text = f" {_words(prompt)} "
        matches: list[tuple[int, Rule, str]] = []
        for rule in self.rules:
            if f"Workshop rule: {rule.name}" in systems:
                matches.append((4, rule, "explicit rule selection"))
            elif any(system in systems for system in rule.systems):
                matches.append((3, rule, "system instruction"))
            elif rule.protocol and rule.protocol(prompt):
                matches.append((2, rule, "structured task input"))
            else:
                for topic, phrase in self._topics[rule.name]:
                    if phrase in text:
                        matches.append((1, rule, f"topic '{topic}'"))
                        break
                else:
                    if rule.intent and rule.intent(prompt):
                        matches.append((0, rule, "general task request"))
        best = max((level for level, _, _ in matches), default=0)
        return [(rule, why) for level, rule, why in matches if level == best]

    def complete(self, messages: Sequence[Message]) -> Reply:
        """Answer the latest user turn, using only the supplied conversation."""
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes))
            or not messages
        ):
            raise ValueError(
                "Supply a sequence of chat messages ending in a user turn."
            )
        for m in messages:
            if (
                not isinstance(m, Mapping)
                or m.get("role") not in ("user", "assistant", "system", "developer")
                or not isinstance(m.get("content"), str)
            ):
                raise ValueError(
                    "Each message needs a supported role and text content."
                )
        # Some schema-constrained clients append a final system formatting directive.
        conversational = [
            i for i, m in enumerate(messages) if m["role"] in ("assistant", "user")
        ]
        if not conversational:
            raise ValueError("Include a nonempty user message.")
        last = conversational[-1]
        if messages[last]["role"] != "user" or not messages[last]["content"].strip():
            raise ValueError("End the conversation with a nonempty user message.")
        final_systems = tuple(
            m["content"] for m in messages if m["role"] in ("system", "developer")
        )

        # Reconstruct which task each turn refers to without re-running prior handlers.
        messages = list(messages)
        active: Rule | None = None
        start = 0
        systems: list[str] = []
        matches: list[tuple[Rule, str]] = []
        reason = ""
        for index, message in enumerate(messages):
            if message["role"] in ("system", "developer"):
                systems.append(message["content"])
                continue
            if message["role"] != "user":
                continue
            prompt = message["content"]
            if _is_explanation(prompt):
                continue
            if len(matches) > 1:
                choice = (
                    _words(prompt)
                    .removeprefix("use ")
                    .removeprefix("the ")
                    .removesuffix(" please")
                )
                chosen = [
                    r
                    for r, _ in matches
                    if choice in {_words(t) for t in (r.name, *r.topics)}
                ]
                if len(chosen) == 1:
                    # Run the original request with all of its controls and start
                    # the selected task's history after the clarification.
                    messages[index] = {
                        "role": "user",
                        "content": messages[start]["content"],
                    }
                    active = chosen[0]
                    reason = "participant resolved the task choice"
                    matches = [(active, reason)]
                    start = index
                    continue
            matches = self._matches(
                prompt, final_systems if index == last else tuple(systems)
            )
            follow = bool(active and active.follow_up and active.follow_up(prompt))
            if follow and all(why == "general task request" for _, why in matches):
                reason = "follow-up on the preceding task"
                matches = [(active, reason)]
            if len(matches) == 1:
                selected, reason = matches[0]
                if selected is not active or not follow:
                    start = index
                active = selected
            else:
                active = None
                start = index

        names = tuple(rule.name for rule, _ in matches)
        if _is_explanation(messages[last]["content"]):
            history = _design_history(messages[start:last])
            result = _last_result(history)
            if active is None or result is None:
                return Reply(
                    None,
                    "no recorded task result",
                    {"question": "Which design case should we run first?"},
                    names,
                )
            if _words(messages[last]["content"]).startswith("what changed"):
                # Only compare two recorded results within the current task.
                previous = next(
                    (
                        _last_result((m,))
                        for m in reversed(history[:-1])
                        if m["role"] == "assistant"
                    ),
                    None,
                )
                evidence = compare_results(result, previous)
            elif active.explain:
                evidence = active.explain(result)
            else:
                evidence = {
                    "explanation": "The task returned the following recorded result.",
                    "evidence": result,
                }
            return Reply(active.name, "recorded task evidence", evidence, names)
        if active is None:
            if names:
                question = "Which task should I run? " + ", ".join(names) + "."
                return Reply(
                    None, "multiple tasks match", {"question": question}, names
                )
            choices = {rule.name: rule.description for rule in self.rules}
            question = "Which design task should we work on?"
            return Reply(
                None, "no task matched", {"question": question, "tasks": choices}, ()
            )

        history = _design_history(messages[start:last])
        turn = Turn(messages[last]["content"], final_systems, history)
        content = active.respond(turn)
        if not isinstance(content, dict):
            raise ValueError(f"Rule {active.name!r} must return a JSON object.")
        return Reply(active.name, reason, content, names)

    def reply(
        self, prompt: str, *, history: Sequence[Message] = (), rule: str | None = None
    ) -> Reply:
        """Answer a text prompt. An instructor can select a rule by name for a case."""
        messages = [*history, {"role": "user", "content": prompt}]
        if rule is not None:
            if rule not in self._topics:
                raise ValueError(
                    f"Unknown workshop rule {rule!r}. Choose from {', '.join(self._topics)}."
                )
            messages.append({"role": "system", "content": f"Workshop rule: {rule}"})
        return self.complete(messages)


class WorkshopSession:
    """One participant's turns. The underlying bot can be shared across sessions."""

    def __init__(self, bot: WorkshopPyBot) -> None:
        self.bot = bot
        self._messages: list[Message] = []

    def reply(self, prompt: str) -> Reply:
        result = self.bot.reply(prompt, history=self._messages)
        encoded = json.dumps(result.content, allow_nan=False)
        self._messages.extend(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": encoded},
            )
        )
        return result

    def reset(self) -> None:
        self._messages.clear()
