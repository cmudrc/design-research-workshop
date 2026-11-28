"""Generate design concepts by combining options in an authored morphological chart."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod
from random import Random
from types import MappingProxyType

from design_research.agents import LLMMessage, LLMRequest, OpenAICompatibleHTTPLLMClient


@dataclass(frozen=True, slots=True)
class DesignBrief:
    """Define a design problem and a preferred-first option list for each function."""

    title: str
    statement: str
    functions: Mapping[str, Sequence[str]]
    problem_id: str | None = None

    @classmethod
    def from_problem(
        cls, problem_id: str, functions: Mapping[str, Sequence[str]]
    ) -> DesignBrief:
        """Bind an installed Design Research Problems ideation brief to a mechanism chart."""
        from design_research.problems import TextProblem, get_problem

        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("Supply a Design Research Problems catalog ID.")
        try:
            problem = get_problem(problem_id)
        except KeyError as error:
            raise ValueError(
                f"Unknown Design Research Problems catalog ID: {problem_id}."
            ) from error
        if (
            not isinstance(problem, TextProblem)
            or "ideation" not in problem.metadata.taxonomy.tags
        ):
            raise ValueError(
                "Choose a text ideation problem from the installed problem catalog."
            )
        return cls(
            problem.metadata.title, problem.statement_markdown, functions, problem_id
        )

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Give the brief a title.")
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("Give the brief a design statement.")
        if not isinstance(self.functions, Mapping) or not self.functions:
            raise ValueError("Define at least one function in the chart.")
        chart = {}
        for name, options in self.functions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Give each function a label.")
            if (
                isinstance(options, str)
                or not isinstance(options, Sequence)
                or not options
            ):
                raise ValueError(f"Give {name} a nonempty list of options.")
            if any(not isinstance(item, str) or not item.strip() for item in options):
                raise ValueError(f"Use nonempty option names for {name}.")
            choices = tuple(item.strip() for item in options)
            if len({item.casefold() for item in choices}) != len(choices):
                raise ValueError(f"Use distinct options for {name}.")
            chart[name.strip()] = choices
        if len({name.casefold() for name in chart}) != len(self.functions):
            raise ValueError("Use distinct function labels.")
        object.__setattr__(self, "functions", MappingProxyType(chart))


PROBLEM_ID = "ideation_one_handed_lidded_container_opening"
JAR_OPENER = DesignBrief(
    title="One-handed jar opener",
    problem_id=PROBLEM_ID,
    statement=(
        "Design a countertop device that lets someone working with one hand open "
        "a screw-top food jar. Consider how the person positions the jar, breaks "
        "the lid seal, and takes the jar back out."
    ),
    functions={
        "Hold the jar": (
            "High-friction cradle",
            "Counter-mounted V-block",
            "Self-centering clamp",
        ),
        "Grip the lid": (
            "Tapered rubber socket",
            "Lever-lock strap",
            "Spring-loaded jaws",
        ),
        "Turn the lid": ("T-handle", "Offset lever", "Ratcheting arm"),
    },
)

BRAINSTORM_SYSTEM = "Brainstorm one-handed jar opener concepts from the workshop chart."
DIRECT = "Suggest complete one-handed jar opener concepts."
FUNCTION_FIRST = (
    "Brainstorm one-handed jar openers function by function. Vary how each "
    "concept holds the jar, grips the lid, and turns the lid."
)

# A chat request can supply its own chart. Bound the exhaustive search before expansion.
MAX_COMBINATIONS = 4096
MAX_CONCEPTS = 24
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()
_COUNTS = {name: i for i, name in enumerate(_NUMBER_WORDS)}
_COUNTS.update(
    {f"twenty-{name}": 20 + i for i, name in enumerate(_NUMBER_WORDS[1:5], 1)}
)
_COUNTS["dozen"] = 12
_COUNTS["another"] = 1
_NUMBER = r"-?\d+|" + "|".join(
    word.replace("-", "[ -]") for word in sorted(_COUNTS, key=len, reverse=True)
)
_COUNT = re.compile(
    rf"(?:\b(?:concept|idea)\s+count\s*[:=]\s*(?P<label>{_NUMBER})(?![\w-]|\.\d)"
    rf"|(?<![\w.-])(?P<noun>{_NUMBER})\s+(?:(?:distinct|different|complete|new|more|additional|further|other)\s+)*"
    rf"(?:ideas?|concepts?|designs?|solutions?|options?|ways?|alternatives?)\b"
    rf"|(?<![\w.-])(?P<extra>{_NUMBER})\s+more\b"
    rf"|\b(?P<single>another)\s+one\b"
    rf"|\b(?:give me|show me|(?:can|could) I (?:see|have|get)|list|brainstorm|suggest|generate|make it|try|"
    rf"(?:narrow|reduce|cut|limit) (?:it|that|these|them|the list)(?: down)? to)\s+"
    rf"(?P<verb>{_NUMBER})(?![\w-]|\.\d))",
    re.IGNORECASE,
)
_SEED = re.compile(
    r"\bseed\s*(?:(?:of|is|to)\s+|[=:]\s*)?(-?\d+)(?!\w|\.\d)", re.IGNORECASE
)
_STRATEGY = re.compile(
    r"(?P<coverage>\b(?:functions?[ -](?:by[ -]function|first)|"
    r"by function|(?:each|separate|individual) function|morphological|"
    r"(?:maximize|increase|prioritize|more|greater) (?:mechanism )?(?:coverage|variety|diversity)|"
    r"break (?:it|the (?:task|problem)) (?:down )?into (?:parts|steps|functions))\b)"
    r"|(?P<direct>\bdirect(?:ly|[ -](?:brainstorm|framing|prompt))?\b)",
    re.IGNORECASE,
)
_CHOICE_VERB = re.compile(
    r"\b(?:(?P<exclude>(?:(?:do not|don['’]t|must not|should not|never) (?:use|include|keep)|avoid|exclude|omit|skip|leave out|without))"
    r"|(?P<fix>keep|use|using|require|include|switch to|stick with))\s+"
    r"(?:only\s+)?(?:(?:the|a|an)\s+)?",
    re.IGNORECASE,
)


_MORE = re.compile(
    rf"\b(?:(?:more|additional|other|further|new|another)\s+(?:(?:distinct|design)\s+)?"
    rf"(?:ideas?|concepts?|alternatives?|solutions?|options?)|(?:{_NUMBER})\s+more\b|another one|what else)\b",
    re.I,
)


def wants_more(prompt: str) -> bool:
    """Recognize a request for additional concepts, distinct from a request for more variety."""
    return bool(_MORE.search(prompt))


def is_follow_up(prompt: str) -> bool:
    """Recognize a count, seed, framing, or mechanism control for a previous brief."""
    return any(
        pattern.search(prompt)
        for pattern in (_COUNT, _SEED, _STRATEGY, _CHOICE_VERB, _MORE)
    )


def _count_from_prompt(prompt: str) -> int | None:
    match = next(reversed(list(_COUNT.finditer(prompt))), None)
    if match is None:
        return None
    token = match[match.lastgroup].lower().replace(" ", "-")
    return _COUNTS[token] if token in _COUNTS else int(token)


def _fixed_chart(brief: DesignBrief, prompt: str) -> Mapping[str, Sequence[str]]:
    """Apply named option edits in order, keeping the latest choice per function."""
    if not _CHOICE_VERB.search(prompt):
        return brief.functions
    fixed = {}
    excluded = {name: set() for name in brief.functions}
    named = {}
    for name, options in brief.functions.items():
        for option in options:
            spelling = option.casefold().replace("-", " ")
            named.setdefault(spelling, []).append((name, option))
    alternatives = "|".join(
        re.escape(option) for option in sorted(named, key=len, reverse=True)
    )
    mechanism = re.compile(rf"(?:{alternatives})(?!\w)")
    conjunction = re.compile(r"\s*(?:,\s*(?:and\s+)?|\band\s+)(?:(?:the|a|an)\s+)?")
    for line in prompt.splitlines():
        line_fixed = {}
        for action in _CHOICE_VERB.finditer(line):
            tail = line[action.end() :].casefold().replace("-", " ")
            match = mechanism.match(tail)
            if not match:
                # A standalone command to keep an unknown mechanism needs clarification.
                # The same words can also appear in a general problem statement.
                clause_start = re.search(
                    r"(?:^|[.!?;])\s*(?:please\s*)?$", line[: action.start()], re.I
                )
                explicit = action["exclude"] or action["fix"].casefold() in {
                    "keep",
                    "require",
                    "use",
                    "include",
                    "switch to",
                    "stick with",
                }
                if (
                    clause_start
                    and explicit
                    and not _STRATEGY.match(tail)
                    and not _COUNT.match(tail)
                    and not tail.startswith(
                        (
                            "same ",
                            "number ",
                            "seed ",
                            "count ",
                            "framing ",
                            "chart ",
                            "chart.",
                            "following chart",
                        )
                    )
                ):
                    raise ValueError(
                        "Choose a mechanism from the chart to keep or exclude."
                    )
            while match:
                for name, option in named[match.group()]:
                    if action["exclude"]:
                        excluded[name].add(option)
                        if fixed.get(name) == option:
                            fixed.pop(name)
                    else:
                        if name in line_fixed and line_fixed[name] != option:
                            raise ValueError(f"Choose one fixed option for {name}.")
                        fixed[name] = line_fixed[name] = option
                        excluded[name].discard(option)
                join = conjunction.match(tail, match.end())
                match = mechanism.match(tail, join.end()) if join else None
    chart = {}
    for name, options in brief.functions.items():
        available = (
            [fixed[name]]
            if name in fixed
            else [item for item in options if item not in excluded[name]]
        )
        if not available:
            raise ValueError(f"Allow at least one mechanism for {name}.")
        chart[name] = available
    return chart


def generate_concepts(
    brief: DesignBrief,
    prompt: str,
    *,
    count: int | None = None,
    seed: int | None = None,
    previous: Sequence[Mapping[str, str]] = (),
) -> dict:
    """Select unique combinations and report how each changes mechanism coverage.

    Direct framing uses the sum of the catalog positions. Function-first framing
    picks the most unused options, then the least repeated options. Both resolve
    remaining ties using catalog preference and a seeded shuffle. Pass previous
    concept rows for an additional batch. Matching rows are excluded and their
    mechanisms count toward coverage. Rows outside the current chart are ignored.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Give the bot a brainstorming prompt.")
    chart = _fixed_chart(brief, prompt)
    available = prod(map(len, chart.values()))
    if available > MAX_COMBINATIONS:
        raise ValueError(
            f"Use a chart with at most {MAX_COMBINATIONS} combinations per request."
        )

    shown = {
        tuple(brief.functions[name].index(row[name]) for name in chart)
        for row in previous
        if isinstance(row, Mapping)
        and row.keys() == chart.keys()
        and all(row[name] in options for name, options in chart.items())
    }
    remaining = available - len(shown)
    if not remaining:
        raise ValueError(
            f"All {available} combinations in this chart have already been shown."
        )
    if count is None:
        # An unnumbered “more ideas” turn requests another batch of the remaining choices.
        count_prompt = prompt
        if previous:
            lines = prompt.splitlines()
            for i in reversed(range(len(lines))):
                if wants_more(lines[i]):
                    count_prompt = "\n".join(lines[i:])
                    break
        count = _count_from_prompt(count_prompt)
    if count is None:
        count = min(6, remaining)
    limit = min(remaining, MAX_CONCEPTS)
    if shown and type(count) is int and count > remaining:
        raise ValueError(
            f"Only {remaining} additional combinations remain in this chart."
        )
    if type(count) is not int or not 1 <= count <= limit:
        raise ValueError(
            f"Request 1 to {limit} ideas for this chart and its fixed options."
        )
    if seed is None:
        seeds = _SEED.findall(prompt)
        seed = int(seeds[-1]) if seeds else 7
    if type(seed) is not int:
        raise ValueError("Use an integer seed.")

    strategy_match = next(reversed(list(_STRATEGY.finditer(prompt))), None)
    coverage_first = False
    if strategy_match:
        coverage_first = strategy_match.lastgroup == "coverage"
        # Framing and option controls can both be phrased negatively.
        preceding = prompt[: strategy_match.start()]
        if re.search(
            r"\b(?:not|don['’]t|never|avoid|skip|without)\s+"
            r"(?:(?:use|using|choose|choosing|apply)\s+)?(?:(?:the|a)\s+)?$",
            preceding,
            re.I,
        ):
            coverage_first = not coverage_first
    # Store option indices so repeated names across functions have distinct identities.
    ranked = [
        tuple(
            brief.functions[function].index(option)
            for function, option in zip(chart, combo, strict=True)
        )
        for combo in product(*chart.values())
    ]
    Random(seed).shuffle(ranked)
    ranked.sort(key=sum)
    ranked = [choice for choice in ranked if choice not in shown]
    uses = Counter(item for choice in shown for item in enumerate(choice))

    def novelty_key(combo: tuple[int, ...]) -> tuple[int, int]:
        hits = [uses[item] for item in enumerate(combo)]
        return sum(hit == 0 for hit in hits), -sum(hits)

    concepts = []
    trace = []
    for _ in range(count):
        if coverage_first:
            choice = max(ranked, key=novelty_key)
            ranked.remove(choice)
        else:
            choice = ranked.pop(0)
        concept = {
            name: brief.functions[name][i]
            for name, i in zip(chart, choice, strict=True)
        }
        additions = [
            name for axis, name in enumerate(chart) if not uses[axis, choice[axis]]
        ]
        trace.append(
            {
                "idea": len(concepts) + 1,
                "new_functions": additions,
                "rank_sum": sum(choice),
            }
        )
        concepts.append(concept)
        uses.update(enumerate(choice))

    result = {
        "brief": brief.title,
        "strategy": "function-first" if coverage_first else "direct",
        "seed": seed,
        "available": available,
        "concepts": concepts,
        "trace": trace,
    }
    if shown:
        result["previously_shown"] = len(shown)
    if brief.problem_id is not None:
        result["problem_id"] = brief.problem_id
    return result


def respond(prompt: str, *, previous: Sequence[Mapping[str, str]] = ()) -> dict:
    """Run the built-in one-handed jar-opener chart."""
    return generate_concepts(JAR_OPENER, prompt, previous=previous)


def run_brainstorm(
    client: OpenAICompatibleHTTPLLMClient,
    framing: str,
    *,
    count: int = 6,
    seed: int = 7,
) -> dict:
    """Send the jar-opener brief and the participant's framing through the agent client."""
    if type(count) is not int or type(seed) is not int:
        raise ValueError("Use an integer concept count and seed.")
    prompt = (
        f"{JAR_OPENER.statement}\n\n{framing}\n\nConcept count: {count}\nSeed: {seed}"
    )
    result = client.generate(
        LLMRequest(
            messages=(
                LLMMessage(role="system", content=BRAINSTORM_SYSTEM),
                LLMMessage(role="user", content=prompt),
            )
        )
    )
    return json.loads(result.text)
