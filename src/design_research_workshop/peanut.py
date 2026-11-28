"""A catalog-based peanut-sheller design task with editable review priorities."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from random import Random
from typing import Literal

from design_research.agents import (
    LLMMessage,
    LLMRequest,
    OpenAICompatibleHTTPLLMClient,
    ProposeCriticPattern,
)

from .critique import FEEDBACK_HEADER, PROPOSAL_HEADER, TASK_HEADER, _split_prompt
from .rubric import ProvisionCriterion, apply_revision_goals, review_provisions

PROBLEM_ID = "ideation_peanut_shelling_fu_cagan_kotovsky_2010"
BRIEF = (
    "Design a non-electric tabletop peanut sheller for a small grower. "
    "Aim to keep kernels intact, use readily available materials, and make "
    "food-contact parts easy to clean."
)
DIRECT = "Generate distinct complete concepts from the brief."
FUNCTION_FIRST = (
    "Think through feeding, cracking, and separating as separate functions. "
    "Combine mechanism options across the functions to create distinct concepts."
)
BRAINSTORM_SYSTEM = (
    "Workshop peanut-sheller brainstorm. Return a JSON object with concepts."
)
PEANUT_PROPOSER = (
    "Workshop peanut-sheller proposer. Return the selected concept as JSON."
)
PEANUT_CRITIC = (
    "Workshop peanut-sheller critic. Apply the ordered criteria given in the task."
)

FEEDERS = ("Hand-fed tray", "Gravity hopper", "Indexed feed wheel")
CRACKERS = ("Hinged lever press", "Hand-cranked rollers", "Sliding rub plates")
SEPARATORS = ("Tilted screen", "Hand-shaken sieve", "Hand-cranked air separator")
PROTECTIONS = {
    "Hinged lever press": "Adjustable stop and a compliant pad to limit compression.",
    "Hand-cranked rollers": "Adjustable roller gap and a compliant roller surface.",
    "Sliding rub plates": "Adjustable plate spacing and a stop to limit travel.",
}
CLEANING = {
    "Tilted screen": "Slide the screen and collection tray out for cleaning.",
    "Hand-shaken sieve": "Unclip the sieve and empty the removable collection tray.",
    "Hand-cranked air separator": "Open the debris cover and remove the chute and tray.",
}
FIXED_PROTECTION = "Fixed clearance with no size adjustment during use."
FIXED_CLEANING = "Fixed separation assembly with access from the outlet."
FIXED_FEED = "The feeder has no specified setting or release interval."
METERING = {
    "Hand-fed tray": "Divide the tray into pockets that each hold one pod.",
    "Gravity hopper": "Fit a sliding gate at the hopper throat to meter pods.",
    "Indexed feed wheel": "Fit a detent so each turn releases one pocket of pods.",
}

# Each criterion names a visible design provision that the critic knows how to recognize.
KERNEL = ProvisionCriterion(
    "Kernel protection",
    "kernel_protection",
    "crack",
    "Specify how the operator can limit compression for different pod sizes.",
    PROTECTIONS,
    {
        "Hinged lever press": (
            "adjustable stop",
            "adjustable press stop",
            "adjustable end stop",
        ),
        "Hand-cranked rollers": (
            "adjustable roller gap",
            "adjustable gap",
            "variable roller gap",
            "roller spacing can be adjusted",
            "gap adjusts to pod size",
        ),
        "Sliding rub plates": ("adjustable plate spacing", "adjustable plate gap"),
    },
)
WASH = ProvisionCriterion(
    "Cleaning access",
    "cleaning_access",
    "separate",
    "Specify access to both the separator and its collection tray.",
    CLEANING,
    {
        "Tilted screen": (
            "removable screen and tray",
            "slide-out screen and tray",
            "screen and tray slide out",
        ),
        "Hand-shaken sieve": (
            "removable sieve and tray",
            "lift-out sieve and tray",
            "sieve and tray lift out",
        ),
        "Hand-cranked air separator": ("removable chute and tray",),
    },
)
FEED = ProvisionCriterion(
    "Feed control",
    "feed_control",
    "feed",
    "Specify how the operator can meter pods entering the cracking mechanism.",
    METERING,
    {
        "Hand-fed tray": ("one-pod pockets",),
        "Gravity hopper": (
            "sliding feed gate",
            "adjustable hopper gate",
            "metering gate",
        ),
        "Indexed feed wheel": ("indexed detent", "one-pod detent"),
    },
)
REVIEW_CRITERIA = {item.label.casefold(): item for item in (KERNEL, WASH, FEED)}
DEFAULT_CRITERIA = ("kernel protection", "cleaning access")
GOALS_HEADER = "\n\nRevision goals:\n"


@dataclass(frozen=True, slots=True)
class ShellerConcept:
    """Three chosen mechanisms and the design provisions currently specified for each."""

    feed: str
    crack: str
    separate: str
    kernel_protection: str = FIXED_PROTECTION
    cleaning_access: str = FIXED_CLEANING
    feed_control: str = FIXED_FEED

    def __post_init__(self) -> None:
        if (
            self.feed not in FEEDERS
            or self.crack not in CRACKERS
            or self.separate not in SEPARATORS
        ):
            raise ValueError(
                "Choose feeding, cracking, and separating mechanisms from the workshop catalog."
            )
        for value in (self.kernel_protection, self.cleaning_access, self.feed_control):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Describe each provision with a nonempty string.")


DEFAULT_CONCEPT = ShellerConcept(
    "Gravity hopper", "Hand-cranked rollers", "Hand-shaken sieve"
)


def run_brainstorm(
    client: OpenAICompatibleHTTPLLMClient,
    framing: str,
    *,
    count: int = 6,
    seed: int = 7,
) -> list[ShellerConcept]:
    """Submit the workshop brief and one framing through the Agents client."""
    prompt = f"{BRIEF}\n\n{framing}\n\nConcept count: {count}\nSeed: {seed}"
    response = client.generate(
        LLMRequest(
            messages=(
                LLMMessage(role="system", content=BRAINSTORM_SYSTEM),
                LLMMessage(role="user", content=prompt),
            )
        )
    )
    return [
        ShellerConcept(**concept) for concept in json.loads(response.text)["concepts"]
    ]


def brainstorm(prompt: str) -> dict[str, object]:
    """Rank the same catalog by simplicity or by coverage after a function-first cue."""
    count_match = re.search(r"^Concept count: (\d+)$", prompt, re.MULTILINE)
    seed_match = re.search(r"^Seed: (-?\d+)$", prompt, re.MULTILINE)
    if not count_match or not seed_match or not 1 <= int(count_match[1]) <= 12:
        raise ValueError("Choose a concept count from 1 to 12 and an integer seed.")
    count, seed = int(count_match[1]), int(seed_match[1])
    candidates = list(product(FEEDERS, CRACKERS, SEPARATORS))
    Random(seed).shuffle(candidates)
    # Catalog order moves from fewer to more moving parts or adjustments.
    candidates.sort(
        key=lambda combo: sum(
            axis.index(item)
            for axis, item in zip((FEEDERS, CRACKERS, SEPARATORS), combo, strict=True)
        )
    )
    chosen = []
    if re.search(r"\b(functions?|function-first)\b", prompt, re.IGNORECASE):
        seen: set[str] = set()
        for _ in range(count):
            next_combo = max(candidates, key=lambda combo: len(set(combo) - seen))
            candidates.remove(next_combo)
            chosen.append(next_combo)
            seen.update(next_combo)
    else:
        chosen = candidates[:count]
    return {"concepts": [asdict(ShellerConcept(*combo)) for combo in chosen]}


def sheller_pattern(
    client: OpenAICompatibleHTTPLLMClient, *, max_iterations: int = 2
) -> ProposeCriticPattern:
    """Use the public proposal and critique loop with workshop-specific rules."""
    return ProposeCriticPattern(
        llm_client=client,
        max_iterations=max_iterations,
        proposer_system_prompt=PEANUT_PROPOSER,
        critic_system_prompt=PEANUT_CRITIC,
        proposer_user_prompt_template=(
            f"{TASK_HEADER}$task_prompt{FEEDBACK_HEADER}$prior_feedback"
            f"{GOALS_HEADER}$revision_goals_json"
        ),
        critic_user_prompt_template=f"{TASK_HEADER}$task_prompt{PROPOSAL_HEADER}$proposal",
    )


def _review_order(
    names: Sequence[str], priority: str | None
) -> list[ProvisionCriterion]:
    if (
        isinstance(names, str)
        or not isinstance(names, Sequence)
        or not names
        or any(
            not isinstance(name, str) or name not in REVIEW_CRITERIA for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            f"Choose distinct review criteria from {', '.join(REVIEW_CRITERIA)}."
        )
    if priority is not None and priority not in names:
        raise ValueError(
            "The first review priority must be among the selected criteria."
        )
    order = (
        [priority, *(name for name in names if name != priority)] if priority else names
    )
    return [REVIEW_CRITERIA[name] for name in order]


def sheller_task(
    concept: ShellerConcept,
    priority: str | None = None,
    *,
    criteria: Sequence[str] = DEFAULT_CRITERIA,
    changes_per_revision: int | None = None,
) -> str:
    """Serialize a starting concept, an ordered rubric, and the change budget for each round."""
    ordered = _review_order(criteria, priority)
    budget = len(ordered) if changes_per_revision is None else changes_per_revision
    if type(budget) is not int or budget < 1:
        raise ValueError("Use a positive integer change budget.")
    return json.dumps(
        {
            "brief": BRIEF,
            "concept": asdict(concept),
            "priority": ordered[0].label.casefold(),
            "criteria": [item.label.casefold() for item in ordered],
            "changes_per_revision": budget,
        }
    )


def _task_settings(
    task_text: str,
) -> tuple[ShellerConcept, list[ProvisionCriterion], int]:
    task = json.loads(task_text)
    seed = ShellerConcept(**task["concept"])
    criteria = _review_order(
        task.get("criteria", DEFAULT_CRITERIA), task.get("priority")
    )
    return seed, criteria, task.get("changes_per_revision", len(criteria))


def propose_sheller(prompt: str) -> dict[str, object]:
    task, feedback = _split_prompt(prompt, FEEDBACK_HEADER)
    seed, criteria, _ = _task_settings(task)
    text, separator, serialized = feedback.partition(GOALS_HEADER)
    if separator:
        goals = json.loads(serialized)
        if not isinstance(goals, list) or any(
            not isinstance(goal, str) for goal in goals
        ):
            raise ValueError("Use a JSON list of revision goals.")
    else:
        # The original workshop template passed the critic's feedback text alone.
        base = asdict(seed)
        goals = [
            f"Change {item.label}: {item.repairs[base[item.mechanism_field]]}"
            for item in criteria
            if item.label.casefold() in text.casefold()
        ]
    return apply_revision_goals(asdict(seed), criteria, goals)


def critique_sheller(prompt: str) -> dict[str, object]:
    task, proposal = _split_prompt(prompt, PROPOSAL_HEADER)
    _, criteria, budget = _task_settings(task)
    concept = ShellerConcept(**json.loads(proposal))
    return review_provisions(asdict(concept), criteria, changes_per_revision=budget)


_MECHANISMS = {
    "feed": {
        FEEDERS[0]: ("hand fed tray", "feed tray"),
        FEEDERS[1]: ("hopper",),
        FEEDERS[2]: ("indexed wheel", "feed wheel"),
    },
    "crack": {
        CRACKERS[0]: ("lever press", "hinged press"),
        CRACKERS[1]: ("roller", "rollers"),
        CRACKERS[2]: ("rub plates", "rubbing plates"),
    },
    "separate": {
        SEPARATORS[0]: ("screen",),
        SEPARATORS[1]: ("sieve",),
        SEPARATORS[2]: ("air separator", "winnower"),
    },
}
_FOCUS_WORDS = {
    "kernel protection": (
        "kernel protection",
        "protect kernels",
        "protect the kernels",
        "whole kernels",
        "intact kernels",
        "kernels intact",
        "kernel damage",
        "kernel breakage",
        "kernels whole",
        "reduce breakage",
    ),
    "cleaning access": (
        "cleaning",
        "cleanout",
        "clean out",
        "cleanability",
        "easy to clean",
        "washdown",
        "washability",
        "cleanliness",
        "cleanup",
        "wash down",
    ),
    "feed control": (
        "feed control",
        "controlled feeding",
        "meter pods",
        "metering",
        "pod flow",
        "feed rate",
        "flow of pods",
    ),
}
_FIXED_FEATURES = {
    "kernel_protection": (
        "fixed gap",
        "fixed roller gap",
        "fixed clearance",
        "fixed stop",
    ),
    "cleaning_access": (
        "fixed separation assembly",
        "fixed sieve and tray",
        "sealed debris cover",
    ),
    "feed_control": ("unmetered feed", "uncontrolled feed", "fixed feed rate"),
}


def _words(text: str) -> str:
    expanded = (
        text.casefold()
        .replace("’", "'")
        .replace("n't", " not")
        .replace("cannot", "can not")
    )
    return " " + " ".join(re.findall(r"[a-z0-9]+", expanded)) + " "


def _first_mention(words: str, phrases: Sequence[str]) -> int:
    positions = [words.find(_words(phrase)) for phrase in phrases]
    return min((i for i in positions if i >= 0), default=-1)


def _unknown_focus(prompt: str) -> list[str]:
    """Identify requested checks outside the three available evidence vocabularies."""
    requested = re.findall(
        r"\b(?:especially|prioriti[sz]e|focus(?:ing)? on|(?:special )?attention to|(?:review|check|assess|evaluate)"
        r"(?:\s+[\w-]+){0,7}\s+for|(?:review|check|assess|evaluate) its)\s+([^.;?!\n]+)",
        prompt,
        re.IGNORECASE,
    )
    if not requested:
        direct = re.fullmatch(
            r"\s*(?:please )?(?:review|assess|check|evaluate)\s+([^.;?!]+)[.!?]?\s*",
            prompt,
            re.IGNORECASE,
        )
        if direct and not re.search(
            r"\b(it|this|that|sheller|concept|design|proposal|roller[s]?|press|sieve)\b",
            direct[1],
            re.IGNORECASE,
        ):
            requested = [direct[1]]
    known = [phrase for phrases in _FOCUS_WORDS.values() for phrase in phrases]
    known.extend(
        evidence
        for criterion in REVIEW_CRITERIA.values()
        for options in criterion.alternatives.values()
        for evidence in options
    )
    unknown = []
    for clause in requested:
        # Some approved provisions include a pair of parts, such as a sieve and tray.
        marked = _words(clause.replace(",", " and ").replace("&", " and "))
        for phrase in sorted(known, key=len, reverse=True):
            marked = marked.replace(_words(phrase), " supported ")
        for part in re.split(r"\b(?:and|plus|as well as)\b", marked):
            name = part.strip().removesuffix(" only").strip()
            if (
                name
                and "supported" not in name
                and not re.search(r"\b(growers?|farmers?|workshop)\b", name)
            ):
                unknown.append(name)
    return unknown


def _described_provision(
    prompt: str, item: ProvisionCriterion, mechanism: str
) -> str | None:
    """Extract a stated feature while distinguishing absence from a proposed improvement."""
    observed = None
    options = (item.repairs[mechanism], *item.alternatives.get(mechanism, ()))
    for clause in re.split(r"[.;!?\n]|\bbut\b", prompt, flags=re.IGNORECASE):
        normalized = _words(clause)
        label = re.escape(item.label.casefold())
        missing = re.search(
            rf"\b(?:no|without|lacks?|missing) (?:any )?{label}\b|\b{label} "
            rf"(?:is |was )?(?:missing|absent)\b",
            normalized,
        )
        if missing or _first_mention(normalized, _FIXED_FEATURES[item.field]) >= 0:
            observed = getattr(DEFAULT_CONCEPT, item.field)
        for evidence in options:
            pos = _first_mention(normalized, (evidence,))
            if pos < 0:
                continue
            before = re.split(r"\band\b", normalized[:pos])[-1]
            after = normalized[pos + len(_words(evidence)) - 1 :]
            negative = re.search(
                r"\b(no|non|not|without|lacks?|lacking|missing)\b", before
            )
            if negative or re.match(r"\s*(?:is |was )?(?:missing|absent)\b", after):
                observed = getattr(DEFAULT_CONCEPT, item.field)
                break
            if re.search(
                r"\b(needs?|want|should|could|would|add|consider|require[sd]?)\b",
                before,
            ):
                continue
            if re.search(r"\b(?:whether|check if|see if|does it|is there)\b", before):
                continue
            if re.search(
                r"\b(?:for|especially|prioriti[sz]e)\s*(?:a|an|the)?\s*$", before
            ):
                continue
            if re.match(
                r"\s*(?:(?:is |was )?(?:needed|desired|requested)|would|should|could|might)\b",
                after,
            ):
                continue
            observed = evidence
            break
    return observed


_MECHANISM_NAMES = {
    _words(alias).strip(): name
    for choices in _MECHANISMS.values()
    for name, aliases in choices.items()
    for alias in (name, *aliases)
}
_NAMED_MECHANISM = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(alias) for alias in sorted(_MECHANISM_NAMES, key=len, reverse=True)
    )
    + r")\b"
)
_DETERMINER = r"\s*(?:(?:a|an|any|the|both)(?: pair of)?\s+)?$"
_EXCLUDE = re.compile(
    r"\b(?:without(?: using)?|instead of|rather than|(?:switch|change) from|avoid|skip|"
    r"omit|exclude|remove|drop|replace|no|not(?: use| using| include| want| have)?|never use)"
    + _DETERMINER
)
_KEEP = re.compile(r"\bnot (?:remove|exclude|drop|replace|avoid)" + _DETERMINER)


def _mechanism_requests(prompt: str) -> tuple[set[str], set[str]]:
    """Track named choices and exclusions, including coordinated and replacement phrases."""
    words = _words(re.sub(r"[.;!?]", " next ", prompt.replace(",", " and ")))
    included, excluded = set(), set()
    prior_end, excluding = 0, False
    for match in _NAMED_MECHANISM.finditer(words):
        prefix = words[: match.start()]
        join = words[prior_end : match.start()]
        if not (
            excluding and re.fullmatch(r"\s*(?:and|or)\s+(?:(?:a|an|the)\s+)?", join)
        ):
            excluding = bool(_EXCLUDE.search(prefix) and not _KEEP.search(prefix))
        name = _MECHANISM_NAMES[match[0]]
        (excluded if excluding else included).add(name)
        prior_end = match.end()
    if included & excluded:
        raise ValueError(
            f"Choose whether to keep or exclude {', '.join(sorted(included & excluded))}."
        )
    return included, excluded


def _concept_from_text(
    prompt: str, seed: ShellerConcept, *, propose: bool
) -> ShellerConcept:
    """Apply selected mechanisms, explicit exclusions, and described provisions."""
    included, excluded = _mechanism_requests(prompt)
    revised = asdict(seed)
    for field, choices in _MECHANISMS.items():
        selected = [name for name in choices if name in included]
        allowed = [name for name in choices if name not in excluded]
        if len(selected) > 1:
            raise ValueError(
                f"Choose one {field} mechanism per sheller concept: {', '.join(selected)}."
            )
        if not allowed:
            raise ValueError(
                f"Allow at least one {field} mechanism from {', '.join(choices)}."
            )
        if selected:
            chosen = selected[0]
        elif revised[field] in allowed:
            continue
        elif propose:
            chosen = allowed[0]
        else:
            raise ValueError(
                f"Which {field} mechanism should the critic review? Choose from {', '.join(allowed)}."
            )
        if revised[field] != chosen:
            revised[field] = chosen
            for item in REVIEW_CRITERIA.values():
                if item.mechanism_field == field:
                    revised[item.field] = getattr(DEFAULT_CONCEPT, item.field)
    for item in REVIEW_CRITERIA.values():
        provision = _described_provision(prompt, item, revised[item.mechanism_field])
        if provision is not None:
            revised[item.field] = provision
    return ShellerConcept(**revised)


def _review_directives(words: str) -> tuple[list[str], list[str], list[str]]:
    """Separate positive review cues, dropped checks, and lower-priority checks."""
    ranked = []
    excluded = []
    later = []
    for name, cues in _FOCUS_WORDS.items():
        position = _first_mention(words, cues)
        if position < 0:
            continue
        before = words[:position]
        if re.search(
            r"\b(?:skip|ignore|omit|exclude|leave out|do not (?:review|check|assess))\s*(?:the)?$",
            before,
        ):
            excluded.append(name)
        elif re.search(
            r"\b(?:do not prioriti[sz]e|deprioriti[sz]e)\s*(?:the)?$", before
        ):
            later.append(name)
        else:
            explicit = re.search(
                r"\b(?:especially|prioriti[sz]e|focus on)\s*(?:the)?$", before
            )
            ranked.append((not explicit, position, name))
    return [name for _, _, name in sorted(ranked)], excluded, later


def respond_sheller(
    prompt: str,
    *,
    role: Literal["propose", "critic"],
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Respond to a short natural-language request using the same catalog and rubric as the loop.

    Pass the prior proposal or review payload to resolve references to the current design.
    """
    context = previous or {}
    record = context.get("concept", context)
    if isinstance(record, Mapping) and {"feed", "crack", "separate"} <= record.keys():
        seed = ShellerConcept(
            **{key: record[key] for key in asdict(DEFAULT_CONCEPT) if key in record}
        )
    else:
        seed = DEFAULT_CONCEPT
    concept = asdict(_concept_from_text(prompt, seed, propose=role == "propose"))
    words = _words(prompt)
    mentioned, excluded, later = _review_directives(words)
    prior_criteria = context.get("criteria") or DEFAULT_CRITERIA
    names = (
        mentioned
        if " only " in words and mentioned
        else list(
            dict.fromkeys(
                [
                    *mentioned,
                    *(
                        name
                        for name in prior_criteria
                        if name not in excluded and name not in later
                    ),
                    *later,
                ]
            )
        )
    )
    if not names:
        raise ValueError("Choose at least one remaining sheller review criterion.")
    criteria = _review_order(names, None)
    budget = (
        1
        if " one change " in words
        else context.get("changes_per_revision", len(criteria))
    )
    unknown = _unknown_focus(prompt)
    if unknown:
        scoped = _review_order(mentioned, None) if mentioned else []
        review = (
            review_provisions(concept, scoped, changes_per_revision=budget)
            if scoped
            else {"feedback": "", "revision_goals": []}
        )
        remaining = f"A separate check is needed for {', '.join(unknown)}."
        return {
            **review,
            "approved": False,
            "feedback": " ".join(filter(None, [review["feedback"], remaining])),
            "unassessed": unknown,
            "available_criteria": list(REVIEW_CRITERIA),
            "concept": concept,
            "criteria": mentioned,
            "changes_per_revision": budget,
        }
    if role == "critic":
        return {
            **review_provisions(concept, criteria, changes_per_revision=budget),
            "concept": concept,
            "criteria": names,
            "changes_per_revision": budget,
        }
    if role != "propose":
        raise ValueError("Choose the proposer or critic role.")
    if mentioned:
        review = review_provisions(
            concept,
            _review_order(mentioned, None),
            changes_per_revision=budget,
        )
        goals = review["revision_goals"]
    else:
        goals = context.get("revision_goals", [])
        # Preserve a supplied repair only when it applies to the current mechanisms.
        if goals and concept != asdict(seed):
            goals = review_provisions(concept, criteria, changes_per_revision=budget)[
                "revision_goals"
            ]
    return apply_revision_goals(concept, criteria, goals)
