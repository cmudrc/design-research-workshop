"""Shared peanut-sheller concepts for the brainstorm and revision demos."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from itertools import product
from random import Random

from design_research.agents import (
    LLMMessage,
    LLMRequest,
    OpenAICompatibleHTTPLLMClient,
    ProposeCriticPattern,
)

from .critique import FEEDBACK_HEADER, PROPOSAL_HEADER, TASK_HEADER, _split_prompt

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
    "Workshop peanut-sheller critic. Review kernel protection and cleaning access."
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


@dataclass(frozen=True, slots=True)
class ShellerConcept:
    """A mechanism combination that can pass between the two notebooks."""

    feed: str
    crack: str
    separate: str
    kernel_protection: str = FIXED_PROTECTION
    cleaning_access: str = FIXED_CLEANING

    def __post_init__(self) -> None:
        if (
            self.feed not in FEEDERS
            or self.crack not in CRACKERS
            or self.separate not in SEPARATORS
        ):
            raise ValueError(
                "Choose feeding, cracking, and separating mechanisms from the workshop catalog."
            )
        if self.kernel_protection not in (FIXED_PROTECTION, PROTECTIONS[self.crack]):
            raise ValueError(
                "Use a workshop kernel-protection option for the selected cracker."
            )
        if self.cleaning_access not in (FIXED_CLEANING, CLEANING[self.separate]):
            raise ValueError(
                "Use a workshop cleaning-access option for the selected separator."
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
        proposer_user_prompt_template=f"{TASK_HEADER}$task_prompt{FEEDBACK_HEADER}$prior_feedback",
        critic_user_prompt_template=f"{TASK_HEADER}$task_prompt{PROPOSAL_HEADER}$proposal",
    )


def sheller_task(concept: ShellerConcept, priority: str = "kernel protection") -> str:
    """Pass the participant's chosen concept and review priority to the loop."""
    if priority not in ("kernel protection", "cleaning access"):
        raise ValueError(
            "Use kernel protection or cleaning access as the first review priority."
        )
    return json.dumps(
        {"brief": BRIEF, "concept": asdict(concept), "priority": priority}
    )


def propose_sheller(prompt: str) -> dict[str, object]:
    task, feedback = _split_prompt(prompt, FEEDBACK_HEADER)
    concept = ShellerConcept(**json.loads(task)["concept"])
    if "Kernel protection" in feedback:
        concept = replace(concept, kernel_protection=PROTECTIONS[concept.crack])
    if "Cleaning access" in feedback:
        concept = replace(concept, cleaning_access=CLEANING[concept.separate])
    return asdict(concept)


def critique_sheller(prompt: str) -> dict[str, object]:
    task, proposal = _split_prompt(prompt, PROPOSAL_HEADER)
    concept = ShellerConcept(**json.loads(proposal))
    goals = []
    if concept.kernel_protection == FIXED_PROTECTION:
        goals.append(
            "Kernel protection needs an adjustable limit for different pod sizes."
        )
    if concept.cleaning_access == FIXED_CLEANING:
        goals.append(
            "Cleaning access needs a removable separator and a clear debris path."
        )
    if json.loads(task)["priority"] == "cleaning access":
        goals.reverse()
    return {"approved": not goals, "feedback": " ".join(goals), "revision_goals": goals}
