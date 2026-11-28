"""Review a design record against a catalog of mechanism-specific provisions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict


class ProvisionReview(TypedDict):
    approved: bool
    feedback: str
    revision_goals: list[str]


def _evidence_key(value: str) -> str:
    text = " ".join(re.findall(r"[\w]+", value.casefold()))
    # Accept a direct statement that the feature is present without discarding
    # qualifiers such as "could use", "needs", or "does not have".
    text = re.sub(
        r"^(?:(?:(?:our|this|the) (?:design|device|prototype|mechanism)|it) (?:has|uses|includes)"
        r"|we (?:use|installed|added)) ",
        "",
        text,
    )
    words = text.split()
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words)


@dataclass(frozen=True)
class ProvisionCriterion:
    """One design requirement, its evidence vocabulary, and a repair for each mechanism."""

    label: str
    field: str
    mechanism_field: str
    request: str
    repairs: Mapping[str, str]
    alternatives: Mapping[str, tuple[str, ...]]
    missing_examples: tuple[str, ...] = ()

    def evidence_status(
        self, proposal: Mapping[str, str]
    ) -> Literal["accepted", "missing", "unrecognized"]:
        """Classify the submitted description against this mechanism's evidence vocabulary."""
        mechanism = proposal[self.mechanism_field]
        expected = (self.repairs[mechanism], *self.alternatives.get(mechanism, ()))
        candidate = _evidence_key(proposal[self.field])
        if any(candidate == _evidence_key(value) for value in expected):
            return "accepted"
        absent = (
            "",
            "none",
            "not specified",
            "unspecified",
            "not provided",
            *self.missing_examples,
        )
        if any(candidate == _evidence_key(value) for value in absent):
            return "missing"
        return "unrecognized"

    def accepts(self, proposal: Mapping[str, str]) -> bool:
        return self.evidence_status(proposal) == "accepted"


def review_provisions(
    proposal: Mapping[str, str],
    criteria: Sequence[ProvisionCriterion],
    *,
    changes_per_revision: int,
    revise_unrecognized: bool = True,
) -> ProvisionReview:
    """Report missing provisions in priority order and assign the next round's changes."""
    if not criteria:
        raise ValueError("Choose at least one review criterion.")
    if type(changes_per_revision) is not int or changes_per_revision < 1:
        raise ValueError("Use a positive integer change budget.")
    feedback = []
    goals = []
    changes = 0
    for criterion in criteria:
        current = proposal[criterion.field]
        status = criterion.evidence_status(proposal)
        if status == "accepted":
            # The Agents pattern passes goals and feedback, but no prior draft, to the proposer.
            goals.append(f"Keep {criterion.label.lower()}: {current}")
        else:
            if status == "missing":
                feedback.append(f"{criterion.label} is missing. {criterion.request}")
            else:
                feedback.append(
                    f"{criterion.label} has an unrecognized description. "
                    "A human reviewer can judge the existing feature."
                )
            if changes < changes_per_revision and (
                status == "missing" or revise_unrecognized
            ):
                repair = criterion.repairs[proposal[criterion.mechanism_field]]
                goals.append(f"Change {criterion.label.lower()}: {repair}")
                changes += 1
    return {
        "approved": not feedback,
        "feedback": " ".join(feedback),
        "revision_goals": goals if feedback else [],
    }


def apply_revision_goals(
    seed: Mapping[str, str],
    criteria: Sequence[ProvisionCriterion],
    goals: Iterable[str],
) -> dict[str, str]:
    """Apply the critic's named, supported provisions to a copy of the starting design."""
    revised = dict(seed)
    by_label = {criterion.label.casefold(): criterion for criterion in criteria}
    for goal in goals:
        directive, _, evidence = goal.partition(": ")
        action, _, label = directive.partition(" ")
        if action not in {"Change", "Keep"} or label.casefold() not in by_label:
            raise ValueError(f"Unknown revision goal: {goal!r}.")
        criterion = by_label[label.casefold()]
        candidate = {**revised, criterion.field: evidence}
        if not criterion.accepts(candidate):
            raise ValueError(f"Unrecognized {label.lower()} provision: {evidence!r}.")
        revised = candidate
    return revised
