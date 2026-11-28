"""Review a design record against a catalog of mechanism-specific provisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict


class ProvisionReview(TypedDict):
    approved: bool
    feedback: str
    revision_goals: list[str]


def _evidence_key(value: str) -> str:
    words = value.strip().casefold().rstrip(".").replace("-", " ").split()
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

    def accepts(self, proposal: Mapping[str, str]) -> bool:
        mechanism = proposal[self.mechanism_field]
        expected = (self.repairs[mechanism], *self.alternatives.get(mechanism, ()))
        candidate = _evidence_key(proposal[self.field])
        return any(candidate == _evidence_key(value) for value in expected)


def review_provisions(
    proposal: Mapping[str, str],
    criteria: Sequence[ProvisionCriterion],
    *,
    changes_per_revision: int,
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
        if criterion.accepts(proposal):
            # The Agents pattern passes goals and feedback, but no prior draft, to the proposer.
            goals.append(f"Keep {criterion.label.lower()}: {current}")
        else:
            feedback.append(f"{criterion.label} needs attention. {criterion.request}")
            if changes < changes_per_revision:
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
