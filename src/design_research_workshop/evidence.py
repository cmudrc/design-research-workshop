"""Explain the results already recorded in a workshop conversation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bot import Content


def _record(result: Content) -> Content:
    """The sheller critic wraps a copy of the design it reviewed."""
    concept = result.get("concept")
    return concept if isinstance(concept, dict) else result


def compare_results(current: Content, previous: Content | None) -> Content:
    if previous is None:
        return {
            "explanation": "There is one recorded result for this task. Change an input and run it again to compare."
        }
    before, after = _record(previous), _record(current)
    if "concepts" in before and "concepts" in after:
        keys = (
            "strategy",
            "seed",
            "available",
            "previously_shown",
            "chart",
            "concepts",
        )
    elif "members" in before and "members" in after:
        keys = (
            "status",
            "objective",
            "min_fos",
            "max_mass_kg",
            "strict_mass_cap",
            "strict_fos_floor",
            "best_fos_within_mass_limit",
            "least_mass_kg_meeting_fos_floor",
            "members",
            "mass_kg",
            "factor_of_safety",
            "max_deflection_mm",
        )
    else:
        if "approved" in current and "approved" in previous:
            review_fields = ("approved", "criteria", "unassessed", "revision_goals")
            before = {
                **before,
                **{k: previous[k] for k in review_fields if k in previous},
            }
            after = {**after, **{k: current[k] for k in review_fields if k in current}}
        keys = tuple(dict.fromkeys((*before, *after)))
    changes = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in keys
        if (key in before) != (key in after) or before.get(key) != after.get(key)
    }
    for key in ("mass_kg", "factor_of_safety", "max_deflection_mm"):
        if key in changes and all(
            isinstance(x, (int, float)) for x in changes[key].values()
        ):
            changes[key]["delta"] = after[key] - before[key]
    if changes:
        return {
            "explanation": "These fields changed between the two recorded runs.",
            "changes": changes,
        }
    if "approved" in current:
        return {
            "explanation": "The critic reviewed the same design.",
            "approved": current["approved"],
            "feedback": current.get("feedback", ""),
        }
    return {"explanation": "The recorded design fields are identical.", "changes": {}}


def brainstorm_evidence(result: Content) -> Content:
    concepts = result.get("concepts")
    if not concepts or "strategy" not in result:
        return {
            "explanation": result.get(
                "question", "The response contains no ranked concepts."
            )
        }
    first = concepts[:3]
    used = {(function, option) for idea in first for function, option in idea.items()}
    policy = (
        "The function-first rule gave priority to unused mechanisms, then to less repeated options."
        if result["strategy"] == "function-first"
        else "The direct rule preferred combinations whose mechanisms appear earlier in the chart."
    )
    prefix = result["trace"][: len(first)]
    sentence = f"{policy} The first {len(first)} concepts in this batch used {len(used)} distinct function–mechanism pairs."
    if result.get("previously_shown"):
        introduced = sum(len(step["new_functions"]) for step in prefix)
        sentence += (
            f" They added {introduced} previously unseen pairs to the conversation."
        )
    return {
        "explanation": sentence,
        "evidence": {
            "strategy": result["strategy"],
            "available_combinations": result["available"],
            "seed": result["seed"],
            "mechanisms_in_first_concepts": len(used),
            "selection_trace": prefix,
        },
    }


def sheller_evidence(result: Content) -> Content:
    if "approved" in result:
        feedback = result.get("feedback")
        return {
            "explanation": feedback
            or "The critic recognized a provision for each requested review criterion.",
            "evidence": {
                k: result[k]
                for k in (
                    "approved",
                    "criteria",
                    "revision_goals",
                    "evidence_status",
                    "unassessed",
                    "concept",
                )
                if k in result
            },
        }
    return {
        "explanation": "The proposer selected the following mechanisms and provisions. Ask for a review to check them against the design criteria.",
        "evidence": result,
    }


def truss_evidence(result: Content) -> Content:
    if "evaluated" not in result:
        return {
            "explanation": result.get(
                "message",
                result.get(
                    "reason", "The agent has not returned a structural comparison."
                ),
            )
        }
    objective = result["objective"]
    if result.get("status") == "no feasible completion":
        sentence = (
            "No evaluated layout on these joints satisfied all the requested bounds."
        )
        affordable = result.get("best_fos_within_mass_limit")
        qualifying = result.get("least_mass_kg_meeting_fos_floor")
        if affordable is not None:
            sentence += (
                f" Within the mass limit the best safety factor was {affordable:.3f}."
            )
        if qualifying is not None:
            sentence += f" The lightest layout satisfying the safety floor weighed {qualifying:.2f} kg."
    else:
        sentence = f"The {objective} objective selected a layout with {len(result['members'])} members from the designs that met the bounds."
    evidence = {
        key: result[key]
        for key in (
            "objective",
            "joint_count",
            "span_m",
            "applied_vertical_load_n",
            "status",
            "strict_mass_cap",
            "strict_fos_floor",
            "best_fos_within_mass_limit",
            "least_mass_kg_meeting_fos_floor",
            "min_fos",
            "max_mass_kg",
            "evaluated",
            "members",
            "mass_kg",
            "factor_of_safety",
            "max_deflection_mm",
        )
        if key in result
    }
    evidence["structurally_feasible_before_requested_bounds"] = result[
        "feasible_layouts"
    ]
    return {"explanation": sentence, "evidence": evidence}
