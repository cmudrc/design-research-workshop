"""Rule-based proposals for the packaged student laptop design problem."""

from __future__ import annotations

import re
from operator import ge, le, lt

from design_research.agents import (
    OpenAICompatibleHTTPLLMClient,
    PromptWorkflowAgent,
    build_json_prompt_workflow,
)
from design_research.problems import get_problem

PROBLEM_ID = "decision_laptop_design_profit_maximization"
LAPTOP_TASK = re.compile(
    r"\b(?:student laptop|laptop (?:design|configuration))\b", re.I
)
CHOOSE = re.compile(r"\b(?:choose|select|recommend|suggest|pick|prioritize)\b", re.I)
PROFIT_GOAL = re.compile(
    r"\b(?:profit[- ]focus(?:ed)?|market share|expected demand)\b", re.I
)
DOLLARS = r"\$\s*([\d,]+(?:\.\d+)?)\b"
NUMBER = r"(\d+(?:\.\d+)?)"
CONSTRAINTS = (
    ("z5", 100, lt, re.compile(r"\b(?:under|below|less than)\s+" + DOLLARS, re.I)),
    (
        "z5",
        100,
        le,
        re.compile(
            r"\b(?:at most|up to|no more than|max(?:imum)?(?: price)?(?: of)?|"
            r"price (?:cap|ceiling)(?: of)?)\s+" + DOLLARS,
            re.I,
        ),
    ),
    (
        "z3",
        1,
        ge,
        re.compile(
            r"\b(?:at least|minimum(?: of)?)\s+" + NUMBER + r"\s*(?:hours?|hrs?)\b",
            re.I,
        ),
    ),
    (
        "z4",
        1,
        le,
        re.compile(
            r"\b(?:weigh(?:s|ing)?\s+(?:at most|no more than)|max(?:imum)? weight(?: of)?)\s+"
            + NUMBER
            + r"\s*(?:lbs?|pounds?)\b",
            re.I,
        ),
    ),
)
BALANCED_TARGETS = {"z1": 14.1, "z2": 1.0, "z3": 6.0, "z4": 4.5, "z5": 10.0}
DEMAND_TARGETS = {"z1": 14.1, "z2": 0.75, "z3": 8.0, "z4": 2.5, "z5": 7.5}


class LaptopPolicy:
    """Choose discrete candidates using preferences independent of the study evaluator."""

    def __init__(self) -> None:
        problem = get_problem(PROBLEM_ID)
        self.labels = {factor.key: factor.label for factor in problem.option_factors}
        self.domains = {factor.key: factor.levels for factor in problem.option_factors}

    def allowed_levels(self, prompt: str) -> tuple[str, dict[str, tuple[float, ...]]]:
        # The canonical walkthrough appends a shared brief describing both
        # objectives. Its instruction block determines the requested goal.
        instructions = prompt.strip()
        if instructions.startswith(
            "You are solving a packaged design-research decision problem."
        ):
            instructions = instructions.partition("\n\n")[0]
        if not (
            LAPTOP_TASK.search(prompt)
            and CHOOSE.search(instructions)
            and re.search(r"\bjson\b", prompt, re.I)
        ):
            raise ValueError(
                "Supported task: choose a student laptop configuration as JSON."
            )

        allowed = self.domains.copy()
        for key, scale, accepts, pattern in CONSTRAINTS:
            limits = [
                float(value.replace(",", "")) / scale
                for value in pattern.findall(instructions)
            ]
            if limits:
                allowed[key] = tuple(
                    value
                    for value in allowed[key]
                    if all(accepts(value, limit) for limit in limits)
                )
                if not allowed[key]:
                    raise ValueError("No catalog option satisfies the requested limit.")

        return instructions, allowed

    def complete(self, prompt: str) -> dict[str, float]:
        instructions, allowed = self.allowed_levels(prompt)
        # A classroom policy, with an explicit screen/portability tradeoff.
        # The downstream evaluator remains responsible for market-share results.
        targets = (
            DEMAND_TARGETS if PROFIT_GOAL.search(instructions) else BALANCED_TARGETS
        )
        return {
            key: min(levels, key=lambda level: abs(level - targets[key]))
            for key, levels in allowed.items()
        }


def laptop_study_agent(
    client: OpenAICompatibleHTTPLLMClient, prompt: str
) -> PromptWorkflowAgent:
    """Create a workshop agent whose JSON output follows the catalog schema."""
    factors = get_problem(PROBLEM_ID).option_factors
    schema = {
        "type": "object",
        "properties": {
            factor.key: {"type": "number", "enum": list(factor.levels)}
            for factor in factors
        },
        "required": [factor.key for factor in factors],
        "additionalProperties": False,
    }
    return PromptWorkflowAgent(
        workflow=build_json_prompt_workflow(
            llm_client=client,
            response_schema=schema,
            request_metadata={
                "problem_id": PROBLEM_ID,
                "backend": "workshop-rule-based",
            },
        ),
        prompt_builder=lambda _packet, _run, _condition: prompt,
    )
