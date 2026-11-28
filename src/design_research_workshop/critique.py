"""Workshop proposer and critic policies for the laptop configuration demo."""

from __future__ import annotations

import json

from design_research.agents import OpenAICompatibleHTTPLLMClient, ProposeCriticPattern

from .laptop import LaptopPolicy

PROPOSER = "Workshop laptop proposer. Reply with a JSON configuration."
CRITIC = (
    "Workshop laptop critic. Check the requested limits and return structured feedback."
)
TASK_HEADER = "Task:\n"
FEEDBACK_HEADER = "\n\nPrior feedback:\n"
PROPOSAL_HEADER = "\n\nProposal:\n"
FIRST_DRAFT = "Choose a student laptop configuration as JSON."


def laptop_propose_critic(
    client: OpenAICompatibleHTTPLLMClient,
) -> ProposeCriticPattern:
    """Build the ecosystem's propose–critic loop with the workshop prompts."""
    return ProposeCriticPattern(
        llm_client=client,
        max_iterations=2,
        proposer_system_prompt=PROPOSER,
        critic_system_prompt=CRITIC,
        proposer_user_prompt_template=f"{TASK_HEADER}$task_prompt{FEEDBACK_HEADER}$prior_feedback",
        critic_user_prompt_template=f"{TASK_HEADER}$task_prompt{PROPOSAL_HEADER}$proposal",
    )


def propose(policy: LaptopPolicy, prompt: str) -> dict[str, float]:
    task, feedback = _split_prompt(prompt, FEEDBACK_HEADER)
    if feedback.strip() == "(none)":
        policy.allowed_levels(task)
        return policy.complete(FIRST_DRAFT)
    return policy.complete(task)


def critique(policy: LaptopPolicy, prompt: str) -> dict[str, object]:
    task, proposal = _split_prompt(prompt, PROPOSAL_HEADER)
    _, allowed = policy.allowed_levels(task)
    candidate = json.loads(proposal)
    if not isinstance(candidate, dict) or candidate.keys() != allowed.keys():
        goals = ["Return the five laptop catalog attributes in one JSON object."]
    else:
        goals = [
            f"Choose {policy.labels[key]} from {', '.join(f'{level:g}' for level in levels)}."
            for key, levels in allowed.items()
            if isinstance(candidate[key], bool) or candidate[key] not in levels
        ]
    return {"approved": not goals, "feedback": " ".join(goals), "revision_goals": goals}


def _split_prompt(prompt: str, separator: str) -> tuple[str, str]:
    task, marker, remainder = prompt.removeprefix(TASK_HEADER).partition(separator)
    if not prompt.startswith(TASK_HEADER) or not marker:
        raise ValueError("Use the workshop proposer or critic prompt template.")
    return task, remainder
