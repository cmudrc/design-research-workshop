"""A ReAct-style agent that acts on the packaged planar-truss grammar."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from design_research.agents import (
    CallableToolConfig,
    MultiStepAgent,
    OpenAICompatibleHTTPLLMClient,
    Toolbox,
)
from design_research.problems import get_problem

TRUSS_PROBLEM_ID = "planar_truss_span"
TRUSS_SYSTEM = "Workshop truss agent. Choose one action using the latest observation."
INSPECT = "truss.inspect"
ADD = "truss.add_member"


class TrussDesign:
    """Hold one design and expose only the moves participants practice."""

    def __init__(self) -> None:
        self.grammar = get_problem(TRUSS_PROBLEM_ID)
        self.state = self.grammar.initial_state()

    def _legal_additions(self):
        additions = {}
        for rule in self.grammar.enumerate_transitions(self.state):
            if rule.rule_name == "add_member":
                args = dict(rule.parameters)
                additions[(args["start_joint_id"], args["end_joint_id"])] = (
                    rule.next_state
                )
        return additions

    def inspect(self, _arguments: Mapping[str, object]) -> dict[str, object]:
        return {
            "members": [
                [member.start_joint_id, member.end_joint_id]
                for member in self.state.members
            ],
            "legal_additions": [list(edge) for edge in self._legal_additions()],
        }

    def add(self, arguments: Mapping[str, object]) -> dict[str, object]:
        start, end = arguments["start"], arguments["end"]
        if type(start) is not int or type(end) is not int:
            raise ValueError("Use integer joint IDs.")
        edge = (min(start, end), max(start, end))
        next_state = self._legal_additions().get(edge)
        if next_state is None:
            message = (
                f"Rejected {edge[0]}–{edge[1]}. That member is outside the legal moves."
            )
        else:
            self.state = next_state
            message = f"Added member {edge[0]}–{edge[1]}."
        return {**self.inspect({}), "status": message}


@contextmanager
def truss_agent(
    client: OpenAICompatibleHTTPLLMClient, design: TrussDesign
) -> Iterator[MultiStepAgent]:
    """Bind two real tools to the ecosystem's JSON action–observation loop."""
    with Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableToolConfig(
                name=INSPECT,
                description="Inspect the members and legal additions.",
                handler=design.inspect,
            ),
            CallableToolConfig(
                name=ADD,
                description="Attempt to add a member between two joints.",
                handler=design.add,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end"],
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                },
            ),
        ),
    ) as tools:
        yield MultiStepAgent(
            mode="json",
            llm_client=client,
            tool_runtime=tools,
            allowed_tools=(INSPECT, ADD),
            max_steps=7,
            step_memory_tail_items=30,
            tool_calling_system_prompt=TRUSS_SYSTEM,
            step_user_prompt_template="Task JSON: $task_prompt\nMemory JSON: $memory_tail",
            tool_calling_user_prompt_template="$user_prompt\n$choices_block",
        )


def truss_action(prompt: str) -> dict[str, object]:
    """Return a short decision and action based on the agent's observed tool state."""
    task = _json_line(prompt, "Task JSON: ")
    memory = _json_line(prompt, "Memory JSON: ")
    observations = [
        record["final_output"]
        for record in memory
        if record.get("kind") == "observation"
    ]
    if not observations:
        return _action(INSPECT, {}, "Inspect current geometry and allowed edits.")

    observation = observations[-1]
    # The action runner wraps the callable's return value in a tool envelope.
    state = observation.get("result", observation)
    attempted = task.get("try_first")
    if len(observations) == 1 and attempted is not None:
        return _add_action(
            attempted, "Check the proposed first member against the grammar."
        )

    current = {tuple(sorted(edge)) for edge in state["members"]}
    legal = {tuple(edge) for edge in state["legal_additions"]}
    needed = {tuple(sorted(edge)) for edge in task["target_members"]} - current
    available = needed & legal
    if available:
        return _add_action(
            min(available), "Add a requested edge that the last observation allows."
        )
    return _action(
        "final_answer",
        {"members": sorted(current), "missing": sorted(needed)},
        "Report the observed design and any requested edges that remain unavailable.",
    )


def _json_line(prompt: str, label: str):
    return json.loads(
        next(
            line[len(label) :] for line in prompt.splitlines() if line.startswith(label)
        )
    )


def _add_action(edge, reason: str) -> dict[str, object]:
    return _action(ADD, {"start": edge[0], "end": edge[1]}, reason)


def _action(name: str, arguments: dict[str, object], reason: str) -> dict[str, object]:
    return {"tool_name": name, "tool_input": arguments, "reason": reason}
