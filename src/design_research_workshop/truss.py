"""A ReAct agent that compares and builds designs from the planar-truss grammar."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from itertools import combinations

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
COMPARE = "truss.compare_layouts"
ADD = "truss.add_member"

MASS = "minimum mass"
SAFETY = "maximum factor of safety"
DEFLECTION = "minimum deflection"
GOALS = (MASS, SAFETY, DEFLECTION)
GOAL_PROMPT = (
    "This exercise uses four fixed joints on a 10 m span, one 1,000 N downward "
    "load plus self-weight, and uniform member stock. Choose minimum mass, maximum factor of safety, "
    "or minimum deflection. You can also specify a mass limit in kg "
    "or a minimum factor of safety."
)
# Match objective phrases with their requested direction.
_LOW = r"(?:minimi[sz](?:e|ing)|reduc(?:e|ing)|lower(?:ing)?|min|minimum|low|lowest|smallest|least|less)"
_HIGH = r"(?:maximi[sz](?:e|ing)|increas(?:e|ing)|improv(?:e|ing)|high|highest|greatest|largest|max|maximum|most)"
_PREFIX = (
    r"\s+(?:(?:the|its|truss|total|overall|peak|joint|possible|minimum|maximum)\s+)*"
)
_FOS = r"(?:factor of safety|safety factor|safety margin|fos)"
_GOAL_WORDS = {
    MASS: rf"\b(?:lightest|lightweight|lighter|as light as possible|weighs? as little as possible|{_LOW}{_PREFIX}(?:mass|weight|(?:amount of )?material))\b",
    SAFETY: rf"\b(?:safest|safer|{_HIGH}{_PREFIX}{_FOS})\b",
    DEFLECTION: rf"\b(?:stiffest|stiffer|{_HIGH}{_PREFIX}stiffness|{_LOW}{_PREFIX}(?:deflection|displacement))\b",
}
_UNSUPPORTED = re.compile(
    rf"\b(?:cost|price|cheapest|heaviest|heavier|"
    rf"{_HIGH}{_PREFIX}(?:mass|weight|deflection|displacement)|"
    rf"(?:minimi[sz](?:e|ing)|reduc(?:e|ing)|lower(?:ing)?|lowest){_PREFIX}(?:{_FOS}|stiffness)|"
    rf"at least as stiff|as stiff as|not|never|don['’]t)\b"
)
# Additional design requirements must not silently disappear when an objective
# matches. The workshop varies member connectivity on a fixed analysis model.
_FIXED_FEATURE = (
    r"loads?|loading|forces?|materials?|stock|sections?|cross sections?|geometry|"
    r"joints?|supports?|span|height|length|angles?|coordinates|diameters?|thickness"
)
_UNMODELED = re.compile(
    rf"\b(?:{_FIXED_FEATURE}|lateral|horizontal|wind|seismic|thermal|impact|"
    r"fatigue|life cycle|lifespan|durability|carbon|emissions|vibration|buckling|"
    r"gravity|self weight|symmetr(?:ic(?:al)?|y)|cantilever(?:ed)?|longer bars?|shorter bars?|"
    r"yielding|resonance|alumin(?:i)?um|steel|timber|wood|plastic|composite)\b"
)
_SAME_MODEL = re.compile(
    rf"\b(?:given|same|fixed|existing|packaged|default) (?:{_FIXED_FEATURE})\b"
)
_OTHER_OBJECTIVE = re.compile(
    r"\b(?:maximi[sz](?:e|ing)|minimi[sz](?:e|ing)|increas(?:e|ing)|reduc(?:e|ing))\b"
)
_NUMBER = r"(?<![-−.\d])(?P<value>\d+(?:\.\d+)?)"
_MASS_CAP = re.compile(
    rf"\b(?:(?P<strict>under|below|less than)|at most|no more than|not more than|do not exceed|up to|"
    rf"max(?:imum)?(?: (?:mass|weight))?(?: of)?|cap (?:it|the (?:mass|weight)) at|"
    rf"(?:mass|weight) (?:cap|limit|budget)(?: of| is| to| at)?)\s+{_NUMBER}\s*(?:kg|kilograms?)\b"
)
_SAFETY_FLOOR = re.compile(
    rf"\b(?:{_FOS}\s+(?:of\s+)?(?:(?P<strict>above|over|greater than)|at least|no less than)|"
    rf"minimum {_FOS}(?: of)?)\s+{_NUMBER}\b"
)
_MASS_SUFFIX = re.compile(
    rf"\b{_NUMBER}\s*(?:kg|kilograms?) (?:mass |weight )?(?:limit|cap|budget)\b"
)
_SAFETY_SUFFIX = re.compile(
    rf"\b(?:(?P<strict>above|over)|at least|no less than)\s+{_NUMBER}\s+(?:{_FOS})\b"
)
_CLEAR_MASS = re.compile(
    r"\b(?:remove|drop|lift|clear) (?:the )?(?:mass|weight) (?:limit|cap|budget)\b"
)
_CLEAR_FOS = re.compile(
    rf"\b(?:remove|drop|clear) (?:the )?(?:minimum {_FOS}|{_FOS} (?:limit|floor))\b"
)


@dataclass(frozen=True)
class TrussBrief:
    objective: str
    max_mass_kg: float | None = None
    min_fos: float = 1.0
    strict_mass_cap: bool = False
    strict_fos_floor: bool = False

    def __post_init__(self) -> None:
        if self.objective not in GOALS:
            raise ValueError(GOAL_PROMPT)
        if (
            type(self.min_fos) not in (int, float)
            or not math.isfinite(self.min_fos)
            or self.min_fos < 1
        ):
            raise ValueError(
                "Set the minimum factor of safety to a finite value of at least 1."
            )
        if self.max_mass_kg is not None and (
            type(self.max_mass_kg) not in (int, float)
            or not math.isfinite(self.max_mass_kg)
            or self.max_mass_kg <= 0
        ):
            raise ValueError("Set the mass limit to a positive, finite value in kg.")

    @classmethod
    def from_text(cls, text: str, *, previous: TrussBrief | None = None) -> TrussBrief:
        """Read an objective or revise the bounds of a prior truss brief."""
        text = re.sub(r"(?<=\w)[-‐‑](?=[a-z])", " ", text.casefold())
        text = text.replace(">=", " at least ").replace("≥", " at least ")
        text = text.replace("<=", " at most ").replace("≤", " at most ")
        mass = _MASS_CAP.search(text) or _MASS_SUFFIX.search(text)
        safety = _SAFETY_FLOOR.search(text) or _SAFETY_SUFFIX.search(text)
        clear_mass = _CLEAR_MASS.search(text)
        clear_fos = _CLEAR_FOS.search(text)
        # Ignore bounds while identifying the objective. Any leftover number
        # signals a constraint or unit outside the supported vocabulary.
        objective_text = text
        for clause in (mass, safety, clear_mass, clear_fos):
            if clause:
                objective_text = objective_text.replace(clause[0], "", 1)
        if re.search(r"\d", objective_text) or _UNSUPPORTED.search(objective_text):
            raise ValueError(GOAL_PROMPT)
        found = [
            goal
            for goal, pattern in _GOAL_WORDS.items()
            if re.search(pattern, objective_text)
        ]
        remaining = objective_text
        for pattern in _GOAL_WORDS.values():
            remaining = re.sub(pattern, "", remaining)
        remaining = _SAME_MODEL.sub("", remaining)
        if _UNMODELED.search(remaining) or _OTHER_OBJECTIVE.search(remaining):
            raise ValueError(GOAL_PROMPT)
        if (
            not found
            and previous
            and any((mass, safety, clear_mass, clear_fos))
            and not re.search(rf"\b(?:{_LOW}|{_HIGH})\b", objective_text)
        ):
            found = [previous.objective]
        if len(found) != 1:
            raise ValueError(GOAL_PROMPT)

        settings = asdict(previous) if previous else {}
        settings["objective"] = found[0]
        if clear_mass:
            settings.update(max_mass_kg=None, strict_mass_cap=False)
        if clear_fos:
            settings.update(min_fos=1.0, strict_fos_floor=False)
        if mass:
            settings.update(
                max_mass_kg=float(mass["value"]),
                strict_mass_cap=bool(mass.groupdict().get("strict")),
            )
        if safety:
            settings.update(
                min_fos=float(safety["value"]), strict_fos_floor=bool(safety["strict"])
            )
        return cls(**settings)

    def select(self, layouts: list[dict], current: set[tuple]) -> dict | None:
        """Rank feasible completions that preserve every member already placed."""
        max_mass = math.inf if self.max_mass_kg is None else self.max_mass_kg
        if self.strict_mass_cap:
            max_mass = math.nextafter(max_mass, -math.inf)
        min_fos = self.min_fos
        if self.strict_fos_floor:
            min_fos = math.nextafter(min_fos, math.inf)
        compatible = [
            layout
            for layout in layouts
            if current.issubset(map(tuple, layout["members"]))
            and layout["factor_of_safety"] >= min_fos
            and layout["mass_kg"] <= max_mass
        ]
        if not compatible:
            return None

        # Resolve mirror-image ties by deflection, then by canonical joint IDs.
        def rank(layout):
            metric = {
                MASS: layout["mass_kg"],
                SAFETY: -layout["factor_of_safety"],
                DEFLECTION: layout["max_deflection_mm"],
            }[self.objective]
            return (
                round(metric, 9),
                round(layout["mass_kg"], 9),
                round(layout["max_deflection_mm"], 9),
                layout["members"],
            )

        return min(compatible, key=rank)


class TrussDesign:
    """Hold a design, enumerate member additions, and evaluate fixed-joint layouts."""

    def __init__(self, *, center_joint: bool = False) -> None:
        self.grammar = get_problem(TRUSS_PROBLEM_ID)
        self.state = self.grammar.initial_state()
        if center_joint:
            # Use a candidate emitted by the same grammar that supplies member edits.
            center = {"x": self.state.span / 2, "y": self.state.max_height / 2}
            self.state = next(
                t.next_state
                for t in self.grammar.enumerate_transitions(self.state)
                if t.rule_name == "add_joint" and dict(t.parameters) == center
            )

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
            "members": [[m.start_joint_id, m.end_joint_id] for m in self.state.members],
            "legal_additions": [list(edge) for edge in self._legal_additions()],
        }

    def compare(self, _arguments: Mapping[str, object]) -> dict:
        """Analyze every potentially rigid completion on this exercise's joint set."""
        observation = self.inspect({})
        legal = observation["legal_additions"]
        # With one pin and one roller, at least 2j - 3 bars are needed for rigidity.
        minimum = max(0, 2 * len(self.state.joints) - 3 - len(self.state.members))
        layouts = []
        evaluated = 0
        for count in range(minimum, len(legal) + 1):
            for edges in combinations(legal, count):
                candidate = self.state
                for start, end in edges:
                    candidate = self.grammar.add_member(
                        candidate, start_joint_id=start, end_joint_id=end
                    )
                result = self.grammar.evaluate(candidate)
                evaluated += 1
                metrics = (result.mass, result.fos, result.deflection)
                if result.is_feasible and all(
                    math.isfinite(v) and v > 0 for v in metrics
                ):
                    layouts.append(
                        {
                            "members": sorted(
                                [m.start_joint_id, m.end_joint_id]
                                for m in candidate.members
                            ),
                            "mass_kg": result.mass,
                            "factor_of_safety": result.fos,
                            "max_deflection_mm": result.deflection * 1_000,
                        }
                    )
        return {**observation, "evaluated": evaluated, "feasible_layouts": layouts}

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
    """Bind geometry inspection, layout analysis and member editing to a ReAct loop."""
    with Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableToolConfig(
                name=INSPECT,
                description="Inspect current members and legal additions.",
                handler=design.inspect,
            ),
            CallableToolConfig(
                name=COMPARE,
                description="Compare the mass, safety factor and deflection of feasible completions.",
                handler=design.compare,
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
            allowed_tools=(INSPECT, COMPARE, ADD),
            max_steps=10,
            step_memory_tail_items=30,
            tool_calling_system_prompt=TRUSS_SYSTEM,
            step_user_prompt_template="Task JSON: $task_prompt\nMemory JSON: $memory_tail",
            tool_calling_user_prompt_template="$user_prompt\n$choices_block",
        )


def _no_design(brief: TrussBrief, current: set[tuple]) -> dict:
    return {
        "status": "no feasible completion",
        **asdict(brief),
        "members": sorted(current),
    }


def truss_reply(prompt: str, *, previous: Mapping | None = None) -> dict:
    """Compare four-joint layouts, preserving prior truss constraints on follow-ups."""
    last = None
    if previous and previous.get("objective") in GOALS:
        last = TrussBrief(
            **{
                f.name: previous[f.name]
                for f in fields(TrussBrief)
                if f.name in previous
            }
        )
    try:
        brief = TrussBrief.from_text(prompt, previous=last)
    except ValueError as error:
        return {
            **(asdict(last) if last else {}),
            "status": "clarification needed",
            "message": str(error),
        }
    design = TrussDesign(center_joint=True)
    analysis = design.compare({})
    choice = brief.select(analysis["feasible_layouts"], set())
    return {
        **asdict(brief),
        "joint_count": len(design.state.joints),
        "span_m": design.state.span,
        "applied_vertical_load_n": design.state.load_vector[1],
        "evaluated": analysis["evaluated"],
        "feasible_layouts": len(analysis["feasible_layouts"]),
        **(choice if choice is not None else _no_design(brief, set())),
    }


def truss_action(prompt: str) -> dict[str, object]:
    """Choose the next tool or final answer from the objective and actual observations."""
    task_text = prompt.removeprefix("Task JSON: ").split("\nMemory JSON:", 1)[0]
    if task_text.lstrip().startswith("{"):
        task = json.loads(task_text)
    else:
        task = {"brief": task_text}
    memory = json.loads(
        next(
            line.removeprefix("Memory JSON: ")
            for line in prompt.splitlines()
            if line.startswith("Memory JSON: ")
        )
    )
    observations = [r["final_output"] for r in memory if r.get("kind") == "observation"]
    if not observations:
        return _action(INSPECT, {}, "Inspect current geometry and allowed edits.")
    states = [o.get("result", o) for o in observations]
    current = {tuple(sorted(edge)) for edge in states[-1]["members"]}
    legal = {tuple(edge) for edge in states[-1]["legal_additions"]}
    attempted = task.get("try_first")

    # Older notebooks and callers can still request a specific member list.
    if "target_members" in task:
        if len(states) == 1 and attempted is not None:
            return _add_action(
                attempted, "Check the proposed first member against the grammar."
            )
        needed = {tuple(sorted(edge)) for edge in task["target_members"]} - current
        if available := needed & legal:
            return _add_action(
                min(available), "Add a requested edge allowed by the observation."
            )
        return _action(
            "final_answer",
            {"members": sorted(current), "missing": sorted(needed)},
            "Report the observed design and any requested edges that remain unavailable.",
        )

    try:
        if "objective" in task:
            brief = TrussBrief(
                **{f.name: task[f.name] for f in fields(TrussBrief) if f.name in task}
            )
        else:
            brief = TrussBrief.from_text(task.get("brief", ""))
    except ValueError as error:
        return _action(
            "final_answer",
            {"status": "clarification needed", "message": str(error)},
            "Request a supported engineering objective.",
        )
    analysis = next(
        (state for state in reversed(states) if "feasible_layouts" in state), None
    )
    if analysis is None:
        return _action(
            COMPARE, {}, "Ask the structural solver to evaluate possible completions."
        )
    if len(states) == 2 and attempted is not None:
        return _add_action(
            attempted, "Try the first joint pair proposed in the design brief."
        )
    choice = brief.select(analysis["feasible_layouts"], current)
    if choice is None:
        return _action(
            "final_answer",
            _no_design(brief, current),
            "No analyzed completion satisfies the brief and the existing geometry.",
        )
    missing = {tuple(edge) for edge in choice["members"]} - current
    if available := missing & legal:
        return _add_action(
            min(available), f"Add a legal member of the layout with {brief.objective}."
        )
    if missing:
        return _action(
            "final_answer",
            {
                "status": "unavailable edits",
                "members": sorted(current),
                "missing": sorted(missing),
            },
            "Report the blocked edits from the latest observation.",
        )
    return _action(
        "final_answer",
        {**asdict(brief), **choice},
        "Report the measured layout after observing every chosen member.",
    )


def _add_action(edge, reason: str) -> dict[str, object]:
    return _action(ADD, {"start": edge[0], "end": edge[1]}, reason)


def _action(name: str, arguments: dict[str, object], reason: str) -> dict[str, object]:
    return {"tool_name": name, "tool_input": arguments, "reason": reason}
