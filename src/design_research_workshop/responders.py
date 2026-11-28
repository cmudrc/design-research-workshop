"""Built-in task rules for the workshop and user-supplied design briefs."""

from __future__ import annotations

import json
import re

from . import brainstorm, peanut, truss
from .bot import Content, Rule, Turn
from .brainstorm import DesignBrief, generate_concepts
from .critique import (
    CRITIC,
    FEEDBACK_HEADER,
    PROPOSAL_HEADER,
    PROPOSER,
    critique,
    propose,
)
from .functions_catalog import infer_functions
from .laptop import CHOOSE, LAPTOP_TASK, LaptopPolicy


def _matches(expression: str):
    pattern = re.compile(expression, re.I)
    return lambda prompt: bool(pattern.search(prompt))


_chart_line = re.compile(r"^([^:\n|]{1,80}):[ \t]*(.*\|.*)$", re.M)
_ideation = _matches(
    r"\b(brainstorm(?:ing)?|ideate|ideation|(?:suggest|generate|give me) (?:\w+ ){0,3}(?:concepts|ideas)|"
    r"(?:show|build|make|create|draw) (?:me )?(?:a |the )?(?:function |morphological )?chart)\b"
)
_sheller_action = re.compile(
    r"\b(review|critique|evaluate|assess|check|audit|what.s missing|"
    r"revise|improve|amend|update|rework|apply|propose)\b",
    re.I,
)
_revision_verbs = {"revise", "improve", "amend", "update", "rework", "apply", "propose"}
_new_design = re.compile(
    r"\b(?:brainstorm|design|devise|invent|sketch) (?:an? |my |our |another |new )"
    r"(?!(?:few|couple|dozen|direct|function[ -]first)\b)",
    re.I,
)


def _sheller_role(prompt: str):
    for verb in _sheller_action.finditer(prompt):
        # A previous critique is evidence for a revision, not a second review request.
        if re.search(
            r"\b(?:the|your|prior|previous|last|that|this)\s+$",
            prompt[: verb.start()],
            re.I,
        ):
            continue
        return "propose" if verb.group().casefold() in _revision_verbs else "critic"
    return "critic" if re.search(r"\bcritic\b", prompt, re.I) else "propose"


def _brainstorm_edit(prompt: str) -> bool:
    # A newly named design brief starts its own conversation scope.
    if _new_design.search(prompt) or (_ideation(prompt) and infer_functions(prompt)):
        return False
    if re.search(
        r"^brief:|\b(?:ideas?|concepts?|designs?|solutions?|ways) (?:for|to) "
        r"(?!(?:it|that|this|the same|those|these)\b)",
        prompt,
        re.I | re.M,
    ):
        return False
    return brainstorm.is_follow_up(prompt)


def _chart_edit(prompt: str) -> bool:
    if re.search(r"^brief:", prompt, re.I | re.M):
        return False
    if _chart_line.search(prompt):
        return not any(
            _ideation(line) for line in prompt.splitlines() if "|" not in line
        )
    return _brainstorm_edit(prompt)


def brainstorm_rule(brief: DesignBrief, *, name: str, topics: tuple[str, ...]) -> Rule:
    """Register another hand-authored function/option chart with the same engine."""
    return Rule(
        name,
        brief.statement,
        topics,
        lambda turn: generate_concepts(brief, turn.user_text),
        follow_up=_brainstorm_edit,
    )


def _open_brainstorm(turn: Turn) -> Content:
    """Build a chart from participant input. Later turns can update individual rows."""
    chart: dict[str, list[str]] = {}
    prompts = [m["content"] for m in turn.history if m["role"] == "user"] + [
        turn.prompt
    ]
    sources: dict[str, dict[str, str]] = {}
    title = ""
    controls: list[str] = []
    for prompt in prompts:
        if not chart and not _chart_line.search(prompt):
            for match in infer_functions(prompt):
                chart[match.name] = list(match.options)
                sources[match.name] = {
                    "source": "workshop catalog",
                    "matched_phrase": match.phrase,
                }
        supplied: set[str] = set()
        for line in prompt.splitlines():
            line = line.strip()
            row = _chart_line.fullmatch(line)
            if row:
                label, raw = row.groups()
                key = label.strip().casefold()
                if key in supplied:
                    raise ValueError("Give each function one row per message.")
                supplied.add(key)
                name = next((k for k in chart if k.casefold() == key), label.strip())
                chart[name] = [item.strip() for item in raw.split("|")]
                sources[name] = {"source": "participant"}
            elif line.casefold().startswith("brief:"):
                title = line.partition(":")[2].strip()
            elif line:
                controls.append(line)
                if not title:
                    title = line
    if not chart:
        return {
            "brief": title,
            "question": "Which functions and mechanism options should the concepts combine?",
            "format": "Use one line per function, for example 'Collect water: tray | gutter'.",
        }
    brief = DesignBrief(
        title or "Your design brief", title or "Combine the supplied options.", chart
    )
    result = generate_concepts(brief, "\n".join(controls) or "Generate concepts.")
    result["chart"] = {
        name: {"options": options, **sources[name]} for name, options in chart.items()
    }
    return result


def _previous_result(turn: Turn) -> Content | None:
    for message in reversed(turn.history):
        if message["role"] == "assistant":
            try:
                result = json.loads(message["content"])
            except ValueError:
                return None
            return result if isinstance(result, dict) else None
    return None


def _peanut(turn: Turn) -> Content:
    if peanut.BRAINSTORM_SYSTEM in turn.systems:
        return peanut.brainstorm(turn.prompt)
    if peanut.PEANUT_CRITIC in turn.systems or PROPOSAL_HEADER in turn.prompt:
        return peanut.critique_sheller(turn.prompt)
    if peanut.PEANUT_PROPOSER in turn.systems or FEEDBACK_HEADER in turn.prompt:
        return peanut.propose_sheller(turn.prompt)
    role = _sheller_role(turn.prompt)
    return peanut.respond_sheller(
        turn.prompt, role=role, previous=_previous_result(turn)
    )


def _truss_protocol(prompt: str) -> bool:
    return (
        prompt.startswith("Task JSON: ")
        and "\nMemory JSON: " in prompt
        and ("truss.inspect" in prompt or "target_members" in prompt)
    )


def _truss(turn: Turn) -> Content:
    if turn.prompt.startswith("Task JSON: "):
        return truss.truss_action(turn.prompt)
    return truss.truss_reply(turn.prompt, previous=_previous_result(turn))


def workshop_rules() -> tuple[Rule, ...]:
    """Construct the responders shared by the Python and loopback interfaces."""
    laptop = LaptopPolicy()

    def laptop_response(turn: Turn) -> Content:
        if CRITIC in turn.systems or PROPOSAL_HEADER in turn.prompt:
            return critique(laptop, turn.prompt)
        if PROPOSER in turn.systems or FEEDBACK_HEADER in turn.prompt:
            return dict(propose(laptop, turn.prompt))
        prompt = turn.user_text
        # The original packaged study explicitly asks for the JSON catalog schema.
        # Direct chat exposes that same output without making users name the format.
        if "json" not in prompt.casefold():
            prompt = "Choose a student laptop configuration as JSON. " + prompt
        return dict(laptop.complete(prompt))

    return (
        Rule(
            "jar-opener",
            brainstorm.JAR_OPENER.statement,
            (
                "jar opener",
                "jar openers",
                "jar lid",
                "jar lids",
                "open jars",
                "open a jar",
                "opening jars",
                "opening a jar",
                "open food jars",
                "open screw top jars",
                "open screw top containers",
                "screw top food jar",
                "opening glass jars",
                "unscrew jar lids",
                "jar opening",
                "lidded containers with one hand",
                "open the jar",
                "lidded container opening",
                "one handed container opener",
            ),
            lambda turn: brainstorm.respond(turn.user_text),
            systems=(brainstorm.BRAINSTORM_SYSTEM,),
            follow_up=_brainstorm_edit,
        ),
        Rule(
            "peanut-sheller",
            "Propose, revise, or review a tabletop peanut sheller.",
            (
                "peanut sheller",
                "peanut shellers",
                "shell peanuts",
                "shelling peanuts",
                "peanut shelling",
                "groundnut sheller",
                "groundnut shellers",
            ),
            _peanut,
            systems=(
                peanut.PEANUT_PROPOSER,
                peanut.PEANUT_CRITIC,
                peanut.BRAINSTORM_SYSTEM,
            ),
            follow_up=_matches(
                r"\b(review|critique|revise|improve|amend|update|rework|apply|cleanout|cleaning access|"
                r"kernel protection|whole kernels|feed control)\b"
            ),
        ),
        Rule(
            "planar-truss",
            "Choose a truss topology for a structural objective.",
            ("truss", "trusses"),
            _truss,
            systems=(truss.TRUSS_SYSTEM,),
            protocol=_truss_protocol,
            follow_up=_matches(
                r"\b(lightest|lighter|weight|safest|safer|stiffest|stiffer|stiffness|deflection|mass|factor of safety|safety factor|fos|cap it|limit it|\d+(?:\.\d+)?\s*kg)\b"
            ),
        ),
        Rule(
            "laptop",
            "Select a catalog laptop configuration subject to price, mass, and battery limits.",
            (
                "student laptop",
                "laptop configuration",
                "laptop configurations",
                "catalog laptop",
                "recommend a laptop",
                "choose a laptop",
                "select a laptop",
            ),
            laptop_response,
            systems=(PROPOSER, CRITIC),
            protocol=lambda prompt: bool(
                LAPTOP_TASK.search(prompt)
                and CHOOSE.search(prompt)
                and re.search(r"\bjson\b", prompt, re.I)
            ),
            follow_up=_matches(
                r"(?:\$\s*\d|\b(?:battery life|weighs? at most|maximum weight)\b)"
            ),
        ),
        Rule(
            "morphology",
            "Brainstorm on another design brief using your own function and option table.",
            (),
            _open_brainstorm,
            protocol=lambda prompt: bool(_chart_line.search(prompt)),
            intent=lambda prompt: (
                _ideation(prompt)
                or (
                    bool(
                        re.search(r"\b(?:design|devise|invent|sketch)\b", prompt, re.I)
                    )
                    and bool(infer_functions(prompt))
                )
            ),
            follow_up=_chart_edit,
        ),
    )
