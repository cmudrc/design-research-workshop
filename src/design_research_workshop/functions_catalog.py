"""Suggest mechanism options for mechanical functions explicitly named in a brief.

The catalog matches authored verb–object phrases. Two distinct affirmative
functions are required. The result carries each original phrase and the options
the catalog supplied, so a participant can inspect or replace an inferred row.

>>> brief = "Build a downspout attachment to collect rainwater and filter debris."
>>> [(m.name, m.phrase) for m in infer_functions(brief)]
[('Collect water', 'collect rainwater'), ('Filter debris', 'filter debris')]
>>> infer_functions("Brainstorm a better garden downspout.")
()
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FunctionMatch:
    """A chart row, the words supporting it, and the workshop's suggested options."""

    name: str
    phrase: str
    options: tuple[str, ...]


_ARTICLE = r"(?:(?:the|a|an|any|some|its|their)\s+)?"
_WATER = r"(?:rain\s?water|water|runoff|run-off|meltwater|condensate|drips)"
_PANEL = r"(?:(?:drawing|display|flat|wooden)\s+)?(?:panels?|boards?)"
_PART = r"(?:(?:small|loose|flat)\s+)?(?:workpieces?|parts?|blanks?)"

# Each rule requires both an operation and its physical object.
_ROWS = (
    (
        "Collect water",
        rf"(?:collect(?:s|ing)?|catch(?:es|ing)?|captur(?:e|es|ing)|gather(?:s|ing)?)\s+{_ARTICLE}"
        rf"(?:(?:falling|dripping|excess|roof|rain)\s+)?{_WATER}",
        (
            "Open catch basin",
            "Collection tank with inlet",
            "Flexible collection bladder",
        ),
    ),
    (
        "Filter debris",
        rf"(?:filter(?:s|ing)?|screen(?:s|ing)?|strain(?:s|ing)?)\s+(?:out\s+)?{_ARTICLE}"
        r"(?:(?:fallen|coarse|floating|loose)\s+)?(?:debris|leaves|particles|sediment)",
        ("Perforated basket", "Removable mesh screen", "Slotted grate"),
    ),
    (
        "Guide water",
        rf"(?:rout(?:e|es|ing)|redirect(?:s|ing)?|drain(?:s|ing)?|"
        rf"channel(?:s|l?ing)?|divert(?:s|ing)?)\s+{_ARTICLE}(?:excess\s+)?(?:{_WATER}|overflow)",
        ("Sloped trough", "Flexible drain hose", "Enclosed pipe"),
    ),
    (
        "Support a panel",
        rf"(?:support(?:s|ing)?|hold(?:s|ing)?|carr(?:y|ies|ying))\s+{_ARTICLE}{_PANEL}",
        ("Edge ledge", "Three-point cradle", "Pair of support brackets"),
    ),
    (
        "Adjust tilt",
        rf"(?:(?:adjust(?:s|ing)?|chang(?:e|es|ing)|set(?:s|ting)?|var(?:y|ies|ying))\s+"
        r"(?:(?:its|their|the|a)\s+)?(?:(?:panel|board|viewing|tilt)\s+)?(?:angle|tilt|inclination)"
        rf"|tilt(?:s|ing)?\s+{_ARTICLE}{_PANEL})",
        ("Friction pivot", "Pinned folding stay", "Indexed hinge"),
    ),
    (
        "Grip a workpiece",
        rf"(?:clamp(?:s|ing)?|grip(?:s|ping)?|hold(?:s|ing)?)\s+{_ARTICLE}{_PART}",
        ("Toggle clamp", "Lead-screw vise", "Wedge clamp"),
    ),
    (
        "Locate a workpiece",
        rf"(?:align(?:s|ing)?|position(?:s|ing)?|register(?:s|ing)?)\s+{_ARTICLE}{_PART}",
        ("Edge fence", "Corner nest", "Adjustable pin stops"),
    ),
    (
        "Move packages",
        rf"(?:mov(?:e|es|ing)|transport(?:s|ing)?|convey(?:s|ing)?)\s+{_ARTICLE}"
        r"(?:(?:small|packed|loaded)\s+)?(?:boxes|packages|totes|cartons)",
        ("Roller bed", "Belt conveyor", "Wheeled cart"),
    ),
    (
        "Dispense dry material",
        rf"(?:dispens(?:e|es|ing)|meter(?:s|ing)?|portion(?:s|ing)?)\s+{_ARTICLE}"
        r"(?:(?:dry|small)\s+)?(?:granules|pellets|seeds|grain)",
        ("Rotary pocket feeder", "Slide gate", "Metering auger"),
    ),
)
_CATALOG = tuple(
    (name, re.compile(rf"\b(?:{expression})\b", re.I), options)
    for name, expression, options in _ROWS
)
_OWN_CHART = re.compile(
    r"\b(?:(?:do not|don['’]t|never)\s+(?:infer|suggest|supply|provide|generate|guess)\s+"
    r"(?:(?:a|the|any|new|mechanism)\s+)*(?:chart|mechanisms?|options?|functions?)"
    r"|(?:I(?:'ll| will) (?:supply|provide)|ask me for)\s+(?:(?:my|the|a|own)\s+)*"
    r"(?:chart|mechanisms?|options?|functions?)|use (?:only (?:my|the supplied|user-supplied)|my own)\s+"
    r"(?:(?:own|mechanism)\s+)*(?:chart|mechanisms?|options?|functions?))\b",
    re.I,
)
_CLAUSE = re.compile(r"[.!?;\n]|\b(?:but|however|instead|while)\b", re.I)
_NEGATION = re.compile(
    r"\b(?:no need to|not(?! (?:only|just)\b)|never|without|avoid(?:s|ing)?|"
    r"exclud(?:e|es|ing)|omit(?:s|ting)?|skip(?:s|ping)?|[a-z]+n['’]t)\b",
    re.I,
)
_ALTERNATIVE = re.compile(r"\s*,?\s*(?:or|versus|vs\.?)\s*(?:to\s+)?", re.I)
_EXPLICIT_ROW = re.compile(r"^\s*[^:|\n]+:[^\n]*\|")


def infer_functions(text: str) -> tuple[FunctionMatch, ...]:
    """Return evidenced starter rows, or an empty tuple when a chart is needed.

    Explicit chart rows are input data and remain outside this suggestion pass.
    A request to supply one's own chart and a single matching function both
    leave the selection of functions and options to the participant.
    """
    if not isinstance(text, str):
        raise ValueError("Supply the design brief as text.")
    if _OWN_CHART.search(text):
        return ()
    # Remove chart rows, retaining the exact spelling of every prose line.
    prose = "\n".join(
        line for line in text.splitlines() if not _EXPLICIT_ROW.match(line)
    )
    rows = {}
    for clause in _CLAUSE.split(prose):
        negation = _NEGATION.search(clause)
        mentions = sorted(
            (
                (m.start(), m.end(), FunctionMatch(name, m.group(), options))
                for name, pattern, options in _CATALOG
                for m in pattern.finditer(clause)
            ),
            key=lambda entry: entry[0],
        )
        # Alternatives cannot safely become two required chart rows.
        alternatives = set()
        for i in range(1, len(mentions)):
            _, end, previous = mentions[i - 1]
            start, _, current = mentions[i]
            if previous.name != current.name and _ALTERNATIVE.fullmatch(
                clause[end:start]
            ):
                alternatives.update((i - 1, i))
        for i, (start, _, row) in enumerate(mentions):
            if negation and start >= negation.start():
                rows.pop(row.name, None)
            elif i not in alternatives:
                rows.setdefault(row.name, row)
    return tuple(rows.values()) if len(rows) >= 2 else ()
