"""Reusable source code for the Design Research workshop demos."""

from .bot import Reply, Rule, Turn, WorkshopPyBot, WorkshopSession
from .brainstorm import DesignBrief
from .responders import brainstorm_rule
from .critique import laptop_propose_critic
from .laptop import PROBLEM_ID, laptop_study_agent
from .server import workshop_client

__all__ = [
    "PROBLEM_ID",
    "DesignBrief",
    "Reply",
    "Rule",
    "Turn",
    "WorkshopPyBot",
    "WorkshopSession",
    "brainstorm_rule",
    "laptop_propose_critic",
    "laptop_study_agent",
    "workshop_client",
]
