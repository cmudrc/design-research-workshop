"""Reusable source code for the Design Research workshop demos."""

from .critique import laptop_propose_critic
from .laptop import PROBLEM_ID, laptop_study_agent
from .server import workshop_client

__all__ = [
    "PROBLEM_ID",
    "laptop_propose_critic",
    "laptop_study_agent",
    "workshop_client",
]
