"""Strategy registry: variants.json refers to these names."""

from .base import Session, Strategy, Trace
from .decision_actions import DecisionActions
from .decision_rules import DecisionRules
from .decision_stub import DecisionStub
from .llm_tools import LLMTools
from .router import Router

REGISTRY: dict[str, type[Strategy]] = {
    c.name: c for c in (LLMTools, Router, DecisionActions, DecisionRules, DecisionStub)
}

__all__ = ["REGISTRY", "Session", "Strategy", "Trace"]
