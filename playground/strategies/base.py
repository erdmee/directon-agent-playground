"""Strategy interface: every agent variant (free LLM, decision model, ...) implements `turn`."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import load_manual, openai_tools
from ..tools import ToolRunner


@dataclass
class Session:
    """Everything a strategy may read for one call; `tools.run` is the only way to touch the backend."""

    now: str
    tools: ToolRunner
    caller: dict | None = None
    manual: str = field(default_factory=load_manual)
    tool_schemas: list[dict] = field(default_factory=openai_tools)


class Trace:
    """Per-turn event log; `emit("text", delta=...)` also marks time-to-first-spoken-text."""

    def __init__(self, on_text: Callable[[str], None] | None = None) -> None:
        self.on_text = on_text  # Live reply-text deltas, e.g. to feed streaming TTS.
        self.t0 = time.perf_counter()
        self.events: list[dict] = []
        self.first_text_ms: float | None = None
        self.llm_calls = 0
        self.tokens = {"prompt": 0, "completion": 0}

    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def emit(self, kind: str, **data) -> None:
        if kind == "text" and self.first_text_ms is None and data.get("delta", "").strip():
            self.first_text_ms = self.ms()
        if kind == "text" and self.on_text:
            self.on_text(data.get("delta", ""))
        if kind != "text":
            self.events.append({"t_ms": self.ms(), "kind": kind, **data})

    def llm(self, label: str, res) -> None:
        self.llm_calls += 1
        self.tokens["prompt"] += res.usage.get("prompt_tokens", 0)
        self.tokens["completion"] += res.usage.get("completion_tokens", 0)
        self.emit(
            "llm",
            label=label,
            first_token_ms=res.first_token_ms,
            total_ms=round(res.total_ms, 1),
            tool_calls=[c["name"] for c in res.tool_calls],
            usage=res.usage,
        )


class Strategy:
    """Subclass, set `name`, and implement `turn`. Options come from variants.json."""

    name = "base"

    def __init__(self, session: Session, **options):
        self.s = session
        self.options = options

    async def turn(self, user_text: str, trace: Trace) -> str:
        """
        Handle one caller utterance and return the reply that will be spoken.

        Call `self.s.tools.run(name, args)` for business tools and `trace.emit(...)` for anything
        you want in the trace; emit("text", delta=...) as reply text streams out.
        """
        raise NotImplementedError
