"""
Template for your decision-model strategy; copy this file and fill in `decide` and `phrase`.

Suggested shape: a decision model picks ONE action from a fixed set, your code validates it
against the manual and runs the tool, then a responder phrases the reply. Register the class
in strategies/__init__.py and add a variant to variants.json.
"""

from __future__ import annotations

from .base import Session, Strategy, Trace

ACTIONS = ("ask", "call_tool", "answer", "transfer")


class DecisionStub(Strategy):
    name = "decision_stub"

    def __init__(self, session: Session, **options):
        super().__init__(session, **options)
        self.state: dict = {}  # slots, verified customer, pending confirmation, ...
        self.history: list[dict] = []

    async def decide(self, user_text: str, trace: Trace) -> dict:
        """
        Return e.g. {"action": "call_tool", "tool": "get_order", "args": {...}} or {"action": "ask", "slot": "phone"}.

        Read DECISION_API_KEY / DECISION_BASE_URL / DECISION_MODEL via `llm.chat(..., prefix="DECISION")`.
        """
        raise NotImplementedError("implement your decision model here")

    async def phrase(self, decision: dict, tool_result: dict | None, trace: Trace) -> str:
        """Turn the decision (and tool result) into the spoken reply: template or LLM."""
        raise NotImplementedError

    async def turn(self, user_text: str, trace: Trace) -> str:
        self.history.append({"role": "user", "content": user_text})
        result = None
        for _ in range(4):  # allow tool -> decide again, e.g. verify then get_order
            d = await self.decide(user_text, trace)
            trace.emit("decision", **d)
            if d.get("action") != "call_tool":
                break
            result = await self.s.tools.run(d["tool"], d.get("args", {}))
            trace.emit("tool", name=d["tool"], args=d.get("args", {}), result=result)
        reply = await self.phrase(d, result, trace)
        trace.emit("text", delta=reply)
        self.history.append({"role": "assistant", "content": reply})
        return reply
