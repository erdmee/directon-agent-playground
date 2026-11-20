"""Baseline: the LLM gets the manual (or a decision-tree prompt) plus all tools and decides everything."""

from __future__ import annotations

import asyncio
import json

from ..config import PG
from ..llm import chat
from .base import Session, Strategy, Trace


class LLMTools(Strategy):
    """
    Options: prompt (file in prompts/), prefix (env prefix LLM/DECISION), model,
    reasoning_effort, temperature, max_rounds, tools (subset of names)."""

    name = "llm_tools"

    def __init__(self, session: Session, **options):
        super().__init__(session, **options)
        tpl = (PG / "prompts" / f"{options.get('prompt', 'manual')}.txt").read_text(encoding="utf-8")
        system = tpl.format(
            now=session.now,
            manual=session.manual,
            caller=json.dumps(session.caller, ensure_ascii=False) if session.caller else "없음",
        )
        self.messages: list[dict] = [{"role": "system", "content": system}]
        names = options.get("tools")
        self.tool_schemas = [t for t in session.tool_schemas if not names or t["function"]["name"] in names]

    async def turn(self, user_text: str, trace: Trace) -> str:
        o = self.options
        self.messages.append({"role": "user", "content": user_text})
        spoken: list[str] = []
        for rnd in range(o.get("max_rounds", 5)):
            res = await chat(
                self.messages,
                prefix=o.get("prefix", "LLM"),
                model=o.get("model"),
                tools=self.tool_schemas,
                reasoning_effort=o.get("reasoning_effort"),
                temperature=o.get("temperature"),
                on_text=lambda d: trace.emit("text", delta=d),
            )
            trace.llm(f"round{rnd + 1}", res)
            if res.text.strip():
                spoken.append(res.text.strip())
            if not res.tool_calls:
                self.messages.append({"role": "assistant", "content": res.text})
                break
            self.messages.append(
                {
                    "role": "assistant",
                    "content": res.text or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                        }
                        for c in res.tool_calls
                    ],
                }
            )
            results = await asyncio.gather(*(self._run(c, trace) for c in res.tool_calls))
            for c, r in zip(res.tool_calls, results, strict=True):
                self.messages.append(
                    {"role": "tool", "tool_call_id": c["id"], "content": json.dumps(r, ensure_ascii=False)}
                )
        return " ".join(spoken)

    async def _run(self, c: dict, trace: Trace) -> dict:
        try:
            args = json.loads(c["arguments"] or "{}")
        except json.JSONDecodeError:
            trace.emit("tool", name=c["name"], args=c["arguments"], result="invalid JSON")
            return {"status": "invalid_arguments", "message": "arguments were not valid JSON"}
        r = await self.s.tools.run(c["name"], args)
        trace.emit("tool", name=c["name"], args=args, result=r)
        return r
