"""
Interactive text chat with one variant, printing tool calls and per-turn latency.

uv run python -m playground.chat --variant tree
"""

from __future__ import annotations

import argparse
import asyncio
import json

from .config import load_dotenv, now_default
from .evaluate import load_variants
from .strategies import REGISTRY, Session, Trace
from .tools import ToolRunner


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="manual")
    a = ap.parse_args()
    load_dotenv()
    v = load_variants(a.variant)[0]
    tools = ToolRunner(now_default(), guard=v.get("guard", False))
    agent = REGISTRY[v["strategy"]](Session(now=tools.backend.now, tools=tools), **v.get("options", {}))
    print(f"variant={v['name']}  (empty line to quit)")
    while text := input("\n고객> ").strip():
        tools.new_turn(text)
        trace = Trace()
        reply = await agent.turn(text, trace)
        for e in trace.events:
            if e["kind"] == "tool":
                print(
                    f"  [tool {e['t_ms']:>6.0f}ms] {e['name']} {json.dumps(e['args'], ensure_ascii=False)}"
                    f" -> {json.dumps(e['result'], ensure_ascii=False)[:160]}"
                )
        for c in tools.calls:
            if c.turn == tools.turn and c.violations:
                print(f"  [policy{' BLOCKED' if c.blocked else ''}] {c.violations}")
        print(f"상담원> {reply}\n  first_text={trace.first_text_ms}ms total={trace.ms()}ms llm_calls={trace.llm_calls}")
    await tools.aclose()


if __name__ == "__main__":
    asyncio.run(main())
