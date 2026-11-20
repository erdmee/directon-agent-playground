"""Streaming chat call against any OpenAI-compatible endpoint, with latency and token accounting."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .config import llm_settings
from .model_params import OpenAIChatParams

_clients: dict[tuple, AsyncOpenAI] = {}


def client_for(prefix: str = "LLM") -> tuple[AsyncOpenAI, dict]:
    s = llm_settings(prefix)
    if not s["api_key"]:
        hint = "BAZE_API_KEY, OPENAI_API_KEY or LLM_API_KEY" if prefix == "LLM" else f"{prefix}_API_KEY"
        raise RuntimeError(f"{hint} is not set in .env (see env.example)")
    k = (s["api_key"], s["base_url"])
    if k not in _clients:
        _clients[k] = AsyncOpenAI(api_key=s["api_key"], base_url=s["base_url"])
    return _clients[k], s


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {"id", "name", "arguments"(str)}
    first_token_ms: float | None = None
    first_text_ms: float | None = None
    total_ms: float = 0.0
    usage: dict = field(default_factory=dict)


async def chat(
    messages: list[dict],
    *,
    prefix: str = "LLM",
    model: str | None = None,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    on_text: Callable[[str], None] | None = None,
    **extra,
) -> LLMResult:
    """`extra` takes any other OpenAIChatParams field, e.g. service_tier="priority" or verbosity="low"."""
    client, s = client_for(prefix)
    params = OpenAIChatParams(
        model=model or s["model"],
        reasoning_effort=reasoning_effort if reasoning_effort is not None else s["reasoning_effort"],
    )
    kw = params.to_kwargs(response_format=response_format, temperature=temperature, **extra)
    kw["messages"] = messages
    if tools:
        kw["tools"] = tools

    res, calls = LLMResult(), {}
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(**kw)
    async for chunk in stream:
        if chunk.usage:
            res.usage = chunk.usage.model_dump(exclude_none=True)
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if (d.content or d.tool_calls) and res.first_token_ms is None:
            res.first_token_ms = (time.perf_counter() - t0) * 1000
        if d.content:
            if res.first_text_ms is None:
                res.first_text_ms = (time.perf_counter() - t0) * 1000
            res.text += d.content
            if on_text:
                on_text(d.content)
        for tc in d.tool_calls or []:
            c = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            c["id"] = tc.id or c["id"]
            if tc.function and tc.function.name:
                c["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                c["arguments"] += tc.function.arguments
    res.tool_calls = [calls[k] for k in sorted(calls)]
    res.total_ms = (time.perf_counter() - t0) * 1000
    return res
