"""Helpers shared by the decision strategies: JSON calls, spoken-digit cleanup, the reply phraser."""

from __future__ import annotations

import json
import re

from ..llm import chat
from .base import Trace

PERSONA = (
    "You are a phone customer-service agent for 건강한하루몰, an online health-supplement shop. "
    "Write exactly what will be spoken by TTS: polite spoken Korean (존댓말), one or two short sentences, "
    "no markdown or lists, read IDs and numbers digit by digit."
)

_KO = {"공": "0", "영": "0", "빵": "0", "일": "1", "이": "2", "삼": "3", "사": "4",
       "오": "5", "육": "6", "륙": "6", "칠": "7", "팔": "8", "구": "9"}  # fmt: skip


_ENDING = re.compile(r"(이에요|이예요|이요|이고요|입니다|이야|예요|에요|고요|요|이)\W*$")


def digits(v) -> str | None:
    """
    Phone/ID digits from model output or an extracted span that may hold spoken Korean digits.

    Arabic digits win when present. Otherwise a trailing 이 is ambiguous (오사삼이 ends in 2, but
    삼삼삼삼이요 ends in a particle), so the reading that gives an 11-digit mobile number wins.
    """
    if v is None:
        return None
    s = str(v).strip()
    if re.search(r"\d", s):
        return re.sub(r"\D", "", s) or None

    def conv(t: str) -> str:
        return re.sub(r"\D", "", "".join(_KO.get(ch, ch) for ch in t))

    full, cut = conv(s), conv(_ENDING.sub("", s))
    return next((d for d in (full, cut) if len(d) == 11), full) or None


def order_id(v) -> str | None:
    if not v:
        return None
    s = re.sub(r"[\s\-]", "", str(v)).upper()
    m = re.search(r"A?(\d{4})", s)
    return f"A{m.group(1)}" if m else None


async def json_call(trace: Trace, label: str, system: str, user: str, *, prefix: str = "LLM", **opts) -> dict:
    res = await chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        prefix=prefix,
        response_format={"type": "json_object"},
        **opts,
    )
    trace.llm(label, res)
    try:
        return json.loads(res.text)
    except json.JSONDecodeError:
        trace.emit("parse_error", label=label, text=res.text[:300])
        return {}


async def phrase(trace: Trace, history: list[dict], directives: list[dict], **opts) -> str:
    """Responder: turns the engine's directives into the spoken reply (no tools, streamed)."""
    system = (
        PERSONA + "\nSay exactly what the DIRECTIVES ask, in order, using only their facts. Do not add policies, dates "
        "or amounts that are not in the facts. Combine them naturally into one short reply."
    )
    msgs = [{"role": "system", "content": system}]
    msgs += history[-6:]
    msgs.append({"role": "user", "content": "DIRECTIVES:\n" + json.dumps(directives, ensure_ascii=False)})
    res = await chat(msgs, on_text=lambda d: trace.emit("text", delta=d), **opts)
    trace.llm("phrase", res)
    return res.text.strip()
