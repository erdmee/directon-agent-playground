"""
Inception mercury-decide client (POST /v1/decisions) and its use as the NLU of the rules engine.

Docs (checked 2026-11-20): https://docs.inceptionlabs.ai/capabilities/decisions,
https://docs.inceptionlabs.ai/capabilities/extract, https://docs.inceptionlabs.ai/support/wind-down
Not a chat model: no messages, tools, streaming or temperature. Billing: $0.02 per 1M input tokens.
The whole Mercury API is served only through 2026-12-28. Korean support is not documented.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .config import env
from .model_params import drop_none
from .strategies.common import digits, order_id


@dataclass
class DecideQuestion:
    """One entry of `questions`; every documented field, None = not sent."""

    type: str  # "noul" (yes-probability), "choice", "score" or "extract".
    instructions: str | None = None  # Required for extract when criteria are empty.
    criteria: Any = None  # noul: {"true","false"} texts; choice: {option: description|None} (1-255);
    # score: ordered list of 1-10 level descriptions; extract: {field: description} (1-26).
    source: str | None = None  # extract only: which key of an object `state` to read spans from.
    threshold: float | None = None  # extract only, default 0.
    snap_to_words: bool | None = None  # extract only, default true.
    debug: bool | None = None  # extract only, default false.


@dataclass
class MercuryDecideParams:
    model: str = "mercury-decide"
    state: Any = None  # String, object or array (images as image_url data URLs, not with extract).
    questions: dict[str, DecideQuestion] = field(default_factory=dict)

    def body(self) -> dict:
        return drop_none({"model": self.model, "state": self.state,
                          "questions": {k: asdict(q) for k, q in self.questions.items()}})  # fmt: skip


_client: httpx.AsyncClient | None = None


async def decide(params: MercuryDecideParams) -> tuple[dict, float, dict]:
    """Returns (answers, latency ms, usage)."""
    global _client
    key = env("DECISION_API_KEY")
    if not key:
        raise RuntimeError("DECISION_API_KEY (Inception key) is not set in .env")
    _client = _client or httpx.AsyncClient(timeout=30)
    base = env("DECISION_BASE_URL", "https://api.inceptionlabs.ai").rstrip("/").removesuffix("/v1")
    t0 = time.perf_counter()
    r = await _client.post(f"{base}/v1/decisions", json=params.body(), headers={"Authorization": f"Bearer {key}"})
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise RuntimeError(f"mercury-decide HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data.get("answers", {}), ms, data.get("usage", {})


INTENT_CRITERIA = {
    "order_status": "asks where an order is or when it arrives",
    "cancel": "wants to cancel an order",
    "return": "wants to return or refund a received product",
    "product_info": "asks about a product (dose, ingredients, price, stock)",
    "medical": "asks if a product is safe with a disease, medicine or pregnancy",
    "side_effect": "reports a bad reaction after taking a product",
    "human": "asks to talk to a human agent",
    "callback": "asks to be called back later",
    "faq": "asks a general question such as opening hours",
    "other": "something the shop cannot handle",
    "none": "only answers the agent's question or gives requested info, no new request",
}
SLOT_CRITERIA = {
    "name": "the orderer's name; if corrected, the final one",
    "phone": "the mobile phone number, possibly spoken in Korean digits; if corrected, the final one",
    "order_id": "the order number, e.g. A1001 or 에이 천일",
    "product": "the product the caller finally means, e.g. 오메가3, 유산균, 콜라겐",
    "callback_time": "when the caller wants to be called back; the final one",
}


def _last_span(ans: dict, name: str) -> str | None:
    spans = (ans.get("spans") or {}).get(name) or []
    return spans[-1]["text"] if spans else None


async def mercury_nlu(agent_last_said: str, caller: str) -> tuple[dict, float, dict]:
    """Maps mercury-decide answers onto the same dict the LLM NLU returns."""
    p = MercuryDecideParams(
        model=env("DECISION_MODEL", "mercury-decide"),
        state={"agent_last_said": agent_last_said, "caller": caller},
        questions={
            "intent": DecideQuestion("choice", "What does the caller newly want in this utterance? "
                                     "If they correct themselves, use the corrected request.", INTENT_CRITERIA),
            "confirm": DecideQuestion("choice", "How does the caller answer the agent's last yes/no question?",
                                      {"yes": "agrees / go ahead", "no": "declines or changes their mind",
                                       "none": "not answering a yes/no question"}),
            "opened": DecideQuestion("choice", "Was the product package opened or tried?",
                                     # Null descriptions here plus an extract question make the API return 422.
                                     {"opened": "the package was opened or the product was tried",
                                      "unopened": "the package is still sealed",
                                      "unknown": "not mentioned"}),
            "insists": DecideQuestion("noul", "Does the caller keep demanding something after being refused?"),
            "slots": DecideQuestion("extract", criteria=SLOT_CRITERIA, source="caller"),
        },
    )  # fmt: skip
    a, ms, usage = await decide(p)
    intent = (a.get("intent") or {}).get("choice")
    slots = a.get("slots") or {}
    product, oid = _last_span(slots, "product"), order_id(_last_span(slots, "order_id"))
    opened = {"opened": True, "unopened": False}.get(str((a.get("opened") or {}).get("choice")))
    confirm = (a.get("confirm") or {}).get("choice")
    nlu = {
        "requests": [] if intent in (None, "none") else [{"intent": intent, "product": product, "order_id": oid}],
        "name": _last_span(slots, "name"),
        "phone": digits(_last_span(slots, "phone")),
        "opened": opened,
        "confirm": None if confirm == "none" else confirm,
        "callback_time": _last_span(slots, "callback_time"),
        "insists": ((a.get("insists") or {}).get("noul") or 0) >= 0.5,
        "_raw": a,
    }
    return nlu, ms, usage
