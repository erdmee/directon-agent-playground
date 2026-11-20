"""
Decision model = constrained next-action picker: the model chooses one action per step as JSON.

The frame keeps an explicit state summary, offers a fixed action set, and checks two rules from
its own observations before executing: order tools need a verified customer, and cancel/return
need the previous reply to have asked for consent. The model's "say" text is spoken directly,
so there is no separate responder call.
"""

from __future__ import annotations

import json

from .base import Session, Strategy, Trace
from .common import PERSONA, digits, json_call

ORDER_TOOLS = {"list_orders", "get_order", "cancel_order", "request_return"}

ACTION_PROMPT = (
    PERSONA
    + """
You control the call by choosing ONE next action per step. Return JSON only:
{"action": "say" | "verify_customer" | "list_orders" | "get_order" | "cancel_order" | "request_return"
           | "get_product_info" | "transfer_to_agent" | "request_callback",
 "args": {...tool arguments...}, "say": "spoken reply (only for action=say)",
 "asks_consent": true|false (only for say: you are asking permission to cancel or return)}
After a tool action you will see its result and choose again; end every turn with action "say".
Follow the MANUAL strictly. Phone numbers as digits only. Use the caller's final corrected value.

MANUAL:
"""
)


class DecisionActions(Strategy):
    """Options: prefix (DECISION by default), reasoning_effort, max_steps."""

    name = "decision_actions"

    def __init__(self, session: Session, **options):
        super().__init__(session, **options)
        self.system = ACTION_PROMPT + session.manual + f"\nCurrent time: {session.now}"
        self.log: list[dict] = []  # Dialogue and tool results, shown to the model each step.
        self.verified: str | None = None
        self.asked_consent = False

    async def turn(self, user_text: str, trace: Trace) -> str:
        o = self.options
        self.log.append({"caller": user_text})
        for _ in range(o.get("max_steps", 5)):
            state = {"verified_customer_id": self.verified, "you_asked_consent_last_turn": self.asked_consent}
            user = json.dumps({"state": state, "log": self.log[-14:]}, ensure_ascii=False)
            d = await json_call(trace, "act", self.system, user,
                                prefix=o.get("prefix", "LLM"), reasoning_effort=o.get("reasoning_effort"))  # fmt: skip
            trace.emit("decision", **d)
            action, args = d.get("action"), d.get("args") or {}
            if action == "say" or not action:
                text = (d.get("say") or "").strip() or "죄송합니다, 다시 한 번 말씀해 주시겠어요?"
                self.asked_consent = bool(d.get("asks_consent"))
                self.log.append({"agent": text})
                trace.emit("text", delta=text)
                return text
            if err := self.precheck(action, args):
                self.log.append({"rejected": action, "why": err})
                trace.emit("rejected", action=action, why=err)
                continue
            for k in ("phone",):
                if k in args:
                    args[k] = digits(args[k]) or args[k]
            r = await self.s.tools.run(action, args)
            trace.emit("tool", name=action, args=args, result=r)
            if action == "verify_customer" and r.get("status") == "ok":
                self.verified = r.get("customer_id")
            self.log.append({"tool": action, "args": args, "result": r})
        text = "확인 중에 문제가 있어 상담원에게 연결해 드리겠습니다."
        trace.emit("text", delta=text)
        return text

    def precheck(self, action: str, args: dict) -> str | None:
        if action in ORDER_TOOLS and not self.verified:
            return "verify the customer first"
        if action in ("cancel_order", "request_return") and not self.asked_consent:
            return "ask the caller for explicit consent first (say with asks_consent=true)"
        return None
