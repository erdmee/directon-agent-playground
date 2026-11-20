"""Router: a small intent call picks the tool subset and procedure, then the LLM calls tools as usual."""

from __future__ import annotations

from .base import Session, Trace
from .common import json_call
from .llm_tools import LLMTools

ROUTES = {
    "order": (["verify_customer", "list_orders", "get_order", "transfer_to_agent"],
              "Order status: verify name+phone first, then get_order (or list_orders if no order ID)."),
    "cancel": (["verify_customer", "list_orders", "get_order", "cancel_order", "transfer_to_agent"],
               "Cancel: verify, find the order, only 결제완료/상품준비중 can be cancelled, ask consent, "
               "call cancel_order only after a clear yes in the latest turn."),
    "return": (["verify_customer", "list_orders", "get_order", "request_return", "transfer_to_agent"],
               "Return: verify, find the order, 배송완료 within 7 days and unopened only, ask if opened, ask consent, "
               "then request_return. Not eligible and the caller insists -> transfer_to_agent."),
    "product": (["get_product_info", "transfer_to_agent"],
                "Product: get_product_info, answer only from it. Medical judgement (disease, medicine, pregnancy): "
                "no advice, suggest a doctor or pharmacist. No identity check needed."),
    "handoff": (["transfer_to_agent"], "Side effect, human requested or abuse: call transfer_to_agent now."),
    "callback": (["request_callback", "transfer_to_agent"], "Callback: need phone digits and preferred time."),
    "faq": ([], "General question: answer from the manual, no tool."),
}  # fmt: skip

ROUTER_PROMPT = (
    "Classify the caller's CURRENT need in a Korean customer-service call. Return JSON "
    '{"route": one of ' + ", ".join(ROUTES) + "}. "
    "Use the conversation so far; a short answer like a name, phone number or 네 continues the previous route. "
    "If the caller corrects themselves (아 아니다 ...), use the corrected request."
)


class Router(LLMTools):
    name = "router"

    def __init__(self, session: Session, **options):
        super().__init__(session, **options)
        self.route = "faq"
        self.base_system = self.messages[0]["content"]

    async def turn(self, user_text: str, trace: Trace) -> str:
        recent = [m for m in self.messages[1:] if m["role"] in ("user", "assistant") and m.get("content")][-4:]
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        out = await json_call(
            trace, "route", ROUTER_PROMPT, f"Previous route: {self.route}\n{convo}\nuser: {user_text}"
        )
        self.route = out.get("route") if out.get("route") in ROUTES else self.route
        names, procedure = ROUTES[self.route]
        trace.emit("route", route=self.route)
        self.tool_schemas = [t for t in self.s.tool_schemas if t["function"]["name"] in names]
        self.messages[0] = {
            "role": "system",
            "content": self.base_system + f"\n\nCURRENT TASK ({self.route}): {procedure}",
        }
        return await super().turn(user_text, trace)
