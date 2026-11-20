"""
Decision model = NLU + rules: the LLM only extracts what the caller wants; code decides every tool call.

Per turn: (1) a JSON NLU call reads the utterance into requests and slots, (2) a deterministic
engine walks the manual's procedure, calling tools only through `tools.run`, and collects
directives, (3) the responder phrases the directives. The engine sees only tool results, never
the mock database.
"""

from __future__ import annotations

import json
from datetime import date

from .base import Session, Strategy, Trace
from .common import digits, json_call, order_id, phrase

INTENTS = ["order_status", "cancel", "return", "product_info", "medical", "side_effect",
           "human", "callback", "faq", "other"]  # fmt: skip
CANCELLABLE = {"결제완료", "상품준비중"}

NLU_PROMPT = (
    """You read one caller utterance in a Korean shop phone call and return JSON only:
{"requests": [{"intent": one of INTENT_LIST, "product": str|null, "order_id": str|null}],
 "name": str|null, "phone": digits only|null, "opened": true|false|null, "confirm": "yes"|"no"|null,
 "callback_time": str|null, "reason": "단순변심"|"상품불량"|"오배송"|null, "insists": true|false}
Rules:
- requests: NEW needs stated in this utterance, in order. Empty list if the caller only answers a question.
- The caller may correct themselves (아 아니다, 아 맞다, 잠깐만요): keep only the FINAL value.
- Convert spoken digits to digits (공일공 일이삼사 -> 0101234). Order IDs look like A1001 (에이 천일 -> A1001).
- confirm: answer to the agent's last yes/no question (취소할까요? 진행할까요?). "그냥 둘게요", "하지 마세요" -> "no".
- opened: whether the package/product was opened or tried.
- side_effect: the caller reports a bad reaction after taking a product. medical: asks if it is safe with a
  disease, medicine or pregnancy. human: asks for a person. insists: keeps demanding after being refused."""
).replace("INTENT_LIST", json.dumps(INTENTS))


class DecisionRules(Strategy):
    """
    Options: nlu ("llm" JSON call or "mercury" for mercury-decide), prefix and responder_prefix
    (env prefix of a chat endpoint, LLM by default), reasoning_effort."""

    name = "decision_rules"

    def __init__(self, session: Session, **options):
        super().__init__(session, **options)
        self.history: list[dict] = []
        self.st: dict = {"name": None, "phone": None, "customer_id": None, "tried": set(), "fails": 0,
                         "orders": None, "details": {}, "tasks": [], "pending": None, "opened": None,
                         "callback_time": None, "reason": None, "insists": False}  # fmt: skip

    # --- tool helper: every call goes through the scored ToolRunner ---
    async def call(self, trace: Trace, tool: str, /, **args) -> dict:
        r = await self.s.tools.run(tool, args)
        trace.emit("tool", name=tool, args=args, result=r)
        return r

    async def turn(self, user_text: str, trace: Trace) -> str:
        o = self.options
        last_agent = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")
        if o.get("nlu") == "mercury":
            from ..mercury import mercury_nlu

            nlu, ms, usage = await mercury_nlu(last_agent, user_text)
            trace.llm_calls += 1
            trace.tokens["prompt"] += usage.get("input_tokens", 0)
            trace.emit("llm", label="mercury-decide", total_ms=round(ms, 1), usage=usage)
        else:
            nlu = await json_call(trace, "nlu", NLU_PROMPT,
                                  json.dumps({"agent_last_said": last_agent, "caller": user_text}, ensure_ascii=False),
                                  prefix=o.get("prefix", "LLM"), reasoning_effort=o.get("reasoning_effort"))  # fmt: skip
        trace.emit("nlu", **nlu)
        self.merge(nlu)
        directives = await self.run_engine(trace, nlu.get("confirm"))
        trace.emit("directives", items=directives)
        self.history.append({"role": "user", "content": user_text})
        reply = await phrase(trace, self.history, directives, prefix=o.get("responder_prefix", "LLM"))
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def merge(self, nlu: dict) -> None:
        st = self.st
        if nlu.get("name"):
            st["name"] = nlu["name"]
        if digits(nlu.get("phone")):
            st["phone"] = digits(nlu["phone"])
        for k in ("opened", "callback_time", "reason"):
            if nlu.get(k) is not None:
                st[k] = nlu[k]
        st["insists"] = bool(nlu.get("insists"))
        reqs = [r for r in nlu.get("requests") or [] if r.get("intent") in INTENTS]
        if reqs:
            st["tasks"] = [{**r, "order_id": order_id(r.get("order_id"))} for r in reqs]
            if st["pending"] and not any(r["intent"] == st["pending"]["intent"] for r in reqs):
                st["pending"] = None

    async def run_engine(self, trace: Trace, confirm: str | None) -> list[dict]:
        out: list[dict] = []
        while self.st["tasks"]:
            task = self.st["tasks"][0]
            d = await self.step(trace, task, confirm)
            out.append(d)
            confirm = None  # A yes/no answers only the first pending question.
            if d.get("waiting"):
                break
            self.st["tasks"].pop(0)
        return out or [{"say": "ask how you can help"}]

    async def step(self, trace: Trace, task: dict, confirm: str | None) -> dict:
        st, intent = self.st, task["intent"]
        if intent == "side_effect":
            await self.call(trace, "transfer_to_agent", reason="부작용신고", summary="섭취 후 이상 반응 신고")
            return {"say": "apologize, advise to stop taking it and see a doctor, say you are connecting a human agent"}
        if intent == "human" or (st["insists"] and intent in ("return", "cancel", "other")):
            reason = "고객요청" if intent == "human" else "처리불가"
            await self.call(trace, "transfer_to_agent", reason=reason, summary=f"{intent} 요청")
            return {"say": "say you are connecting a human agent now"}
        if intent == "medical":
            return {
                "say": "you cannot judge medical safety; recommend asking a doctor or pharmacist; offer a human agent"
            }
        if intent == "faq":
            return {"say": "answer the general question from these facts",
                    "facts": "운영시간 평일 09:00~18:00, 점심 12:00~13:00, 주말과 공휴일 휴무"}  # fmt: skip
        if intent == "other":
            return {"say": "say you cannot handle this request and offer a human agent", "waiting": True}
        if intent == "callback":
            if not st["phone"]:
                return {"say": "ask for the phone number for the callback", "waiting": True}
            if not st["callback_time"]:
                return {"say": "ask when they want the callback", "waiting": True}
            r = await self.call(trace, "request_callback", phone=st["phone"], preferred_time=st["callback_time"])
            return {"say": "confirm the callback request", "facts": {"time": st["callback_time"], "result": r}}
        if intent == "product_info":
            if not task.get("product"):
                return {"say": "ask which product", "waiting": True}
            r = await self.call(trace, "get_product_info", product_name=task["product"])
            return {"say": "answer the product question from the facts only", "facts": r}
        return await self.order_step(trace, task, intent, confirm)

    async def order_step(self, trace: Trace, task: dict, intent: str, confirm: str | None) -> dict:
        st = self.st
        if not st["customer_id"]:
            if st["fails"] >= 2:
                await self.call(trace, "transfer_to_agent", reason="본인확인실패", summary="본인 확인 2회 실패")
                return {"say": "identity check failed twice; say you are connecting a human agent"}
            missing = [k for k in ("name", "phone") if not st[k]]
            if missing:
                return {
                    "say": f"for identity verification ask for the orderer's {' and '.join(missing)}",
                    "waiting": True,
                }
            if (st["name"], st["phone"]) in st["tried"]:
                return {
                    "say": "the details did not match; ask them to check the name and phone number again",
                    "waiting": True,
                }
            st["tried"].add((st["name"], st["phone"]))
            r = await self.call(trace, "verify_customer", name=st["name"], phone=st["phone"])
            if r.get("status") != "ok":
                st["fails"] += 1
                st["phone"] = None
                if st["fails"] >= 2:
                    await self.call(trace, "transfer_to_agent", reason="본인확인실패", summary="본인 확인 2회 실패")
                    return {"say": "identity check failed twice; say you are connecting a human agent"}
                return {"say": "no customer matches; ask them to check the name and phone number", "waiting": True}
            st["customer_id"] = r["customer_id"]

        oid = task.get("order_id")
        if not oid:
            if st["orders"] is None:
                r = await self.call(trace, "list_orders", customer_id=st["customer_id"])
                st["orders"] = r.get("orders", [])
            orders = st["orders"]
            want = (task.get("product") or "").replace(" ", "").lower()
            hits = [x for x in orders if want and want[:3] in x["items"].replace(" ", "").lower()] if want else orders
            if len(hits) != 1:
                return {"say": "ask which order they mean", "facts": {"orders": orders}, "waiting": True}
            oid = task["order_id"] = hits[0]["order_id"]

        if oid not in st["details"]:
            r = await self.call(trace, "get_order", order_id=oid)
            if r.get("status") != "ok":
                task["order_id"] = None
                return {"say": "order not found; ask them to check the order number", "waiting": True}
            st["details"][oid] = r["order"]
        od = st["details"][oid]
        facts = {
            k: od.get(k) for k in ("order_id", "status", "items", "carrier", "tracking_no", "delivered_at", "total")
        }

        if intent == "order_status":
            return {"say": "tell the order status (and carrier and tracking number if shipped)", "facts": facts}
        if intent == "cancel":
            return await self.confirm_flow(trace, "cancel", oid, facts, confirm,
                                           None if od["status"] in CANCELLABLE else
                                           f"{od['status']} 상태라 취소 불가, 수령 후 반품 절차 안내")  # fmt: skip
        if intent == "return":
            why = None
            if od["status"] != "배송완료":
                why = "배송완료 전이라 반품 접수 불가"
            elif (date.fromisoformat(self.s.now[:10]) - date.fromisoformat(od["delivered_at"])).days > 7:
                why = "배송완료 후 7일이 지나 반품 불가"
            elif st["opened"]:
                why = "개봉한 상품은 반품 불가"
            if not why and st["opened"] is None:
                return {"say": "ask whether the product was opened", "facts": facts, "waiting": True}
            return await self.confirm_flow(trace, "return", oid, facts, confirm, why)
        return {"say": "ask how you can help with this order", "facts": facts, "waiting": True}

    async def confirm_flow(
        self, trace: Trace, kind: str, oid: str, facts: dict, confirm: str | None, why: str | None
    ) -> dict:
        st = self.st
        if why:
            st["pending"] = None
            return {"say": f"explain that {kind} is not possible and why; offer a human agent if they insist",
                    "facts": {**facts, "reason": why}}  # fmt: skip
        pend = st["pending"]
        if pend and pend["intent"] == kind and pend["order_id"] == oid and confirm == "yes":
            st["pending"] = None
            if kind == "cancel":
                r = await self.call(trace, "cancel_order", order_id=oid, reason="고객 요청")
            else:
                r = await self.call(
                    trace, "request_return", order_id=oid, reason=st["reason"] or "단순변심", opened=False
                )
            return {"say": f"confirm the {kind} result", "facts": r}
        if pend and confirm == "no":
            st["pending"] = None
            return {"say": f"acknowledge that the {kind} will not proceed and the order stays as is"}
        st["pending"] = {"intent": kind, "order_id": oid}
        extra = "mention the 6,000원 return shipping fee for a change of mind" if kind == "return" else ""
        return {
            "say": f"repeat the product and ask for explicit consent to {kind}; {extra}",
            "facts": facts,
            "waiting": True,
        }
