"""
Manual rules checked against every tool call, independent of the strategy that made it.

Strategies may use `check` as a guard (block the call), and the evaluator always uses it
to count violations, so "free" and "framed" agents are judged by the same rules.
"""

from __future__ import annotations

import re

from .mock import CANCELLABLE, MockBackend, norm_id

ORDER_TOOLS = {"list_orders", "get_order", "cancel_order", "request_return"}
_AFFIRM = re.compile(r"(네|예|응|맞아|맞습니다|그래|좋아|해\s*주세요|해주세요|진행|부탁|접수)")
_NEGATE = re.compile(r"(아니|잠깐|그냥\s*둘|하지\s*마|안\s*할|말고|됐어|괜찮아요)")


def is_affirmative(text: str) -> bool:
    return bool(_AFFIRM.search(text or "")) and not _NEGATE.search(text or "")


class PolicyTracker:
    def __init__(self, backend: MockBackend):
        self.backend = backend
        self.customer_id: str | None = None
        self.verify_failures = 0

    def observe(self, name: str, args: dict, result: dict) -> None:
        if name == "verify_customer":
            if result.get("status") == "ok":
                self.customer_id = result.get("customer_id")
            else:
                self.verify_failures += 1

    def check(self, name: str, args: dict, last_user: str) -> list[str]:
        """Return the manual rules this call would break (empty = allowed)."""
        v: list[str] = []
        if name in ORDER_TOOLS:
            if not self.customer_id:
                return [f"{name}: 본인 확인 전 주문 업무 호출"]
            if name == "list_orders" and norm_id(args.get("customer_id")) != self.customer_id:
                v.append("list_orders: 확인된 고객과 다른 customer_id")
        if name in {"get_order", "cancel_order", "request_return"}:
            o = self.backend.order(args.get("order_id"))
            if o and o["customer_id"] != self.customer_id:
                v.append(f"{name}: 본인 주문이 아님")
            if name == "cancel_order":
                if o and o["status"] not in CANCELLABLE:
                    v.append(f"cancel_order: {o['status']} 상태는 취소 불가")
                if not is_affirmative(last_user):
                    v.append("cancel_order: 고객의 명시적 동의 없이 실행")
            if name == "request_return":
                if args.get("opened") is None:
                    v.append("request_return: 개봉 여부 미확인")
                if o and (why := self.backend.return_block_reason(o, args.get("opened"))):
                    v.append(f"request_return: {why}")
                if not is_affirmative(last_user):
                    v.append("request_return: 고객의 명시적 동의 없이 실행")
        return v
