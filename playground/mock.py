"""Stateful in-process mock of the shop backend; one instance per call session."""

from __future__ import annotations

import copy
import json
import re
from datetime import date, datetime

from .config import DOMAIN

CANCELLABLE = {"결제완료", "상품준비중"}
RETURN_DAYS = 7
RETURN_FEE = 6000
_DATA = json.loads((DOMAIN / "data.json").read_text(encoding="utf-8"))

KO_DIGITS = {
    "공": "0",
    "영": "0",
    "빵": "0",
    "일": "1",
    "이": "2",
    "삼": "3",
    "사": "4",
    "오": "5",
    "육": "6",
    "륙": "6",
    "칠": "7",
    "팔": "8",
    "구": "9",
}


def norm_id(v) -> str:
    return re.sub(r"[\s\-]", "", str(v or "")).upper()


def norm_phone(v) -> str:
    """Digits only; Korean spoken digits are not converted, so the agent must normalize them."""
    return re.sub(r"\D", "", str(v or ""))


def norm_text(v) -> str:
    return re.sub(r"\s", "", str(v or "")).lower()


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


class MockBackend:
    def __init__(self, now: str):
        self.now = now
        self.db = copy.deepcopy(_DATA)
        self.callbacks: list[dict] = []
        self.transfers: list[dict] = []

    # --- lookups shared with the policy checker ---
    def order(self, order_id) -> dict | None:
        oid = norm_id(order_id)
        return next((o for o in self.db["orders"] if o["order_id"] == oid), None)

    def customer_by(self, name, phone) -> dict | None:
        n, p = norm_text(name), norm_phone(phone)
        return next((c for c in self.db["customers"] if norm_text(c["name"]) == n and c["phone"] == p), None)

    def product(self, name) -> dict | None:
        q = norm_text(name)
        if not q:
            return None
        for p in self.db["products"]:
            keys = [norm_text(p["name"])] + [norm_text(a) for a in p["aliases"]]
            if any(q == k or q in k or k in q for k in keys):
                return p
        return None

    def return_block_reason(self, o: dict, opened) -> str | None:
        if o["status"] != "배송완료":
            return "배송완료 상태가 아니어서 반품 접수 불가"
        if days_between(o["delivered_at"], self.now) > RETURN_DAYS:
            return "배송완료 후 7일이 지나 반품 불가"
        if opened:
            return "개봉한 상품은 반품 불가"
        return None

    # --- tools ---
    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"status": "error", "message": f"unknown function {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"status": "invalid_arguments", "message": str(e)}

    def t_verify_customer(self, name="", phone="") -> dict:
        c = self.customer_by(name, phone)
        if not c:
            return {"status": "not_found", "message": "일치하는 고객 정보가 없습니다"}
        return {"status": "ok", "customer_id": c["customer_id"], "name": c["name"]}

    def t_list_orders(self, customer_id="") -> dict:
        rows = [o for o in self.db["orders"] if o["customer_id"] == norm_id(customer_id)]
        if not rows:
            return {"status": "not_found", "message": "주문 내역이 없습니다"}
        return {
            "status": "ok",
            "orders": [
                {
                    "order_id": o["order_id"],
                    "status": o["status"],
                    "ordered_at": o["ordered_at"],
                    "items": ", ".join(f"{i['product']} {i['qty']}개" for i in o["items"]),
                }
                for o in rows
            ],
        }

    def t_get_order(self, order_id="") -> dict:
        o = self.order(order_id)
        if not o:
            return {"status": "not_found", "message": "주문번호를 찾을 수 없습니다"}
        return {"status": "ok", "order": {k: v for k, v in o.items() if k != "customer_id"}}

    def t_cancel_order(self, order_id="", reason="") -> dict:
        o = self.order(order_id)
        if not o:
            return {"status": "not_found", "message": "주문번호를 찾을 수 없습니다"}
        if o["status"] not in CANCELLABLE:
            return {"status": "rejected", "message": f"{o['status']} 상태라 취소할 수 없습니다"}
        o["status"] = "취소완료"
        return {
            "status": "ok",
            "order_id": o["order_id"],
            "refund_amount": o["total"],
            "message": "취소 완료, 3영업일 이내 결제 수단으로 환불",
        }

    def t_request_return(self, order_id="", reason="", opened=None) -> dict:
        o = self.order(order_id)
        if not o:
            return {"status": "not_found", "message": "주문번호를 찾을 수 없습니다"}
        if why := self.return_block_reason(o, opened):
            return {"status": "rejected", "message": why}
        fee = 0 if reason in ("상품불량", "오배송") else RETURN_FEE
        o["status"] = "반품접수"
        return {
            "status": "ok",
            "return_id": "R" + o["order_id"][1:],
            "refund_amount": o["total"] - fee,
            "return_fee": fee,
            "message": "회수 후 3영업일 이내 환불",
        }

    def t_get_product_info(self, product_name="") -> dict:
        p = self.product(product_name)
        if not p:
            return {"status": "not_found", "message": "해당 상품을 찾을 수 없습니다"}
        return {"status": "ok", "product": {k: v for k, v in p.items() if k != "aliases"}}

    def t_transfer_to_agent(self, reason="", summary="") -> dict:
        self.transfers.append({"reason": reason, "summary": summary})
        return {"status": "ok", "message": "상담원 연결 대기열에 등록되었습니다", "wait_minutes": 3}

    def t_request_callback(self, phone="", preferred_time="") -> dict:
        if len(norm_phone(phone)) < 10:
            return {"status": "invalid_arguments", "message": "전화번호 형식이 올바르지 않습니다"}
        self.callbacks.append({"phone": norm_phone(phone), "preferred_time": preferred_time})
        return {"status": "ok", "message": "콜백 요청이 접수되었습니다"}


def today(now: str) -> str:
    return datetime.strptime(now, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d (%a) %H:%M")
