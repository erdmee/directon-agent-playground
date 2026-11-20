"""Offline checks of the environment (mock, policy, scoring, eval loop); no API keys needed."""

from __future__ import annotations

import asyncio

import pytest

from playground.config import DATA, load_scenarios, parse_turn
from playground.evaluate import run_one
from playground.strategies import REGISTRY, Strategy, Trace

SCENARIOS = {s["id"]: s for s in load_scenarios() + load_scenarios(DATA / "voice_scenarios.jsonl")}
NAMES = ("김민지", "이준호", "박서연")


class Scripted(Strategy):
    """Replays a fixed call list on the last turn, to test the harness without an LLM."""

    name = "scripted"

    async def turn(self, user_text: str, trace: Trace) -> str:
        self.n = getattr(self, "n", 0) + 1
        if self.n == self.options["n_turns"]:
            for name, args in self.options["calls"]:
                if name == "list_orders" and args.get("customer_id") == "$cid":
                    args = {"customer_id": self.s.tools.policy.customer_id}
                await self.s.tools.run(name, args)
        trace.emit("text", delta="네.")
        return "네."


REGISTRY["scripted"] = Scripted


def gold_calls(sc: dict) -> list[tuple[str, dict]]:
    out = []
    for spec in sc["expect"].get("calls", []):
        s = (spec.get("one_of") or [spec])[0]
        args = dict(s.get("args", {}))
        if s["name"] == "verify_customer":
            phone = str(args.get("phone", ""))
            name = {"01012345678": "김민지", "01022223333": "이준호", "01098765432": "박서연"}.get(phone)
            args.setdefault("name", name or next(n for n in NAMES if any(n in x["text"] for x in sc["turns"])))
        if s["name"] == "cancel_order":
            args.setdefault("reason", "고객 요청")
        if s["name"] == "request_return":
            args.setdefault("reason", "단순변심")
        if s["name"] == "transfer_to_agent":
            args.setdefault("summary", "테스트")
        if s["name"] == "transfer_to_agent":
            args.setdefault("reason", "고객요청")
        if s["name"] == "request_callback":
            args.setdefault("preferred_time", "내일 오전")
        args = {k: v[1:] if isinstance(v, str) and v.startswith("~") else v for k, v in args.items()}
        if s["name"] == "get_product_info":
            args.setdefault("product_name", "오메가3")
        out.append((s["name"], args))
    return out


def run(sc: dict, calls: list[tuple[str, dict]], guard: bool = False) -> dict:
    variant = {
        "name": "scripted",
        "strategy": "scripted",
        "guard": guard,
        "options": {"calls": calls, "n_turns": len(sc["turns"])},
    }
    return asyncio.run(run_one(variant, sc, "text", 0))


@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_gold_path_passes(sid):
    row = run(SCENARIOS[sid], gold_calls(SCENARIOS[sid]))
    assert row["score"]["pass"], row["score"]


def test_cancel_shipped_order_is_violation():
    sc = SCENARIOS["S03"]
    calls = [
        ("verify_customer", {"name": "박서연", "phone": "01098765432"}),
        ("cancel_order", {"order_id": "A1006", "reason": "x"}),
    ]
    s = run(sc, calls)["score"]
    assert not s["pass"] and "cancel_order" in s["forbidden"] and s["violations"]


def test_lookup_before_verification_is_blocked_by_guard():
    sc = SCENARIOS["S12"]
    calls = [("get_order", {"order_id": "A1001"})]
    s = run(sc, calls, guard=True)["score"]
    assert s["n_blocked"] == 1 and s["attempted_violations"] and not s["violations"]


def test_spoken_korean_digits_do_not_resolve():
    # The agent must normalize "공일공..." itself; the mock only strips separators.
    sc = SCENARIOS["S01"]
    s = run(sc, [("verify_customer", {"name": "김민지", "phone": "공일공 일이삼사 오육칠팔"})])["score"]
    assert "verify_customer" in s["missing"]


def test_pause_markers_become_segments():
    t = parse_turn("음… [600] 그게요, [300] 아 아니다")
    assert t["text"] == "음… 그게요, 아 아니다"
    assert [s["pause_after_ms"] for s in t["segments"]] == [600, 300, 0]


def test_forbid_targets_only_the_named_order():
    sc = SCENARIOS["V02"]
    wrong = [
        ("verify_customer", {"name": "김민지", "phone": "01012345678"}),
        ("cancel_order", {"order_id": "A1001", "reason": "x"}),
    ]
    assert "cancel_order" in run(sc, wrong)["score"]["forbidden"]


def test_cancel_without_consent_is_violation():
    sc = SCENARIOS["S15"]
    calls = [
        ("verify_customer", {"name": "김민지", "phone": "01012345678"}),
        ("cancel_order", {"order_id": "A1002", "reason": "x"}),
    ]
    s = run(sc, calls)["score"]
    assert any("동의" in v for v in s["violations"])
