"""Bench integrity and scoring rules (offline)."""

from playground.bench.run import check_frozen, score_run, speed_score
from playground.config import PG, load_scenarios


def test_bench_v1_is_frozen_and_complete():
    assert check_frozen(PG / "bench" / "v1") == []
    ids = [s["id"] for s in load_scenarios(PG / "bench" / "v1" / "bench.jsonl")]
    assert len(ids) == 36 and len(set(ids)) == 36


def test_speed_score_follows_starter_kit_table():
    assert speed_score(1500) == 100 and speed_score(1501) == 90 and speed_score(4000) == 50
    assert speed_score(6500) == 0 and speed_score(None) == 0
    assert speed_score(1000, offset_ms=1000) == 90


def _row(**score):
    base = {"recall": 1.0, "forbidden": [], "violations": [], "extra": [], "duplicates": 0,
            "handoff_ok": True, "bad_phrases": []}  # fmt: skip
    return {"error": None, "score": {**base, **score}, "turns": [{"first_text_ms": 800}]}


def test_score_run_penalties():
    assert score_run(_row(), {}) == {"function": 100, "handling": 100, "speed": 100}
    s = score_run(_row(forbidden=["cancel_order"], violations=["x"]), {})
    assert s["function"] == 0 and s["handling"] == 50
    assert score_run(_row(recall=0.5, extra=["get_order"]), {})["function"] == 40
