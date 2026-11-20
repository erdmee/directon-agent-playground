"""
Run the frozen tool-calling bench and print a leaderboard (scoring rules in bench/v1/SPEC.md).

uv run python -m playground.bench.run --repeat 3
uv run python -m playground.bench.run --variants plain,decision-rules --repeat 3
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

from ..config import DOMAIN, PG, RUNS, load_dotenv, load_scenarios
from ..evaluate import load_variants, run_one

GROUPS = {
    "조회": ("배송", "조회", "주문번호", "본인확인", "고령", "미수령"),
    "취소": ("취소",),
    "반품": ("반품",),
    "상품·의학": ("상품", "재고", "의학"),
    "인계·콜백·FAQ": ("부작용", "상담원", "콜백", "FAQ"),
}


def group_of(category: str) -> str:
    return next((g for g, keys in GROUPS.items() if any(k in (category or "") for k in keys)), "기타")


def check_frozen(bench_dir: Path) -> list[str]:
    """Names of live domain files that differ from the frozen copies."""
    sums = dict(line.split()[::-1] for line in (bench_dir / "SHA256SUMS").read_text().splitlines() if line.strip())
    return [
        n for n in ("manual.md", "tools.json", "data.json")
        if hashlib.sha256((DOMAIN / n).read_bytes()).hexdigest() != sums.get(n)
    ]  # fmt: skip


# Starter-kit demo table (eval/score_demo.py): seconds from caller speech end to first agent sound -> points.
SPEED_TABLE = [(1.5, 100), (2.0, 90), (2.5, 80), (3.0, 70), (3.5, 60), (4.0, 50), (5.0, 30), (6.0, 10)]


def speed_score(ms: float | None, offset_ms: float = 0.0) -> float:
    """Starter-kit points for LLM first-text time plus an estimated STT/TTS offset."""
    if ms is None:
        return 0.0
    t = (ms + offset_ms) / 1000
    return float(next((v for bound, v in SPEED_TABLE if t <= bound), 0))


def score_run(row: dict, expect: dict, offset_ms: float = 0.0) -> dict:
    s = row["score"]
    if row["error"]:
        return {"function": 0.0, "handling": 0.0, "speed": 0.0}
    function = 100 * s["recall"]
    if s["forbidden"]:
        function = 0
    function -= 25 * len(s["violations"]) + 10 * (len(s["extra"]) + s["duplicates"])
    handling = 100 - 50 * (not s["handoff_ok"]) - 50 * bool(s["bad_phrases"]) - 50 * bool(s["violations"])
    speeds = [speed_score(t["first_text_ms"], offset_ms) for t in row["turns"]]
    return {"function": max(0.0, function), "handling": max(0.0, handling),
            "speed": statistics.mean(speeds) if speeds else 0.0}  # fmt: skip


def leaderboard(rows: list[dict], scenarios: dict[str, dict], offset_ms: float = 0.0) -> list[dict]:
    out = []
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        sc = [score_run(r, scenarios[r["scenario"]].get("expect", {}), offset_ms) for r in rs]
        f, h, sp = (statistics.mean(x[k] for x in sc) for k in ("function", "handling", "speed"))
        turns = [t for r in rs for t in r["turns"]]
        lat = sorted(t["first_text_ms"] for t in turns if t["first_text_ms"] is not None)
        by_group = {}
        for g in GROUPS:
            gr = [r for r in rs if group_of(r["category"]) == g]
            if gr:
                by_group[g] = round(100 * sum(r["score"]["pass"] for r in gr) / len(gr))
        out.append({
            "variant": name,
            "total": round(0.5 * h + 0.2 * f + 0.3 * sp, 1),
            "handling": round(h, 1), "function": round(f, 1), "speed": round(sp, 1),
            "pass_%": round(100 * sum(r["score"]["pass"] for r in rs) / len(rs)),
            "p50_ms": round(lat[len(lat) // 2]) if lat else None,
            "p90_ms": round(lat[int(len(lat) * 0.9)]) if lat else None,
            "llm_calls/turn": round(sum(t["llm_calls"] for t in turns) / max(1, len(turns)), 2),
            "tokens/turn": round(sum(t["tokens"]["prompt"] + t["tokens"]["completion"] for t in turns) / max(1, len(turns))),
            "errors": sum(bool(r["error"]) for r in rs),
            **{f"{g} pass_%": v for g, v in by_group.items()},
            "failed": sorted({r["scenario"] for r in rs if not r["score"]["pass"]}),
        })  # fmt: skip
    return sorted(out, key=lambda x: -x["total"])


def to_markdown(board: list[dict]) -> str:
    cols = [c for c in board[0] if c != "failed"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(b.get(c, "")) for c in cols) + " |" for b in board]
    lines += ["", "Failed scenarios (any repeat):"]
    lines += [f"- {b['variant']}: {', '.join(b['failed']) or '-'}" for b in board]
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--variants", help="comma-separated names from variants.json (default: all)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--allow-drift", action="store_true", help="run even if domain files changed since freezing")
    ap.add_argument(
        "--stt-ms", type=float, default=0.0, help="estimated STT wait after speech, added before scoring speed"
    )
    ap.add_argument("--tts-ms", type=float, default=0.0, help="estimated TTS time to first sound, added before scoring")
    ap.add_argument("--rescore", type=Path, help="re-score a saved run folder instead of calling any API")
    a = ap.parse_args()
    load_dotenv()
    bench_dir = PG / "bench" / a.version
    offset = a.stt_ms + a.tts_ms
    title = f"# Bench {a.version}, speed = starter-kit table on LLM first text + {offset:.0f} ms (STT {a.stt_ms:.0f} + TTS {a.tts_ms:.0f})"
    if a.rescore:
        scenarios = {s["id"]: s for s in load_scenarios(bench_dir / "bench.jsonl")}
        rows = [json.loads(x) for x in (a.rescore / "results.jsonl").read_text(encoding="utf-8").splitlines() if x]
        reps = len({r["rep"] for r in rows})
        md = f"{title}, repeat {reps}\n\n" + to_markdown(leaderboard(rows, scenarios, offset))
        name = f"leaderboard_offset{offset:.0f}.md"
        (a.rescore / name).write_text(md + "\n", encoding="utf-8")
        print(md + f"\n\nwrote {a.rescore / name}")
        return
    if (drift := check_frozen(bench_dir)) and not a.allow_drift:
        raise SystemExit(f"domain files changed since bench {a.version} was frozen: {drift} (use --allow-drift)")
    scenarios = {s["id"]: s for s in load_scenarios(bench_dir / "bench.jsonl")}
    variants = load_variants(a.variants)
    out = RUNS / time.strftime(f"bench-{a.version}-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(a.concurrency)
    done = 0
    total = len(variants) * len(scenarios) * a.repeat

    async def one(v: dict, sc: dict, rep: int) -> dict:
        nonlocal done
        async with sem:
            row = await run_one(v, sc, "text", rep)
        done += 1
        if done % 20 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)
        return row

    jobs = [one(v, s, r) for r in range(a.repeat) for v in variants for s in scenarios.values()]
    rows = await asyncio.gather(*jobs)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    board = leaderboard(rows, scenarios, offset)
    (out / "leaderboard.json").write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"{title}, repeat {a.repeat}\n\n" + to_markdown(board)
    (out / "leaderboard.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md + f"\n\nresults: {out}")


if __name__ == "__main__":
    asyncio.run(main())
