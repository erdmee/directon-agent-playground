"""
Run variants x scenarios, score tool-call accuracy and latency, write a comparison table.

uv run --with-requirements playground/requirements.txt python -m playground.evaluate \
    --variants manual,tree --scenarios S01,S02 --repeat 2 --input text
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import traceback
from pathlib import Path

from .config import DATA, PG, RUNS, load_dotenv, load_scenarios, now_default
from .scoring import score
from .strategies import REGISTRY, Session, Trace
from .tools import ToolRunner


def load_variants(names: str | None) -> list[dict]:
    allv = json.loads((PG / "variants.json").read_text(encoding="utf-8"))
    if not names:
        return allv
    wanted = names.split(",")
    return [v for v in allv if v["name"] in wanted]


def turn_input(turn: dict, mode: str) -> str:
    """text = scripted text; stt / stt_phone = Soniox transcript of clean / phone-band audio."""
    if mode != "text" and not turn.get(mode):
        raise ValueError(f"turn has no {mode!r}; run `playground.dataset.audio transcribe` first")
    return turn["text"] if mode == "text" else turn[mode]


async def run_one(variant: dict, sc: dict, mode: str, rep: int) -> dict:
    tools = ToolRunner(sc.get("now") or now_default(), guard=variant.get("guard", False))
    session = Session(now=tools.backend.now, tools=tools, caller=sc.get("caller"))
    agent = REGISTRY[variant["strategy"]](session, **variant.get("options", {}))
    turns, replies, error = [], [], None
    try:
        for t in sc["turns"]:
            text = turn_input(t, mode)
            tools.new_turn(text)
            trace = Trace()
            reply = await agent.turn(text, trace)
            replies.append(reply)
            turns.append(
                {
                    "user": text,
                    "reply": reply,
                    "first_text_ms": trace.first_text_ms,
                    "total_ms": trace.ms(),
                    "llm_calls": trace.llm_calls,
                    "tokens": trace.tokens,
                    "events": trace.events,
                }
            )
    except Exception as e:  # keep the batch going; the row records the failure
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        await tools.aclose()
    calls = tools.log()
    s = score(sc.get("expect", {}), calls, replies)
    if error:
        s["pass"] = False
    return {
        "variant": variant["name"],
        "scenario": sc["id"],
        "category": sc.get("category"),
        "rep": rep,
        "input": mode,
        "error": error,
        "score": s,
        "calls": calls,
        "turns": turns,
    }


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 0)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        ts = [t for r in rs for t in r["turns"]]
        out.append(
            {
                "variant": name,
                "runs": len(rs),
                "pass_rate": round(sum(r["score"]["pass"] for r in rs) / len(rs), 3),
                "call_recall": round(statistics.mean(r["score"]["recall"] for r in rs), 3),
                "violations/run": round(sum(len(r["score"]["violations"]) for r in rs) / len(rs), 2),
                "blocked/run": round(sum(r["score"]["n_blocked"] for r in rs) / len(rs), 2),
                "forbidden/run": round(sum(len(r["score"]["forbidden"]) for r in rs) / len(rs), 2),
                "extra/run": round(sum(len(r["score"]["extra"]) + r["score"]["duplicates"] for r in rs) / len(rs), 2),
                "errors": sum(bool(r["error"]) for r in rs),
                "first_text_p50_ms": pct([t["first_text_ms"] for t in ts], 0.5),
                "first_text_p90_ms": pct([t["first_text_ms"] for t in ts], 0.9),
                "turn_p50_ms": pct([t["total_ms"] for t in ts], 0.5),
                "llm_calls/turn": round(sum(t["llm_calls"] for t in ts) / max(1, len(ts)), 2),
                "tokens/turn": round(
                    sum(t["tokens"]["prompt"] + t["tokens"]["completion"] for t in ts) / max(1, len(ts))
                ),
                "failed": sorted({r["scenario"] for r in rs if not r["score"]["pass"]}),
            }
        )
    return out


def to_markdown(summary: list[dict]) -> str:
    cols = [k for k in summary[0] if k != "failed"] + ["failed"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summary:
        lines.append("| " + " | ".join(", ".join(s[c]) if c == "failed" else str(s[c]) for c in cols) + " |")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", help="comma-separated names from variants.json (default: all)")
    ap.add_argument("--scenarios", help="comma-separated scenario ids (default: all)")
    ap.add_argument(
        "--file", type=Path, nargs="+", help="scenario jsonl file(s) (default: scenarios.jsonl + voice_scenarios.jsonl)"
    )
    ap.add_argument("--input", choices=["text", "stt", "stt_phone"], default="text")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    load_dotenv()

    variants = load_variants(a.variants)
    files = a.file or [DATA / "scenarios.jsonl", DATA / "voice_scenarios.jsonl"]
    scenarios = [s for f in files for s in load_scenarios(f)]
    if a.scenarios:
        ids = set(a.scenarios.split(","))
        scenarios = [s for s in scenarios if s["id"] in ids]
    out = a.out or RUNS / time.strftime("eval-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(a.concurrency)
    jobs = [(v, s, r) for r in range(a.repeat) for v in variants for s in scenarios]

    async def guarded(v, s, r):
        async with sem:
            row = await run_one(v, s, a.input, r)
            mark = "PASS" if row["score"]["pass"] else "FAIL"
            print(
                f"{mark} {v['name']:<18} {s['id']} rep{r} {row['score']['missing'] or ''}"
                f"{row['score']['violations'] or ''}{row['error'] or ''}",
                flush=True,
            )
            return row

    rows = await asyncio.gather(*(guarded(*j) for j in jobs))
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(summary)
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md + f"\n\nresults: {out}")


if __name__ == "__main__":
    asyncio.run(main())
