"""
Replay the starter kit's three demo clips through STT -> LLM (+demo tools) -> TTS, one wav per call.

Uses the starter kit's own demo cases, tools and mock data (data/demo_*.json*), the manual prompt
with the shop manual plus a short demo addendum, and the models set in .env.

uv run python -m playground.demo_calls --variant plain --tts full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import time
from pathlib import Path

from .config import ROOT, RUNS, env, load_dotenv, load_manual
from .evaluate import load_variants
from .pipeline import run_call
from .speech import RATE, write_wav16k
from .tools import CallRecord, ToolRunner

DATA = ROOT / "data"
DEMO_ADDENDUM = """

## Demo call addendum (starter kit demo tools)
- Only `get_order_info` and `request_callback` exist in this call; the other tools above are unavailable.
- Order status: call `get_order_info` with the order ID the caller says; the demo needs no identity check.
- Callback: use the caller profile's phone number when it is given, otherwise ask for it, then call `request_callback`.
"""


def _norm(v: object) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(v).upper())


class DemoTools(ToolRunner):
    """Answers tool calls from the starter kit's demo mock data by exact (normalized) argument match."""

    def __init__(self, now: str):
        super().__init__(now, tools_url="")
        self.mock = json.loads((DATA / "demo_mock_data.json").read_text(encoding="utf-8"))

    async def run(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        want = {k: _norm(v) for k, v in args.items()}
        result = next(
            (
                row["result"]
                for row in self.mock.get(name, [])
                if {k: _norm(v) for k, v in row["arguments"].items()} == want
            ),
            {"status": "not_found", "message": "no matching record"},
        )
        self.calls.append(CallRecord(name, args, result, (time.perf_counter() - t0) * 1000, self.turn))
        return result


def demo_scenarios() -> list[dict]:
    meta = json.loads((DATA / "audio" / "demo" / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (DATA / "demo_cases.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return [
        {
            "id": c["case_id"],
            "now": c["now"],
            "caller": c.get("customer_profile"),
            "turns": [
                {
                    "text": c["customer_utterance"],
                    "audio": f"data/audio/demo/{c['case_id']}.wav",
                    "speech_end_s": meta[c["case_id"]]["speech_end"],
                }
            ],
        }
        for c in rows
    ]


async def main_async(variant_name: str, mode: str, out_dir: Path, copy_to: Path | None) -> None:
    variant = load_variants(variant_name)[0]
    tool_schemas = [
        {"type": "function", "function": t} for t in json.loads((DATA / "demo_tools.json").read_text(encoding="utf-8"))
    ]
    voice = env("GEMINI_TTS_VOICE", "Kore")
    out_dir.mkdir(parents=True, exist_ok=True)
    for sc in demo_scenarios():
        print(f"\n=== {sc['id']} ({variant['name']}, tts {mode})", flush=True)
        extra = {"caller": sc["caller"], "manual": load_manual() + DEMO_ADDENDUM, "tool_schemas": tool_schemas}
        pcm, timeline = await run_call(sc, variant, voice, mode, tools=DemoTools(sc["now"]), session_extra=extra)
        out = out_dir / f"{sc['id']}_{variant['name']}_{mode}.wav"
        write_wav16k(out, pcm)
        out.with_suffix(".json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{out} ({len(pcm) / 2 / RATE:.1f}s)", flush=True)
        if copy_to:
            shutil.copy(out, copy_to / out.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--tts", choices=["full", "stream"], default="full")
    ap.add_argument("--out", type=Path, default=RUNS / "calls" / "demo")
    ap.add_argument("--copy-to", type=Path, help="also copy each wav here, e.g. ~/Downloads")
    a = ap.parse_args()
    load_dotenv()
    asyncio.run(main_async(a.variant, a.tts, a.out, a.copy_to.expanduser() if a.copy_to else None))


if __name__ == "__main__":
    main()
