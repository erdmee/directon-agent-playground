"""
Replay a voice scenario through caller audio -> STT -> LLM (+tools) -> TTS and save the whole call as one wav.

Each caller turn is streamed to Soniox in real time. All agent timings are measured from the
moment the caller's speech ends, and every agent audio chunk is placed in the output at the
time it actually arrived, so the silences you hear are what a caller would hear.

--tts full:   wait for the whole reply, then synthesize it in one request (simple, slow).
--tts stream: split the reply into sentences while the LLM is still writing, stream TTS per
              sentence and play chunks as they arrive (gaps appear if synthesis falls behind).

uv run python -m playground.pipeline --file playground/data/voice_scenarios_audio.jsonl --id V02 --tts stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from .config import ROOT, RUNS, env, load_dotenv, load_scenarios, now_default
from .dataset.audio import STT_TERMS, silence
from .evaluate import load_variants
from .speech import RATE, gemini_tts, gemini_tts_stream, read_wav16k, soniox_transcribe, trim_silence, write_wav16k
from .strategies import REGISTRY, Session, Trace
from .tools import ToolRunner

AGENT_STYLE = "Speak as a calm, friendly Korean customer-service agent on the phone, at a natural pace"
GREETING = "안녕하세요, 건강한하루몰 고객센터입니다. 무엇을 도와드릴까요?"
_SENT_END = re.compile(r"[.?!。？！](?=\s|$)")
FIRST_CLAUSE_MIN = 8  # Speak a first comma clause early once it has this many characters.


class SentenceSplitter:
    """Cuts streamed reply text into speakable pieces as soon as each one is complete."""

    def __init__(self) -> None:
        self.buf = ""
        self.emitted = 0

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            m = _SENT_END.search(self.buf)
            if not m and self.emitted == 0 and (c := self.buf.find(",")) >= FIRST_CLAUSE_MIN:
                end = c + 1
            elif m:
                end = m.end()
            else:
                break
            piece, self.buf = self.buf[:end].strip(), self.buf[end:]
            if piece:
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        piece, self.buf = self.buf.strip(), ""
        return [piece] if piece else []


class Playout:
    """Places audio chunks on a timeline: a chunk starts when it arrived or when the previous one ends."""

    def __init__(self, t_zero: float) -> None:
        self.t_zero = t_zero
        self.cursor_s = 0.0
        self.chunks: list[tuple[float, bytes]] = []

    def add(self, pcm: bytes) -> None:
        arrived = time.perf_counter() - self.t_zero
        start = max(self.cursor_s, arrived)
        self.chunks.append((start, pcm))
        self.cursor_s = start + len(pcm) / 2 / RATE

    def render(self) -> tuple[bytes, float | None, float]:
        """Return (audio from t_zero, first-audio delay s, total stall s after the first chunk)."""
        out, pos, stall = bytearray(), 0.0, 0.0
        for i, (start, pcm) in enumerate(self.chunks):
            gap = start - pos
            if i and gap > 0.02:
                stall += gap
            out += silence(max(0.0, gap))
            out += pcm
            pos = start + len(pcm) / 2 / RATE
        return bytes(out), (self.chunks[0][0] if self.chunks else None), stall


async def agent_turn_streaming(agent, heard: str, voice: str, t_zero: float) -> tuple[str, Trace, Playout, list]:
    playout, splitter = Playout(t_zero), SentenceSplitter()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    pieces: list[dict] = []

    async def speaker() -> None:
        while (text := await queue.get()) is not None:
            t0 = time.perf_counter() - t_zero
            first = None
            async for pcm in gemini_tts_stream(text, voice, AGENT_STYLE):
                first = first if first is not None else time.perf_counter() - t_zero
                playout.add(pcm)
            pieces.append({"text": text, "queued_s": round(t0, 2), "first_audio_s": first and round(first, 2)})

    task = asyncio.create_task(speaker())

    def on_text(delta: str) -> None:
        for s in splitter.push(delta):
            queue.put_nowait(s)

    trace = Trace(on_text=on_text)
    reply = await agent.turn(heard, trace)
    trace.emit("llm_done")  # Marks LLM end before waiting for the remaining TTS.
    for s in splitter.flush():
        queue.put_nowait(s)
    queue.put_nowait(None)
    await task
    return reply, trace, playout, pieces


async def run_call(sc: dict, variant: dict, voice: str, mode: str) -> tuple[bytes, list[dict]]:
    tools = ToolRunner(sc.get("now") or now_default(), guard=variant.get("guard", False))
    agent = REGISTRY[variant["strategy"]](Session(now=tools.backend.now, tools=tools), **variant.get("options", {}))
    timeline: list[dict] = [{"speaker": "agent", "text": GREETING}]
    audio = [silence(0.3), trim_silence(await gemini_tts(GREETING, voice, AGENT_STYLE)), silence(0.6)]

    for i, turn in enumerate(sc["turns"], 1):
        caller_pcm = read_wav16k(ROOT / turn["audio"])
        file_s = len(caller_pcm) / 2 / RATE
        speech_end_s = turn.get("speech_end_s") or file_s
        audio.append(caller_pcm)

        t0 = time.perf_counter()
        t_zero = t0 + speech_end_s  # Wall-clock moment the caller stops talking (real-time streaming).
        heard = await soniox_transcribe(caller_pcm, STT_TERMS, speed=1.0)
        stt_s = time.perf_counter() - t_zero
        tools.new_turn(heard)

        pieces: list = []
        if mode == "stream":
            reply, trace, playout, pieces = await agent_turn_streaming(agent, heard, voice, t_zero)
        else:
            trace = Trace()
            reply = await agent.turn(heard, trace)
            trace.emit("llm_done")
            playout = Playout(t_zero)
            if reply.strip():
                playout.add(trim_silence(await gemini_tts(reply, voice, AGENT_STYLE)))
        agent_pcm, first_s, stall_s = playout.render()

        # The caller file already holds its trailing silence; only add what is left of the delay.
        audio += [silence(max(0.0, (first_s or 0) - (file_s - speech_end_s))), agent_pcm[len(silence(first_s or 0)) :]]
        audio.append(silence(0.6))

        calls = [
            {"name": c.name, "args": c.args, "status": c.result.get("status")}
            for c in tools.calls
            if c.turn == tools.turn
        ]
        ms = {
            "stt_final_after_speech": round(stt_s * 1000),
            "llm_first_text_after_speech": round(stt_s * 1000 + (trace.first_text_ms or 0)),
            "llm_done_after_speech": round(
                stt_s * 1000 + next((e["t_ms"] for e in trace.events if e["kind"] == "llm_done"), trace.ms())
            ),
            "first_audio_after_speech": first_s and round(first_s * 1000),
            "stall_during_reply": round(stall_s * 1000),
        }
        timeline.append(
            {
                "turn": i,
                "script": turn["text"],
                "stt": heard,
                "agent_reply": reply,
                "tool_calls": calls,
                "tts_pieces": pieces,
                "ms": ms,
            }
        )
        print(f"[{i}] 고객(STT): {heard}", flush=True)
        for c in calls:
            print(f"     tool {c['name']} {json.dumps(c['args'], ensure_ascii=False)} -> {c['status']}")
        print(f"     상담원: {reply}")
        print("     " + "  ".join(f"{k}={v}" for k, v in ms.items()), flush=True)
    await tools.aclose()
    return b"".join(audio), timeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True, help="scenario jsonl whose turns have audio")
    ap.add_argument("--id", required=True)
    ap.add_argument("--variant", default="tree")
    ap.add_argument("--tts", choices=["full", "stream"], default="stream")
    ap.add_argument("--out", type=Path, help="default: runs/playground/calls/<id>_<variant>_<tts>.wav")
    a = ap.parse_args()
    load_dotenv()
    sc = next(s for s in load_scenarios(a.file) if s["id"] == a.id)
    variant = load_variants(a.variant)[0]
    pcm, timeline = asyncio.run(run_call(sc, variant, env("GEMINI_TTS_VOICE", "Kore"), a.tts))
    out = a.out or RUNS / "calls" / f"{a.id}_{variant['name']}_{a.tts}.wav"
    write_wav16k(out, pcm)
    out.with_suffix(".json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{out} ({len(pcm) / 2 / RATE:.1f}s)\n{out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
