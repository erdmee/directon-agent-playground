"""
Synthesize realistic Korean caller audio for scenarios, then add Soniox transcripts.

  synth:      Gemini TTS per turn; "[600]" in a turn script becomes 600 ms of silence, so
              mid-utterance pauses and self-corrections land exactly where the script puts them.
              Writes <out_dir>/<id>_<n>.wav per caller turn and <id>_full.wav (turns joined) for listening.
              --phone also writes an 8 kHz telephone-band copy (<id>_<n>_phone.wav).
  transcribe: Soniox real-time STT on every turn's audio -> turn["stt"] (and turn["stt_phone"]).

uv run python -m playground.dataset.audio synth --in playground/data/voice_scenarios.jsonl --phone
uv run python -m playground.dataset.audio transcribe --in playground/data/voice_scenarios_audio.jsonl
Real recordings work too: put a wav path in turn["audio"] and run only `transcribe`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from ..config import DATA, ROOT, env, load_dotenv, load_scenarios, soniox_settings
from ..speech import RATE, gemini_tts, read_wav16k, resample, soniox_transcribe, trim_silence, write_wav16k

LEAD_S, TAIL_S, TURN_GAP_S = 0.4, 0.8, 1.5
STT_TERMS = ["건강한하루몰", "오메가3", "알티지", "프로바이오틱스", "루테인", "지아잔틴", "콜라겐", "홍삼정", "운송장"]


def style_for(sc: dict) -> str:
    persona = sc.get("persona") or "an ordinary Korean customer"
    return (
        f"You are {persona}, calling an online shop's customer center on the phone. "
        "Speak this in natural, casual spoken Korean, not like reading a script; "
        "fillers and self-corrections should sound spontaneous"
    )


def silence(sec: float) -> bytes:
    return b"\0" * (int(RATE * sec) * 2)


def phone_channel(pcm: bytes, seed: int = 0) -> bytes:
    """Telephone band (300-3400 Hz), 8 kHz round trip and faint line noise, back at 16 kHz."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    a = np.asarray(sosfilt(butter(6, [300, 3400], btype="band", fs=RATE, output="sos"), x))
    a += np.random.default_rng(seed).normal(0, 60, len(x))
    narrow = np.clip(a, -32768, 32767).astype(np.int16).tobytes()
    return resample(resample(narrow, RATE, 8000), 8000, RATE)


async def tts_retry(text: str, voice: str, style: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            return await gemini_tts(text, voice, style)
        except Exception as e:  # rate limits on the preview model are common
            # Billing (402), auth and unknown-model errors will not fix themselves on retry.
            if i == tries - 1 or any(f"Error code: {c}" in str(e) for c in (400, 401, 402, 403, 404)):
                raise
            wait = 8 * (i + 1)
            print(f"  retry in {wait}s: {type(e).__name__}: {str(e)[:120]}", flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


async def synth(rows: list[dict], out_dir: Path, default_voice: str, phone: bool, force: bool) -> None:
    for n, sc in enumerate(rows):
        voice, style = sc.get("voice") or default_voice, style_for(sc)
        full = []
        for i, t in enumerate(sc["turns"], 1):
            path = out_dir / f"{sc['id']}_{i}.wav"
            if t.get("audio") and not force:
                full.append(read_wav16k(ROOT / t["audio"]))
                continue
            speech, pauses = b"", []
            for seg in t.get("segments") or [{"text": t["text"], "pause_after_ms": 0}]:
                speech += trim_silence(await tts_retry(seg["text"], voice, style))
                if seg["pause_after_ms"]:
                    speech += silence(seg["pause_after_ms"] / 1000)
                    pauses.append(seg["pause_after_ms"])
            pcm = silence(LEAD_S) + speech + silence(TAIL_S)
            write_wav16k(path, pcm)
            t["audio"] = rel(path)
            t["speech_start_s"], t["speech_end_s"] = LEAD_S, round(LEAD_S + len(speech) / 2 / RATE, 3)
            t["pauses_ms"] = pauses
            if phone:
                p = out_dir / f"{sc['id']}_{i}_phone.wav"
                write_wav16k(p, phone_channel(pcm, seed=n * 10 + i))
                t["audio_phone"] = rel(p)
            full.append(pcm)
            print(f"{sc['id']} turn {i}: {t['speech_end_s'] - LEAD_S:.1f}s speech, voice={voice}", flush=True)
        write_wav16k(out_dir / f"{sc['id']}_full.wav", silence(TURN_GAP_S).join(full))
        sc["caller_speech_s"] = round(sum(t.get("speech_end_s", 0) - LEAD_S for t in sc["turns"]), 1)


async def transcribe(rows: list[dict], force: bool) -> None:
    for sc in rows:
        for i, t in enumerate(sc["turns"], 1):
            for src, dst in (("audio", "stt"), ("audio_phone", "stt_phone")):
                if not t.get(src) or (t.get(dst) and not force):
                    continue
                path = Path(t[src])
                t[dst] = await soniox_transcribe(read_wav16k(path if path.is_absolute() else ROOT / path), STT_TERMS)
                print(f"{sc['id']} {i} {dst:<9}| {t['text']}\n{'':>16}-> {t[dst]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth", "transcribe"])
    ap.add_argument("--in", dest="inp", type=Path, default=DATA / "voice_scenarios.jsonl")
    ap.add_argument("--out", type=Path, help="default: synth -> <in>_audio.jsonl, transcribe -> overwrite --in")
    ap.add_argument("--audio-dir", type=Path, default=DATA / "audio" / "voice")
    ap.add_argument("--voice", default=None, help="fallback when a scenario has no voice")
    ap.add_argument("--phone", action="store_true", help="also write a telephone-band copy")
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--force", action="store_true", help="redo turns that already have audio/stt")
    a = ap.parse_args()
    load_dotenv()
    rows = load_scenarios(a.inp)
    todo = [r for r in rows if not a.only or r["id"] in a.only.split(",")]
    if a.cmd == "synth":
        if not env("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY is not set in .env")
        asyncio.run(synth(todo, a.audio_dir, a.voice or env("GEMINI_CALLER_VOICE", "Puck"), a.phone, a.force))
        out = a.out or a.inp.with_name(a.inp.stem + "_audio.jsonl")
    else:
        if not soniox_settings()["key"]:
            raise SystemExit("Set BAZE_API_KEY or SONIOX_API_KEY in .env")
        asyncio.run(transcribe(todo, a.force))
        out = a.out or a.inp
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    total = sum(r.get("caller_speech_s", 0) for r in rows)
    print(f"wrote {out}" + (f" ({total:.0f}s caller speech in total)" if total else ""))


if __name__ == "__main__":
    main()
