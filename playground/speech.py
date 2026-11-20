"""Speech I/O for dataset building: Gemini TTS (caller audio) and Soniox real-time STT on files."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import websockets
from scipy.signal import resample_poly

from .config import env
from .model_params import GeminiTTSParams, SonioxContext, SonioxRealtimeParams

RATE = 16000


def resample(pcm: bytes, src: int, dst: int) -> bytes:
    if src == dst or not pcm:
        return pcm
    a = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)
    g = np.gcd(src, dst)
    y = resample_poly(a, dst // g, src // g)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()


def read_wav16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav is supported")
        pcm = w.readframes(w.getnframes())
        if w.getnchannels() == 2:
            pcm = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2).mean(axis=1).astype(np.int16).tobytes()
        return resample(pcm, w.getframerate(), RATE)


def write_wav16k(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def trim_silence(pcm: bytes, thr: float = 300.0) -> bytes:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    hop = RATE // 50
    loud = [i for i in range(0, len(a), hop) if np.sqrt(np.mean(a[i : i + hop] ** 2)) >= thr]
    return pcm[loud[0] * 2 : (loud[-1] + hop) * 2] if loud else b""


def tts_params() -> GeminiTTSParams:
    """Our Gemini TTS settings; every other API field stays None in GeminiTTSParams."""
    model = env("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
    return GeminiTTSParams(
        model=model,
        voice=env("GEMINI_TTS_VOICE", "Kore"),
        supports_annotations=not model.startswith("gemini-2.5"),
    )


async def gemini_tts(
    text: str, voice: str | None = None, style: str = "", params: GeminiTTSParams | None = None
) -> bytes:
    """Whole-utterance Gemini TTS via the Interactions API, returned as 16 kHz mono s16le."""
    from google import genai

    client = genai.Client(api_key=env("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
    it = await client.aio.interactions.create(**(params or tts_params()).to_kwargs(text, style, voice))
    audio = getattr(it, "output_audio", None)
    if audio is None or not audio.data:
        raise RuntimeError(f"no audio in TTS response for {text[:30]!r}")
    data = base64.b64decode(audio.data)
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data), "rb") as w:
            return resample(w.readframes(w.getnframes()), w.getframerate(), RATE)
    m = re.search(r"rate=(\d+)", getattr(audio, "mime_type", "") or "")
    return resample(data, int(m.group(1)) if m else 24000, RATE)


async def gemini_tts_stream(
    text: str, voice: str | None = None, style: str = "", params: GeminiTTSParams | None = None
) -> AsyncIterator[bytes]:
    """
    Streaming Gemini TTS: yields 16 kHz s16le chunks as they arrive.

    The stream's default format is headerless 24 kHz L16; 2.5 Flash TTS rejects an explicit
    audio/l16 mime_type, so the format is left unset.
    """
    from google import genai

    client = genai.Client(api_key=env("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
    p = params or tts_params()
    p.stream = True
    stream = await client.aio.interactions.create(**p.to_kwargs(text, style, voice))
    rest, first = b"", True
    async for ev in stream:  # pyrefly: ignore[not-iterable]
        if getattr(ev, "event_type", "") == "error":
            raise RuntimeError(f"TTS stream error: {str(ev)[:200]}")
        delta = getattr(ev, "delta", None)
        if delta is None or not getattr(delta, "data", None):
            continue
        data = base64.b64decode(delta.data)
        if first and data[:4] == b"RIFF":
            data = data[44:]
        first = False
        # Resample whole 3-sample groups so 24k -> 16k needs no carried filter state.
        data = rest + data
        cut = len(data) // 6 * 6
        rest = data[cut:]
        if cut:
            yield resample(data[:cut], 24000, RATE)
    if len(rest) >= 2:
        yield resample(rest[: len(rest) // 2 * 2], 24000, RATE)


async def soniox_transcribe(pcm: bytes, terms: list[str] | None = None, speed: float = 4.0) -> str:
    """Stream a 16 kHz file through the same real-time model the live agent uses."""
    url = env("SONIOX_URL", "wss://stt-rt.soniox.com/transcribe-websocket")
    params = SonioxRealtimeParams(
        model=env("SONIOX_MODEL", "stt-rt-v6"),
        sample_rate=RATE,
        context=SonioxContext(terms=terms) if terms else None,
    )
    final: list[str] = []
    headers = {"Authorization": f"Bearer {env('SONIOX_API_KEY')}"}
    async with websockets.connect(url, max_size=None, additional_headers=headers) as ws:
        await ws.send(json.dumps(params.to_message()))

        async def sender():
            step = RATE * 2 // 10  # 100 ms
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i : i + step])
                await asyncio.sleep(0.1 / speed)
            await ws.send("")  # end of audio

        send_task = asyncio.create_task(sender())
        try:
            async for raw in ws:
                ev = json.loads(raw)
                if ev.get("error_code"):
                    raise RuntimeError(f"Soniox {ev.get('error_code')}: {ev.get('error_message')}")
                final += [
                    t["text"] for t in ev.get("tokens", []) if t.get("is_final") and not t["text"].startswith("<")
                ]
                if ev.get("finished"):
                    break
            await send_task
        finally:
            send_task.cancel()
    return "".join(final).strip()
