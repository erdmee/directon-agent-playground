"""Paths, shared .env loading and settings for the playground."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PG = Path(__file__).resolve().parent
DOMAIN = PG / "domain"
DATA = PG / "data"
RUNS = ROOT / "runs" / "playground"

KEY_NAMES = ("SONIOX_API_KEY", "BAZE_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "DECISION_API_KEY")
SONIOX_DIRECT_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
# BAZE gateway (docs.factchat.kr): one key for Soniox STT, OpenAI-compatible chat and Gemini TTS.
BAZE_GATEWAY = "https://api.baze.co/v1/gateway"
BAZE_SONIOX_URL = "wss://api.baze.co/v1/gateway/soniox/transcribe-websocket"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader; values already in the shell environment win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def key_status() -> dict[str, bool]:
    """Which keys are set, without exposing any value."""
    return {k: bool(env(k)) for k in KEY_NAMES}


def soniox_settings() -> dict[str, str]:
    """
    Endpoint, key and model for Soniox real-time STT.

    A BAZE_API_KEY routes Soniox through the BAZE gateway and takes precedence over
    SONIOX_URL / SONIOX_API_KEY; without it the direct Soniox endpoint is used.
    """
    model = env("SONIOX_MODEL", "stt-rt-v6")
    if env("BAZE_API_KEY"):
        return {
            "via": "baze",
            "url": env("BAZE_SONIOX_URL", BAZE_SONIOX_URL),
            "key": env("BAZE_API_KEY"),
            "model": model,
        }
    return {"via": "soniox", "url": env("SONIOX_URL", SONIOX_DIRECT_URL), "key": env("SONIOX_API_KEY"), "model": model}


def llm_settings(prefix: str = "LLM") -> dict:
    """
    Settings for an OpenAI-compatible endpoint.

    LLM_* uses, in order: LLM_API_KEY, then BAZE_API_KEY (BAZE gateway as base URL), then
    OPENAI_API_KEY. Any other prefix (e.g. DECISION) with no key of its own falls back to the
    LLM_* settings, so decision strategies run before a separate model is set up.
    """
    if prefix != "LLM" and not env(f"{prefix}_API_KEY"):
        return llm_settings("LLM")
    key, base_url, via = env(f"{prefix}_API_KEY"), env(f"{prefix}_BASE_URL") or None, prefix.lower()
    if prefix == "LLM" and not key:
        if env("BAZE_API_KEY"):
            key, base_url, via = env("BAZE_API_KEY"), base_url or BAZE_GATEWAY, "baze"
        else:
            key, via = env("OPENAI_API_KEY"), "openai"
    # "off" omits reasoning_effort; other providers (e.g. Mercury) default to off.
    effort = env(f"{prefix}_REASONING_EFFORT", "none" if prefix == "LLM" else "off")
    return {
        "via": via,
        "api_key": key,
        "base_url": base_url,
        "model": env(f"{prefix}_MODEL", "gpt-5.4-mini" if prefix == "LLM" else ""),
        "reasoning_effort": None if effort == "off" else effort,
    }


def now_default() -> str:
    return env("PLAYGROUND_NOW", "2026-11-21 11:00")


def load_tools() -> list[dict]:
    return json.loads((DOMAIN / "tools.json").read_text(encoding="utf-8"))


def openai_tools(names: set[str] | None = None) -> list[dict]:
    return [{"type": "function", "function": t} for t in load_tools() if names is None or t["name"] in names]


def load_manual() -> str:
    return (DOMAIN / "manual.md").read_text(encoding="utf-8")


def load_scenarios(path: Path | None = None) -> list[dict]:
    path = path or DATA / "scenarios.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        # Turns may be plain strings or {"text", "audio", "stt"} objects.
        r["turns"] = [parse_turn(t) if isinstance(t, str) else t for t in r["turns"]]
    return rows


_PAUSE = re.compile(r"\s*\[(\d+)\]\s*")


def parse_turn(script: str) -> dict:
    """Split "음… [600] 그게요" into TTS segments with pauses; `text` is what the caller says."""
    parts = _PAUSE.split(script)
    segments = [
        {"text": parts[i].strip(), "pause_after_ms": int(parts[i + 1]) if i + 1 < len(parts) else 0}
        for i in range(0, len(parts), 2)
        if parts[i].strip()
    ]
    return {"text": " ".join(s["text"] for s in segments), "script": script, "segments": segments}
