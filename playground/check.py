"""
Show which .env keys are set (never their values) and which route each provider takes.

uv run python -m playground.check
"""

from __future__ import annotations

from .config import ROOT, env, llm_settings, load_dotenv, soniox_settings

FEATURES = [
    ("BAZE_API_KEY", "one key for STT, LLM and TTS via BAZE", "everything below except DECISION"),
    ("SONIOX_API_KEY", "STT direct to Soniox", "used only without BAZE_API_KEY"),
    ("GEMINI_API_KEY", "TTS direct to Gemini", "used only without BAZE_API_KEY"),
    ("OPENAI_API_KEY", "LLM direct to OpenAI", "used only without BAZE/LLM_API_KEY"),
    ("DECISION_API_KEY", "your decision model (Mercury)", 'llm.chat(..., prefix="DECISION")'),
]


def main() -> None:
    path = ROOT / ".env"
    print(f".env: {'found' if path.exists() else 'MISSING (cp env.example .env)'}")
    load_dotenv()
    for key, what, note in FEATURES:
        print(f"  {'OK ' if env(key) else '-- '} {key:<17} {what:<40} {note}")
    baze = bool(env("BAZE_API_KEY"))
    s = llm_settings("LLM")
    stt = soniox_settings()
    tts_key = env("BAZE_API_KEY") or env("GEMINI_API_KEY")
    print(f"\nSTT  {stt['model']:<30} via {stt['via']:<7} key {'set' if stt['key'] else 'MISSING'}")
    print(
        f"LLM  {s['model']:<30} via {s['via']:<7} key {'set' if s['api_key'] else 'MISSING'}  effort {s['reasoning_effort']}"
    )
    print(
        f"TTS  {env('GEMINI_TTS_MODEL', 'gemini-2.5-flash-preview-tts'):<30} via {'baze' if baze else 'gemini':<7} "
        f"key {'set' if tts_key else 'MISSING'}"
    )


if __name__ == "__main__":
    main()
