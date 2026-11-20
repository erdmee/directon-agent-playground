"""
Show which .env keys are set (never their values) and which commands they unlock.

uv run python -m playground.check
"""

from __future__ import annotations

from .config import ROOT, env, llm_settings, load_dotenv, soniox_settings

FEATURES = [
    ("GEMINI_API_KEY", "caller audio synthesis", "playground.dataset.audio synth"),
    ("BAZE_API_KEY", "STT via BAZE (overrides SONIOX_*)", "playground.pipeline, playground.demo_calls"),
    ("SONIOX_API_KEY", "STT direct to Soniox", "playground.dataset.audio transcribe"),
    ("OPENAI_API_KEY", "baseline agent / eval / labeling", "playground.evaluate, playground.chat"),
    ("DECISION_API_KEY", "your decision model", 'llm.chat(..., prefix="DECISION")'),
]


def main() -> None:
    path = ROOT / ".env"
    print(f".env: {'found' if path.exists() else 'MISSING (cp env.example .env)'}")
    load_dotenv()
    for key, what, cmd in FEATURES:
        ok = bool(env(key)) or (key == "OPENAI_API_KEY" and bool(env("LLM_API_KEY")))
        print(f"  {'OK ' if ok else '-- '} {key:<17} {what:<34} {cmd}")
    s = llm_settings("LLM")
    print(f"\nLLM model: {s['model']}  base_url: {s['base_url'] or 'OpenAI default'}  effort: {s['reasoning_effort']}")
    stt = soniox_settings()
    print(f"TTS model: {env('GEMINI_TTS_MODEL', 'gemini-2.5-flash-preview-tts')}")
    print(f"STT model: {stt['model']}  via: {stt['via']}  url: {stt['url']}  key: {'set' if stt['key'] else 'MISSING'}")


if __name__ == "__main__":
    main()
