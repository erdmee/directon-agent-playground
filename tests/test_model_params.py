"""Unused (None) parameters must never reach the provider; used ones must."""

from playground.model_params import (
    GeminiTTSParams,
    OpenAIChatParams,
    SonioxContext,
    SonioxRealtimeParams,
)


def test_soniox_message_has_only_set_fields():
    msg = SonioxRealtimeParams(context=SonioxContext(terms=["오메가3"])).to_message()
    assert msg == {
        "model": "stt-rt-v6",
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "language_hints": ["ko"],
        "context": {"terms": ["오메가3"]},
    }
    assert "api_key" not in msg


def test_openai_kwargs_drop_unused_and_apply_overrides():
    kw = OpenAIChatParams().to_kwargs(service_tier="priority", temperature=None)
    assert kw == {
        "model": "gpt-5.4-mini",
        "reasoning_effort": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
        "service_tier": "priority",
    }


def test_gemini_25_puts_style_in_text_not_annotations():
    kw = GeminiTTSParams().to_kwargs("안녕하세요", style="calm", voice="Puck")
    item = kw["input"][0]["content"][0]
    assert item == {"type": "text", "text": "calm: 안녕하세요"}
    assert kw["generation_config"] == {"speech_config": [{"voice": "Puck"}]}
    assert kw["response_format"] == {"type": "audio"}


def test_gemini_38_uses_speech_metadata():
    kw = GeminiTTSParams(model="gemini-3.8-flash-tts", supports_annotations=True).to_kwargs("네", style="calm")
    assert kw["input"][0]["content"][0]["annotations"] == [{"type": "speech_metadata", "style": "calm"}]
