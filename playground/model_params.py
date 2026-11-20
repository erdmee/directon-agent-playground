"""
Every request parameter of the three model APIs we call, written out explicitly.

A field set to None is not sent, so the provider's default applies; that marks "not used by us".
Values we do use are filled in. Docs were checked on 2026-11-20; re-check the source URLs
before relying on a field that is still None.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def drop_none(obj: Any) -> Any:
    """Recursively remove None values and empty containers so unused fields are not sent."""
    if isinstance(obj, dict):
        out = {k: drop_none(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v is not None and v != {} and v != []}
    if isinstance(obj, list):
        return [drop_none(v) for v in obj if v is not None]
    return obj


# ---------------------------------------------------------------------------
# Soniox real-time STT (WebSocket first message)
# https://docs.soniox.com/stt/api-reference/websocket-api
# https://docs.soniox.com/stt/rt/endpoint-detection
# https://docs.soniox.com/speech-to-text/core-concepts/audio-formats
# https://docs.soniox.com/stt/models  (stt-rt-v6 current since 2026-11-03; v5 removed 2026-12-15)
# ---------------------------------------------------------------------------


@dataclass
class SonioxContext:
    """Domain context to improve recognition; whole object max 8000 tokens."""

    general: list[dict[str, str]] | None = None  # [{"key": "domain", "value": "..."}] key/value hints.
    text: str | None = None  # Free-form background text.
    terms: list[str] | None = None  # Words to favor, e.g. product names.
    translation_terms: list[dict[str, str]] | None = None  # [{"source", "target"}] for translation.


@dataclass
class SonioxTranslation:
    type: str | None = None  # "one_way" or "two_way".
    target_language: str | None = None  # Required for one_way.
    language_a: str | None = None  # Required for two_way.
    language_b: str | None = None  # Required for two_way.


@dataclass
class SonioxRealtimeParams:
    """
    Config message for wss://stt-rt.soniox.com/transcribe-websocket.

    The API key goes in the connection's Authorization header; the `api_key` field is deprecated.
    Control messages after this: binary audio frames, {"type": "finalize"} (emits <fin>),
    {"type": "keepalive"} (at least every 20 s when idle), {"type": "agent_text", ...} (v6 only),
    and an empty text frame "" to end the audio.
    """

    model: str = "stt-rt-v6"
    audio_format: str = "pcm_s16le"  # "auto" for containers, or raw: pcm_s16le, pcm_f32le, mulaw, alaw, ...
    sample_rate: int | None = 16000  # Required for raw formats.
    num_channels: int | None = 1  # Required for raw formats.
    language_hints: list[str] | None = field(default_factory=lambda: ["ko"])  # Unique language codes.
    language_hints_strict: bool | None = None  # Restrict output to the hinted languages; not on all models.
    context: SonioxContext | None = None
    enable_speaker_diarization: bool | None = None  # Adds token.speaker; endpointing lowers its accuracy.
    enable_language_identification: bool | None = None  # Adds token.language.
    enable_endpoint_detection: bool | None = None  # Emits an <end> token when the speaker finishes.
    max_endpoint_delay_ms: int | None = None  # 500-3000, default 2000.
    endpoint_sensitivity: float | None = None  # -1.0 to 1.0, default 0.0; v5/v6 only.
    endpoint_latency_adjustment_level: int | None = None  # 0-3, default 0; v5/v6 only.
    enable_agent_text_input: bool | None = None  # Accept agent_text messages; v6 only.
    client_reference_id: str | None = None  # Up to 256 chars, for your own tracking.
    translation: SonioxTranslation | None = None

    def to_message(self) -> dict:
        return drop_none(asdict(self))


# ---------------------------------------------------------------------------
# OpenAI Chat Completions, model gpt-5.4-mini
# https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
# https://developers.openai.com/api/docs/models/gpt-5.4-mini
# gpt-5.4-mini: 400k context (272k input, 128k output), knowledge cutoff 2025-08-31.
# Price per 1M tokens: standard in $0.75 / cached $0.075 / out $4.50; flex and batch half; priority ("fast") double.
# ---------------------------------------------------------------------------


@dataclass
class OpenAIChatParams:
    """
    Everything except `messages` and `tools`, which the strategy builds per call.

    temperature, top_p and logprobs are accepted only with reasoning_effort="none" (GPT-5.4 guide).
    Deprecated and left out on purpose: max_tokens, user, functions, function_call.
    """

    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = "none"  # none (model default), low, medium, high, xhigh.
    verbosity: str | None = None  # low, medium (default), high.
    tool_choice: Any = None  # "none", "auto" (default with tools), "required", or a specific tool.
    parallel_tool_calls: bool | None = None  # Default true.
    response_format: dict | None = None  # {"type": "text" | "json_object" | "json_schema", ...}.
    max_completion_tokens: int | None = None  # Output cap, reasoning tokens included.
    temperature: float | None = None  # 0-2, default 1; only with reasoning_effort="none".
    top_p: float | None = None  # 0-1, default 1; only with reasoning_effort="none".
    n: int | None = None  # Newer models reject an explicit n.
    stop: list[str] | None = None  # Up to 4 sequences.
    seed: int | None = None  # Best-effort determinism (beta).
    presence_penalty: float | None = None  # -2 to 2, default 0.
    frequency_penalty: float | None = None  # -2 to 2, default 0.
    logit_bias: dict[str, int] | None = None  # Token id -> -100..100.
    logprobs: bool | None = None  # Only with reasoning_effort="none".
    top_logprobs: int | None = None  # 0-20, needs logprobs.
    stream: bool = True  # We always stream to measure time to first token.
    stream_options: dict | None = field(default_factory=lambda: {"include_usage": True})  # Also include_obfuscation.
    service_tier: str | None = None  # auto (default), default, flex, scale, priority/fast.
    prompt_cache_key: str | None = None  # Groups requests for prompt caching.
    prompt_cache_retention: str | None = None  # in_memory or 24h; deprecated for prompt_cache_options.
    prompt_cache_options: dict | None = None  # {"mode": "implicit" | "explicit", "ttl": "30m"}; gpt-5.6+.
    safety_identifier: str | None = None  # Stable hashed end-user id, max 128 chars.
    store: bool | None = None  # Default false.
    metadata: dict[str, str] | None = None  # Up to 16 pairs.
    modalities: list[str] | None = None  # ["text"] default; audio needs an audio model.
    audio: dict | None = None  # {"format", "voice"} when modalities has audio.
    prediction: dict | None = None  # {"type": "content", "content": ...} predicted output.
    web_search_options: dict | None = None  # {"search_context_size", "user_location"}.
    moderation: dict | None = None  # {"model", "policy": {...}}.

    def to_kwargs(self, **overrides: Any) -> dict:
        d = asdict(self)
        d.update({k: v for k, v in overrides.items() if v is not None})
        if not d["stream"]:
            d["stream_options"] = None
        return drop_none(d)


# ---------------------------------------------------------------------------
# Gemini TTS via the Interactions API (client.interactions.create, POST /v1beta/interactions)
# https://aistudio.google.com/docs/speech-generation
# https://aistudio.google.com/api-reference/interactions-api
# https://aistudio.google.com/docs/models/gemini-2.5-flash-preview-tts  (8,192 input / 16,384 output tokens)
# https://aistudio.google.com/docs/deprecations  (2.5 Flash TTS earliest shutdown 2026-12-18 -> gemini-3.8-flash-tts)
# Field names also checked against google-genai 3.6.1 (_gaos/types/interactions).
# ---------------------------------------------------------------------------


@dataclass
class GeminiAudioFormat:
    type: str = "audio"
    mime_type: str | None = (
        None  # audio/wav (unary default, 24 kHz), audio/l16 (stream default), audio/mulaw, audio/alaw (8 kHz), audio/mp3, audio/ogg_opus.
    )
    sample_rate: int | None = None  # Default 24000; e.g. 16000, 8000.
    delivery: str | None = None  # inline or uri.
    bit_rate: int | None = None  # bps, compressed formats only.


@dataclass
class GeminiGenerationConfig:
    # Single voice: [{"voice": "Kore"}]; two speakers: {"speakers": [{"speaker", "voice"}, ...]}; each entry also takes "language".
    speech_config: Any = None
    temperature: float | None = None  # In the SDK schema; TTS behavior not documented.
    top_p: float | None = None  # In the SDK schema; TTS behavior not documented.
    seed: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    # Not for TTS models: tool_choice, thinking_level, thinking_summaries, image/video/transcription configs.


@dataclass
class GeminiTTSParams:
    """
    Request for one TTS call; `input` is built per call from the text and style.

    gemini-2.5-flash-preview-tts rejects speech_metadata annotations (style, speaker), so for it the
    style is put in front of the text. It also does not support tools, caching, thinking, Batch, Flex or Priority.
    Inline tags like <short pause> or <sigh> are documented for 3.8; untested on 2.5.
    """

    model: str = "gemini-2.5-flash-preview-tts"
    voice: str = "Kore"  # 30 prebuilt voices: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, ...
    language: str | None = None  # Per-speaker language code; auto-detected when unset.
    supports_annotations: bool = False  # True for gemini-3.8-flash-tts / -lite-tts.
    response_format: GeminiAudioFormat = field(default_factory=GeminiAudioFormat)
    generation_config: GeminiGenerationConfig = field(default_factory=GeminiGenerationConfig)
    stream: bool | None = None  # Streams step.delta audio chunks.
    service_tier: str | None = None  # flex, standard, priority; 2.5 Flash TTS supports neither flex nor priority.
    system_instruction: str | None = None  # In the schema; TTS use not documented.
    store: bool | None = None  # Keep the interaction server-side.
    safety_settings: list[dict] | None = None
    labels: dict[str, str] | None = None  # Billing labels.
    # Agent/conversation fields, unused for TTS: previous_interaction_id, background, tools, webhook_config, environment.

    def build_input(self, text: str, style: str = "") -> list[dict]:
        item: dict[str, Any] = {"type": "text", "text": text}
        if style and self.supports_annotations:
            item["annotations"] = [{"type": "speech_metadata", "style": style}]
        elif style:
            item["text"] = f"{style}: {text}"
        return [{"type": "user_input", "content": [item]}]

    def to_kwargs(self, text: str, style: str = "", voice: str | None = None) -> dict:
        gen = asdict(self.generation_config)
        if gen["speech_config"] is None:
            gen["speech_config"] = [drop_none({"voice": voice or self.voice, "language": self.language})]
        return drop_none(
            {
                "model": self.model,
                "input": self.build_input(text, style),
                "response_format": asdict(self.response_format),
                "generation_config": gen,
                "stream": self.stream,
                "service_tier": self.service_tier,
                "system_instruction": self.system_instruction,
                "store": self.store,
                "safety_settings": self.safety_settings,
                "labels": self.labels,
            }
        )
