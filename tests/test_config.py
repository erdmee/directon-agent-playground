"""Provider routing between the BAZE gateway and direct keys (offline)."""

from playground.config import BAZE_GATEWAY, BAZE_SONIOX_URL, SONIOX_DIRECT_URL, llm_settings, soniox_settings


def test_llm_routes_through_baze_unless_llm_key_set(monkeypatch):
    for k in ("LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BAZE_API_KEY", "baze-test")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    s = llm_settings("LLM")
    assert (s["via"], s["api_key"], s["base_url"]) == ("baze", "baze-test", BAZE_GATEWAY)
    monkeypatch.setenv("LLM_API_KEY", "llm-test")
    assert llm_settings("LLM")["api_key"] == "llm-test"
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.delenv("BAZE_API_KEY")
    s = llm_settings("LLM")
    assert (s["via"], s["api_key"], s["base_url"]) == ("openai", "openai-test", None)


def test_baze_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("BAZE_API_KEY", "baze-test")
    monkeypatch.setenv("SONIOX_API_KEY", "soniox-test")
    monkeypatch.setenv("SONIOX_URL", SONIOX_DIRECT_URL)
    monkeypatch.delenv("BAZE_SONIOX_URL", raising=False)
    s = soniox_settings()
    assert (s["via"], s["url"], s["key"]) == ("baze", BAZE_SONIOX_URL, "baze-test")


def test_direct_soniox_without_baze(monkeypatch):
    monkeypatch.delenv("BAZE_API_KEY", raising=False)
    monkeypatch.delenv("SONIOX_URL", raising=False)
    monkeypatch.setenv("SONIOX_API_KEY", "soniox-test")
    s = soniox_settings()
    assert (s["via"], s["url"], s["key"]) == ("soniox", SONIOX_DIRECT_URL, "soniox-test")
