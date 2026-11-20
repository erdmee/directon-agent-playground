"""STT routing between the BAZE gateway and direct Soniox (offline)."""

from playground.config import BAZE_SONIOX_URL, SONIOX_DIRECT_URL, soniox_settings


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
