"""Pour-probe geometry config: radius bump + env-configurable offset parsing."""

import importlib
import logging
import sys


def _reload_config(monkeypatch, **env):
    """Reload config.py hermetically, without touching the real .env file."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    env.setdefault("HICON_API_URL", "http://example.invalid")
    env.setdefault("HICON_CUSTOMER_ID", "test-customer")
    env.setdefault("HICON_ENABLE_SYNC", "false")
    env.setdefault("HICON_MOULD_GIE_ENABLED", "true")
    env.setdefault("HICON_MOULD_COUNT_MODE", "tracker")

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))

    sys.modules.pop("config", None)
    return importlib.import_module("config")


def test_default_probe_offsets_unchanged_radius_increased(monkeypatch):
    """Offset list stays at the original 5-point ±24px spread — the probe-miss
    fix is in _measure_multi_probe_brightness's aggregation (mean -> max), not
    in widening this list (see that function's docstring for why widening
    alone would have diluted a legitimate centered hit)."""
    config = _reload_config(monkeypatch)

    assert config.POUR_PROBE_OFFSETS == [(0, 0), (12, 0), (-12, 0), (24, 0), (-24, 0)]
    assert config.POUR_PROBE_RADIUS_PX == 12


def test_probe_offsets_env_override_parses_pairs(monkeypatch):
    config = _reload_config(monkeypatch, HICON_POUR_PROBE_OFFSETS="0:0,10:5,-10:-5")

    assert config.POUR_PROBE_OFFSETS == [(0, 0), (10, 5), (-10, -5)]


def test_probe_offsets_env_malformed_falls_back_to_default(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    config = _reload_config(monkeypatch, HICON_POUR_PROBE_OFFSETS="not-a-valid-offset-list")

    assert config.POUR_PROBE_OFFSETS == [(0, 0), (12, 0), (-12, 0), (24, 0), (-24, 0)]
    assert "Could not parse HICON_POUR_PROBE_OFFSETS" in caplog.text
