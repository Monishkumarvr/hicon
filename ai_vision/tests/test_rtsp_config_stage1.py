import importlib
import logging
import sys


def _reload_config(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))

    sys.modules.pop("config", None)
    return importlib.import_module("config")


def test_config_uses_new_rtsp_envs_and_ignores_legacy_timeout(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    config = _reload_config(
        monkeypatch,
        HICON_RTSP_TIMEOUT_SEC="20",
        HICON_RTSP_PROTOCOL_0="udp",
        HICON_RTSP_PROTOCOL_1="tcp",
        HICON_RTSP_PROTOCOL_2="auto",
        HICON_RTSP_TCP_TIMEOUT_US="65000000",
        HICON_RTSP_UDP_TIMEOUT_US="1234567",
        HICON_RTSP_PORT_RETRY="7",
        HICON_RTSP_DO_RETRANSMISSION="false",
    )

    assert config.RTSP_PROTOCOL_0 == "udp"
    assert config.RTSP_PROTOCOL_1 == "tcp"
    assert config.RTSP_PROTOCOL_2 == "auto"
    assert config.RTSP_TCP_TIMEOUT_US == 65000000
    assert config.RTSP_UDP_TIMEOUT_US == 1234567
    assert config.RTSP_PORT_RETRY == 7
    assert config.RTSP_DO_RETRANSMISSION is False
    assert "HICON_RTSP_TIMEOUT_SEC is obsolete and ignored" in caplog.text


def test_config_reads_stream0_diagnostic_bypass_flags(monkeypatch):
    config = _reload_config(
        monkeypatch,
        HICON_BYPASS_STREAM_0_TRACKER="true",
        HICON_BYPASS_STREAM_0_PGIE="false",
        HICON_STREAM_0_DECODE_ONLY_MODE="true",
        HICON_STREAM_0_POSTMUX_ONLY_MODE="true",
        HICON_STREAM_0_POSTCONV_ONLY_MODE="true",
        HICON_STREAM_0_PREOSD_ONLY_MODE="true",
        HICON_STREAM_0_DECOUPLED_ANALYSIS_MODE="true",
    )

    assert config.STREAM_0_BYPASS_TRACKER is True
    assert config.STREAM_0_BYPASS_PGIE is False
    assert config.STREAM_0_DECODE_ONLY_MODE is True
    assert config.STREAM_0_POSTMUX_ONLY_MODE is True
    assert config.STREAM_0_POSTCONV_ONLY_MODE is True
    assert config.STREAM_0_PREOSD_ONLY_MODE is True
    assert config.STREAM_0_DECOUPLED_ANALYSIS_MODE is True


def test_config_reads_stream0_processor_diagnostic_flags(monkeypatch):
    config = _reload_config(
        monkeypatch,
        HICON_ENABLE_STREAM_0_POURING_PROCESSOR="false",
        HICON_ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR="true",
    )

    assert config.ENABLE_STREAM_0_POURING_PROCESSOR is False
    assert config.ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR is True


def test_config_reads_stream0_segment_buffer_flags(monkeypatch):
    config = _reload_config(
        monkeypatch,
        HICON_USE_SEGMENT_BUFFER_0="true",
        HICON_SEGMENT_BUFFER_DIR_0="/dev/shm/test-stream0-buffer",
        HICON_SEGMENT_BUFFER_SEGMENT_SEC_0="3",
        HICON_SEGMENT_BUFFER_DELAY_SEC_0="45",
        HICON_SEGMENT_BUFFER_RETENTION_SEC_0="90",
    )

    assert config.USE_SEGMENT_BUFFER_0 is True
    assert config.SEGMENT_BUFFER_DIR_0 == "/dev/shm/test-stream0-buffer"
    assert config.SEGMENT_BUFFER_SEGMENT_SEC_0 == 3
    assert config.SEGMENT_BUFFER_DELAY_SEC_0 == 45
    assert config.SEGMENT_BUFFER_RETENTION_SEC_0 == 90
