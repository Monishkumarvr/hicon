import importlib
import logging
import sys


def _reload_config(monkeypatch, *, load_env_file=True, **env):
    if not load_env_file:
        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
        env.setdefault("HICON_API_URL", "http://example.invalid")
        env.setdefault("HICON_CUSTOMER_ID", "test-customer")
        env.setdefault("HICON_ENABLE_SYNC", "false")

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
    )

    assert config.STREAM_0_BYPASS_TRACKER is True
    assert config.STREAM_0_BYPASS_PGIE is False
    assert config.STREAM_0_DECODE_ONLY_MODE is True
    assert config.STREAM_0_POSTMUX_ONLY_MODE is True
    assert config.STREAM_0_POSTCONV_ONLY_MODE is True
    assert config.STREAM_0_PREOSD_ONLY_MODE is True


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


def test_config_reads_stream0_local_relay_flags(monkeypatch):
    config = _reload_config(
        monkeypatch,
        HICON_ENABLE_STREAM0_LOCAL_RELAY="true",
        HICON_STREAM0_REMOTE_RELAY_URL="rtsp://example.com/live/stream0",
    )

    assert config.ENABLE_STREAM0_LOCAL_RELAY is True
    assert config.STREAM0_REMOTE_RELAY_URL == "rtsp://example.com/live/stream0"


def test_config_defaults_stream0_remote_relay_url_to_empty(monkeypatch):
    config = _reload_config(
        monkeypatch,
        HICON_ENABLE_STREAM0_LOCAL_RELAY="false",
        HICON_STREAM0_REMOTE_RELAY_URL=None,
    )

    assert config.ENABLE_STREAM0_LOCAL_RELAY is False
    assert config.STREAM0_REMOTE_RELAY_URL == ""


def test_config_inference_video_stream_flags_follow_global_by_default(monkeypatch):
    config = _reload_config(
        monkeypatch,
        load_env_file=False,
        HICON_ENABLE_INFERENCE_VIDEO="true",
        HICON_ENABLE_INFERENCE_VIDEO_STREAM_0=None,
        HICON_ENABLE_INFERENCE_VIDEO_STREAM_1=None,
        HICON_ENABLE_INFERENCE_VIDEO_STREAM_2=None,
    )

    assert config.ENABLE_INFERENCE_VIDEO is True
    assert config.ENABLE_INFERENCE_VIDEO_STREAM_0 is True
    assert config.ENABLE_INFERENCE_VIDEO_STREAM_1 is True
    assert config.ENABLE_INFERENCE_VIDEO_STREAM_2 is True


def test_config_inference_video_stream_flags_override_global(monkeypatch):
    config = _reload_config(
        monkeypatch,
        HICON_ENABLE_INFERENCE_VIDEO="true",
        HICON_ENABLE_INFERENCE_VIDEO_STREAM_0="true",
        HICON_ENABLE_INFERENCE_VIDEO_STREAM_1="false",
        HICON_ENABLE_INFERENCE_VIDEO_STREAM_2="false",
    )

    assert config.ENABLE_INFERENCE_VIDEO is True
    assert config.ENABLE_INFERENCE_VIDEO_STREAM_0 is True
    assert config.ENABLE_INFERENCE_VIDEO_STREAM_1 is False
    assert config.ENABLE_INFERENCE_VIDEO_STREAM_2 is False


def test_config_reads_live_stream_timestamp_overlay_flag(monkeypatch):
    config = _reload_config(
        monkeypatch,
        load_env_file=False,
        HICON_LIVE_STREAM_TIMESTAMP_OVERLAY="true",
    )

    assert config.LIVE_STREAM_TIMESTAMP_OVERLAY is True
