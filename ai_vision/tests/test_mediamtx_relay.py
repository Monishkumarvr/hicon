from tools.mediamtx_env import build_mediamtx_env, emit_shell_exports
from tools.stream0_overlay_remote_relay import build_ffmpeg_command


def test_build_mediamtx_env_ignores_missing_stream0_source():
    overrides = build_mediamtx_env(
        {
            "HICON_CPPLUS_SOURCE_STREAM_0": "",
            "HICON_USE_SEGMENT_BUFFER_0": "true",
            "HICON_STREAM0_REMOTE_RELAY_URL": "",
        }
    )

    assert "MTX_PATHS_STREAM0_SOURCE" not in overrides
    assert "MTX_PATHS_STREAM0_SOURCEONDEMAND" not in overrides


def test_build_mediamtx_env_sets_stream0_source_and_ondemand():
    overrides = build_mediamtx_env(
        {
            "HICON_CPPLUS_SOURCE_STREAM_0": "rtsp://camera/stream0",
            "HICON_USE_SEGMENT_BUFFER_0": "true",
            "HICON_STREAM0_REMOTE_RELAY_URL": "",
        }
    )

    assert overrides["MTX_PATHS_STREAM0_SOURCE"] == "rtsp://camera/stream0"
    assert overrides["MTX_PATHS_STREAM0_SOURCEONDEMAND"] == "yes"


def test_build_mediamtx_env_enables_overlay_hook_only_when_remote_url_is_set():
    without_remote = build_mediamtx_env(
        {
            "HICON_CPPLUS_SOURCE_STREAM_0": "",
            "HICON_USE_SEGMENT_BUFFER_0": "false",
            "HICON_STREAM0_REMOTE_RELAY_URL": "",
        }
    )
    with_remote = build_mediamtx_env(
        {
            "HICON_CPPLUS_SOURCE_STREAM_0": "",
            "HICON_USE_SEGMENT_BUFFER_0": "false",
            "HICON_STREAM0_REMOTE_RELAY_URL": "rtsp://remote.example/live/stream0",
        }
    )

    assert "MTX_PATHS_STREAM0_OVERLAY_RUNONREADY" not in without_remote
    assert with_remote["MTX_PATHS_STREAM0_OVERLAY_RUNONREADY"].endswith(
        "/ai_vision/tools/stream0_overlay_remote_relay.sh"
    )
    assert with_remote["MTX_PATHS_STREAM0_OVERLAY_RUNONREADYRESTART"] == "yes"
    assert "MTX_PATHS_STREAM0_OVERLAY_RUNONREADY" in emit_shell_exports(
        {
            "HICON_STREAM0_REMOTE_RELAY_URL": "rtsp://remote.example/live/stream0",
        }
    )


def test_build_stream0_overlay_remote_relay_command_uses_local_overlay_path_and_remote_url():
    cmd = build_ffmpeg_command(
        "8554",
        "stream0_overlay",
        "rtsp://remote.example/live/stream0",
    )

    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "rtsp://127.0.0.1:8554/stream0_overlay" in cmd
    assert "-c" in cmd and "copy" in cmd
    assert cmd[-1] == "rtsp://remote.example/live/stream0"
