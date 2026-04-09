import json
import logging
import time

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from pipeline.bus_handler import BusHandler


class FakeBus:
    def add_signal_watch(self):
        return None

    def connect(self, _signal, _callback):
        return None


class FakePipeline:
    def get_bus(self):
        return FakeBus()


class FakeLoop:
    def __init__(self):
        self.quit_called = False

    def quit(self):
        self.quit_called = True
        return None


def test_update_stream0_stage_time_records_latest_timestamp():
    handler = BusHandler(FakePipeline(), FakeLoop(), stream0_decoupled_analysis_mode=True)

    handler.update_stream0_stage_time("mux_src")

    assert "mux_src" in handler.stream0_stage_last_time
    assert abs(time.time() - handler.stream0_stage_last_time["mux_src"]) < 1.0


def test_update_stream0_stage_sample_tracks_pts_delta():
    handler = BusHandler(FakePipeline(), FakeLoop(), stream0_decoupled_analysis_mode=True)

    handler.update_stream0_stage_sample("decoder_src", 1_000_000_000)
    handler.update_stream0_stage_sample("decoder_src", 1_040_000_000)

    state = handler.stream0_stage_pts["decoder_src"]
    assert state["last_pts_ns"] == 1_040_000_000
    assert state["delta_ns"] == 40_000_000
    assert state["regressed"] is False


def test_fps_logger_emits_stream0_stage_ages(monkeypatch, caplog):
    scheduled = {}

    def fake_timeout_add_seconds(_interval, callback):
        scheduled["callback"] = callback
        return 1

    monkeypatch.setattr("pipeline.bus_handler.GLib.timeout_add_seconds", fake_timeout_add_seconds)

    handler = BusHandler(FakePipeline(), FakeLoop(), stream0_decoupled_analysis_mode=True)
    now = time.time()
    handler.last_frame_time[0] = now
    handler._frame_counts[0] = 25
    handler.stream0_analysis_last_time = now
    handler.stream0_stage_last_time = {
        "decoder_src": now,
        "nvvidconv_src": now,
        "caps_src": now,
        "premuxq_src": now,
        "mux_src": now,
        "postmuxq_src": now,
        "pgie_sink": now,
        "pgie_src": now,
        "tracker_sink": now,
        "tracker_src": now,
    }
    handler.stream0_stage_pts = {
        "decoder_src": {"delta_ns": 40_000_000},
        "nvvidconv_src": {"delta_ns": 40_000_000},
        "caps_src": {"delta_ns": 40_000_000},
        "premuxq_src": {"delta_ns": 40_000_000},
    }

    caplog.set_level(logging.INFO)
    handler.start_fps_logger()

    assert "callback" in scheduled
    assert scheduled["callback"]() is True
    assert "[S0-DIAG]" in caplog.text
    assert "[S0-STAGES] decoder_src_age=" in caplog.text
    assert "nvvidconv_src_age=" in caplog.text
    assert "caps_src_age=" in caplog.text
    assert "premuxq_src_age=" in caplog.text
    assert "mux_src_age=" in caplog.text
    assert "postmuxq_src_age=" in caplog.text
    assert "pgie_sink_age=" in caplog.text
    assert "pgie_src_age=" in caplog.text
    assert "tracker_sink_age=" in caplog.text
    assert "tracker_src_age=" in caplog.text
    assert "[S0-PTS] decoder_src_pts_delta=40.00ms" in caplog.text
    assert "nvvidconv_src_pts_delta=40.00ms" in caplog.text
    assert "caps_src_pts_delta=40.00ms" in caplog.text
    assert "premuxq_src_pts_delta=40.00ms" in caplog.text


def test_fps_logger_suppresses_stream0_watchdog_during_segment_buffer_rebuffer(monkeypatch, tmp_path, caplog):
    scheduled = {}

    def fake_timeout_add_seconds(_interval, callback):
        scheduled["callback"] = callback
        return 1

    monkeypatch.setattr("pipeline.bus_handler.GLib.timeout_add_seconds", fake_timeout_add_seconds)

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({
            "mode": "rebuffering",
            "pending_segments": 14,
            "target_segments": 30,
            "updated_at": time.time(),
            "active_epoch": 0,
        }),
        encoding="utf-8",
    )

    handler = BusHandler(
        FakePipeline(),
        FakeLoop(),
        stream0_segment_buffer_mode=True,
        stream0_segment_buffer_state_path=str(state_path),
        stream0_startup_grace_sec=70,
    )
    handler._startup_time = time.time() - 120
    handler.last_frame_time[0] = time.time() - 5
    handler._frame_counts[0] = 0

    caplog.set_level(logging.INFO)
    handler.start_fps_logger()

    assert scheduled["callback"]() is True
    assert "[FPS-WATCHDOG] Stream 0 at 0fps" not in caplog.text
    assert handler._zero_fps_counts.get(0, 0) == 0


def test_fps_logger_keeps_stream0_watchdog_active_while_segment_buffer_playing(monkeypatch, tmp_path, caplog):
    scheduled = {}

    def fake_timeout_add_seconds(_interval, callback):
        scheduled["callback"] = callback
        return 1

    monkeypatch.setattr("pipeline.bus_handler.GLib.timeout_add_seconds", fake_timeout_add_seconds)

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({
            "mode": "playing",
            "pending_segments": 30,
            "target_segments": 30,
            "updated_at": time.time(),
            "active_epoch": 0,
        }),
        encoding="utf-8",
    )

    handler = BusHandler(
        FakePipeline(),
        FakeLoop(),
        stream0_segment_buffer_mode=True,
        stream0_segment_buffer_state_path=str(state_path),
        stream_policies={0: "warn"},
    )
    handler._startup_time = time.time() - 120
    handler.last_frame_time[0] = time.time() - 5
    handler._frame_counts[0] = 0

    caplog.set_level(logging.INFO)
    handler.start_fps_logger()

    assert scheduled["callback"]() is True
    assert "[FPS-WATCHDOG] Stream 0 at 0fps for 5s" in caplog.text


class FakeSrc:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class FakeMessage:
    def __init__(self, message_type, src_name):
        self.type = message_type
        self.src = FakeSrc(src_name)

    def parse_error(self):
        raise AssertionError("parse_error should not be called for EOS")


def test_eos_from_restartable_source_schedules_stream_restart(monkeypatch):
    scheduled = []

    def fake_timeout_add_seconds(_interval, callback):
        callback()
        return 1

    def fake_restart(stream_id, reason):
        scheduled.append((stream_id, reason))
        return True

    monkeypatch.setattr("pipeline.bus_handler.GLib.timeout_add_seconds", fake_timeout_add_seconds)

    loop = FakeLoop()
    handler = BusHandler(
        FakePipeline(),
        loop,
        stream_restart_cb=fake_restart,
        restartable_stream_ids={0},
        rtsp_restart_backoff_sec=0,
    )

    handler._on_bus_message(None, FakeMessage(Gst.MessageType.EOS, "source0"))

    assert scheduled == [(0, "EOS from source0")]
    assert handler.fatal_exit is False
    assert loop.quit_called is False


def test_eos_from_non_restartable_stream_remains_fatal():
    loop = FakeLoop()
    handler = BusHandler(FakePipeline(), loop)

    handler._on_bus_message(None, FakeMessage(Gst.MessageType.EOS, "pipeline0"))

    assert handler.fatal_exit is True
    assert loop.quit_called is True
