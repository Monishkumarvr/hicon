import json
from collections import deque
from pathlib import Path

import pytest

from pipeline.segment_buffer_helper import (
    SegmentBufferHelper,
    SegmentRef,
    list_complete_segments,
    parse_segment_ref,
    should_rebuffer,
)


def _touch(path: Path, content: bytes = b"ts"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_parse_segment_ref_reads_epoch_and_index(tmp_path):
    path = tmp_path / "epoch_000007" / "seg_000123.ts"
    _touch(path)

    ref = parse_segment_ref(path)

    assert ref == SegmentRef(epoch=7, index=123, path=path)


def test_list_complete_segments_skips_active_epoch_tail(tmp_path):
    segments_root = tmp_path / "segments"
    _touch(segments_root / "epoch_000000" / "seg_000000.ts")
    _touch(segments_root / "epoch_000001" / "seg_000000.ts")
    _touch(segments_root / "epoch_000001" / "seg_000001.ts")

    refs = list_complete_segments(
        segments_root,
        active_epoch=1,
        finalized_epochs=set(),
    )

    assert [(ref.epoch, ref.index) for ref in refs] == [(0, 0), (1, 0)]


def test_should_rebuffer_uses_target_before_primed_and_low_watermark_after():
    assert should_rebuffer(5, False, target_segments=30, low_watermark_segments=15) is True
    assert should_rebuffer(30, False, target_segments=30, low_watermark_segments=15) is False
    assert should_rebuffer(14, True, target_segments=30, low_watermark_segments=15) is True
    assert should_rebuffer(15, True, target_segments=30, low_watermark_segments=15) is False


def test_build_ffmpeg_cmd_keeps_continuous_timestamps(tmp_path):
    helper = SegmentBufferHelper(
        stream_id=0,
        rtsp_url="rtsp://example/substream",
        codec="h264",
        buffer_dir=str(tmp_path),
        segment_seconds=2,
        delay_seconds=60,
        retention_seconds=120,
    )

    cmd = helper._build_ffmpeg_cmd(tmp_path / "segments" / "epoch_000000")

    assert "-reset_timestamps" not in cmd


def test_wait_for_feed_slot_paces_by_segment_interval(monkeypatch, tmp_path):
    helper = SegmentBufferHelper(
        stream_id=0,
        rtsp_url="rtsp://example/substream",
        codec="h264",
        buffer_dir=str(tmp_path),
        segment_seconds=2,
        delay_seconds=60,
        retention_seconds=120,
    )

    class FakeStopEvent:
        def __init__(self):
            self.now = 100.0
            self.wait_calls = []

        def is_set(self):
            return False

        def wait(self, duration):
            self.wait_calls.append(duration)
            self.now += duration
            return False

    fake_event = FakeStopEvent()
    helper._stop_event = fake_event

    import pipeline.segment_buffer_helper as helper_mod

    monkeypatch.setattr(helper_mod.time, "monotonic", lambda: fake_event.now)

    assert helper._wait_for_feed_slot(102.0) is True
    assert sum(fake_event.wait_calls) == pytest.approx(2.0)
    assert all(duration <= helper_mod.POLL_INTERVAL_SEC for duration in fake_event.wait_calls)


def test_publish_state_tracks_buffering_playing_and_rebuffering(tmp_path):
    helper = SegmentBufferHelper(
        stream_id=0,
        rtsp_url="rtsp://example/substream",
        codec="h264",
        buffer_dir=str(tmp_path),
        segment_seconds=2,
        delay_seconds=60,
        retention_seconds=120,
    )
    helper.buffer_dir.mkdir(parents=True, exist_ok=True)

    helper._publish_state("buffering", pending_segments=0, active_epoch=None)
    helper._publish_state("playing", pending_segments=30, active_epoch=0)
    helper._publish_state("rebuffering", pending_segments=14, active_epoch=1)
    helper._publish_state("playing", pending_segments=30, active_epoch=1)

    state = json.loads(helper.state_path.read_text(encoding="utf-8"))
    assert state["mode"] == "playing"
    assert state["pending_segments"] == 30
    assert state["target_segments"] == 30
    assert state["active_epoch"] == 1


def test_prune_fed_segments_removes_old_files_after_retention(tmp_path):
    helper = SegmentBufferHelper(
        stream_id=0,
        rtsp_url="rtsp://example/substream",
        codec="h264",
        buffer_dir=str(tmp_path),
        segment_seconds=2,
        delay_seconds=60,
        retention_seconds=10,
    )
    segment = tmp_path / "segments" / "epoch_000000" / "seg_000000.ts"
    _touch(segment)
    helper._fed_history = deque([
        (100.0, SegmentRef(epoch=0, index=0, path=segment)),
    ])

    import pipeline.segment_buffer_helper as helper_mod

    original_time = helper_mod.time.time
    helper_mod.time.time = lambda: 111.0
    try:
        helper._prune_fed_segments()
    finally:
        helper_mod.time.time = original_time

    assert not segment.exists()
