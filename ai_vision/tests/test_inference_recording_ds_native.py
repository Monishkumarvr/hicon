import types
from datetime import datetime
from pathlib import Path

import pytest

from pipeline.recording import RecordingManager
import processors.pouring_processor as pouring_mod
from processors.pouring_processor import PouringProcessor


class DummyDB:
    def insert_pouring_event(self, **kwargs):
        return 1

    def update_pouring_end(self, **kwargs):
        return None

    def insert_heat_cycle(self, **kwargs):
        return 1


class DummyConfig:
    CUSTOMER_ID = "C1"
    LOCATION = "Loc"
    CAMERA_ID_STREAM_0 = "Cam-0"
    MOUTH_CONFIDENCE = 0.4
    TROLLEY_CONFIDENCE = 0.25
    SESSION_START_DURATION = 1.0
    SESSION_END_DURATION = 1.5
    POUR_PROBE_BELOW_PX = 50
    POUR_PROBE_OFFSETS = [(20, 0), (30, 0), (40, 0)]
    POUR_PROBE_RADIUS_PX = 6
    POUR_BRIGHTNESS_START = 230
    POUR_BRIGHTNESS_END = 180
    POUR_START_DURATION = 0.25
    POUR_END_DURATION = 1.0
    POUR_MIN_DURATION = 2.0
    MOULD_DISPLACEMENT_THRESHOLD = 0.15
    MOULD_SUSTAINED_DURATION = 1.5
    CLUSTER_R_CLUSTER = 0.08
    CLUSTER_R_MERGE = 0.05
    MOULD_SWITCH_MIN_POUR_S = 2.0
    MIN_CLUSTER_POUR_S = 1.5
    EDGE_EXPAND_PX = 200
    MOUTH_MISSING_TOL_S = 0.8
    POURING_CYCLE_TIMEOUT_S = 300.0
    ENABLE_INFERENCE_VIDEO = True
    VIDEO_DIR = Path("/tmp")


def _make_proc(tmp_path):
    return PouringProcessor(
        db_manager=DummyDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=None,
    )


def test_osd_overlay_metadata_contains_text_and_probe_rects(tmp_path, monkeypatch):
    proc = _make_proc(tmp_path)
    proc._last_probe_base = (100, 120)
    proc._last_probe_brightness = 260.0
    proc.trolley_locked = True
    proc.locked_trolley_bbox = (80, 90, 200, 240)
    proc.locked_trolley_id = 5
    proc.cycle_start_time = 1.0
    proc.mouth_last_seen_in_trolley = 9.0

    class _Color:
        def __init__(self):
            self.value = None

        def set(self, r, g, b, a):
            self.value = (r, g, b, a)

    class _FontParams:
        def __init__(self):
            self.font_name = ""
            self.font_size = 0
            self.font_color = _Color()

    class _TextParams:
        def __init__(self):
            self.display_text = ""
            self.x_offset = 0
            self.y_offset = 0
            self.font_params = _FontParams()
            self.set_bg_clr = 0
            self.text_bg_clr = _Color()

    class _RectParams:
        def __init__(self):
            self.left = 0
            self.top = 0
            self.width = 0
            self.height = 0
            self.border_width = 0
            self.has_bg_color = 0
            self.border_color = _Color()
            self.bg_color = _Color()

    class _DisplayMeta:
        def __init__(self):
            self.num_labels = 0
            self.num_rects = 0
            self.text_params = [_TextParams() for _ in range(16)]
            self.rect_params = [_RectParams() for _ in range(16)]

    fake_display = _DisplayMeta()

    def _acquire(_batch_meta):
        return fake_display

    def _attach(frame_meta, display_meta):
        frame_meta._display_meta = display_meta

    monkeypatch.setattr(
        pouring_mod,
        "pyds",
        types.SimpleNamespace(
            nvds_acquire_display_meta_from_pool=_acquire,
            nvds_add_display_meta_to_frame=_attach,
        ),
    )

    frame_meta = types.SimpleNamespace()
    proc._add_inference_display_meta(
        batch_meta=object(),
        frame_meta=frame_meta,
        mouths=[],
        trolleys=[],
        target_trolley=None,
        timestamp=10.0,
        datetime_obj=datetime.now(),
    )

    assert hasattr(frame_meta, "_display_meta")
    assert frame_meta._display_meta.num_labels >= 1
    # 3 probe points + 1 expanded locked trolley zone
    assert frame_meta._display_meta.num_rects >= 4
    assert "POURING INFERENCE" in frame_meta._display_meta.text_params[0].display_text


def test_recording_manager_creates_inference_mp4(tmp_path):
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GStreamer unavailable: {exc}")

    Gst.init(None)

    required = ["videotestsrc", "capsfilter", "tee", "queue", "fakesink", "x264enc", "mp4mux", "nvvideoconvert"]
    for name in required:
        if Gst.ElementFactory.find(name) is None:
            pytest.skip(f"{name} plugin unavailable")

    pipeline = Gst.Pipeline.new("recording-test")
    src = Gst.ElementFactory.make("videotestsrc", "src")
    src.set_property("num-buffers", 45)
    caps = Gst.ElementFactory.make("capsfilter", "caps")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=NV12,width=320,height=240,framerate=10/1"))
    tee = Gst.ElementFactory.make("tee", "t")
    q_display = Gst.ElementFactory.make("queue", "q-display")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    sink.set_property("sync", False)

    for el in (src, caps, tee, q_display, sink):
        assert el is not None
        pipeline.add(el)

    assert src.link(caps)
    assert caps.link(tee)
    tee_pad = tee.request_pad_simple("src_%u")
    q_pad = q_display.get_static_pad("sink")
    assert tee_pad.link(q_pad) == Gst.PadLinkReturn.OK
    assert q_display.link(sink)

    manager = RecordingManager(output_dir=str(tmp_path), stream_id=0, target_fps=10)
    assert manager.setup_recording_branch(pipeline, tee)
    manager.start_recording(event_prefix="inference_test")

    assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        10 * Gst.SECOND,
        Gst.MessageType.EOS | Gst.MessageType.ERROR,
    )
    assert msg is not None
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        pytest.fail(f"GStreamer error: {err}; {dbg}")

    recorded_path = manager.stop_recording()
    pipeline.set_state(Gst.State.NULL)

    assert recorded_path is not None
    out = Path(recorded_path)
    assert out.exists()
    assert out.stat().st_size > 0
