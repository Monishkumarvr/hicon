import logging

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
from gi.repository import GstRtsp

import pipeline.gst_builder as gst_builder_mod
import pipeline.recording as recording_mod
from pipeline.gst_builder import DeepStreamPipelineBuilder
from pipeline.recording import RecordingManager


class DummySource:
    def __init__(self):
        self.props = {}

    def set_property(self, name, value):
        self.props[name] = value


class DummyFileSink:
    def __init__(self):
        self.props = {}

    def set_property(self, name, value):
        self.props[name] = value


class DummyValve:
    def __init__(self):
        self.props = {}

    def set_property(self, name, value):
        self.props[name] = value


class FakeStaticPad:
    def __init__(self, name):
        self.name = name
        self.linked_to = None

    def link(self, other):
        self.linked_to = other
        return 0

    def is_linked(self):
        return self.linked_to is not None


class FakeElement:
    def __init__(self, factory_name, name):
        self.factory_name = factory_name
        self.name = name
        self.props = {}
        self.links = []
        self._pads = {
            "sink": FakeStaticPad(f"{name}:sink"),
            "src": FakeStaticPad(f"{name}:src"),
        }
        self.connected_signals = []

    def set_property(self, name, value):
        self.props[name] = value

    def link(self, other):
        self.links.append(other.name)
        return True

    def get_static_pad(self, name):
        return self._pads[name]

    def request_pad_simple(self, name):
        return FakeStaticPad(f"{self.name}:{name}")

    def connect(self, signal, callback, stream_id):
        self.connected_signals.append((signal, callback, stream_id))

    def sync_state_with_parent(self):
        return True


class FakeStructure:
    def __init__(self, encoding):
        self.encoding = encoding

    def get_name(self):
        return "application/x-rtp"

    def get_string(self, name):
        if name == "encoding-name":
            return self.encoding
        return None


class FakeCaps:
    def __init__(self, encoding):
        self.encoding = encoding

    def get_size(self):
        return 1

    def get_structure(self, index):
        return FakeStructure(self.encoding)


class FakeSrcPad:
    def __init__(self, encoding):
        self.encoding = encoding
        self.linked_to = None

    def get_current_caps(self):
        return FakeCaps(self.encoding)

    def query_caps(self, _filter):
        return FakeCaps(self.encoding)

    def link(self, sink_pad):
        self.linked_to = sink_pad
        sink_pad.linked_to = self
        return 0


class FakeTsStructure:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class FakeTsCaps:
    def __init__(self, name):
        self.name = name

    def get_size(self):
        return 1

    def get_structure(self, index):
        return FakeTsStructure(self.name)


class FakeTsPad:
    def __init__(self, caps_name):
        self.caps_name = caps_name
        self.linked_to = None

    def get_current_caps(self):
        return FakeTsCaps(self.caps_name)

    def query_caps(self, _filter):
        return FakeTsCaps(self.caps_name)

    def link(self, sink_pad):
        self.linked_to = sink_pad
        sink_pad.linked_to = self
        return 0


class FakePipeline:
    def __init__(self):
        self.added = []

    def add(self, element):
        self.added.append(element.name)


def _make_builder(**overrides):
    config = {
        "rtsp_protocol_0": "tcp",
        "rtsp_protocol_1": "auto",
        "rtsp_protocol_2": "auto",
        "rtsp_tcp_timeout_us": 60000000,
        "rtsp_udp_timeout_us": 0,
        "rtsp_port_retry": 20,
        "rtsp_do_retransmission": True,
    }
    config.update(overrides)
    return DeepStreamPipelineBuilder(config)


def test_rtsp_builder_sets_tcp_only_properties():
    builder = _make_builder(
        rtsp_protocol_0="tcp",
        rtsp_tcp_timeout_us=60000000,
        rtsp_udp_timeout_us=5000000,
    )
    source = DummySource()

    builder._configure_rtsp_source(source, "rtsp://example/stream0", 0)

    assert source.props["protocols"] == GstRtsp.RTSPLowerTrans.TCP
    assert source.props["tcp-timeout"] == 60000000
    assert "timeout" not in source.props
    assert source.props["retry"] == 20


def test_rtsp_builder_sets_udp_only_properties():
    builder = _make_builder(
        rtsp_protocol_0="udp",
        rtsp_tcp_timeout_us=60000000,
        rtsp_udp_timeout_us=5000000,
    )
    source = DummySource()

    builder._configure_rtsp_source(source, "rtsp://example/stream0", 0)

    assert source.props["protocols"] == GstRtsp.RTSPLowerTrans.UDP
    assert source.props["timeout"] == 5000000
    assert "tcp-timeout" not in source.props
    assert source.props["retry"] == 20


def test_rtsp_builder_leaves_protocol_unset_in_auto_mode():
    builder = _make_builder(
        rtsp_protocol_0="auto",
        rtsp_tcp_timeout_us=60000000,
        rtsp_udp_timeout_us=5000000,
    )
    source = DummySource()

    builder._configure_rtsp_source(source, "rtsp://example/stream0", 0)

    assert "protocols" not in source.props
    assert source.props["timeout"] == 5000000
    assert source.props["tcp-timeout"] == 60000000


def test_recording_logs_include_stream_id(tmp_path, caplog):
    manager = RecordingManager(output_dir=str(tmp_path), stream_id=7, target_fps=10)
    manager.filesink = DummyFileSink()

    caplog.set_level(logging.INFO)
    manager.start_recording(event_prefix="stage1_test")
    manager._on_branch_buffer(None, None)
    manager._branch_buffer_count = 299
    manager._on_branch_buffer(None, None)
    manager.stop_recording()

    messages = "\n".join(record.message for record in caplog.records)
    assert "Stream 7 recording started:" in messages
    assert "Stream 7 recording: first buffer received" in messages
    assert "Stream 7 recording: received 300 buffers" in messages
    assert "Stream 7 recording stopped:" in messages


def test_recording_start_keeps_branch_dormant_outside_schedule(tmp_path, monkeypatch):
    manager = RecordingManager(output_dir=str(tmp_path), stream_id=7, target_fps=10)
    manager.filesink = DummyFileSink()
    manager.record_valve = DummyValve()

    monkeypatch.setattr(recording_mod, "is_in_schedule", lambda _windows: False)

    manager.start_recording(event_prefix="stage1_test")

    assert manager.is_recording is False
    assert manager.current_file is None
    assert manager.filesink.props["location"] == "/dev/null"
    assert manager.record_valve.props["drop"] is True


def test_recording_start_and_stop_toggle_valve_inside_schedule(tmp_path, monkeypatch):
    manager = RecordingManager(output_dir=str(tmp_path), stream_id=7, target_fps=10)
    manager.filesink = DummyFileSink()
    manager.record_valve = DummyValve()

    monkeypatch.setattr(recording_mod, "is_in_schedule", lambda _windows: True)

    manager.start_recording(event_prefix="stage1_test")

    assert manager.is_recording is True
    assert manager.current_file is not None
    assert manager.filesink.props["location"] != "/dev/null"
    assert manager.record_valve.props["drop"] is False

    recorded_path = manager.stop_recording()

    assert recorded_path is not None
    assert manager.is_recording is False
    assert manager.current_file is None
    assert manager.filesink.props["location"] == "/dev/null"
    assert manager.record_valve.props["drop"] is True


def test_stream0_decode_chain_creates_isolation_queues(monkeypatch, caplog):
    created = {}

    def fake_make(factory_name, name):
        element = FakeElement(factory_name, name)
        created[name] = element
        return element

    builder = _make_builder(rtsp_codec_0="h265")
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_configure_rtsp_source", lambda source, location, stream_id: None)
    caplog.set_level(logging.INFO)

    builder._create_decode_chain(0, "rtsp://example/stream0")

    assert "srcq0" in builder.elements
    assert "premuxq0" in builder.elements
    assert builder.elements["srcq0"].props["leaky"] == 2
    assert builder.elements["srcq0"].props["max-size-buffers"] == 16
    assert builder.elements["srcq0"].props["max-size-bytes"] == 0
    assert builder.elements["srcq0"].props["max-size-time"] == 0
    assert builder.elements["premuxq0"].props["leaky"] == 2
    assert "Stream 0 (CP Plus): source isolation queues enabled" in caplog.text


def test_create_all_elements_adds_stream0_post_mux_isolation_queues(monkeypatch, caplog):
    created = {}

    def fake_make(factory_name, name):
        element = FakeElement(factory_name, name)
        created[name] = element
        return element

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        use_cpp_pouring_plugin=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert builder.elements["hicon_pouring_0"].props["enable-osd"] is True
    assert "Stream 0: C++ pouring OSD enabled on main path before nvosd_0" in caplog.text
    assert "postmuxq0" in builder.elements
    assert "preosdq0" in builder.elements
    assert builder.elements["postmuxq0"].props["leaky"] == 2
    assert builder.elements["postmuxq0"].props["max-size-buffers"] == 16
    assert builder.elements["postmuxq0"].props["max-size-bytes"] == 0
    assert builder.elements["postmuxq0"].props["max-size-time"] == 0
    assert builder.elements["preosdq0"].props["leaky"] == 2
    assert builder.elements["nvosd_0"].props["process-mode"] == 0
    assert "Stream 0 (CP Plus): post-mux isolation queues enabled" in caplog.text


def test_tune_stream0_mux_for_cp_plus_sets_async_process_and_pool_size():
    mux = FakeElement("nvstreammux", "mux_0")

    DeepStreamPipelineBuilder._tune_stream0_mux_for_cp_plus(mux)

    assert mux.props["async-process"] is False
    assert mux.props["buffer-pool-size"] == 32


def test_tune_stream0_postmux_convert_for_cp_plus_sets_gpu_and_buffers():
    convert = FakeElement("nvvideoconvert", "nvvidconv_osd_0")

    DeepStreamPipelineBuilder._tune_stream0_postmux_convert_for_cp_plus(convert)

    assert convert.props["compute-hw"] == 1
    assert convert.props["copy-hw"] == 1
    assert convert.props["output-buffers"] == 32
    assert convert.props["disable-passthrough"] is True


def test_create_all_elements_skips_stream0_tracker_when_bypassed(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_bypass_tracker=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "tracker_0" not in builder.elements
    assert "pgie_pouring" in builder.elements
    assert "bypassing tracker_0 for diagnostic run" in caplog.text


def test_create_all_elements_skips_stream0_pgie_and_tracker_when_pgie_bypassed(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_bypass_pgie=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "pgie_pouring" not in builder.elements
    assert "tracker_0" not in builder.elements
    assert "bypassing pgie_pouring and tracker_0 for diagnostic run" in caplog.text


def test_create_all_elements_sets_stream0_decode_only_mode(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_decode_only_mode=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "decode_sink_0" in builder.elements
    assert "mux_0" not in builder.elements
    assert "nvosd_0" not in builder.elements
    assert "tee_0" not in builder.elements
    assert "decode-only diagnostic mode enabled" in caplog.text


def test_create_all_elements_sets_stream0_postmux_only_mode(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_postmux_only_mode=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "mux_0" in builder.elements
    assert "postmuxq0" in builder.elements
    assert "postmux_sink_0" in builder.elements
    assert "nvosd_0" not in builder.elements
    assert "tee_0" not in builder.elements
    assert "post-mux-only diagnostic mode enabled" in caplog.text


def test_create_all_elements_sets_stream0_postconv_only_mode(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_postconv_only_mode=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "mux_0" in builder.elements
    assert "postmuxq0" in builder.elements
    assert "nvvidconv_osd_0" in builder.elements
    assert "postconv_sink_0" in builder.elements
    assert "caps_osd_0" not in builder.elements
    assert "nvosd_0" not in builder.elements
    assert "post-convert-only diagnostic mode enabled" in caplog.text


def test_create_all_elements_sets_stream0_preosd_only_mode(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_preosd_only_mode=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "mux_0" in builder.elements
    assert "postmuxq0" in builder.elements
    assert "nvvidconv_osd_0" in builder.elements
    assert "caps_osd_0" in builder.elements
    assert "preosdq0" in builder.elements
    assert "preosd_sink_0" in builder.elements
    assert "nvosd_0" not in builder.elements
    assert "pre-OSD-only diagnostic mode enabled" in caplog.text


def test_create_all_elements_sets_stream0_decoupled_analysis_mode(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_decoupled_analysis_mode=True,
        use_cpp_pouring_plugin=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert builder.elements["hicon_pouring_0"].props["enable-osd"] is False
    assert "Stream 0: C++ pouring OSD disabled on analysis branch (no downstream nvdsosd)" in caplog.text
    assert "tee_stream0_analysis" in builder.elements
    assert "displayq0" in builder.elements
    assert "analysisq0" in builder.elements
    assert "analysis_conv0" not in builder.elements
    assert "analysis_caps0" not in builder.elements
    assert "analysis_sink0" in builder.elements
    assert "nvvidconv_osd_0" not in builder.elements
    assert "caps_osd_0" not in builder.elements
    assert "preosdq0" not in builder.elements
    assert builder.elements["displayq0"].props["leaky"] == 0
    assert builder.elements["displayq0"].props["max-size-buffers"] == 16
    assert builder.elements["analysisq0"].props["leaky"] == 2
    assert builder.elements["analysisq0"].props["max-size-buffers"] == 2
    assert builder.elements["nvosd_0"].props["process-mode"] == 0
    assert "Stream 0: decoupled analysis mode — NV12 tee" in caplog.text


def test_create_all_elements_omits_stream0_analysis_branch_when_disabled(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_decoupled_analysis_mode=True,
        stream_0_analysis_branch_enabled=False,
        use_cpp_pouring_plugin=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "tee_stream0_analysis" in builder.elements
    assert "displayq0" in builder.elements
    assert "analysisq0" not in builder.elements
    assert "analysis_conv0" not in builder.elements
    assert "analysis_caps0" not in builder.elements
    assert "analysis_sink0" not in builder.elements
    assert "hicon_pouring_0" not in builder.elements
    assert "nvvidconv_osd_0" not in builder.elements
    assert "caps_osd_0" not in builder.elements
    assert "preosdq0" not in builder.elements
    assert builder.elements["nvosd_0"].props["process-mode"] == 0
    assert (
        "Stream 0: C++ pouring plugin skipped because analysis branch is disabled "
        "for isolation"
    ) in caplog.text
    assert (
        "Stream 0: decoupled analysis mode — NV12 tee "
        "(analysis branch disabled for isolation)"
    ) in caplog.text


def test_create_all_elements_builds_shell_only_stream0_analysis_branch(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_decoupled_analysis_mode=True,
        stream_0_analysis_branch_enabled=True,
        stream_0_analysis_rgba_enabled=False,
        stream_0_analysis_cpp_plugin_enabled=False,
        use_cpp_pouring_plugin=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "analysisq0" in builder.elements
    assert "analysis_sink0" in builder.elements
    assert "analysis_conv0" not in builder.elements
    assert "analysis_caps0" not in builder.elements
    assert "hicon_pouring_0" not in builder.elements
    assert "nvvidconv_osd_0" not in builder.elements
    assert "caps_osd_0" not in builder.elements
    assert "preosdq0" not in builder.elements
    assert (
        "Stream 0: decoupled analysis mode — NV12 tee "
        "(display NV12 → nvosd_0, leaky NV12 analysis branch)"
    ) in caplog.text
    assert "Stream 0: C++ pouring plugin skipped on analysis branch for staged isolation" in caplog.text


def test_create_all_elements_keeps_stream0_cpp_plugin_for_main_path_fallback(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_decoupled_analysis_mode=True,
        stream_0_analysis_branch_enabled=True,
        stream_0_analysis_probe_enabled=False,
        stream_0_analysis_cpp_plugin_enabled=True,
        use_cpp_pouring_plugin=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "hicon_pouring_0" in builder.elements
    assert builder.elements["hicon_pouring_0"].props["enable-osd"] is False
    assert "analysisq0" in builder.elements
    assert "analysis_sink0" in builder.elements
    assert "Stream 0: C++ pouring plugin created (hicon_pouring_detect)" in caplog.text


def test_create_all_elements_sets_stream1_nvosd_cpu_mode(monkeypatch):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_1="rtsp://example/stream1",
        config_pyrometer="/tmp/config_pyrometer.txt",
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)

    assert builder._create_all_elements() is True
    assert builder.elements["nvosd_1"].props["process-mode"] == 0


def test_create_all_elements_sets_stream2_cpp_plugin_osd_on_main_path(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_2="rtsp://example/stream2",
        config_pouring_2="/tmp/config_pouring_2.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        use_cpp_pouring_plugin=True,
    )
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert builder.elements["hicon_pouring_2"].props["enable-osd"] is True
    assert builder.elements["nvosd_2"].props["process-mode"] == 0
    assert "Stream 2: C++ pouring OSD enabled on main path before nvosd_2" in caplog.text



def test_link_decode_chain_for_stream0_uses_isolation_queues():
    builder = _make_builder()
    builder.elements = {
        "srcq0": FakeElement("queue", "srcq0"),
        "depay0": FakeElement("rtph265depay", "depay0"),
        "parser0": FakeElement("h265parse", "parser0"),
        "vidcaps0": FakeElement("capsfilter", "vidcaps0"),
        "decoder0": FakeElement("nvv4l2decoder", "decoder0"),
        "nvvidconv0": FakeElement("nvvideoconvert", "nvvidconv0"),
        "caps0": FakeElement("capsfilter", "caps0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
    }

    assert builder._link_decode_chain(0) is True
    assert builder.elements["srcq0"].links == ["depay0"]
    assert builder.elements["depay0"].links == ["parser0"]
    assert builder.elements["parser0"].links == ["vidcaps0"]
    assert builder.elements["vidcaps0"].links == ["decoder0"]
    assert builder.elements["decoder0"].links == ["nvvidconv0"]
    assert builder.elements["nvvidconv0"].links == ["caps0"]
    assert builder.elements["caps0"].links == ["premuxq0"]


def test_link_decode_chain_for_stream1_stays_unchanged():
    builder = _make_builder()
    builder.elements = {
        "depay1": FakeElement("rtph265depay", "depay1"),
        "parser1": FakeElement("h265parse", "parser1"),
        "vidcaps1": FakeElement("capsfilter", "vidcaps1"),
        "decoder1": FakeElement("nvv4l2decoder", "decoder1"),
        "nvvidconv1": FakeElement("nvvideoconvert", "nvvidconv1"),
        "caps1": FakeElement("capsfilter", "caps1"),
    }

    assert builder._link_decode_chain(1) is True
    assert builder.elements["depay1"].links == ["parser1"]
    assert builder.elements["parser1"].links == ["vidcaps1"]
    assert builder.elements["vidcaps1"].links == ["decoder1"]
    assert builder.elements["decoder1"].links == ["nvvidconv1"]
    assert builder.elements["nvvidconv1"].links == ["caps1"]


def test_cb_newpad_links_stream0_video_to_srcq0_and_stream1_to_depay():
    builder = _make_builder(rtsp_codec_0="h265", rtsp_codec_1="h265")
    builder.elements = {
        "srcq0": FakeElement("queue", "srcq0"),
        "depay0": FakeElement("rtph265depay", "depay0"),
        "depay1": FakeElement("rtph265depay", "depay1"),
    }

    stream0_pad = FakeSrcPad("H265")
    builder._cb_newpad(None, stream0_pad, 0)
    assert stream0_pad.linked_to is builder.elements["srcq0"].get_static_pad("sink")

    stream1_pad = FakeSrcPad("H265")
    builder._cb_newpad(None, stream1_pad, 1)
    assert stream1_pad.linked_to is builder.elements["depay1"].get_static_pad("sink")


def test_link_all_branches_uses_premuxq0_for_stream0(monkeypatch):
    builder = _make_builder()
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "source0": FakeElement("rtspsrc", "source0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "pgie_pouring": FakeElement("nvinfer", "pgie_pouring"),
        "tracker_0": FakeElement("nvtracker", "tracker_0"),
        "hicon_pouring_0": FakeElement("hicon_pouring_detect", "hicon_pouring_0"),
        "nvvidconv_osd_0": FakeElement("nvvideoconvert", "nvvidconv_osd_0"),
        "caps_osd_0": FakeElement("capsfilter", "caps_osd_0"),
        "preosdq0": FakeElement("queue", "preosdq0"),
        "nvosd_0": FakeElement("nvdsosd", "nvosd_0"),
        "sink_0": FakeElement("fakesink", "sink_0"),
    }

    mux_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert builder.elements["mux_0"].links == ["postmuxq0"]
    assert builder.elements["postmuxq0"].links == ["pgie_pouring"]
    assert builder.elements["pgie_pouring"].links == ["tracker_0"]
    assert builder.elements["tracker_0"].links == ["hicon_pouring_0"]
    assert builder.elements["hicon_pouring_0"].links == ["nvvidconv_osd_0"]
    assert builder.elements["nvvidconv_osd_0"].links == ["caps_osd_0"]
    assert builder.elements["caps_osd_0"].links == ["preosdq0"]
    assert builder.elements["preosdq0"].links == ["nvosd_0"]


def test_link_all_branches_skips_stream0_tracker_when_bypassed(monkeypatch):
    builder = _make_builder(stream_0_bypass_tracker=True)
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "source0": FakeElement("rtspsrc", "source0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "pgie_pouring": FakeElement("nvinfer", "pgie_pouring"),
        "nvvidconv_osd_0": FakeElement("nvvideoconvert", "nvvidconv_osd_0"),
        "caps_osd_0": FakeElement("capsfilter", "caps_osd_0"),
        "preosdq0": FakeElement("queue", "preosdq0"),
        "nvosd_0": FakeElement("nvdsosd", "nvosd_0"),
        "sink_0": FakeElement("fakesink", "sink_0"),
    }

    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(builder, "_link_to_mux", lambda src_name, mux_name: True)

    assert builder._link_all_branches() is True


def test_link_all_branches_uses_decoupled_stream0_analysis_split(monkeypatch):
    builder = _make_builder(stream_0_decoupled_analysis_mode=True)
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "source0": FakeElement("rtspsrc", "source0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "pgie_pouring": FakeElement("nvinfer", "pgie_pouring"),
        "tracker_0": FakeElement("nvtracker", "tracker_0"),
        "hicon_pouring_0": FakeElement("hicon_pouring_detect", "hicon_pouring_0"),
        "tee_stream0_analysis": FakeElement("tee", "tee_stream0_analysis"),
        "displayq0": FakeElement("queue", "displayq0"),
        "analysisq0": FakeElement("queue", "analysisq0"),
        "analysis_sink0": FakeElement("fakesink", "analysis_sink0"),
        "nvosd_0": FakeElement("nvdsosd", "nvosd_0"),
        "sink_0": FakeElement("fakesink", "sink_0"),
    }

    mux_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert builder.elements["mux_0"].links == ["postmuxq0"]
    assert builder.elements["postmuxq0"].links == ["pgie_pouring"]
    assert builder.elements["pgie_pouring"].links == ["tracker_0"]
    assert builder.elements["tracker_0"].links == ["tee_stream0_analysis"]
    assert builder.elements["displayq0"].links == ["nvosd_0"]
    assert builder.elements["analysisq0"].links == ["hicon_pouring_0"]
    assert builder.elements["hicon_pouring_0"].links == ["analysis_sink0"]
    assert builder.elements["nvosd_0"].links == ["sink_0"]


def test_link_all_branches_omits_stream0_analysis_split_when_disabled(monkeypatch):
    builder = _make_builder(
        stream_0_decoupled_analysis_mode=True,
        stream_0_analysis_branch_enabled=False,
    )
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "source0": FakeElement("rtspsrc", "source0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "pgie_pouring": FakeElement("nvinfer", "pgie_pouring"),
        "tracker_0": FakeElement("nvtracker", "tracker_0"),
        "tee_stream0_analysis": FakeElement("tee", "tee_stream0_analysis"),
        "displayq0": FakeElement("queue", "displayq0"),
        "nvosd_0": FakeElement("nvdsosd", "nvosd_0"),
        "sink_0": FakeElement("fakesink", "sink_0"),
    }

    mux_links = []
    tee_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )
    monkeypatch.setattr(
        builder,
        "_link_tee_src_to_element",
        lambda tee_name, dst_name: tee_links.append((tee_name, dst_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert tee_links == [("tee_stream0_analysis", "displayq0")]
    assert builder.elements["tracker_0"].links == ["tee_stream0_analysis"]
    assert builder.elements["displayq0"].links == ["nvosd_0"]
    assert builder.elements["nvosd_0"].links == ["sink_0"]


def test_link_all_branches_builds_shell_only_stream0_analysis_split(monkeypatch):
    builder = _make_builder(
        stream_0_decoupled_analysis_mode=True,
        stream_0_analysis_branch_enabled=True,
        stream_0_analysis_rgba_enabled=False,
        stream_0_analysis_cpp_plugin_enabled=False,
    )
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "source0": FakeElement("rtspsrc", "source0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "pgie_pouring": FakeElement("nvinfer", "pgie_pouring"),
        "tracker_0": FakeElement("nvtracker", "tracker_0"),
        "tee_stream0_analysis": FakeElement("tee", "tee_stream0_analysis"),
        "displayq0": FakeElement("queue", "displayq0"),
        "analysisq0": FakeElement("queue", "analysisq0"),
        "analysis_sink0": FakeElement("fakesink", "analysis_sink0"),
        "nvosd_0": FakeElement("nvdsosd", "nvosd_0"),
        "sink_0": FakeElement("fakesink", "sink_0"),
    }

    mux_links = []
    tee_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )
    monkeypatch.setattr(
        builder,
        "_link_tee_src_to_element",
        lambda tee_name, dst_name: tee_links.append((tee_name, dst_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert tee_links == [
        ("tee_stream0_analysis", "displayq0"),
        ("tee_stream0_analysis", "analysisq0"),
    ]
    assert builder.elements["tracker_0"].links == ["tee_stream0_analysis"]
    assert builder.elements["displayq0"].links == ["nvosd_0"]
    assert builder.elements["analysisq0"].links == ["analysis_sink0"]
    assert builder.elements["nvosd_0"].links == ["sink_0"]



def test_link_all_branches_places_stream0_cpp_plugin_on_display_path_fallback(monkeypatch):
    builder = _make_builder(
        stream_0_decoupled_analysis_mode=True,
        stream_0_analysis_branch_enabled=True,
        stream_0_analysis_probe_enabled=False,
        stream_0_analysis_cpp_plugin_enabled=True,
    )
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "source0": FakeElement("rtspsrc", "source0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "pgie_pouring": FakeElement("nvinfer", "pgie_pouring"),
        "tracker_0": FakeElement("nvtracker", "tracker_0"),
        "tee_stream0_analysis": FakeElement("tee", "tee_stream0_analysis"),
        "displayq0": FakeElement("queue", "displayq0"),
        "hicon_pouring_0": FakeElement("hicon_pouring_detect", "hicon_pouring_0"),
        "analysisq0": FakeElement("queue", "analysisq0"),
        "analysis_sink0": FakeElement("fakesink", "analysis_sink0"),
        "nvosd_0": FakeElement("nvdsosd", "nvosd_0"),
        "sink_0": FakeElement("fakesink", "sink_0"),
    }

    mux_links = []
    tee_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )
    monkeypatch.setattr(
        builder,
        "_link_tee_src_to_element",
        lambda tee_name, dst_name: tee_links.append((tee_name, dst_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert tee_links == [
        ("tee_stream0_analysis", "displayq0"),
        ("tee_stream0_analysis", "analysisq0"),
    ]
    assert builder.elements["displayq0"].links == ["hicon_pouring_0"]
    assert builder.elements["hicon_pouring_0"].links == ["nvosd_0"]
    assert builder.elements["analysisq0"].links == ["analysis_sink0"]
    assert builder.elements["nvosd_0"].links == ["sink_0"]


def test_link_all_branches_skips_stream0_pgie_and_tracker_when_bypassed(monkeypatch):
    builder = _make_builder(stream_0_bypass_pgie=True)
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "source0": FakeElement("rtspsrc", "source0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "nvvidconv_osd_0": FakeElement("nvvideoconvert", "nvvidconv_osd_0"),
        "caps_osd_0": FakeElement("capsfilter", "caps_osd_0"),
        "preosdq0": FakeElement("queue", "preosdq0"),
        "nvosd_0": FakeElement("nvdsosd", "nvosd_0"),
        "sink_0": FakeElement("fakesink", "sink_0"),
    }

    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(builder, "_link_to_mux", lambda src_name, mux_name: True)

    assert builder._link_all_branches() is True
    assert builder.elements["postmuxq0"].links == ["nvvidconv_osd_0"]


def test_link_all_branches_uses_decode_sink_in_stream0_decode_only_mode(monkeypatch):
    builder = _make_builder(stream_0_decode_only_mode=True)
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "premuxq0": FakeElement("queue", "premuxq0"),
        "decode_sink_0": FakeElement("fakesink", "decode_sink_0"),
    }

    mux_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert builder.elements["premuxq0"].links == ["decode_sink_0"]
    assert mux_links == []


def test_link_all_branches_uses_postmux_sink_in_stream0_postmux_only_mode(monkeypatch):
    builder = _make_builder(stream_0_postmux_only_mode=True)
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "postmux_sink_0": FakeElement("fakesink", "postmux_sink_0"),
    }

    mux_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert builder.elements["mux_0"].links == ["postmuxq0"]
    assert builder.elements["postmuxq0"].links == ["postmux_sink_0"]


def test_link_all_branches_uses_postconv_sink_in_stream0_postconv_only_mode(monkeypatch):
    builder = _make_builder(stream_0_postconv_only_mode=True)
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "nvvidconv_osd_0": FakeElement("nvvideoconvert", "nvvidconv_osd_0"),
        "postconv_sink_0": FakeElement("fakesink", "postconv_sink_0"),
    }

    mux_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert builder.elements["mux_0"].links == ["postmuxq0"]
    assert builder.elements["postmuxq0"].links == ["nvvidconv_osd_0"]
    assert builder.elements["nvvidconv_osd_0"].links == ["postconv_sink_0"]


def test_link_all_branches_uses_preosd_sink_in_stream0_preosd_only_mode(monkeypatch):
    builder = _make_builder(stream_0_preosd_only_mode=True)
    builder.enabled_streams = [0]
    builder.enable_inference_video = False
    builder.elements = {
        "premuxq0": FakeElement("queue", "premuxq0"),
        "mux_0": FakeElement("nvstreammux", "mux_0"),
        "postmuxq0": FakeElement("queue", "postmuxq0"),
        "nvvidconv_osd_0": FakeElement("nvvideoconvert", "nvvidconv_osd_0"),
        "caps_osd_0": FakeElement("capsfilter", "caps_osd_0"),
        "preosdq0": FakeElement("queue", "preosdq0"),
        "preosd_sink_0": FakeElement("fakesink", "preosd_sink_0"),
    }

    mux_links = []
    monkeypatch.setattr(builder, "_link_decode_chain", lambda stream_id: True)
    monkeypatch.setattr(
        builder,
        "_link_to_mux",
        lambda src_name, mux_name: mux_links.append((src_name, mux_name)) or True,
    )

    assert builder._link_all_branches() is True
    assert mux_links == [("premuxq0", "mux_0")]
    assert builder.elements["mux_0"].links == ["postmuxq0"]
    assert builder.elements["postmuxq0"].links == ["nvvidconv_osd_0"]
    assert builder.elements["nvvidconv_osd_0"].links == ["caps_osd_0"]
    assert builder.elements["caps_osd_0"].links == ["preosdq0"]
    assert builder.elements["preosdq0"].links == ["preosd_sink_0"]


def test_segment_buffer_mode_takes_priority_over_other_stream0_sources():
    builder = _make_builder(
        use_segment_buffer_0=True,
        use_udp_loopback_0=True,
        use_ffmpeg_src_0=True,
        use_nvurisrcbin_0=True,
    )

    assert builder.use_segment_buffer_0 is True
    assert builder.use_udp_loopback_0 is False
    assert builder.use_ffmpeg_src_0 is False
    assert builder.use_nvurisrcbin_0 is False


def test_create_all_elements_uses_segment_buffer_chain_for_stream0(monkeypatch):
    builder = _make_builder(
        rtsp_stream_0="rtsp://example/substream",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        use_segment_buffer_0=True,
        segment_buffer_dir_0="/dev/shm/test-stream0-buffer",
        segment_buffer_segment_sec_0=2,
        segment_buffer_delay_sec_0=60,
        segment_buffer_retention_sec_0=120,
    )
    builder.pipeline = FakePipeline()

    called = {}

    def fake_segment_chain(stream_id, rtsp_url, buffer_dir, segment_sec, delay_sec, retention_sec):
        called["args"] = (stream_id, rtsp_url, buffer_dir, segment_sec, delay_sec, retention_sec)
        return True

    monkeypatch.setattr(builder, "_create_segment_buffer_chain", fake_segment_chain)
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", lambda factory_name, name: FakeElement(factory_name, name))

    assert builder._create_all_elements() is True
    assert called["args"] == (
        0,
        "rtsp://example/substream",
        "/dev/shm/test-stream0-buffer",
        2,
        60,
        120,
    )


def test_link_decode_chain_for_segment_buffer_stream0_links_fdsrc_through_tsdemux():
    builder = _make_builder(use_segment_buffer_0=True)
    builder.elements = {
        "source0": FakeElement("fdsrc", "source0"),
        "tsparse0": FakeElement("tsparse", "tsparse0"),
        "tspace0": FakeElement("identity", "tspace0"),
        "tsdemux0": FakeElement("tsdemux", "tsdemux0"),
    }

    assert builder._link_decode_chain(0) is True
    assert builder.elements["source0"].links == ["tsparse0"]
    assert builder.elements["tsparse0"].links == ["tspace0"]
    assert builder.elements["tspace0"].links == ["tsdemux0"]


def test_cb_tsdemux_pad_added_links_segment_buffer_stream0_chain(caplog):
    builder = _make_builder(use_segment_buffer_0=True)
    builder.elements = {
        "demuxq0": FakeElement("queue", "demuxq0"),
        "parser0": FakeElement("h264parse", "parser0"),
        "vidcaps0": FakeElement("capsfilter", "vidcaps0"),
        "decoder0": FakeElement("nvv4l2decoder", "decoder0"),
        "nvvidconv0": FakeElement("nvvideoconvert", "nvvidconv0"),
        "caps0": FakeElement("capsfilter", "caps0"),
        "premuxq0": FakeElement("queue", "premuxq0"),
    }

    caplog.set_level(logging.INFO)
    builder._cb_tsdemux_pad_added(None, FakeTsPad("video/x-h264"), 0)

    assert builder.elements["demuxq0"].links == ["parser0"]
    assert builder.elements["parser0"].links == ["vidcaps0"]
    assert builder.elements["vidcaps0"].links == ["decoder0"]
    assert builder.elements["decoder0"].links == ["nvvidconv0"]
    assert builder.elements["nvvidconv0"].links == ["caps0"]
    assert builder.elements["caps0"].links == ["premuxq0"]
    assert "Stream 0: segment buffer chain fully linked via tsdemux pad-added" in caplog.text
