import logging

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
from gi.repository import Gst, GstRtsp

Gst.init(None)

import pipeline.gst_builder as gst_builder_mod
from pipeline.gst_builder import DeepStreamPipelineBuilder


class DummySource:
    def __init__(self):
        self.props = {}
        self.connected_signals = []

    def set_property(self, name, value):
        self.props[name] = value

    def connect(self, signal, callback, stream_id):
        self.connected_signals.append((signal, callback, stream_id))


class FakeStaticPad:
    def __init__(self, name):
        self.name = name
        self.linked_to = None

    def link(self, other):
        self.linked_to = other
        other.linked_to = self
        return Gst.PadLinkReturn.OK

    def is_linked(self):
        return self.linked_to is not None


class FakeElement:
    def __init__(self, factory_name, name):
        self.factory_name = factory_name
        self.name = name
        self.props = {}
        self.links = []
        self.connected_signals = []
        self._pads = {
            "sink": FakeStaticPad(f"{name}:sink"),
            "src": FakeStaticPad(f"{name}:src"),
        }

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


class FakeSourceElement(FakeElement):
    def __init__(self, name):
        super().__init__("rtspsrc", name)
        self.state_history = []

    def set_state(self, state):
        self.state_history.append(state)
        return Gst.StateChangeReturn.SUCCESS

    def get_state(self, _timeout):
        return (Gst.StateChangeReturn.SUCCESS, None, None)


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
        "rtsp_udp_timeout_us": 5000000,
        "rtsp_port_retry": 20,
        "rtsp_do_retransmission": True,
    }
    config.update(overrides)
    return DeepStreamPipelineBuilder(config)


def test_rtsp_builder_sets_tcp_only_properties():
    builder = _make_builder(rtsp_protocol_0="tcp")
    source = DummySource()

    builder._configure_rtsp_source(source, "rtsp://example/stream0", 0)

    assert source.props["protocols"] == GstRtsp.RTSPLowerTrans.TCP
    assert source.props["tcp-timeout"] == 60000000
    assert "timeout" not in source.props
    assert source.props["retry"] == 20
    assert source.props["do-retransmission"] is True
    assert source.connected_signals
    assert source.connected_signals[0][0] == "new-manager"


def test_rtsp_builder_sets_udp_only_properties():
    builder = _make_builder(rtsp_protocol_0="udp")
    source = DummySource()

    builder._configure_rtsp_source(source, "rtsp://example/stream0", 0)

    assert source.props["protocols"] == GstRtsp.RTSPLowerTrans.UDP
    assert source.props["timeout"] == 5000000
    assert "tcp-timeout" not in source.props
    assert source.props["retry"] == 20


def test_rtsp_builder_leaves_protocol_unset_in_auto_mode():
    builder = _make_builder(rtsp_protocol_0="auto")
    source = DummySource()

    builder._configure_rtsp_source(source, "rtsp://example/stream0", 0)

    assert "protocols" not in source.props
    assert source.props["timeout"] == 5000000
    assert source.props["tcp-timeout"] == 60000000


def test_stream0_decode_chain_creates_isolation_queues_and_extra_surfaces(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(rtsp_codec_0="h265")
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_configure_rtsp_source", lambda source, location, stream_id: None)
    caplog.set_level(logging.INFO)

    builder._create_decode_chain(0, "rtsp://example/stream0")

    assert "srcq0" in builder.elements
    assert "premuxq0" in builder.elements
    assert builder.elements["srcq0"].props["leaky"] == 2
    assert builder.elements["premuxq0"].props["leaky"] == 2
    assert builder.elements["decoder0"].props["num-extra-surfaces"] == 8
    assert "source isolation queues enabled" in caplog.text


def test_create_all_elements_decoupled_analysis_mode_builds_current_analysis_branch(monkeypatch, caplog):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_decoupled_analysis_mode=True,
    )
    builder.pipeline = FakePipeline()
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)
    caplog.set_level(logging.INFO)

    assert builder._create_all_elements() is True
    assert "displayq0" in builder.elements
    assert "analysisq0" in builder.elements
    assert "analysis_sink0" in builder.elements
    assert "nvvidconv_osd_0" not in builder.elements
    assert builder.elements["displayq0"].props["leaky"] == 0
    assert builder.elements["analysisq0"].props["leaky"] == 2
    assert builder.elements["analysisq0"].props["max-size-buffers"] == 2
    assert builder.elements["nvosd_0"].props["process-mode"] == 0
    assert "decoupled analysis mode" in caplog.text


def test_stream0_local_relay_enables_annotated_tee_without_recording(monkeypatch):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        config_pouring="/tmp/config_pouring.txt",
        tracker_lib="/tmp/libtracker.so",
        tracker_config="/tmp/tracker.yml",
        stream_0_decoupled_analysis_mode=True,
        enable_inference_video=False,
        enable_live_stream_0=False,
        enable_stream0_local_relay=True,
    )
    builder.pipeline = FakePipeline()
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)

    assert builder._create_all_elements() is True
    assert "post_osd_conv_0" in builder.elements
    assert "post_osd_caps_0" in builder.elements
    assert "tee_0" in builder.elements
    assert "queue_display_0" in builder.elements


def test_stream1_per_stream_recording_flag_skips_recording_topology(monkeypatch):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    builder = _make_builder(
        rtsp_stream_1="rtsp://example/stream1",
        config_pyrometer="/tmp/config_pyrometer.txt",
        enable_inference_video=True,
        enable_inference_video_stream_1=False,
    )
    builder.pipeline = FakePipeline()
    monkeypatch.setattr(gst_builder_mod.Gst.ElementFactory, "make", fake_make)
    monkeypatch.setattr(builder, "_create_decode_chain", lambda stream_id, rtsp_url: None)

    assert builder._create_all_elements() is True
    assert "post_osd_conv_1" not in builder.elements
    assert "post_osd_caps_1" not in builder.elements
    assert "tee_1" not in builder.elements
    assert "queue_display_1" not in builder.elements


def test_get_restartable_stream_ids_only_returns_native_rtsp_streams():
    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        rtsp_stream_1="rtsp://example/stream1",
        rtsp_stream_2="rtsp://example/stream2",
        use_ffmpeg_src_0=True,
        use_segment_buffer_2=True,
    )

    assert builder.get_restartable_stream_ids() == {1}


def test_schedule_stream_restart_cycles_native_source_states():
    builder = _make_builder(rtsp_stream_0="rtsp://example/stream0")
    source = FakeSourceElement("source0")
    builder.elements["source0"] = source

    assert builder.schedule_stream_restart(0, "unit test") is True
    assert source.state_history == [Gst.State.NULL, Gst.State.READY, Gst.State.PLAYING]


def test_schedule_stream_restart_rejects_non_native_rtsp_stream():
    builder = _make_builder(
        rtsp_stream_0="rtsp://example/stream0",
        use_ffmpeg_src_0=True,
    )
    builder.elements["source0"] = FakeSourceElement("source0")

    assert builder.schedule_stream_restart(0, "unit test") is False
