import logging

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
from gi.repository import Gst, GstRtsp

Gst.init(None)

import pipeline.stream0_local_relay as relay_mod
from pipeline.stream0_local_relay import Stream0LocalRelayManager


class FakeStaticPad:
    def __init__(self, name):
        self.name = name
        self.linked_to = None

    def link(self, other):
        self.linked_to = other
        other.linked_to = self
        return Gst.PadLinkReturn.OK


class FakeElement:
    def __init__(self, factory_name, name):
        self.factory_name = factory_name
        self.name = name
        self.props = {}
        self.links = []
        self.signals = []
        self.requested_pad_names = []
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
        self.requested_pad_names.append(name)
        return FakeStaticPad(f"{self.name}:{name}")

    def connect(self, signal, callback):
        self.signals.append((signal, callback))


class FakePipeline:
    def __init__(self):
        self.added = []

    def add(self, element):
        self.added.append(element.name)


def test_stream0_local_relay_branch_builds_expected_publish_chain(monkeypatch):
    def fake_make(factory_name, name):
        return FakeElement(factory_name, name)

    monkeypatch.setattr(relay_mod.Gst.ElementFactory, "make", fake_make)

    pipeline = FakePipeline()
    tee = FakeElement("tee", "tee-0")
    manager = Stream0LocalRelayManager(
        stream_id=0,
        target_fps=10,
        target_width=640,
        target_height=360,
    )

    assert manager.setup_relay_branch(pipeline, tee) is True
    assert manager.publish_uri == "rtsp://127.0.0.1:8554/stream0_overlay"
    assert tee.requested_pad_names == ["src_%u"]
    assert manager.elements["sink"].props["location"] == manager.publish_uri
    assert manager.elements["sink"].props["protocols"] == GstRtsp.RTSPLowerTrans.TCP
    assert manager.elements["parser"].props["config-interval"] == -1
    assert manager.elements["encoder"].props["insert-sps-pps"] is True
    assert manager.elements["queue"].props["leaky"] == 2
    assert "relay-sink-0" in pipeline.added


def test_stream0_local_relay_handles_missing_rtspclientsink(monkeypatch, caplog):
    def fake_make(factory_name, name):
        if factory_name == "rtspclientsink":
            return None
        return FakeElement(factory_name, name)

    monkeypatch.setattr(relay_mod.Gst.ElementFactory, "make", fake_make)
    pipeline = FakePipeline()
    tee = FakeElement("tee", "tee-0")
    manager = Stream0LocalRelayManager(stream_id=0, target_fps=10)
    caplog.set_level(logging.ERROR)

    assert manager.setup_relay_branch(pipeline, tee) is False
    assert "rtspclientsink unavailable" in caplog.text
    assert pipeline.added == []
