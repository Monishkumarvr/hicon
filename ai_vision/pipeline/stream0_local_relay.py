"""
Stream 0 local RTSP relay branch.

Publishes the post-OSD annotated Stream 0 output to the local MediaMTX
server so any remote fan-out happens outside the main DeepStream pipeline.
"""
import logging

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
from gi.repository import Gst, GstRtsp


logger = logging.getLogger(__name__)


class Stream0LocalRelayManager:
    """Attach a local RTSP publishing branch to the post-OSD Stream 0 tee."""

    LOCAL_HOST = "127.0.0.1"
    LOCAL_PORT = 8554
    LOCAL_PATH = "stream0_overlay"

    def __init__(
        self,
        *,
        stream_id: int = 0,
        target_fps: float = 0,
        target_width: int = 640,
        target_height: int = 360,
    ):
        self.stream_id = int(stream_id)
        self.target_fps = float(target_fps or 0)
        self.target_width = int(target_width or 640)
        self.target_height = int(target_height or 360)
        self.elements = {}

    @property
    def publish_uri(self) -> str:
        return (
            f"rtsp://{self.LOCAL_HOST}:{self.LOCAL_PORT}/{self.LOCAL_PATH}"
        )

    @staticmethod
    def _safe_set_property(element, name, value) -> None:
        try:
            element.set_property(name, value)
        except Exception:
            logger.debug(
                "Relay branch: property %s unsupported on %s",
                name,
                getattr(element, "name", element),
                exc_info=True,
            )

    def _build_caps_string(self) -> str:
        caps = [
            "video/x-raw(memory:NVMM)",
            "format=NV12",
            f"width={self.target_width}",
            f"height={self.target_height}",
        ]
        fps = int(round(self.target_fps))
        if fps > 0:
            caps.append(f"framerate={fps}/1")
        return ", ".join(caps)

    def _configure_encoder(self, encoder) -> None:
        # Insert codec config regularly so new readers can attach cleanly.
        self._safe_set_property(encoder, "insert-sps-pps", True)
        if self.target_fps > 0:
            frame_interval = max(1, int(round(self.target_fps)))
            self._safe_set_property(encoder, "iframeinterval", frame_interval)
            self._safe_set_property(encoder, "idrinterval", frame_interval)
        self._safe_set_property(encoder, "bitrate", 2_000_000)
        self._safe_set_property(encoder, "maxperf-enable", True)

    def _on_new_payloader(self, _sink, payloader) -> None:
        self._safe_set_property(payloader, "config-interval", 1)

    def setup_relay_branch(self, pipeline, tee_element) -> bool:
        sid = str(self.stream_id)

        queue = Gst.ElementFactory.make("queue", f"relay-queue-{sid}")
        if queue:
            queue.set_property("max-size-buffers", 8)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)
            queue.set_property("leaky", 2)

        conv = Gst.ElementFactory.make("nvvideoconvert", f"relay-conv-{sid}")

        capsfilter = Gst.ElementFactory.make("capsfilter", f"relay-caps-{sid}")
        if capsfilter:
            capsfilter.set_property("caps", Gst.Caps.from_string(self._build_caps_string()))

        encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"relay-enc-{sid}")
        if encoder:
            self._configure_encoder(encoder)

        parser = Gst.ElementFactory.make("h264parse", f"relay-parse-{sid}")
        if parser:
            self._safe_set_property(parser, "config-interval", -1)
            self._safe_set_property(parser, "disable-passthrough", True)

        sink = Gst.ElementFactory.make("rtspclientsink", f"relay-sink-{sid}")
        if sink is None:
            logger.error(
                "Stream %s: rtspclientsink unavailable; local MediaMTX relay disabled "
                "(install the GStreamer RTSP server runtime)",
                sid,
            )
            return False
        sink.set_property("location", self.publish_uri)
        self._safe_set_property(sink, "protocols", GstRtsp.RTSPLowerTrans.TCP)
        self._safe_set_property(sink, "sync", False)
        self._safe_set_property(sink, "async", False)
        try:
            sink.connect("new-payloader", self._on_new_payloader)
        except Exception:
            logger.debug("Relay branch: rtspclientsink.new-payloader unavailable", exc_info=True)

        elements = {
            "queue": queue,
            "conv": conv,
            "capsfilter": capsfilter,
            "encoder": encoder,
            "parser": parser,
            "sink": sink,
        }
        self.elements = elements

        missing = [name for name, element in elements.items() if element is None]
        if missing:
            logger.error(
                "Stream %s: local relay disabled; failed to create elements: %s",
                sid,
                ", ".join(missing),
            )
            return False

        for element in elements.values():
            pipeline.add(element)

        if not (
            queue.link(conv)
            and conv.link(capsfilter)
            and capsfilter.link(encoder)
            and encoder.link(parser)
            and parser.link(sink)
        ):
            logger.error("Stream %s: failed to link local relay branch", sid)
            return False

        tee_pad = tee_element.request_pad_simple("src_%u")
        relay_pad = queue.get_static_pad("sink")
        if tee_pad is None or relay_pad is None:
            logger.error("Stream %s: failed to get tee pad for local relay branch", sid)
            return False
        if tee_pad.link(relay_pad) != Gst.PadLinkReturn.OK:
            logger.error("Stream %s: failed to link tee to local relay branch", sid)
            return False

        logger.info(
            "Stream %s: local MediaMTX relay branch configured (%s)",
            sid,
            self.publish_uri,
        )
        return True
