"""
GStreamer Bus Handler - Error recovery and pipeline monitoring
Handles EOS, errors, warnings, and state change messages.
"""
import logging
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logger = logging.getLogger(__name__)


class BusHandler:
    """Handle GStreamer bus messages for error recovery and monitoring."""

    def __init__(self, pipeline, loop, frame_counters=None):
        """
        Args:
            pipeline: GStreamer pipeline
            loop: GLib MainLoop
            frame_counters: Optional dict to track per-stream frame counts
        """
        self.pipeline = pipeline
        self.loop = loop
        self.frame_counters = frame_counters or {}
        self.last_frame_time = {}
        self.stale_threshold_sec = 600  # 10 min watchdog

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        logger.info("Bus handler attached")

    def _on_bus_message(self, bus, message):
        """Process bus messages."""
        t = message.type

        if t == Gst.MessageType.EOS:
            logger.info("End of stream received")
            self.loop.quit()

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src_name = message.src.get_name() if message.src else "unknown"
            logger.error(f"Pipeline error from {src_name}: {err.message}")
            if debug:
                logger.debug(f"Debug info: {debug}")
            self.loop.quit()

        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            src_name = message.src.get_name() if message.src else "unknown"
            logger.warning(f"Pipeline warning from {src_name}: {err.message}")

        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                logger.info(
                    f"Pipeline state: {old.value_nick} -> {new.value_nick}"
                )

    def update_frame_time(self, stream_id):
        """Call from probe to update last frame timestamp."""
        self.last_frame_time[stream_id] = time.time()

    def check_stale_streams(self):
        """Check if any stream has gone stale (no frames for threshold)."""
        now = time.time()
        stale = []
        for sid, last_t in self.last_frame_time.items():
            if now - last_t > self.stale_threshold_sec:
                stale.append(sid)
                logger.warning(
                    f"Stream {sid} stale: no frames for {now - last_t:.0f}s"
                )
        return stale
