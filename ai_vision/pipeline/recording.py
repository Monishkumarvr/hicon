"""
Recording Module - DS-native MJPEG MKV recording via tee → nvjpegenc → matroskamux.
Uses NVMM path to avoid costly/fragile NVMM->CPU conversion on Jetson.
"""
import logging
import time
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

logger = logging.getLogger(__name__)


class RecordingManager:
    """Manage DS-native recording branch for a stream (continuous)."""

    def __init__(self, output_dir: str, stream_id: int = 0, target_fps: float = 0,
                 target_width: int = 640, target_height: int = 360):
        """
        Args:
            output_dir: Directory for recording files
            stream_id: Stream identifier
            target_fps: Optional output FPS cap (0 = inherit source cadence)
            target_width: Output width
            target_height: Output height
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_id = stream_id
        self.target_fps = float(target_fps or 0)
        self.target_width = int(target_width or 640)
        self.target_height = int(target_height or 360)
        self.is_recording = False
        self.current_file = None
        self.filesink = None
        self._branch_buffer_count = 0
        logger.info(
            f"RecordingManager initialized for stream {stream_id}, "
            f"target_fps={self.target_fps or 'source'}, "
            f"size={self.target_width}x{self.target_height}"
        )

    def setup_recording_branch(self, pipeline, tee_element):
        """
        Add recording branch to pipeline: tee → queue → nvvideoconvert → nvjpegenc → matroskamux → filesink

        Args:
            pipeline: GStreamer pipeline
            tee_element: Tee element to branch from

        Returns:
            True if setup successful
        """
        sid = str(self.stream_id)

        # Create recording elements
        queue = Gst.ElementFactory.make("queue", f"rec-queue-{sid}")
        if queue:
            # Drop old buffers if encoder lags so live pipeline keeps moving.
            queue.set_property("max-size-buffers", 120)
            queue.set_property("leaky", 2)

        encoder = Gst.ElementFactory.make("nvjpegenc", f"rec-enc-{sid}")
        if not encoder:
            logger.error("nvjpegenc not available - recording disabled")
            return False
        encoder.set_property("quality", 85)

        muxer = Gst.ElementFactory.make("matroskamux", f"rec-mux-{sid}")
        if not muxer:
            logger.error("matroskamux not available - recording disabled")
            return False

        self.filesink = Gst.ElementFactory.make("filesink", f"rec-sink-{sid}")
        self.filesink.set_property("location", "/dev/null")
        self.filesink.set_property("sync", False)
        self.filesink.set_property("async", False)

        elements = [queue, encoder, muxer, self.filesink]

        for el in elements:
            if not el:
                logger.error("Failed to create recording element")
                return False
            pipeline.add(el)

        # Link: queue -> nvjpegenc -> matroskamux -> filesink
        if not (queue.link(encoder) and
                encoder.link(muxer) and
                muxer.link(self.filesink)):
            logger.error("Failed to link recording branch")
            return False

        # Link tee to queue
        tee_pad = tee_element.request_pad_simple("src_%u")
        queue_pad = queue.get_static_pad("sink")
        if tee_pad.link(queue_pad) != Gst.PadLinkReturn.OK:
            logger.error("Failed to link tee to recording queue")
            return False

        # Diagnostics: verify recording branch is receiving buffers on live runs.
        queue_src = queue.get_static_pad("src")
        if queue_src:
            queue_src.add_probe(Gst.PadProbeType.BUFFER, self._on_branch_buffer)

        logger.info(f"Recording branch set up for stream {sid}")
        return True

    def _on_branch_buffer(self, pad, info):
        """Pad probe for recording branch diagnostics."""
        self._branch_buffer_count += 1
        if self._branch_buffer_count == 1:
            logger.info("Recording branch: first buffer received")
        elif self._branch_buffer_count % 300 == 0:
            logger.info(f"Recording branch: received {self._branch_buffer_count} buffers")
        return Gst.PadProbeReturn.OK

    def start_recording(self, event_prefix="event"):
        """Start recording to a new MKV file."""
        if self.is_recording:
            logger.warning("Already recording")
            return

        if not self.filesink:
            logger.error("Recording branch not set up")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{event_prefix}_s{self.stream_id}_{timestamp}.mkv"
        filepath = self.output_dir / filename

        self.filesink.set_property("location", str(filepath))

        self.is_recording = True
        self.current_file = filepath
        logger.info(f"Recording started: {filepath}")

    def stop_recording(self):
        """Stop recording and return output path."""
        if not self.is_recording:
            return None

        self.is_recording = False

        filepath = self.current_file
        self.current_file = None
        logger.info(
            f"Recording stopped: {filepath} (branch_buffers={self._branch_buffer_count})"
        )
        return str(filepath) if filepath else None
