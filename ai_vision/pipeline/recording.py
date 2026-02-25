"""
Recording Module - DS-native MJPEG MKV recording via tee → nvjpegenc → splitmuxsink.

Uses NVMM path (nvjpegenc on GPU) to avoid CPU conversion on Jetson.
splitmuxsink auto-rotates files at the configured segment duration so the
pipeline never needs to restart to start a new recording file.
"""
import logging
import time
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

logger = logging.getLogger(__name__)


class RecordingManager:
    """
    Manage DS-native recording branch for a stream (continuous, auto-segmented).

    Pipeline branch:
        tee → queue → nvvideoconvert → capsfilter(NV12) → nvjpegenc → splitmuxsink
                                                                           └── muxer: matroskamux
                                                                           └── rotates every segment_duration_s
    """

    def __init__(self, output_dir: str, stream_id: int = 0, target_fps: float = 0,
                 target_width: int = 640, target_height: int = 360,
                 segment_duration_s: int = 3600):
        """
        Args:
            output_dir:         Directory for recording files.
            stream_id:          Stream identifier (0 = Process, 1 = Pyrometer).
            target_fps:         Optional output FPS cap (0 = inherit source cadence).
            target_width:       Output width in pixels.
            target_height:      Output height in pixels.
            segment_duration_s: Seconds per output file. splitmuxsink rotates
                                automatically at this boundary without pipeline restart.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_id = stream_id
        self.target_fps = float(target_fps or 0)
        self.target_width = int(target_width or 640)
        self.target_height = int(target_height or 360)
        self.segment_duration_s = int(segment_duration_s or 3600)
        self.is_recording = False
        self._prefix = 'inference'
        self.splitmuxsink = None
        self._branch_buffer_count = 0
        logger.info(
            f"RecordingManager initialized for stream {stream_id}, "
            f"target_fps={self.target_fps or 'source'}, "
            f"size={self.target_width}x{self.target_height}, "
            f"segment={self.segment_duration_s}s"
        )

    def setup_recording_branch(self, pipeline, tee_element) -> bool:
        """
        Add recording branch to pipeline:
            tee → queue → nvvideoconvert → capsfilter → nvjpegenc → splitmuxsink

        splitmuxsink wraps matroskamux internally and rotates files at
        segment_duration_s intervals.  The format-location signal generates
        timestamp-based filenames for each segment.

        Args:
            pipeline:     GStreamer pipeline.
            tee_element:  Tee element to branch from (post-OSD annotated frames).

        Returns:
            True if setup successful.
        """
        sid = str(self.stream_id)

        queue = Gst.ElementFactory.make("queue", f"rec-queue-{sid}")
        if queue:
            # Drop old buffers if encoder lags — live pipeline must never block.
            queue.set_property("max-size-buffers", 120)
            queue.set_property("leaky", 2)

        conv = Gst.ElementFactory.make("nvvideoconvert", f"rec-conv-{sid}")
        if not conv:
            logger.error("nvvideoconvert not available — recording disabled")
            return False

        capsfilter = Gst.ElementFactory.make("capsfilter", f"rec-caps-{sid}")
        if not capsfilter:
            logger.error("capsfilter not available — recording disabled")
            return False
        caps = (
            f"video/x-raw(memory:NVMM), format=NV12, "
            f"width={self.target_width}, height={self.target_height}"
        )
        capsfilter.set_property("caps", Gst.Caps.from_string(caps))

        encoder = Gst.ElementFactory.make("nvjpegenc", f"rec-enc-{sid}")
        if not encoder:
            logger.error("nvjpegenc not available — recording disabled")
            return False
        encoder.set_property("quality", 85)

        # splitmuxsink replaces the old matroskamux + filesink pair.
        # It creates a new MKV file every segment_duration_s seconds automatically.
        splitmuxsink = Gst.ElementFactory.make("splitmuxsink", f"rec-split-{sid}")
        if not splitmuxsink:
            logger.error("splitmuxsink not available — recording disabled")
            return False

        splitmuxsink.set_property("muxer-factory", "matroskamux")
        # max-size-time is in nanoseconds
        splitmuxsink.set_property("max-size-time", self.segment_duration_s * Gst.SECOND)
        # Finalize previous segment in a background thread so the pipeline
        # is never stalled while matroskamux writes its index.
        splitmuxsink.set_property("async-finalize", True)
        # Fallback location pattern (used only if format-location signal is not connected)
        splitmuxsink.set_property(
            "location",
            str(self.output_dir / f"inference_s{sid}_%05d.mkv")
        )
        # Override naming with wall-clock timestamps
        splitmuxsink.connect("format-location", self._on_format_location)

        self.splitmuxsink = splitmuxsink

        elements = [queue, conv, capsfilter, encoder, splitmuxsink]
        for el in elements:
            if not el:
                logger.error("Failed to create recording element")
                return False
            pipeline.add(el)

        # Link: queue → nvvideoconvert → capsfilter → nvjpegenc → splitmuxsink
        if not (queue.link(conv) and
                conv.link(capsfilter) and
                capsfilter.link(encoder) and
                encoder.link(splitmuxsink)):
            logger.error("Failed to link recording branch")
            return False

        # Link tee → queue
        tee_pad = tee_element.request_pad_simple("src_%u")
        queue_pad = queue.get_static_pad("sink")
        if tee_pad.link(queue_pad) != Gst.PadLinkReturn.OK:
            logger.error("Failed to link tee to recording queue")
            return False

        # Diagnostics: verify branch is receiving buffers on live runs
        queue_src = queue.get_static_pad("src")
        if queue_src:
            queue_src.add_probe(Gst.PadProbeType.BUFFER, self._on_branch_buffer)

        logger.info(f"Recording branch set up for stream {sid} ({self.segment_duration_s}s segments)")
        return True

    def _on_format_location(self, splitmux, fragment_id: int) -> str:
        """
        splitmuxsink 'format-location' signal — called at the start of each new segment.
        Returns the full output file path for that segment.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self._prefix}_s{self.stream_id}_{timestamp}_{fragment_id:05d}.mkv"
        path = str(self.output_dir / filename)
        logger.info(f"Recording segment {fragment_id}: {path}")
        return path

    def _on_branch_buffer(self, pad, info):
        """Pad probe for recording branch diagnostics."""
        self._branch_buffer_count += 1
        if self._branch_buffer_count == 1:
            logger.info(f"Stream {self.stream_id} recording branch: first buffer received")
        elif self._branch_buffer_count % 300 == 0:
            logger.info(
                f"Stream {self.stream_id} recording branch: "
                f"{self._branch_buffer_count} buffers received"
            )
        return Gst.PadProbeReturn.OK

    def start_recording(self, event_prefix: str = "inference"):
        """
        Mark recording as active and set the filename prefix for new segments.
        splitmuxsink starts writing as soon as the pipeline reaches PLAYING —
        no manual file-path assignment needed.
        """
        if self.is_recording:
            logger.warning(f"Stream {self.stream_id}: already recording")
            return

        if not self.splitmuxsink:
            logger.error(f"Stream {self.stream_id}: recording branch not set up")
            return

        self._prefix = event_prefix
        self.is_recording = True
        logger.info(
            f"Stream {self.stream_id}: segmented recording started "
            f"(prefix='{event_prefix}', segment={self.segment_duration_s}s)"
        )

    def stop_recording(self):
        """
        Mark recording as stopped.
        The pipeline EOS event finalizes the current MKV segment in splitmuxsink.
        """
        if not self.is_recording:
            return None

        self.is_recording = False
        logger.info(
            f"Stream {self.stream_id}: recording stopped "
            f"(branch_buffers={self._branch_buffer_count})"
        )
        return str(self.output_dir)
