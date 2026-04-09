"""
Recording Module - DS-native MJPEG AVI recording via tee → valve → nvjpegenc → avimux → filesink.
Uses NVMM path to avoid costly/fragile NVMM->CPU conversion on Jetson.

avimux accepts image/jpeg natively; filesink location is set once at setup time so no
dynamic element-state management is needed while the pipeline is running.

The valve is the sole control point: closed during warmup and outside schedule windows,
open when recording is active.  avimux only writes to the file when buffers arrive,
so keeping the valve closed produces no output in the file (the file is created on disk
at pipeline startup but stays empty until the first recording window opens).

Supports scheduled recording windows (morning/evening shifts or 24/7).
File rotation is handled at the schedule-boundary level (stop + restart recording
re-arms the buffer count; the file continues to grow — one AVI file per pipeline run).
"""
import logging
import time
from datetime import datetime
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

logger = logging.getLogger(__name__)


def parse_schedule(schedule_str: str) -> list:
    """Parse schedule string into list of (start_hour, start_min, end_hour, end_min) tuples.

    Args:
        schedule_str: 'always' or comma-separated 'HH:MM-HH:MM' windows

    Returns:
        Empty list for 'always' (record 24/7), or list of (sh, sm, eh, em) tuples.
    """
    s = schedule_str.strip().lower()
    if not s or s == 'always':
        return []
    windows = []
    for part in s.split(','):
        part = part.strip()
        if '-' not in part:
            logger.warning(f"Invalid schedule window '{part}', skipping")
            continue
        try:
            start_s, end_s = part.split('-', 1)
            sh, sm = int(start_s.split(':')[0]), int(start_s.split(':')[1])
            eh, em = int(end_s.split(':')[0]), int(end_s.split(':')[1])
            windows.append((sh, sm, eh, em))
        except (ValueError, IndexError):
            logger.warning(f"Invalid schedule window '{part}', skipping")
    return windows


def is_in_schedule(windows: list, now: datetime = None) -> bool:
    """Check if current time falls within any recording window.

    Args:
        windows: Output of parse_schedule(). Empty = always record.
        now: Optional datetime for testing.

    Returns:
        True if recording should be active.
    """
    if not windows:
        return True
    if now is None:
        now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    for sh, sm, eh, em in windows:
        start_m = sh * 60 + sm
        end_m = eh * 60 + em
        if end_m > start_m:
            # Same-day window: e.g. 06:00-08:00
            if start_m <= current_minutes < end_m:
                return True
        else:
            # Overnight window: e.g. 22:00-06:00
            if current_minutes >= start_m or current_minutes < end_m:
                return True
    return False


class RecordingManager:
    """Manage DS-native recording branch for a stream with schedule-based valve control."""

    def __init__(self, output_dir: str, stream_id: int = 0, target_fps: float = 0,
                 target_width: int = 640, target_height: int = 360,
                 schedule: str = 'always', max_duration_s: int = 3600,
                 retention_days: int = 3):
        """
        Args:
            output_dir: Directory for recording files
            stream_id: Stream identifier
            target_fps: Optional output FPS cap (0 = inherit source cadence)
            target_width: Output width
            target_height: Output height
            schedule: 'always' or 'HH:MM-HH:MM,...' windows
            max_duration_s: Max seconds per recording session (0 = no limit)
            retention_days: Delete recordings older than N days (0 = keep forever)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_id = stream_id
        self.target_fps = float(target_fps or 0)
        self.target_width = int(target_width or 640)
        self.target_height = int(target_height or 360)
        self.is_recording = False
        self.current_file = None
        self.record_valve = None
        self._filesink = None
        self._branch_buffer_count = 0
        self._schedule_probe_count = 0
        self._recording_start_time = 0.0
        self._schedule_windows = parse_schedule(schedule)
        self._max_duration_s = int(max_duration_s or 0)
        self._retention_days = int(retention_days or 0)
        self._last_cleanup_time = 0.0
        self._event_prefix = "inference"
        schedule_desc = 'always' if not self._schedule_windows else schedule
        retention_desc = f"{self._retention_days}d" if self._retention_days > 0 else 'forever'
        logger.info(
            f"RecordingManager initialized for stream {stream_id}, "
            f"target_fps={self.target_fps or 'source'}, "
            f"size={self.target_width}x{self.target_height}, "
            f"schedule={schedule_desc}, max_duration={self._max_duration_s}s, "
            f"retention={retention_desc}"
        )

    def setup_recording_branch(self, pipeline, tee_element):
        """
        Add recording branch to pipeline:
        tee → valve → queue → nvvideoconvert → capsfilter(NV12) → nvjpegenc → avimux → filesink

        Filename is generated once at setup time (startup timestamp).
        The valve is the sole runtime control — no dynamic element reconfiguration.

        Args:
            pipeline: GStreamer pipeline
            tee_element: Tee element to branch from

        Returns:
            True if setup successful
        """
        sid = str(self.stream_id)
        in_window = is_in_schedule(self._schedule_windows)

        # Filename is fixed at pipeline setup time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"inference_s{self.stream_id}_{timestamp}.avi"
        self.current_file = filepath

        # Create recording elements
        self.record_valve = Gst.ElementFactory.make("valve", f"rec-valve-{sid}")
        if not self.record_valve:
            logger.error("valve not available - recording disabled")
            return False
        # Keep the branch closed until start_recording() explicitly opens it.
        self.record_valve.set_property("drop", True)

        queue = Gst.ElementFactory.make("queue", f"rec-queue-{sid}")
        if queue:
            queue.set_property("max-size-buffers", 16)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)
            queue.set_property("leaky", 2)

        conv = Gst.ElementFactory.make("nvvideoconvert", f"rec-conv-{sid}")
        if not conv:
            logger.error("nvvideoconvert not available - recording disabled")
            return False

        capsfilter = Gst.ElementFactory.make("capsfilter", f"rec-caps-{sid}")
        if not capsfilter:
            logger.error("capsfilter not available - recording disabled")
            return False
        caps = (f"video/x-raw(memory:NVMM), format=NV12, "
                f"width={self.target_width}, height={self.target_height}")
        capsfilter.set_property("caps", Gst.Caps.from_string(caps))

        encoder = Gst.ElementFactory.make("nvjpegenc", f"rec-enc-{sid}")
        if not encoder:
            logger.error("nvjpegenc not available - recording disabled")
            return False
        encoder.set_property("quality", 85)

        # avimux accepts image/jpeg directly — no jpegparse needed
        muxer = Gst.ElementFactory.make("avimux", f"rec-mux-{sid}")
        if not muxer:
            logger.error("avimux not available - recording disabled")
            return False
        muxer.set_property("bigfile", True)

        filesink = Gst.ElementFactory.make("filesink", f"rec-sink-{sid}")
        if not filesink:
            logger.error("filesink not available - recording disabled")
            return False
        filesink.set_property("location", str(filepath))
        filesink.set_property("sync", False)
        filesink.set_property("async", False)
        self._filesink = filesink

        elements = [self.record_valve, queue, conv, capsfilter, encoder, muxer, filesink]

        for el in elements:
            if not el:
                logger.error("Failed to create recording element")
                return False
            pipeline.add(el)

        if not (self.record_valve.link(queue) and
                queue.link(conv) and
                conv.link(capsfilter) and
                capsfilter.link(encoder) and
                encoder.link(muxer) and
                muxer.link(filesink)):
            logger.error("Failed to link recording branch")
            return False

        tee_pad = tee_element.request_pad_simple("src_%u")
        valve_pad = self.record_valve.get_static_pad("sink")
        if tee_pad.link(valve_pad) != Gst.PadLinkReturn.OK:
            logger.error("Failed to link tee to recording valve")
            return False

        # Schedule probe: drives valve open/close from the upstream (pre-valve) pad
        schedule_pad = self.record_valve.get_static_pad("sink")
        if schedule_pad:
            schedule_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_schedule_probe)

        # Diagnostics probe: counts encoded frames downstream of nvjpegenc
        probe_pad = encoder.get_static_pad("src")
        if probe_pad:
            probe_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_branch_buffer)

        if in_window:
            logger.info(
                f"Stream {sid} recording branch primed at startup "
                f"(inside schedule, awaiting deferred start)"
            )
        else:
            logger.info(f"Stream {sid} recording branch dormant at startup (outside schedule)")
        logger.info(
            f"Recording branch set up for stream {sid}: {filepath}"
        )
        return True

    def _set_branch_flow_enabled(self, enabled: bool):
        """Open or close the recording branch without relinking the pipeline."""
        if self.record_valve:
            self.record_valve.set_property("drop", not enabled)

    def _on_schedule_probe(self, pad, info):
        """Lightweight pre-valve probe to drive schedule transitions while dormant."""
        self._schedule_probe_count += 1
        if self._schedule_probe_count % 100 == 0:
            self._check_schedule_and_rotation()
        return Gst.PadProbeReturn.OK

    def _on_branch_buffer(self, pad, info):
        """Pad probe for diagnostics on the active recording branch."""
        self._branch_buffer_count += 1
        if self._branch_buffer_count == 1:
            logger.info(f"Stream {self.stream_id} recording: first buffer received")
        elif self._branch_buffer_count % 300 == 0:
            logger.info(
                f"Stream {self.stream_id} recording: {self._branch_buffer_count} buffers encoded"
            )
        return Gst.PadProbeReturn.OK

    def _check_schedule_and_rotation(self):
        """Check if recording should start/stop based on schedule."""
        in_window = is_in_schedule(self._schedule_windows)
        now = time.monotonic()

        if self.is_recording and not in_window:
            logger.info(f"Stream {self.stream_id}: outside recording window, stopping")
            self.stop_recording()
            return

        if not self.is_recording and in_window:
            logger.info(f"Stream {self.stream_id}: entering recording window, starting")
            self.start_recording(event_prefix=self._event_prefix)
            return

        if (self.is_recording and self._max_duration_s > 0 and
                (now - self._recording_start_time) >= self._max_duration_s):
            logger.info(
                f"Stream {self.stream_id}: max duration {self._max_duration_s}s reached, "
                f"resetting recording start time"
            )
            # Cannot rotate file without pipeline restart; just reset the timer
            self._recording_start_time = now

        # Run cleanup once per hour
        if self._retention_days > 0 and (now - self._last_cleanup_time) >= 3600:
            self._last_cleanup_time = now
            self._cleanup_old_recordings()

    def _cleanup_old_recordings(self):
        """Delete .avi files older than retention_days in output_dir."""
        if self._retention_days <= 0:
            return
        cutoff = time.time() - (self._retention_days * 86400)
        deleted = 0
        for f in self.output_dir.glob("*.avi"):
            if f == self.current_file:
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    deleted += 1
            except OSError:
                pass
        if deleted > 0:
            logger.info(
                f"Stream {self.stream_id}: cleaned up {deleted} recordings "
                f"older than {self._retention_days} days"
            )

    def start_recording(self, event_prefix="event"):
        """Start recording by opening the valve."""
        if self.is_recording:
            logger.warning("Already recording")
            return

        if not self._filesink:
            logger.error("Recording branch not set up")
            return

        if not is_in_schedule(self._schedule_windows):
            self._set_branch_flow_enabled(False)
            logger.info(f"Stream {self.stream_id}: not in recording window, skipping start")
            return

        self._event_prefix = event_prefix
        self._set_branch_flow_enabled(True)
        self._branch_buffer_count = 0
        self.is_recording = True
        self._recording_start_time = time.monotonic()
        logger.info(f"Stream {self.stream_id} recording started: {self.current_file}")

    def stop_recording(self):
        """Stop recording by closing the valve. Returns output file path."""
        if not self.is_recording:
            return None

        self.is_recording = False
        self._set_branch_flow_enabled(False)

        duration = time.monotonic() - self._recording_start_time
        logger.info(
            f"Stream {self.stream_id} recording stopped: "
            f"{self.current_file} ({duration:.0f}s, "
            f"encoded_buffers={self._branch_buffer_count})"
        )
        return str(self.current_file) if self.current_file else None
