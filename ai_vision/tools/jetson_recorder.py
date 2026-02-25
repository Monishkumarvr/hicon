#!/usr/bin/env python3
"""
Jetson Video Recorder — Standalone multi-stream timed recording utility.

Records any number of video streams for a specified duration and saves them
as MKV files (or MP4 with H.264) in the output directory.

Codec auto-detection via GStreamer decodebin handles any input format:
  H.264, H.265, MJPEG, VP8/VP9, MPEG-4, RAW, USB/V4L2, MIPI CSI, file sources.

Encoder selection order (hardware-first for Jetson):
  1. nvv4l2h264enc  — Jetson HW H.264 encode  → MP4 output
  2. nvjpegenc      — Jetson HW MJPEG encode   → MKV output
  3. x264enc        — Software H.264 encode    → MP4 output (fallback)

Usage examples:
  # Single RTSP stream, 60 seconds
  python3 jetson_recorder.py --streams rtsp://192.168.1.10/stream1 --duration 60

  # Multiple RTSP streams, 30 seconds, custom output folder
  python3 jetson_recorder.py \\
      --streams rtsp://cam1/live rtsp://cam2/live rtsp://cam3/live \\
      --duration 30 --output /home/user/recordings

  # Local file sources (any codec)
  python3 jetson_recorder.py --streams /path/to/a.mp4 /path/to/b.mkv --duration 20

  # V4L2 USB cameras
  python3 jetson_recorder.py --streams /dev/video0 /dev/video1 --duration 60

  # Force a specific output codec
  python3 jetson_recorder.py --streams rtsp://... --duration 60 --codec mjpeg

  # Resize output
  python3 jetson_recorder.py --streams rtsp://... --duration 60 --width 1280 --height 720

  # Schedule: start at specific wallclock time
  python3 jetson_recorder.py --streams rtsp://... --duration 60 --start-at 14:30:00
"""

import argparse
import datetime
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jetson_recorder")


# ---------------------------------------------------------------------------
# Encoder capability probe (run once at startup)
# ---------------------------------------------------------------------------

def _probe_element(name: str) -> bool:
    """Return True if GStreamer can create the named element."""
    el = Gst.ElementFactory.make(name, None)
    if el:
        el.set_state(Gst.State.NULL)
        return True
    return False


def detect_best_encoder() -> str:
    """
    Return the best available encoder name for this platform.

    Priority: nvv4l2h264enc (Jetson HW) > nvjpegenc (Jetson HW MJPEG)
              > x264enc (software).
    """
    for enc in ("nvv4l2h264enc", "nvjpegenc", "x264enc"):
        if _probe_element(enc):
            logger.info(f"Encoder selected: {enc}")
            return enc
    raise RuntimeError(
        "No supported encoder found. Install x264enc: apt install gstreamer1.0-plugins-ugly"
    )


# ---------------------------------------------------------------------------
# Source URI helper
# ---------------------------------------------------------------------------

def _source_type(uri: str):
    """
    Classify a source URI and return (source_type, gst_source_description).

    source_type: 'rtsp' | 'file' | 'v4l2' | 'test'
    """
    u = uri.lower()
    if u.startswith("rtsp://"):
        return "rtsp", uri
    if u.startswith("/dev/video"):
        return "v4l2", uri
    if u.startswith("videotestsrc"):
        return "test", uri
    # Assume local file
    return "file", os.path.abspath(uri)


# ---------------------------------------------------------------------------
# Per-stream pipeline
# ---------------------------------------------------------------------------

class StreamRecorder:
    """
    Manages a single GStreamer recording pipeline for one source.

    Pipeline (universal):
      [source] → decodebin
                   └─ [pad-added] → nvvideoconvert/videoconvert
                                  → [optional capsfilter for resize]
                                  → [encoder]
                                  → [muxer]
                                  → filesink
    """

    # Encoders that produce a bitstream needing a muxer wrapper
    _H264_ENCODERS = {"nvv4l2h264enc", "x264enc"}
    _MJPEG_ENCODERS = {"nvjpegenc"}

    def __init__(
        self,
        stream_id: int,
        uri: str,
        output_dir: Path,
        duration_s: float,
        encoder: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        prefix: str = "rec",
        rtsp_latency_ms: int = 200,
        bitrate_kbps: int = 4000,
    ):
        self.stream_id = stream_id
        self.uri = uri
        self.output_dir = output_dir
        self.duration_s = duration_s
        self.encoder = encoder
        self.width = width
        self.height = height
        self.prefix = prefix
        self.rtsp_latency_ms = rtsp_latency_ms
        self.bitrate_kbps = bitrate_kbps

        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop: Optional[GLib.MainLoop] = None
        self._eos_sent = False
        self._started_at: Optional[float] = None
        self._output_path: Optional[Path] = None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._done = threading.Event()

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------

    def _make_output_path(self) -> Path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "mp4" if self.encoder in self._H264_ENCODERS else "mkv"
        fname = f"{self.prefix}_s{self.stream_id}_{ts}.{ext}"
        return self.output_dir / fname

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> bool:
        self.pipeline = Gst.Pipeline.new(f"recorder-{self.stream_id}")
        if not self.pipeline:
            self._error = "Failed to create pipeline"
            return False

        src_type, src_val = _source_type(self.uri)

        # ---- Source element ----
        if src_type == "rtsp":
            source = Gst.ElementFactory.make("rtspsrc", f"src-{self.stream_id}")
            if not source:
                self._error = "rtspsrc unavailable"
                return False
            source.set_property("location", src_val)
            source.set_property("latency", self.rtsp_latency_ms)
            source.set_property("drop-on-latency", True)
            source.set_property("buffer-mode", 0)
            source.set_property("do-rtsp-keep-alive", True)
            # Try TCP — more reliable for recording
            try:
                import gi
                gi.require_version("GstRtsp", "1.0")
                from gi.repository import GstRtsp
                source.set_property("protocols", GstRtsp.RTSPLowerTrans.TCP)
            except Exception:
                try:
                    source.set_property("protocols", 0x4)  # TCP bitmask
                except Exception:
                    pass
            # Use decodebin to handle any codec in the RTSP stream
            decodebin = Gst.ElementFactory.make("decodebin", f"dec-{self.stream_id}")
            if not decodebin:
                self._error = "decodebin unavailable"
                return False
            self.pipeline.add(source)
            self.pipeline.add(decodebin)
            source.connect("pad-added", self._on_rtsp_pad, decodebin)
            decodebin.connect("pad-added", self._on_decoded_pad)
            self._decodebin = decodebin

        elif src_type == "v4l2":
            source = Gst.ElementFactory.make("v4l2src", f"src-{self.stream_id}")
            if not source:
                self._error = "v4l2src unavailable"
                return False
            source.set_property("device", src_val)
            decodebin = Gst.ElementFactory.make("decodebin", f"dec-{self.stream_id}")
            if not decodebin:
                self._error = "decodebin unavailable"
                return False
            self.pipeline.add(source)
            self.pipeline.add(decodebin)
            if not source.link(decodebin):
                self._error = "Failed to link v4l2src → decodebin"
                return False
            decodebin.connect("pad-added", self._on_decoded_pad)
            self._decodebin = decodebin

        elif src_type == "test":
            # videotestsrc — for local testing without real cameras
            source = Gst.ElementFactory.make("videotestsrc", f"src-{self.stream_id}")
            if not source:
                self._error = "videotestsrc unavailable"
                return False
            source.set_property("is-live", True)
            self.pipeline.add(source)
            # videotestsrc outputs raw, no decodebin needed — connect directly
            if not self._attach_encode_chain(source, raw=True):
                return False
            return True

        else:  # file
            source = Gst.ElementFactory.make("filesrc", f"src-{self.stream_id}")
            if not source:
                self._error = "filesrc unavailable"
                return False
            source.set_property("location", src_val)
            decodebin = Gst.ElementFactory.make("decodebin", f"dec-{self.stream_id}")
            if not decodebin:
                self._error = "decodebin unavailable"
                return False
            self.pipeline.add(source)
            self.pipeline.add(decodebin)
            if not source.link(decodebin):
                self._error = "Failed to link filesrc → decodebin"
                return False
            decodebin.connect("pad-added", self._on_decoded_pad)
            self._decodebin = decodebin

        return True

    # ------------------------------------------------------------------
    # Dynamic pad callbacks
    # ------------------------------------------------------------------

    def _on_rtsp_pad(self, rtspsrc, pad, decodebin):
        """Link rtspsrc dynamic pad → decodebin sink."""
        dec_sink = decodebin.get_static_pad("sink")
        if dec_sink and not dec_sink.is_linked():
            ret = pad.link(dec_sink)
            if ret != Gst.PadLinkReturn.OK:
                logger.warning(f"Stream {self.stream_id}: rtspsrc→decodebin link failed: {ret}")

    def _on_decoded_pad(self, decodebin, pad):
        """Link decodebin decoded video pad → encode chain."""
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        name = structure.get_name()

        if not name.startswith("video/"):
            # Skip audio pads silently
            return

        logger.info(f"Stream {self.stream_id}: decoded pad caps = {caps.to_string()}")
        self._attach_encode_chain_from_pad(pad)

    # ------------------------------------------------------------------
    # Encode chain construction
    # ------------------------------------------------------------------

    def _attach_encode_chain(self, src_element, raw: bool = False):
        """
        Build and link encode chain starting from src_element.
        Used when we already have a raw video element (e.g. videotestsrc).
        """
        # Create a ghost/queue as bridge
        queue = Gst.ElementFactory.make("queue", f"q-enc-{self.stream_id}")
        if queue:
            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", 60)
            self.pipeline.add(queue)

        # Build the encode tail
        tail = self._make_encode_tail()
        if tail is None:
            return False

        if queue:
            if not src_element.link(queue):
                self._error = "Failed to link src → queue"
                return False
            if not queue.link(tail["convert"]):
                self._error = "Failed to link queue → convert"
                return False
        else:
            if not src_element.link(tail["convert"]):
                self._error = "Failed to link src → convert"
                return False

        return True

    def _attach_encode_chain_from_pad(self, decoded_pad):
        """
        Build and link encode chain from a dynamic decoded pad.
        Called from decodebin pad-added callback.
        """
        # Already linked (only use first video pad)
        if hasattr(self, "_encode_linked") and self._encode_linked:
            return
        self._encode_linked = True

        tail = self._make_encode_tail()
        if tail is None:
            logger.error(f"Stream {self.stream_id}: Failed to build encode chain")
            self._error = "Failed to build encode chain"
            return

        convert_sink = tail["convert"].get_static_pad("sink")
        if not convert_sink:
            logger.error(f"Stream {self.stream_id}: convert element has no sink pad")
            return

        ret = decoded_pad.link(convert_sink)
        if ret != Gst.PadLinkReturn.OK:
            logger.error(f"Stream {self.stream_id}: decoded→convert link failed: {ret}")
            self._error = f"Pad link failed: {ret}"

    def _make_encode_tail(self):
        """
        Create and add encode chain elements to pipeline.

        Returns dict with {"convert": <first element>} or None on failure.

        Chain:
          nvvideoconvert/videoconvert
          → [optional capsfilter for resize + format]
          → [encoder]
          → [optional h264parse for H.264]
          → [muxer]
          → filesink
        """
        sid = str(self.stream_id)

        # --- Video convert (prefer NVMM path for Jetson) ---
        convert = Gst.ElementFactory.make("nvvideoconvert", f"conv-{sid}")
        convert_is_hw = True
        if not convert:
            logger.warning(f"Stream {self.stream_id}: nvvideoconvert unavailable, using videoconvert")
            convert = Gst.ElementFactory.make("videoconvert", f"conv-{sid}")
            convert_is_hw = False
        if not convert:
            self._error = "No video converter available"
            return None
        self.pipeline.add(convert)

        # --- Caps filter (format + optional resize) ---
        capsfilter = Gst.ElementFactory.make("capsfilter", f"caps-{sid}")
        if capsfilter:
            if convert_is_hw:
                # Jetson NVMM path
                caps_str = "video/x-raw(memory:NVMM), format=NV12"
            else:
                # CPU path
                caps_str = "video/x-raw, format=I420"

            if self.width and self.height:
                caps_str += f", width={self.width}, height={self.height}"

            capsfilter.set_property("caps", Gst.Caps.from_string(caps_str))
            self.pipeline.add(capsfilter)

        # --- Encoder ---
        encoder = Gst.ElementFactory.make(self.encoder, f"enc-{sid}")
        if not encoder:
            self._error = f"Encoder {self.encoder} failed to create"
            return None

        if self.encoder == "nvv4l2h264enc":
            encoder.set_property("bitrate", self.bitrate_kbps * 1000)
            encoder.set_property("preset-level", 1)   # UltraFastPreset
            encoder.set_property("insert-sps-pps", True)
            encoder.set_property("bufapi-version", True)
        elif self.encoder == "x264enc":
            encoder.set_property("bitrate", self.bitrate_kbps)
            encoder.set_property("tune", 0x00000004)  # zerolatency
            encoder.set_property("speed-preset", 1)   # ultrafast
        elif self.encoder == "nvjpegenc":
            encoder.set_property("quality", 85)
        self.pipeline.add(encoder)

        # --- H.264 parse (needed before muxer for H.264 streams) ---
        h264parse = None
        if self.encoder in self._H264_ENCODERS:
            h264parse = Gst.ElementFactory.make("h264parse", f"parse-{sid}")
            if h264parse:
                self.pipeline.add(h264parse)

        # --- Muxer ---
        if self.encoder in self._H264_ENCODERS:
            muxer = Gst.ElementFactory.make("mp4mux", f"mux-{sid}")
            if not muxer:
                muxer = Gst.ElementFactory.make("matroskamux", f"mux-{sid}")
        else:
            # MJPEG → MKV
            muxer = Gst.ElementFactory.make("matroskamux", f"mux-{sid}")
        if not muxer:
            self._error = "No muxer available (mp4mux / matroskamux)"
            return None
        self.pipeline.add(muxer)

        # --- File sink ---
        self._output_path = self._make_output_path()
        filesink = Gst.ElementFactory.make("filesink", f"sink-{sid}")
        if not filesink:
            self._error = "filesink unavailable"
            return None
        filesink.set_property("location", str(self._output_path))
        filesink.set_property("sync", False)
        filesink.set_property("async", False)
        self.pipeline.add(filesink)
        self._filesink = filesink

        # --- Link chain ---
        prev = convert

        if capsfilter:
            if not prev.link(capsfilter):
                self._error = "convert → capsfilter link failed"
                return None
            prev = capsfilter

        if not prev.link(encoder):
            self._error = f"capsfilter → {self.encoder} link failed"
            return None
        prev = encoder

        if h264parse:
            if not prev.link(h264parse):
                self._error = f"{self.encoder} → h264parse link failed"
                return None
            prev = h264parse

        if not prev.link(muxer):
            self._error = f"→ muxer link failed"
            return None

        if not muxer.link(filesink):
            self._error = "muxer → filesink link failed"
            return None

        logger.info(
            f"Stream {self.stream_id}: encode chain = "
            f"{'nvvideoconvert' if convert_is_hw else 'videoconvert'} → "
            f"{self.encoder} → {'h264parse → ' if h264parse else ''}"
            f"{muxer.get_factory().get_name()} → {self._output_path.name}"
        )
        return {"convert": convert}

    # ------------------------------------------------------------------
    # Bus handler
    # ------------------------------------------------------------------

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info(f"Stream {self.stream_id}: EOS received — recording complete")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self._error = f"{err.message} ({debug})"
            logger.error(f"Stream {self.stream_id}: Pipeline error: {self._error}")
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"Stream {self.stream_id}: {warn.message} ({debug})")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, _ = message.parse_state_changed()
                if new == Gst.State.PLAYING and self._started_at is None:
                    self._started_at = time.monotonic()
                    logger.info(f"Stream {self.stream_id}: Recording PLAYING → {self._output_path}")
        return True

    # ------------------------------------------------------------------
    # Duration timeout
    # ------------------------------------------------------------------

    def _on_duration_elapsed(self):
        """GLib timeout callback — send EOS to stop recording cleanly."""
        elapsed = time.monotonic() - (self._started_at or time.monotonic())
        logger.info(
            f"Stream {self.stream_id}: Duration {self.duration_s}s elapsed "
            f"(actual {elapsed:.1f}s) — sending EOS"
        )
        if not self._eos_sent:
            self._eos_sent = True
            self.pipeline.send_event(Gst.Event.new_eos())
        return False  # Do not repeat

    # ------------------------------------------------------------------
    # Run (blocks until done)
    # ------------------------------------------------------------------

    def run(self):
        """Build pipeline, run main loop for duration_s, then clean up."""
        try:
            Gst.init(None)
            if not self._build_pipeline():
                logger.error(f"Stream {self.stream_id}: Build failed: {self._error}")
                self._done.set()
                return

            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            # Schedule EOS after duration (in ms)
            # Add 2 s grace for pipeline PLAYING ramp-up
            timeout_ms = int((self.duration_s + 2) * 1000)
            self.loop = GLib.MainLoop()
            GLib.timeout_add(timeout_ms, self._on_duration_elapsed)

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                self._error = "Pipeline failed to reach PLAYING state"
                logger.error(f"Stream {self.stream_id}: {self._error}")
                self._done.set()
                return

            logger.info(f"Stream {self.stream_id}: Pipeline started, recording {self.duration_s}s → {self._output_path}")
            self.loop.run()

        except Exception as exc:
            self._error = str(exc)
            logger.exception(f"Stream {self.stream_id}: Unexpected error")
        finally:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            self._done.set()

    def start_in_thread(self):
        """Start recording in a background thread."""
        self._thread = threading.Thread(
            target=self.run,
            name=f"recorder-s{self.stream_id}",
            daemon=True,
        )
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until recording is finished. Returns True if finished cleanly."""
        finished = self._done.wait(timeout=timeout)
        if self._thread:
            self._thread.join(timeout=1)
        return finished and self._error is None

    def stop_now(self):
        """Force-stop the pipeline immediately (e.g. on Ctrl-C)."""
        if not self._eos_sent:
            self._eos_sent = True
            if self.pipeline:
                self.pipeline.send_event(Gst.Event.new_eos())
        if self.loop and self.loop.is_running():
            # Give EOS 2 s to flush, then hard-quit
            def _force_quit():
                if self.loop.is_running():
                    self.loop.quit()
                return False
            GLib.timeout_add(2000, _force_quit)

    @property
    def output_path(self) -> Optional[Path]:
        return self._output_path

    @property
    def error(self) -> Optional[str]:
        return self._error


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------

def _wait_until(time_str: str):
    """Block until a HH:MM:SS wallclock time today (or tomorrow if past)."""
    now = datetime.datetime.now()
    target = datetime.datetime.strptime(time_str, "%H:%M:%S").replace(
        year=now.year, month=now.month, day=now.day
    )
    if target <= now:
        target += datetime.timedelta(days=1)
    wait_s = (target - now).total_seconds()
    logger.info(f"Waiting {wait_s:.0f}s until {target.strftime('%Y-%m-%d %H:%M:%S')} to start recording")
    time.sleep(wait_s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Jetson multi-stream video recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--streams", "-s",
        nargs="+",
        required=True,
        metavar="URI",
        help=(
            "One or more source URIs. Supported: "
            "rtsp://..., /dev/videoN (V4L2), /path/to/file.mp4, videotestsrc"
        ),
    )
    p.add_argument(
        "--duration", "-d",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Recording duration in seconds (e.g. 60, 300, 3600)",
    )
    p.add_argument(
        "--output", "-o",
        default="output/videos",
        metavar="DIR",
        help="Output directory for recorded files (default: output/videos)",
    )
    p.add_argument(
        "--codec",
        choices=["auto", "nvv4l2h264", "mjpeg", "x264"],
        default="auto",
        metavar="CODEC",
        help=(
            "Output encoder: auto (detect best), nvv4l2h264 (Jetson HW H.264), "
            "mjpeg (Jetson HW MJPEG/MKV), x264 (software H.264). Default: auto"
        ),
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="PX",
        help="Output width in pixels (default: keep source resolution)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="PX",
        help="Output height in pixels (default: keep source resolution)",
    )
    p.add_argument(
        "--bitrate",
        type=int,
        default=4000,
        metavar="KBPS",
        help="H.264 output bitrate in kbps (default: 4000). Ignored for MJPEG.",
    )
    p.add_argument(
        "--prefix",
        default="rec",
        metavar="STR",
        help="Filename prefix for output files (default: rec)",
    )
    p.add_argument(
        "--rtsp-latency",
        type=int,
        default=200,
        metavar="MS",
        help="RTSP jitter buffer latency in ms (default: 200)",
    )
    p.add_argument(
        "--start-at",
        metavar="HH:MM:SS",
        default=None,
        help="Wallclock time to start recording (today or tomorrow if past)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and GST_DEBUG=3",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ.setdefault("GST_DEBUG", "3")

    # GStreamer init
    Gst.init(None)

    # Wait for scheduled start
    if args.start_at:
        _wait_until(args.start_at)

    # Resolve output directory (relative to script CWD or ai_vision root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    # Resolve encoder
    codec_map = {
        "nvv4l2h264": "nvv4l2h264enc",
        "mjpeg":      "nvjpegenc",
        "x264":       "x264enc",
    }
    if args.codec == "auto":
        encoder = detect_best_encoder()
    else:
        encoder = codec_map[args.codec]
        if not _probe_element(encoder):
            logger.warning(f"Requested encoder {encoder} not available, falling back to auto-detect")
            encoder = detect_best_encoder()

    logger.info(
        f"Recording {len(args.streams)} stream(s) for {args.duration}s "
        f"using {encoder} → {output_dir.resolve()}"
    )

    # Build one StreamRecorder per source
    recorders = []
    for idx, uri in enumerate(args.streams):
        rec = StreamRecorder(
            stream_id=idx,
            uri=uri,
            output_dir=output_dir,
            duration_s=args.duration,
            encoder=encoder,
            width=args.width,
            height=args.height,
            prefix=args.prefix,
            rtsp_latency_ms=args.rtsp_latency,
            bitrate_kbps=args.bitrate,
        )
        recorders.append(rec)

    # Graceful shutdown on Ctrl-C / SIGTERM
    def _signal_handler(signum, frame):
        logger.info(f"Signal {signum} received — stopping all streams")
        for r in recorders:
            r.stop_now()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start all recorders in parallel threads
    for rec in recorders:
        rec.start_in_thread()

    # Wait for all to complete (duration + generous headroom)
    total_timeout = args.duration + 30
    all_ok = True
    for rec in recorders:
        finished = rec.wait(timeout=total_timeout)
        if not finished:
            logger.error(f"Stream {rec.stream_id}: Timed out waiting for completion")
            all_ok = False
        elif rec.error:
            logger.error(f"Stream {rec.stream_id}: Finished with error: {rec.error}")
            all_ok = False
        else:
            sz = rec.output_path.stat().st_size if rec.output_path and rec.output_path.exists() else 0
            logger.info(
                f"Stream {rec.stream_id}: Saved {rec.output_path} "
                f"({sz / 1024 / 1024:.1f} MB)"
            )

    # Summary
    print("\n" + "=" * 60)
    print("Recording Summary")
    print("=" * 60)
    for rec in recorders:
        status = "OK" if not rec.error else f"ERROR: {rec.error}"
        path_str = str(rec.output_path) if rec.output_path else "N/A"
        sz_mb = 0.0
        if rec.output_path and rec.output_path.exists():
            sz_mb = rec.output_path.stat().st_size / 1024 / 1024
        print(f"  Stream {rec.stream_id}: {status}")
        print(f"    Source : {rec.uri}")
        print(f"    Output : {path_str}")
        print(f"    Size   : {sz_mb:.1f} MB")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
