"""
GStreamer Bus Handler - Error recovery and pipeline monitoring.

Handles EOS, errors, warnings, and state change messages with
RTSP-aware error classification:
- RTSP source errors: non-fatal, logged and tracked (rtspsrc retries internally)
- Fatal errors (nvinfer, decoder, mux): quit pipeline (systemd restarts)
- Stale stream watchdog: quit if ALL streams silent for 10 min
- healthchecks.io heartbeat: ping every 60s, /fail on fatal errors
"""
import logging
import time
import threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logger = logging.getLogger(__name__)

# RTSP source element name prefix (source0, source1, ...)
_RTSP_SOURCE_PREFIX = "source"

# Error rate-limit: N errors within M seconds → escalate to fatal
_RTSP_ERROR_LIMIT = 3
_RTSP_ERROR_WINDOW_SEC = 60.0


class BusHandler:
    """Handle GStreamer bus messages with RTSP-aware error recovery."""

    def __init__(self, pipeline, loop, healthcheck_url=""):
        """
        Args:
            pipeline: GStreamer pipeline
            loop: GLib MainLoop
            healthcheck_url: healthchecks.io ping URL (empty = disabled)
        """
        self.pipeline = pipeline
        self.loop = loop
        self.last_frame_time = {}
        self.stale_threshold_sec = 600  # 10 min watchdog
        self._frame_counts = {}       # {stream_id: int} — rolling 10s counter
        self._fps_log_interval = 10   # seconds
        self._zero_fps_counts = {}    # {stream_id: int} — consecutive 0fps intervals
        self._zero_fps_limit = 3      # 3 × 10s = 30s of 0fps → restart
        self._healthcheck_url = (healthcheck_url or "").rstrip("/")

        # Per-source RTSP error timestamps for rate-limiting
        self._rtsp_errors = {}  # {source_name: [timestamp, ...]}

        # Set True when quitting due to error so hicon_pipeline can sys.exit(1)
        # for systemd Restart=on-failure to trigger. EOS/SIGTERM stay False (clean exit).
        self.fatal_exit = False

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        logger.info("Bus handler attached")
        if self._healthcheck_url:
            logger.info(f"Healthcheck enabled: {self._healthcheck_url}")

    # ------------------------------------------------------------------
    # healthchecks.io heartbeat
    # ------------------------------------------------------------------
    def _ping_healthcheck(self, suffix=""):
        """
        Fire-and-forget ping to healthchecks.io.

        Args:
            suffix: "" for success, "/fail" for failure
        """
        if not self._healthcheck_url:
            return
        url = self._healthcheck_url + suffix

        def _do_ping():
            try:
                import requests
                requests.get(url, timeout=5)
            except Exception:
                pass  # Never block pipeline on healthcheck failure

        # Run in background thread so GLib main loop is never blocked
        threading.Thread(target=_do_ping, daemon=True).start()

    def _is_rtsp_source(self, src_name: str) -> bool:
        """Check if the error source is an rtspsrc element."""
        return src_name.startswith(_RTSP_SOURCE_PREFIX)

    def _track_rtsp_error(self, src_name: str) -> bool:
        """
        Track an RTSP error and check if rate limit is exceeded.

        Returns:
            True if error rate exceeded (should escalate to fatal).
        """
        now = time.time()
        if src_name not in self._rtsp_errors:
            self._rtsp_errors[src_name] = []

        # Prune old entries outside the window
        cutoff = now - _RTSP_ERROR_WINDOW_SEC
        self._rtsp_errors[src_name] = [
            t for t in self._rtsp_errors[src_name] if t > cutoff
        ]

        # Record this error
        self._rtsp_errors[src_name].append(now)

        count = len(self._rtsp_errors[src_name])
        if count >= _RTSP_ERROR_LIMIT:
            logger.error(
                f"[RTSP-FATAL] {src_name}: {count} errors in "
                f"{_RTSP_ERROR_WINDOW_SEC:.0f}s — escalating to fatal"
            )
            return True

        logger.warning(
            f"[RTSP-RECOVERABLE] {src_name}: error {count}/{_RTSP_ERROR_LIMIT} "
            f"(window {_RTSP_ERROR_WINDOW_SEC:.0f}s) — pipeline continues"
        )
        return False

    def _on_bus_message(self, bus, message):
        """Process bus messages with RTSP-aware error classification."""
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

            # Classify: RTSP source errors are non-fatal (rtspsrc retries)
            if self._is_rtsp_source(src_name):
                if self._track_rtsp_error(src_name):
                    # Rate limit exceeded → fatal
                    self._ping_healthcheck("/fail")
                    self.fatal_exit = True
                    self.loop.quit()
                # Otherwise: suppress, let rtspsrc handle reconnection
            else:
                # Non-RTSP error (nvinfer, decoder, mux, etc.) → fatal
                logger.error(f"[FATAL] Non-recoverable error from {src_name}")
                self._ping_healthcheck("/fail")
                self.fatal_exit = True
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
        """Call from probe to update last frame timestamp and increment frame counter."""
        self.last_frame_time[stream_id] = time.time()
        self._frame_counts[stream_id] = self._frame_counts.get(stream_id, 0) + 1

    def check_stale_streams(self):
        """Check if any stream has gone stale (no frames for threshold)."""
        now = time.time()
        stale = []
        for sid, last_t in self.last_frame_time.items():
            elapsed = now - last_t
            if elapsed > self.stale_threshold_sec:
                stale.append(sid)
                logger.warning(
                    f"Stream {sid} stale: no frames for {elapsed:.0f}s"
                )
        return stale

    def start_fps_logger(self):
        """Log per-stream frame count and FPS every 10 seconds via GLib timer."""
        from gi.repository import GLib

        def _fps_tick():
            if not self._frame_counts:
                return True
            parts = []
            for sid in sorted(self._frame_counts):
                count = self._frame_counts[sid]
                fps = count / self._fps_log_interval
                parts.append(f"Stream {sid}: {count} frames ({fps:.1f} fps)")
                self._frame_counts[sid] = 0  # reset for next interval

                # Dead stream detection: N consecutive 0fps intervals → restart
                if count == 0 and sid in self.last_frame_time:
                    self._zero_fps_counts[sid] = self._zero_fps_counts.get(sid, 0) + 1
                    if self._zero_fps_counts[sid] >= self._zero_fps_limit:
                        logger.info("[FPS] " + " | ".join(parts))
                        logger.critical(
                            f"[FPS-WATCHDOG] Stream {sid} at 0fps for "
                            f"{self._zero_fps_limit * self._fps_log_interval}s — restarting"
                        )
                        self._ping_healthcheck("/fail")
                        self.fatal_exit = True
                        self.loop.quit()
                        return False
                else:
                    self._zero_fps_counts[sid] = 0  # reset on any frames

            logger.info("[FPS] " + " | ".join(parts))
            return True  # keep timer alive

        GLib.timeout_add_seconds(self._fps_log_interval, _fps_tick)
        logger.info(f"FPS logger started (every {self._fps_log_interval}s)")

    def start_watchdog(self, interval_sec=60):
        """
        Start periodic stale stream watchdog via GLib timer.

        Checks every interval_sec seconds. If ALL tracked streams are stale
        (no frames for stale_threshold_sec), quits the pipeline.
        """
        def _watchdog_tick():
            if not self.last_frame_time:
                # No streams registered yet — skip
                return True

            stale = self.check_stale_streams()
            total = len(self.last_frame_time)

            if stale:
                logger.critical(
                    f"[WATCHDOG] Stream(s) {stale} stale for "
                    f">{self.stale_threshold_sec}s — quitting pipeline"
                )
                self._ping_healthcheck("/fail")
                self.fatal_exit = True
                self.loop.quit()
                return False  # Stop timer

            # Pipeline healthy — send heartbeat ping
            self._ping_healthcheck()
            return True  # Keep timer alive

        GLib.timeout_add_seconds(interval_sec, _watchdog_tick)
        logger.info(
            f"Watchdog started: check every {interval_sec}s, "
            f"stale threshold {self.stale_threshold_sec}s"
        )
