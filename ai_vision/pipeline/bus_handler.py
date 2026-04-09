"""
GStreamer Bus Handler - Error recovery and pipeline monitoring.

Handles EOS, errors, warnings, and state change messages with
RTSP-aware error classification:
- RTSP source errors: non-fatal, logged and tracked (rtspsrc retries internally)
- Fatal errors (nvinfer, decoder, mux): quit pipeline (systemd restarts)
- Stale stream watchdog: quit if ALL streams silent for 10 min
- healthchecks.io heartbeat: ping every 60s, /fail on fatal errors
"""
import json
import logging
import time
import threading
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logger = logging.getLogger(__name__)

# RTSP source element name prefix (source0, source1, ...)
_RTSP_SOURCE_PREFIX = "source"

# Error rate-limit: N errors within M seconds → escalate to fatal
_RTSP_ERROR_LIMIT = 3
_RTSP_ERROR_WINDOW_SEC = 60.0

# Grace period: skip 0fps watchdog during startup (network may not be ready)
_STARTUP_GRACE_SEC = 30
_STREAM0_STAGE_ORDER = (
    "decoder_src",
    "nvvidconv_src",
    "caps_src",
    "premuxq_src",
    "mux_src",
    "postmuxq_src",
    "pgie_sink",
    "pgie_src",
    "tracker_sink",
    "tracker_src",
)
_STREAM0_UPSTREAM_PTS_ORDER = (
    "decoder_src",
    "nvvidconv_src",
    "caps_src",
    "premuxq_src",
)


class BusHandler:
    """Handle GStreamer bus messages with RTSP-aware error recovery."""

    def __init__(self, pipeline, loop, healthcheck_url="",
                 stream0_decoupled_analysis_mode=False, stream_policies=None,
                 stream0_segment_buffer_mode=False, stream0_segment_buffer_state_path="",
                 stream0_startup_grace_sec=None, stream_startup_grace_overrides=None,
                 stream_segment_buffer_state_paths=None,
                 warn_safety_cap_sec=90,
                 rtsp_restart_stale_sec=90,
                 rtsp_restart_cooldown_sec=60,
                 rtsp_restart_backoff_sec=5,
                 stream_restart_cb=None,
                 restartable_stream_ids=None):
        """
        Args:
            pipeline: GStreamer pipeline
            loop: GLib MainLoop
            healthcheck_url: healthchecks.io ping URL (empty = disabled)
            stream0_decoupled_analysis_mode: Whether Stream 0 uses a side analysis branch
            stream_policies: {stream_id: 'restart'|'warn'} per-stream 0fps policy
            stream0_segment_buffer_mode: Whether Stream 0 uses delayed segment buffering
            stream0_segment_buffer_state_path: JSON state file published by the helper
            stream0_startup_grace_sec: Optional Stream 0 startup grace override
            stream_startup_grace_overrides: {stream_id: int} per-stream grace overrides
            stream_segment_buffer_state_paths: {stream_id: str} state.json paths for
                any segment-buffer stream (enables watchdog suppression during rebuffering)
        """
        self.pipeline = pipeline
        self.loop = loop
        self.last_frame_time = {}
        self.stream0_decoupled_analysis_mode = bool(stream0_decoupled_analysis_mode)
        self.stream0_segment_buffer_mode = bool(stream0_segment_buffer_mode)
        self.stream0_segment_buffer_state_path = (
            Path(stream0_segment_buffer_state_path)
            if stream0_segment_buffer_state_path else None
        )
        self.stream0_analysis_last_time = None
        self.stream0_analysis_count = 0
        self.stream0_stage_last_time = {}
        self.stream0_stage_pts = {}
        self.stale_threshold_sec = 600  # 10 min watchdog
        self._frame_counts = {}       # {stream_id: int} — rolling 10s counter
        self._fps_log_interval = 5    # seconds (tightened for ~5min CP Plus drops)
        self._zero_fps_counts = {}    # {stream_id: int} — consecutive 0fps intervals
        self._zero_fps_limit = 1      # 1 × 10s = 10s of 0fps → restart
        self._stream_zero_fps_policy = stream_policies or {}  # {0: 'warn', 1: 'restart'}
        self._warn_safety_cap_sec = warn_safety_cap_sec  # max 0fps in warn mode before restart
        self._startup_time = time.time()  # Grace period: skip 0fps watchdog at boot
        self._stream0_startup_grace_sec = max(
            _STARTUP_GRACE_SEC,
            int(stream0_startup_grace_sec or _STARTUP_GRACE_SEC),
        )
        self._stream_startup_grace_overrides = dict(stream_startup_grace_overrides or {})
        self._stream_segment_buffer_state_paths = {
            sid: Path(p)
            for sid, p in (stream_segment_buffer_state_paths or {}).items()
            if p
        }
        self._healthcheck_url = (healthcheck_url or "").rstrip("/")
        self._rtsp_restart_stale_sec = max(1, int(rtsp_restart_stale_sec or 90))
        self._rtsp_restart_cooldown_sec = max(0, int(rtsp_restart_cooldown_sec or 60))
        self._rtsp_restart_backoff_sec = max(0, int(rtsp_restart_backoff_sec or 5))
        self._stream_restart_cb = stream_restart_cb
        self._restartable_stream_ids = set(restartable_stream_ids or [])
        self._last_stream_restart = {}
        self._pending_stream_restarts = set()

        # Per-source RTSP error timestamps for rate-limiting
        self._rtsp_errors = {}  # {source_name: [timestamp, ...]}

        # Set True when quitting due to error so hicon_pipeline can sys.exit(1)
        # for systemd Restart=on-failure to trigger. EOS/SIGTERM stay False (clean exit).
        self.fatal_exit = False

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        logger.info("Bus handler attached")
        if self.stream0_segment_buffer_mode:
            logger.info(
                "Stream 0 segment buffer watchdog suppression enabled "
                "(state=%s, startup_grace=%ss)",
                self.stream0_segment_buffer_state_path,
                self._stream0_startup_grace_sec,
            )
        for sid, grace in self._stream_startup_grace_overrides.items():
            logger.info("Stream %s: startup grace extended to %ss", sid, grace)
        if self._healthcheck_url:
            logger.info(f"Healthcheck enabled: {self._healthcheck_url}")
        if self._restartable_stream_ids:
            logger.info(
                "Per-stream restart enabled for streams=%s (stale=%ss cooldown=%ss backoff=%ss)",
                sorted(self._restartable_stream_ids),
                self._rtsp_restart_stale_sec,
                self._rtsp_restart_cooldown_sec,
                self._rtsp_restart_backoff_sec,
            )

    def _stream_startup_grace_sec(self, stream_id: int) -> int:
        if stream_id in self._stream_startup_grace_overrides:
            return self._stream_startup_grace_overrides[stream_id]
        if stream_id == 0 and self.stream0_segment_buffer_mode:
            return self._stream0_startup_grace_sec
        return _STARTUP_GRACE_SEC

    def _read_stream0_segment_buffer_state(self) -> dict | None:
        if not self.stream0_segment_buffer_mode or self.stream0_segment_buffer_state_path is None:
            return None
        try:
            payload = self.stream0_segment_buffer_state_path.read_text(encoding="utf-8")
            data = json.loads(payload)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _stream0_watchdog_suppressed(self) -> bool:
        state = self._read_stream0_segment_buffer_state()
        if not state:
            return False
        return state.get("mode") in {"buffering", "rebuffering"}

    def _segment_buffer_watchdog_suppressed(self, stream_id: int) -> bool:
        """Return True if stream_id's segment buffer is buffering/rebuffering."""
        state_path = self._stream_segment_buffer_state_paths.get(stream_id)
        if state_path is None:
            return False
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        return isinstance(data, dict) and data.get("mode") in {"buffering", "rebuffering"}

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

    @staticmethod
    def _stream_id_from_name(src_name: str) -> int:
        if src_name.startswith(_RTSP_SOURCE_PREFIX):
            suffix = src_name[len(_RTSP_SOURCE_PREFIX):]
            return int(suffix) if suffix.isdigit() else -1
        if src_name.startswith("mux_"):
            suffix = src_name.split("_", 1)[1]
            return int(suffix) if suffix.isdigit() else -1
        return -1

    def _schedule_stream_restart(self, stream_id: int, reason: str) -> bool:
        if stream_id not in self._restartable_stream_ids or self._stream_restart_cb is None:
            return False

        now = time.time()
        if stream_id in self._pending_stream_restarts:
            logger.warning("[RTSP-RESTART] Stream %s restart already pending (%s)", stream_id, reason)
            return True

        last_restart = self._last_stream_restart.get(stream_id)
        if last_restart is not None:
            elapsed = now - last_restart
            if elapsed < self._rtsp_restart_cooldown_sec:
                logger.warning(
                    "[RTSP-RESTART] Stream %s restart suppressed by cooldown (%.1fs < %ss): %s",
                    stream_id,
                    elapsed,
                    self._rtsp_restart_cooldown_sec,
                    reason,
                )
                return True

        self._pending_stream_restarts.add(stream_id)
        delay_sec = self._rtsp_restart_backoff_sec
        logger.warning(
            "[RTSP-RESTART] Stream %s restart scheduled in %ss (%s)",
            stream_id,
            delay_sec,
            reason,
        )

        def _run_restart():
            try:
                restarted = bool(self._stream_restart_cb(stream_id, reason))
                if restarted:
                    self._last_stream_restart[stream_id] = time.time()
                    self._zero_fps_counts[stream_id] = 0
                    self.last_frame_time[stream_id] = time.time()
                else:
                    logger.error("[RTSP-RESTART] Stream %s restart callback failed (%s)", stream_id, reason)
            except Exception as exc:
                logger.error(
                    "[RTSP-RESTART] Stream %s restart raised: %s",
                    stream_id,
                    exc,
                    exc_info=True,
                )
            finally:
                self._pending_stream_restarts.discard(stream_id)
            return False

        GLib.timeout_add_seconds(delay_sec, _run_restart)
        return True

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
            src_name = message.src.get_name() if message.src else "unknown"
            stream_id = self._stream_id_from_name(src_name)
            if self._schedule_stream_restart(stream_id, f"EOS from {src_name}"):
                logger.warning("[RTSP-EOS] Recoverable EOS from %s; stream restart scheduled", src_name)
                return
            logger.error("Unexpected end of stream received from %s", src_name)
            self._ping_healthcheck("/fail")
            self.fatal_exit = True
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
                    stream_id = self._stream_id_from_name(src_name)
                    if self._schedule_stream_restart(stream_id, f"RTSP error rate from {src_name}"):
                        return
                    # Rate limit exceeded — check per-stream policy
                    try:
                        stream_id = int(src_name.replace(_RTSP_SOURCE_PREFIX, ''))
                    except ValueError:
                        stream_id = -1
                    policy = self._stream_zero_fps_policy.get(stream_id, 'restart')
                    if policy == 'warn':
                        logger.warning(
                            f"[RTSP-RATE] {src_name}: rate limit exceeded "
                            f"but policy=warn, continuing"
                        )
                    else:
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

    def update_stream0_analysis_time(self):
        """Call from the Stream 0 CPU analysis branch to track side-branch liveness."""
        self.stream0_analysis_last_time = time.time()
        self.stream0_analysis_count += 1

    def update_stream0_stage_time(self, stage_name):
        """Track liveness of a specific Stream 0 pipeline stage."""
        self.update_stream0_stage_sample(stage_name, None)

    def update_stream0_stage_sample(self, stage_name, pts_ns):
        """Track liveness and latest PTS delta of a specific Stream 0 pipeline stage."""
        self.stream0_stage_last_time[stage_name] = time.time()
        if pts_ns is None or pts_ns == Gst.CLOCK_TIME_NONE:
            return

        prev = self.stream0_stage_pts.get(stage_name)
        delta_ns = None
        regressed = False
        if prev and prev.get("last_pts_ns") is not None:
            delta_ns = pts_ns - prev["last_pts_ns"]
            regressed = delta_ns < 0

        self.stream0_stage_pts[stage_name] = {
            "last_pts_ns": pts_ns,
            "delta_ns": delta_ns,
            "regressed": regressed,
        }

    @staticmethod
    def _format_pts_delta(delta_ns):
        """Render a PTS delta in milliseconds for diagnostic logging."""
        if delta_ns is None:
            return "n/a"
        return f"{delta_ns / 1_000_000.0:.2f}ms"

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
            now = time.time()
            for sid in sorted(self._frame_counts):
                count = self._frame_counts[sid]
                fps = count / self._fps_log_interval
                parts.append(f"Stream {sid}: {count} frames ({fps:.1f} fps)")
                self._frame_counts[sid] = 0  # reset for next interval

                # Dead stream detection: N consecutive 0fps intervals
                # Skip during startup grace period (network may not be ready at boot)
                if count == 0 and sid in self.last_frame_time and (
                    now - self._startup_time
                ) >= self._stream_startup_grace_sec(sid):
                    if (
                        (sid == 0 and self._stream0_watchdog_suppressed())
                        or self._segment_buffer_watchdog_suppressed(sid)
                    ):
                        self._zero_fps_counts[sid] = 0
                        continue
                    self._zero_fps_counts[sid] = self._zero_fps_counts.get(sid, 0) + 1
                    if self._zero_fps_counts[sid] >= self._zero_fps_limit:
                        stall_sec = self._zero_fps_counts[sid] * self._fps_log_interval
                        if sid in self._restartable_stream_ids:
                            logger.warning(
                                f"[FPS-WATCHDOG] Stream {sid} at 0fps for {stall_sec}s — "
                                f"waiting for per-stream restart threshold"
                            )
                            if stall_sec >= self._rtsp_restart_stale_sec:
                                if self._schedule_stream_restart(
                                    sid,
                                    f"0fps for {stall_sec}s",
                                ):
                                    self._zero_fps_counts[sid] = 0
                            continue
                        policy = self._stream_zero_fps_policy.get(sid, 'restart')
                        if policy == 'warn':
                            logger.warning(
                                f"[FPS-WATCHDOG] Stream {sid} at 0fps for "
                                f"{stall_sec}s — policy=warn, waiting for recovery"
                            )
                            # Safety cap: stale even in warn mode → restart
                            # Default 90s for segment buffer; raised to 300s for nvurisrcbin
                            if stall_sec >= self._warn_safety_cap_sec:
                                logger.critical(
                                    f"[FPS-WATCHDOG] Stream {sid} stale {stall_sec}s — escalating to restart"
                                )
                                self._ping_healthcheck("/fail")
                                self.fatal_exit = True
                                self.loop.quit()
                                return False
                        else:
                            logger.info("[FPS] " + " | ".join(parts))
                            logger.critical(
                                f"[FPS-WATCHDOG] Stream {sid} at 0fps for "
                                f"{stall_sec}s — restarting"
                            )
                            self._ping_healthcheck("/fail")
                            self.fatal_exit = True
                            self.loop.quit()
                            return False
                else:
                    if self._zero_fps_counts.get(sid, 0) > 0:
                        stall_sec = self._zero_fps_counts[sid] * self._fps_log_interval
                        logger.info(f"[FPS-WATCHDOG] Stream {sid} recovered after {stall_sec}s stall")
                    self._zero_fps_counts[sid] = 0

            logger.info("[FPS] " + " | ".join(parts))
            if self.stream0_decoupled_analysis_mode and 0 in self.last_frame_time:
                main_last = self.last_frame_time.get(0)
                analysis_last = self.stream0_analysis_last_time
                main_age = f"{(now - main_last):.2f}s" if main_last else "n/a"
                analysis_age = f"{(now - analysis_last):.2f}s" if analysis_last else "n/a"
                logger.info(f"[S0-DIAG] main_age={main_age} analysis_age={analysis_age}")
                self.stream0_analysis_count = 0
            if self.stream0_stage_last_time:
                stage_parts = []
                for stage_name in _STREAM0_STAGE_ORDER:
                    last_t = self.stream0_stage_last_time.get(stage_name)
                    age = f"{(now - last_t):.2f}s" if last_t else "n/a"
                    stage_parts.append(f"{stage_name}_age={age}")
                logger.info("[S0-STAGES] " + " ".join(stage_parts))
            if self.stream0_stage_pts:
                pts_parts = []
                for stage_name in _STREAM0_UPSTREAM_PTS_ORDER:
                    state = self.stream0_stage_pts.get(stage_name) or {}
                    delta_ns = state.get("delta_ns")
                    pts_parts.append(
                        f"{stage_name}_pts_delta={self._format_pts_delta(delta_ns)}"
                    )
                logger.info("[S0-PTS] " + " ".join(pts_parts))
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
