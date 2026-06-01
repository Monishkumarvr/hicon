"""
Sentry integration for HiCon AI Vision Pipeline.

Call sentry_config.init() at the very top of hicon_pipeline.py, before
any other imports or logging setup, so all startup errors are captured.

Incident boundary
-----------------
SEND to Sentry  → startup failures, RTSP/stream errors, GStreamer errors,
                  API/sync failures, abnormal shutdowns, unexpected exceptions.
DO NOT SEND     → normal detections, routine state transitions (tapping ON/OFF,
                  pouring session start/end), expected sync activity.
"""

import os
import logging

logger = logging.getLogger(__name__)

_initialized = False


def init() -> None:
    global _initialized
    if _initialized:
        return

    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        logger.info("Sentry disabled (SENTRY_DSN not set)")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_logging = LoggingIntegration(
            level=logging.WARNING,     # Breadcrumbs from WARNING+
            event_level=logging.ERROR, # Send to Sentry from ERROR+
        )

        sentry_sdk.init(
            dsn=dsn,
            integrations=[sentry_logging],
            traces_sample_rate=0.0,    # No performance tracing — inference pipeline is latency-critical
            before_send=_before_send,
            send_default_pii=False,
            environment=os.environ.get("HICON_ENVIRONMENT", "production"),
            release=os.environ.get("HICON_RELEASE", "unknown"),
        )

        _initialized = True
        logger.info("Sentry initialised (DSN configured)")

    except Exception as exc:
        # Never let Sentry init crash the pipeline.
        logger.warning("Sentry init failed (continuing without it): %s", exc)


# ---------------------------------------------------------------------------
# Noise filter — keep Sentry clean
# ---------------------------------------------------------------------------

_IGNORE_SUBSTRINGS = (
    # GStreamer / RTSP internal noise that self-recovers
    "Internal data stream error",     # udpsrc RTP jitter — rtspsrc recovers
    "EOS dropped on rtspsrc pad",     # false rtpsession timeout — handled
    "rtcp-min-interval",              # routine RTCP config
    # NvBufSurface transient mmap failures from MJPEG probe under memory pressure
    "NvBufSurfaceMap function failed",
    "Failed to map buffer to CPU",
    "Failed to sync buffer to CPU",
    "NvMapMemCacheMaint",
    # Expected pipeline shutdown messages
    "Shutting down pipeline",
    "Sync thread stopped",
    # Expected per-stream RTSP reconnect activity
    "Could not write to resource",
    "No data from source since last",
    # Normal detection log noise occasionally logged at ERROR by third-party libs
    "ErrorType=",
)

_IGNORE_LOGGERS = {
    # High-frequency probe loggers — noisy at DEBUG, don't Sentry-spam
    "processors.brightness_processor",
    "processors.pyrometer_processor",
    "processors.pouring_processor",
    "processors.melting_meta_reader",
}


def _before_send(event, hint):
    """Drop events that are expected operational noise."""
    # Filter by logger name
    logger_name = event.get("logger", "")
    if logger_name in _IGNORE_LOGGERS:
        return None

    # Drop werkzeug malformed-request noise. External scanners / NVRs probing the
    # optional MJPEG HTTP port send non-HTTP payloads (TLS ClientHello, HTTP/0.9),
    # which werkzeug logs as "code 400" at ERROR level. These are benign — the
    # request is correctly rejected and no data is served. The MJPEG server is an
    # internal access-log server (app.logger disabled); genuine server faults
    # surface as Python exceptions on other loggers, not via werkzeug's request log.
    if logger_name == "werkzeug":
        return None

    # Filter by message content
    msg = ""
    if "logentry" in event:
        msg = event["logentry"].get("message", "")
    elif "exception" in event:
        values = event["exception"].get("values", [])
        if values:
            msg = values[-1].get("value", "")

    for substr in _IGNORE_SUBSTRINGS:
        if substr in msg:
            return None

    return event


# ---------------------------------------------------------------------------
# Manual capture helpers (call these at incident sites)
# ---------------------------------------------------------------------------

def capture_pipeline_error(message: str, **extras) -> None:
    """Capture a non-exception pipeline incident (stream stall, watchdog, etc.)."""
    if not _initialized:
        return
    try:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            for k, v in extras.items():
                scope.set_extra(k, v)
            event_id = sentry_sdk.capture_message(message, level="error")
            logger.debug("Sentry event sent: %s (%s)", message, event_id)
    except Exception:
        pass


def capture_exception(exc: Exception, **extras) -> None:
    """Capture an exception at an operational incident site."""
    if not _initialized:
        return
    try:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            for k, v in extras.items():
                scope.set_extra(k, v)
            event_id = sentry_sdk.capture_exception(exc)
            logger.debug("Sentry exception sent: %s", event_id)
    except Exception:
        pass
