"""
WebRTC streaming server backed by jetson-utils videoOutput.

Drop-in replacement for MJPEGServer — same public interface
(update_frame, has_active_subscribers, register_stream, start, stop).

Falls back to a silent no-op when jetson_utils is unavailable
(dev machines, CI environments).

Individual stream URLs (after accepting self-signed cert once):
    https://<jetson-ip>:<WEBRTC_PORT>/stream0
    https://<jetson-ip>:<WEBRTC_PORT>/stream1
    https://<jetson-ip>:<WEBRTC_PORT>/stream2

Multi-stream dashboard (plain HTTP, no cert required):
    http://<jetson-ip>:<WEBRTC_DASHBOARD_PORT>/

SSL: jetson-utils auto-generates a self-signed cert at
~/.jetson-inference/ssl/ on first run. To use a custom cert,
set SSL_KEY and SSL_CERT environment variables before launch.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

try:
    import jetson_utils
    _JETSON_UTILS_AVAILABLE = True
except (ImportError, SystemError, AttributeError):
    _JETSON_UTILS_AVAILABLE = False
    logger.debug("jetson_utils not available — WebRTCServer will run as no-op")

_STREAM_NAMES = {
    0: "Stream 0 — Process Camera (Pouring / Tapping / Deslagging)",
    1: "Stream 1 — Pyrometer Camera (Rod Detection)",
    2: "Stream 2 — Pouring2",
}

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>HiCon Live Inference — WebRTC</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #111; color: #eee; font-family: monospace; padding: 16px; }
        h1 { color: #0f0; text-align: center; margin-bottom: 16px; font-size: 1.3em; }
        .streams { display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }
        .card {
            background: #1a1a1a; border: 1px solid #333; padding: 10px;
            min-width: 320px; max-width: 640px; flex: 1;
        }
        .card-title { margin-bottom: 8px; }
        .card-title h2 { color: #0ff; font-size: 0.9em; }
        iframe {
            display: block; width: 100%; height: 360px;
            border: 1px solid #222; background: #000;
        }
        .links { font-size: 0.7em; color: #555; margin-top: 4px; }
        .links a { color: #777; }
        .note {
            margin-top: 20px; text-align: center; font-size: 0.75em; color: #555;
        }
        .note a { color: #777; }
    </style>
</head>
<body>
    <h1>&#9650; HiCon Live Inference — WebRTC</h1>
    <div class="streams">
        {% for sid in stream_ids %}
        <div class="card">
            <div class="card-title">
                <h2>{{ stream_names[sid] }}</h2>
            </div>
            <iframe
                src="https://{{ host }}:{{ webrtc_port }}/stream{{ sid }}"
                allow="autoplay"
                allowfullscreen
            ></iframe>
            <div class="links">
                <a href="https://{{ host }}:{{ webrtc_port }}/stream{{ sid }}" target="_blank">
                    open full page &#8599;
                </a>
            </div>
        </div>
        {% endfor %}
    </div>
    <div class="note">
        If iframes are blank: visit each stream link above once to accept the self-signed cert,
        then reload this page. &nbsp;|&nbsp;
        Individual streams: {% for sid in stream_ids %}
        <a href="https://{{ host }}:{{ webrtc_port }}/stream{{ sid }}" target="_blank">stream{{ sid }}</a>{% if not loop.last %} &nbsp;{% endif %}
        {% endfor %}
    </div>
</body>
</html>"""


class WebRTCServer:
    """
    Multi-stream WebRTC server using jetson-utils videoOutput.

    Encodes annotated BGR frames via NVENC (hardware H.264) and serves
    them to browsers over WebRTC. The jetson-utils embedded webserver
    handles signalling and the browser UI for individual streams.

    A separate plain-HTTP Flask dashboard serves all streams in a grid
    at http://<host>:<dashboard_port>/.
    """

    def __init__(self, host='0.0.0.0', port=8554, max_fps=25,
                 dashboard_port=8555, **kwargs):
        """
        Args:
            host: Bind address
            port: WebRTC signalling + media port (HTTPS)
            max_fps: Maximum frames rendered per stream per second
            dashboard_port: Plain HTTP port for the multi-stream grid dashboard
            **kwargs: Absorbs unused MJPEGServer params (jpeg_quality,
                      demand_driven, idle_grace_sec, timestamp_overlay)
        """
        self.host = host
        self.port = port
        self.dashboard_port = dashboard_port
        self.frame_delay = 1.0 / max_fps if max_fps > 0 else 0.0

        self._outputs = {}         # stream_id → jetson_utils.videoOutput
        self._enabled = {}         # stream_id → bool
        self._last_emit = {}       # stream_id → float (monotonic timestamp)
        self._stream_ids = []      # ordered list for dashboard
        self._lock = threading.Lock()

    def register_stream(self, stream_id):
        """Register a stream and create its videoOutput endpoint."""
        if not _JETSON_UTILS_AVAILABLE:
            return
        with self._lock:
            if stream_id in self._outputs:
                return
            uri = f"webrtc://@:{self.port}/stream{stream_id}"
            output = jetson_utils.videoOutput(uri)
            self._outputs[stream_id] = output
            self._enabled[stream_id] = True
            self._last_emit[stream_id] = 0.0
            self._stream_ids.append(stream_id)
        logger.info("WebRTC registered stream %s → %s", stream_id, uri)

    def update_frame(self, stream_id, frame_bgr):
        """
        Push an annotated BGR frame to the WebRTC stream.

        Args:
            stream_id: Stream identifier (0, 1, 2)
            frame_bgr: Annotated BGR numpy array (from pad probe)
        """
        if not _JETSON_UTILS_AVAILABLE or frame_bgr is None:
            return
        if not self._enabled.get(stream_id, False):
            return

        now = time.monotonic()
        if (now - self._last_emit.get(stream_id, 0.0)) < self.frame_delay:
            return

        with self._lock:
            output = self._outputs.get(stream_id)
        if output is None:
            return

        try:
            cuda_img = jetson_utils.cudaFromNumpy(frame_bgr)
            output.Render(cuda_img)
            self._last_emit[stream_id] = now
        except Exception:
            logger.exception("WebRTC Render failed for stream %s", stream_id)

    def has_active_subscribers(self, stream_id):
        """
        Always returns True for enabled streams.

        jetson-utils manages its own client tracking internally. NVENC idle
        cost when no browser is connected is negligible, so we skip the
        demand-driven guard complexity used by the MJPEG server.
        """
        return self._enabled.get(stream_id, False)

    def start(self):
        """Start the WebRTC server and the multi-stream dashboard."""
        if not _JETSON_UTILS_AVAILABLE:
            logger.warning("WebRTC server: jetson_utils not available, streaming disabled")
            return
        logger.info(
            "WebRTC server ready — open https://<jetson-ip>:%d/stream<id> in browser",
            self.port,
        )
        self._start_dashboard()

    def _start_dashboard(self):
        """Start a plain-HTTP Flask server serving the multi-stream grid dashboard."""
        try:
            from flask import Flask, render_template_string
        except ImportError:
            logger.warning("Flask not available — WebRTC dashboard disabled")
            return

        import logging as _pylog
        _pylog.getLogger('werkzeug').setLevel(_pylog.ERROR)

        server = self

        app = Flask(__name__)
        app.logger.disabled = True

        @app.route('/')
        def index():
            with server._lock:
                stream_ids = sorted(server._stream_ids)
            # Use the actual host if bound to a specific IP, otherwise let the
            # browser use whatever hostname it used to reach the dashboard.
            _host = server.host if server.host != '0.0.0.0' else 'localhost'
            return render_template_string(
                _DASHBOARD_HTML,
                stream_ids=stream_ids,
                stream_names=_STREAM_NAMES,
                host=_host,
                webrtc_port=server.port,
            )

        threading.Thread(
            target=lambda: app.run(
                host=self.host,
                port=self.dashboard_port,
                threaded=True,
                debug=False,
            ),
            daemon=True,
            name='webrtc-dashboard',
        ).start()
        logger.info(
            "WebRTC dashboard: http://<jetson-ip>:%d/",
            self.dashboard_port,
        )

    def stop(self):
        """Stop the server (videoOutput cleans up on garbage collection)."""
        with self._lock:
            self._outputs.clear()
