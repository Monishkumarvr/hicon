"""
MJPEG HTTP Streaming Server for HiCon Live Inference Monitoring

Provides HTTP endpoints for live MJPEG streams from DeepStream pipeline.
Runs in background thread, serves annotated frames from configured streams.

Usage:
    from streaming.mjpeg_server import MJPEGServer
    server = MJPEGServer(host='0.0.0.0', port=8080)
    server.start()

    # In pad probe:
    server.update_frame(stream_id=0, frame=annotated_bgr_frame)

    # Browser:
    # http://jetson-ip:8080/stream0
    # http://jetson-ip:8080/stream1
    # http://jetson-ip:8080/stream2
"""

import cv2
import time
import threading
import logging
from datetime import datetime
from flask import Flask, Response, render_template_string

logger = logging.getLogger(__name__)

_MJPEG_AGE_LOG_INTERVAL_S = 10.0


class MJPEGServer:
    """
    Multi-stream MJPEG server for live inference monitoring.

    Serves annotated frames from DeepStream pipeline as MJPEG streams
    accessible via HTTP (no plugins required, works in any browser).
    """

    def __init__(
        self,
        host='0.0.0.0',
        port=8080,
        jpeg_quality=85,
        max_fps=30,
        timestamp_overlay=False,
        demand_driven=True,
        idle_grace_sec=5.0,
    ):
        """
        Initialize MJPEG server.

        Args:
            host: Bind address (0.0.0.0 for all interfaces)
            port: HTTP port
            jpeg_quality: JPEG compression quality (0-100)
            max_fps: Maximum FPS for stream (throttles to save bandwidth)
            timestamp_overlay: Whether to draw source timestamp / age on frames
        """
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.frame_delay = 1.0 / max_fps if max_fps > 0 else 0.0
        self.timestamp_overlay = bool(timestamp_overlay)
        self.demand_driven = bool(demand_driven)
        self.idle_grace_sec = max(0.0, float(idle_grace_sec))

        # Frame storage per stream
        self.frames = {}    # stream_id → (frame_bgr, timestamp)
        self.locks = {}     # stream_id → threading.Lock
        self.enabled = {}   # stream_id → bool
        self._age_log_last_time = {}
        self._frame_counts = {}  # stream_id → int (for fps estimate)
        self._active_clients = {}
        self._last_client_disconnect = {}
        self._snapshot_demand_until = {}

        # Flask app
        self.app = Flask(__name__)
        self.app.logger.disabled = True  # Suppress Flask logs

        # Register routes
        self.app.add_url_rule('/stream<int:stream_id>', 'stream',
                              self._stream_route, methods=['GET'])
        self.app.add_url_rule('/snapshot<int:stream_id>', 'snapshot',
                              self._snapshot_route, methods=['GET'])
        self.app.add_url_rule('/api/status', 'status',
                              self._status_route, methods=['GET'])
        self.app.add_url_rule('/api/stream/<int:stream_id>/enable', 'enable',
                              self._enable_route, methods=['POST'])
        self.app.add_url_rule('/api/stream/<int:stream_id>/disable', 'disable',
                              self._disable_route, methods=['POST'])
        self.app.add_url_rule('/', 'index', self._index_route, methods=['GET'])

        # Background thread
        self.thread = None
        self.running = False

        logger.info(f"MJPEG server initialized: http://{host}:{port}/")

    def register_stream(self, stream_id):
        """Register a new stream ID (call before updating frames)."""
        if stream_id not in self.frames:
            self.frames[stream_id] = (None, 0.0)
            self.locks[stream_id] = threading.Lock()
            self.enabled[stream_id] = True
            self._age_log_last_time[stream_id] = 0.0
            self._frame_counts[stream_id] = 0
            self._active_clients[stream_id] = 0
            self._last_client_disconnect[stream_id] = 0.0
            self._snapshot_demand_until[stream_id] = 0.0
            logger.info(f"Registered stream {stream_id}")

    def has_active_subscribers(self, stream_id):
        """Whether the pipeline should pay the cost of extracting this preview."""
        if not self.demand_driven:
            return True
        if stream_id not in self.locks or not self.enabled.get(stream_id, True):
            return False

        now = time.monotonic()
        with self.locks[stream_id]:
            clients = self._active_clients.get(stream_id, 0)
            disconnected = self._last_client_disconnect.get(stream_id, 0.0)
            snapshot_until = self._snapshot_demand_until.get(stream_id, 0.0)
        return bool(
            clients > 0
            or now <= snapshot_until
            or (disconnected > 0 and (now - disconnected) <= self.idle_grace_sec)
        )

    def _client_connected(self, stream_id):
        with self.locks[stream_id]:
            self._active_clients[stream_id] = self._active_clients.get(stream_id, 0) + 1
        logger.info("MJPEG stream %s client connected (clients=%d)", stream_id,
                    self._active_clients[stream_id])

    def _client_disconnected(self, stream_id):
        with self.locks[stream_id]:
            self._active_clients[stream_id] = max(
                0, self._active_clients.get(stream_id, 0) - 1
            )
            self._last_client_disconnect[stream_id] = time.monotonic()
            clients = self._active_clients[stream_id]
        logger.info("MJPEG stream %s client disconnected (clients=%d)", stream_id, clients)

    def request_snapshot_frame(self, stream_id, timeout_sec=1.0):
        """Temporarily request extraction so an idle demand-driven stream can snapshot."""
        if stream_id not in self.locks:
            return
        with self.locks[stream_id]:
            self._snapshot_demand_until[stream_id] = max(
                self._snapshot_demand_until.get(stream_id, 0.0),
                time.monotonic() + max(0.1, float(timeout_sec)),
            )

    def update_frame(self, stream_id, frame_bgr):
        """
        Update frame for a stream (call from pad probe).

        Args:
            stream_id: Stream identifier (0, 1, etc.)
            frame_bgr: Annotated BGR frame (numpy array)
        """
        if stream_id not in self.locks:
            self.register_stream(stream_id)

        if not self.enabled.get(stream_id, True):
            return  # stream disabled — drop frame

        # Ownership is transferred: the producer creates a fresh array and never
        # mutates it after this call. Holding the NumPy reference is therefore safe
        # and avoids another full-frame allocation/copy on the streaming thread.
        with self.locks[stream_id]:
            self.frames[stream_id] = (frame_bgr, time.time())
            self._frame_counts[stream_id] = self._frame_counts.get(stream_id, 0) + 1

    def get_latest_frame_age(self, stream_id):
        """Return age in seconds for the newest cached frame of a stream."""
        _frame, _timestamp, frame_age = self._get_frame_snapshot(stream_id)
        return frame_age

    def _get_frame_snapshot(self, stream_id):
        """Return the latest cached frame, source timestamp, and age."""
        if stream_id not in self.locks:
            return None, 0.0, None

        with self.locks[stream_id]:
            frame, timestamp = self.frames.get(stream_id, (None, 0.0))

        if frame is None:
            return None, timestamp, None

        frame_age = max(0.0, time.time() - timestamp) if timestamp else None
        return frame, timestamp, frame_age

    def _maybe_log_frame_age(self, stream_id, frame_age, frame_timestamp):
        """Emit occasional debug logs for preview latency diagnostics."""
        if frame_age is None or not logger.isEnabledFor(logging.DEBUG):
            return

        now = time.monotonic()
        last_logged = self._age_log_last_time.get(stream_id, 0.0)
        if (now - last_logged) < _MJPEG_AGE_LOG_INTERVAL_S:
            return

        self._age_log_last_time[stream_id] = now
        ts_text = (
            datetime.fromtimestamp(frame_timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if frame_timestamp else "n/a"
        )
        logger.debug(
            "MJPEG stream %s latest frame age=%.3fs source_ts=%s",
            stream_id,
            frame_age,
            ts_text,
        )

    def _render_frame(self, frame_bgr, frame_timestamp, frame_age):
        """Return the frame to encode, optionally decorated for live diagnostics."""
        if not self.timestamp_overlay:
            return frame_bgr

        rendered = frame_bgr.copy()
        ts_text = (
            datetime.fromtimestamp(frame_timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if frame_timestamp else "n/a"
        )
        age_text = f"{frame_age:.2f}s" if frame_age is not None else "n/a"
        lines = [
            f"SRC {ts_text}",
            f"AGE {age_text}",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                rendered,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                rendered,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 28
        return rendered

    def _generate_mjpeg(self, stream_id):
        """Generator yielding MJPEG frames for a stream."""
        last_emit = 0.0
        self._client_connected(stream_id)

        try:
            while True:
                now = time.time()

                # Throttle to max_fps
                if (now - last_emit) < self.frame_delay:
                    time.sleep(0.01)
                    continue

                if stream_id not in self.locks:
                    time.sleep(0.1)
                    continue

                frame, timestamp, frame_age = self._get_frame_snapshot(stream_id)

                if frame is None:
                    time.sleep(0.1)
                    continue

                self._maybe_log_frame_age(stream_id, frame_age, timestamp)
                render_frame = self._render_frame(frame, timestamp, frame_age)

                ret, jpeg = cv2.imencode('.jpg', render_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ret:
                    logger.error(f"Failed to encode JPEG for stream {stream_id}")
                    time.sleep(0.1)
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

                last_emit = now
        finally:
            self._client_disconnected(stream_id)

    def _stream_route(self, stream_id):
        """Flask route for /stream<id>."""
        if stream_id not in self.frames:
            return f"Stream {stream_id} not available", 404
        if not self.enabled.get(stream_id, True):
            return f"Stream {stream_id} is disabled", 503

        response = Response(
            self._generate_mjpeg(stream_id),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    def _snapshot_route(self, stream_id):
        """Flask route for /snapshot<id> — returns a single JPEG."""
        if stream_id not in self.frames:
            return f"Stream {stream_id} not available", 404
        if not self.enabled.get(stream_id, True):
            return f"Stream {stream_id} is disabled", 503

        previous_timestamp = self.frames.get(stream_id, (None, 0.0))[1]
        self.request_snapshot_frame(stream_id, timeout_sec=1.0)
        deadline = time.monotonic() + 1.0
        frame, timestamp, _ = self._get_frame_snapshot(stream_id)
        while time.monotonic() < deadline and (frame is None or timestamp <= previous_timestamp):
            time.sleep(0.02)
            frame, timestamp, _ = self._get_frame_snapshot(stream_id)
        if frame is None:
            return "No frame available yet", 503

        ret, jpeg = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ret:
            return "Failed to encode frame", 500

        from flask import make_response
        resp = make_response(jpeg.tobytes())
        resp.headers['Content-Type'] = 'image/jpeg'
        resp.headers['Cache-Control'] = 'no-store'
        resp.headers['X-Stream-Id'] = str(stream_id)
        resp.headers['X-Timestamp'] = str(timestamp)
        return resp

    def _status_route(self):
        """Flask route for /api/status — JSON status of all registered streams."""
        from flask import jsonify
        streams = []
        for sid in sorted(self.frames.keys()):
            frame, timestamp, age = self._get_frame_snapshot(sid)
            streams.append({
                'id': sid,
                'enabled': self.enabled.get(sid, True),
                'has_frame': frame is not None,
                'frame_age_s': round(age, 2) if age is not None else None,
                'total_frames': self._frame_counts.get(sid, 0),
                'active_clients': self._active_clients.get(sid, 0),
                'demand_active': self.has_active_subscribers(sid),
                'url': f'/stream{sid}',
                'snapshot_url': f'/snapshot{sid}',
            })
        return jsonify({'streams': streams, 'server': f'{self.host}:{self.port}'})

    def _enable_route(self, stream_id):
        """POST /api/stream/<id>/enable — enable MJPEG capture for stream."""
        from flask import jsonify
        if stream_id not in self.frames:
            return jsonify({'error': f'Stream {stream_id} not registered'}), 404
        self.enabled[stream_id] = True
        logger.info(f"Stream {stream_id} MJPEG enabled")
        return jsonify({'stream_id': stream_id, 'enabled': True})

    def _disable_route(self, stream_id):
        """POST /api/stream/<id>/disable — disable MJPEG capture for stream."""
        from flask import jsonify
        if stream_id not in self.frames:
            return jsonify({'error': f'Stream {stream_id} not registered'}), 404
        self.enabled[stream_id] = False
        logger.info(f"Stream {stream_id} MJPEG disabled")
        return jsonify({'stream_id': stream_id, 'enabled': False})

    def _index_route(self):
        """Flask route for / (index page with all streams, controls, and status)."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>HiCon Live Inference</title>
    <meta http-equiv="refresh" content="60">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #111; color: #eee; font-family: monospace; padding: 16px; }
        h1 { color: #0f0; text-align: center; margin-bottom: 16px; font-size: 1.3em; }
        .streams { display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }
        .card {
            background: #1a1a1a; border: 1px solid #333; padding: 10px;
            min-width: 320px; max-width: 640px; flex: 1;
        }
        .card-title {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 8px;
        }
        .card-title h2 { color: #0ff; font-size: 0.9em; }
        .controls { display: flex; gap: 6px; }
        .btn {
            padding: 3px 10px; font-size: 0.75em; cursor: pointer;
            border: 1px solid #555; background: #333; color: #eee;
            border-radius: 3px;
        }
        .btn:hover { background: #444; }
        .btn-enable { border-color: #0a0; color: #0f0; }
        .btn-disable { border-color: #a00; color: #f44; }
        .btn-snap { border-color: #55f; color: #aaf; }
        img { display: block; width: 100%; border: 1px solid #222; }
        .disabled-overlay {
            width: 100%; height: 180px; background: #000; color: #555;
            display: flex; align-items: center; justify-content: center; font-size: 1.2em;
        }
        .status { font-size: 0.7em; color: #666; margin-top: 4px; }
        .api { margin-top: 20px; text-align: center; font-size: 0.75em; color: #555; }
        .api a { color: #777; }
    </style>
</head>
<body>
    <h1>&#9650; HiCon Live Inference Monitoring</h1>
    <div class="streams">
        {% for sid in stream_ids %}
        <div class="card">
            <div class="card-title">
                <h2>{{ stream_names[sid] }}</h2>
                <div class="controls">
                    <button class="btn btn-snap"
                        onclick="window.open('/snapshot{{ sid }}','_blank')">&#128247; Snap</button>
                    {% if enabled[sid] %}
                    <button class="btn btn-disable"
                        onclick="fetch('/api/stream/{{ sid }}/disable',{method:'POST'}).then(()=>location.reload())">
                        &#9632; Disable</button>
                    {% else %}
                    <button class="btn btn-enable"
                        onclick="fetch('/api/stream/{{ sid }}/enable',{method:'POST'}).then(()=>location.reload())">
                        &#9654; Enable</button>
                    {% endif %}
                </div>
            </div>
            {% if enabled[sid] %}
            <img src="/stream{{ sid }}" alt="Stream {{ sid }}" loading="lazy">
            {% else %}
            <div class="disabled-overlay">Stream {{ sid }} disabled</div>
            {% endif %}
            <div class="status">
                Frames received: {{ frame_counts[sid] }} &nbsp;|&nbsp;
                <a href="/stream{{ sid }}" target="_blank">/stream{{ sid }}</a> &nbsp;|&nbsp;
                <a href="/snapshot{{ sid }}" target="_blank">snapshot</a>
            </div>
        </div>
        {% endfor %}
    </div>
    <div class="api">
        API: <a href="/api/status">/api/status</a> &nbsp;|&nbsp;
        POST /api/stream/&lt;id&gt;/enable|disable &nbsp;|&nbsp;
        Page auto-refreshes every 60s
    </div>
</body>
</html>"""

        _names = {
            0: "Stream 0 — Process Camera (Pouring / Tapping / Deslagging)",
            1: "Stream 1 — Pyrometer Camera (Rod Detection)",
            2: "Stream 2 — Furnace 1 Melting / Shared Spectro",
        }
        stream_ids = sorted(self.frames.keys())
        return render_template_string(
            html,
            stream_ids=stream_ids,
            stream_names={sid: _names.get(sid, f"Stream {sid}") for sid in stream_ids},
            enabled={sid: self.enabled.get(sid, True) for sid in stream_ids},
            frame_counts={sid: self._frame_counts.get(sid, 0) for sid in stream_ids},
        )

    def start(self):
        """Start MJPEG server in background thread."""
        if self.running:
            logger.warning("MJPEG server already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_flask, daemon=True)
        self.thread.start()

        logger.info(f"MJPEG server started: http://{self.host}:{self.port}/")
        logger.info(f"  Index page: http://{self.host}:{self.port}/")
        for sid in self.frames.keys():
            logger.info(f"  Stream {sid}: http://{self.host}:{self.port}/stream{sid}")

    def _run_flask(self):
        """Run Flask app (called in background thread)."""
        # Suppress werkzeug logs
        import logging as py_logging
        log = py_logging.getLogger('werkzeug')
        log.setLevel(py_logging.ERROR)

        self.app.run(host=self.host, port=self.port, threaded=True, debug=False)

    def stop(self):
        """Stop MJPEG server (not implemented — daemon thread dies with process)."""
        self.running = False
        logger.info("MJPEG server stopping (daemon thread)")
