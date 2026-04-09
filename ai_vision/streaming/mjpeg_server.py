"""
MJPEG HTTP Streaming Server for HiCon Live Inference Monitoring

Provides HTTP endpoints for live MJPEG streams from DeepStream pipeline.
Runs in background thread, serves annotated frames from both streams.

Usage:
    from streaming.mjpeg_server import MJPEGServer
    server = MJPEGServer(host='0.0.0.0', port=8080)
    server.start()

    # In pad probe:
    server.update_frame(stream_id=0, frame=annotated_bgr_frame)

    # Browser:
    # http://jetson-ip:8080/stream0
    # http://jetson-ip:8080/stream1
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
        self.frame_delay = 1.0 / max_fps
        self.timestamp_overlay = bool(timestamp_overlay)

        # Frame storage per stream
        self.frames = {}  # stream_id → (frame_bgr, timestamp)
        self.locks = {}   # stream_id → threading.Lock
        self._age_log_last_time = {}

        # Flask app
        self.app = Flask(__name__)
        self.app.logger.disabled = True  # Suppress Flask logs

        # Register routes
        self.app.add_url_rule('/stream<int:stream_id>', 'stream',
                              self._stream_route, methods=['GET'])
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
            self._age_log_last_time[stream_id] = 0.0
            logger.info(f"Registered stream {stream_id}")

    def update_frame(self, stream_id, frame_bgr):
        """
        Update frame for a stream (call from pad probe).

        Args:
            stream_id: Stream identifier (0, 1, etc.)
            frame_bgr: Annotated BGR frame (numpy array)
        """
        if stream_id not in self.locks:
            self.register_stream(stream_id)

        with self.locks[stream_id]:
            self.frames[stream_id] = (frame_bgr.copy(), time.time())

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
                # No frame yet, send placeholder
                time.sleep(0.1)
                continue

            self._maybe_log_frame_age(stream_id, frame_age, timestamp)
            render_frame = self._render_frame(frame, timestamp, frame_age)

            # Encode JPEG
            ret, jpeg = cv2.imencode('.jpg', render_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ret:
                logger.error(f"Failed to encode JPEG for stream {stream_id}")
                time.sleep(0.1)
                continue

            # Yield MJPEG frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            last_emit = now

    def _stream_route(self, stream_id):
        """Flask route for /stream<id>."""
        if stream_id not in self.frames:
            return f"Stream {stream_id} not available", 404

        response = Response(
            self._generate_mjpeg(stream_id),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    def _index_route(self):
        """Flask route for / (index page with all streams)."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HiCon Live Inference</title>
            <style>
                body {
                    margin: 0;
                    padding: 20px;
                    background: #1a1a1a;
                    color: #fff;
                    font-family: Arial, sans-serif;
                }
                h1 {
                    text-align: center;
                    color: #00ff00;
                }
                .streams {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }
                .stream-container {
                    border: 2px solid #00ff00;
                    padding: 10px;
                    background: #000;
                }
                .stream-container h2 {
                    margin: 0 0 10px 0;
                    color: #00ffff;
                }
                img {
                    display: block;
                    max-width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <h1>HiCon Live Inference Monitoring</h1>
            <div class="streams">
                {% for sid in stream_ids %}
                <div class="stream-container">
                    <h2>{{ stream_names[sid] }}</h2>
                    <img src="/stream{{ sid }}" alt="Stream {{ sid }}">
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """

        _names = {
            0: "Process Camera (Pouring + Tapping + Deslagging)",
            1: "Pyrometer Camera (Rod Detection)",
            2: "Second Pouring Camera",
        }
        stream_ids = list(self.frames.keys())
        stream_names = {sid: _names.get(sid, f"Stream {sid}") for sid in stream_ids}

        return render_template_string(html,
                                       stream_ids=stream_ids,
                                       stream_names=stream_names)

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
