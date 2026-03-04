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
from flask import Flask, Response, render_template_string
from pathlib import Path

logger = logging.getLogger(__name__)


class MJPEGServer:
    """
    Multi-stream MJPEG server for live inference monitoring.

    Serves annotated frames from DeepStream pipeline as MJPEG streams
    accessible via HTTP (no plugins required, works in any browser).
    """

    def __init__(self, host='0.0.0.0', port=8080, jpeg_quality=85, max_fps=30):
        """
        Initialize MJPEG server.

        Args:
            host: Bind address (0.0.0.0 for all interfaces)
            port: HTTP port
            jpeg_quality: JPEG compression quality (0-100)
            max_fps: Maximum FPS for stream (throttles to save bandwidth)
        """
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.frame_delay = 1.0 / max_fps

        # Frame storage per stream
        self.frames = {}  # stream_id → (frame_bgr, timestamp)
        self.locks = {}   # stream_id → threading.Lock

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

            with self.locks[stream_id]:
                frame, timestamp = self.frames.get(stream_id, (None, 0.0))

            if frame is None:
                # No frame yet, send placeholder
                time.sleep(0.1)
                continue

            # Encode JPEG
            ret, jpeg = cv2.imencode('.jpg', frame,
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

        return Response(self._generate_mjpeg(stream_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

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
