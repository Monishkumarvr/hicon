import numpy as np

from streaming.mjpeg_server import MJPEGServer


def test_stream_route_sets_anti_buffering_headers():
    server = MJPEGServer(max_fps=15)
    server.register_stream(0)
    server.update_frame(0, np.zeros((48, 64, 3), dtype=np.uint8))

    client = server.app.test_client()
    response = client.get("/stream0", buffered=False)

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["Pragma"] == "no-cache"
    assert response.headers["Expires"] == "0"
    assert response.headers["X-Accel-Buffering"] == "no"

    first_chunk = next(response.response)
    assert first_chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
    response.close()


def test_get_latest_frame_age_reports_recent_frame():
    server = MJPEGServer(max_fps=15)
    server.register_stream(0)
    server.update_frame(0, np.zeros((24, 24, 3), dtype=np.uint8))

    frame_age = server.get_latest_frame_age(0)

    assert frame_age is not None
    assert 0.0 <= frame_age < 1.0


def test_render_frame_adds_timestamp_overlay_when_enabled():
    server = MJPEGServer(max_fps=15, timestamp_overlay=True)
    frame = np.zeros((96, 256, 3), dtype=np.uint8)

    rendered = server._render_frame(frame, 1_744_206_400.123, 0.75)

    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(frame) == 0


def test_demand_driven_stream_skips_idle_and_observes_grace_period(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("streaming.mjpeg_server.time.monotonic", lambda: now[0])
    server = MJPEGServer(max_fps=10, demand_driven=True, idle_grace_sec=5)
    server.register_stream(0)

    assert server.has_active_subscribers(0) is False
    server._client_connected(0)
    assert server.has_active_subscribers(0) is True
    server._client_disconnected(0)
    assert server.has_active_subscribers(0) is True
    now[0] = 106.0
    assert server.has_active_subscribers(0) is False


def test_update_frame_takes_ownership_without_copy():
    server = MJPEGServer(max_fps=10)
    server.register_stream(0)
    frame = np.zeros((12, 16, 3), dtype=np.uint8)

    server.update_frame(0, frame)

    cached, _timestamp, _age = server._get_frame_snapshot(0)
    assert cached is frame
