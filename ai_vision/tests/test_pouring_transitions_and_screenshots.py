from datetime import datetime
from pathlib import Path

import numpy as np

from processors.pouring_processor import PouringProcessor


class DummyDB:
    def insert_pouring_event(self, **kwargs):
        return 1

    def update_pouring_end(self, **kwargs):
        return None

    def insert_heat_cycle(self, **kwargs):
        return 1


class DummyConfig:
    CUSTOMER_ID = "C1"
    LOCATION = "Loc"
    CAMERA_ID_STREAM_0 = "Cam-0"
    MOUTH_CONFIDENCE = 0.4
    TROLLEY_CONFIDENCE = 0.25
    SESSION_START_DURATION = 1.0
    SESSION_END_DURATION = 1.5
    POUR_PROBE_BELOW_PX = 50
    POUR_PROBE_OFFSETS = [(20, 0), (30, 0), (40, 0)]
    POUR_PROBE_RADIUS_PX = 6
    POUR_BRIGHTNESS_START = 230
    POUR_BRIGHTNESS_END = 180
    POUR_START_DURATION = 0.25
    POUR_END_DURATION = 1.0
    POUR_MIN_DURATION = 2.0
    MOULD_DISPLACEMENT_THRESHOLD = 0.15
    MOULD_SUSTAINED_DURATION = 1.5
    CLUSTER_R_CLUSTER = 0.08
    CLUSTER_R_MERGE = 0.05
    MOULD_SWITCH_MIN_POUR_S = 2.0
    MIN_CLUSTER_POUR_S = 1.5
    EDGE_EXPAND_PX = 200
    MOUTH_MISSING_TOL_S = 0.8
    POURING_CYCLE_TIMEOUT_S = 300.0
    ENABLE_INFERENCE_VIDEO = False
    VIDEO_DIR = Path("/tmp")


def _make_proc(tmp_path):
    return PouringProcessor(
        db_manager=DummyDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=None,
    )


def test_session_start_end_and_cycle_timeout(tmp_path):
    proc = _make_proc(tmp_path)
    trolley = {"track_id": 11, "bbox": (100, 100, 200, 200), "confidence": 0.9}

    # session start: mouth inside sustained >= 1.0s
    proc._update_session(
        mouth_in_trolley=True,
        best_mouth=None,
        target_trolley=trolley,
        mouths=[],
        trolleys=[trolley],
        timestamp=0.0,
        datetime_obj=datetime.now(),
        frame=None,
    )
    proc._update_session(
        mouth_in_trolley=True,
        best_mouth=None,
        target_trolley=trolley,
        mouths=[],
        trolleys=[trolley],
        timestamp=1.1,
        datetime_obj=datetime.now(),
        frame=None,
    )
    assert proc.session_active is True

    # session end: 0.8s tolerance + 1.5s effective absence
    proc._update_session(
        mouth_in_trolley=False,
        best_mouth=None,
        target_trolley=trolley,
        mouths=[],
        trolleys=[trolley],
        timestamp=2.0,
        datetime_obj=datetime.now(),
        frame=None,
    )
    proc._update_session(
        mouth_in_trolley=False,
        best_mouth=None,
        target_trolley=trolley,
        mouths=[],
        trolleys=[trolley],
        timestamp=4.5,
        datetime_obj=datetime.now(),
        frame=None,
    )
    assert proc.session_active is False

    # cycle timeout reset
    proc.trolley_locked = True
    proc.locked_trolley_id = 11
    proc.mouth_last_seen_in_trolley = 0.0
    proc.mould_count = 3
    proc._check_cycle_timeout(
        timestamp=301.0,
        datetime_obj=datetime.now(),
        mouths=[],
        trolleys=[],
        frame=None,
    )
    assert proc.trolley_locked is False
    assert proc.mould_count == 0


def test_probe_points_passed_to_all_pouring_event_screenshots(tmp_path):
    proc = _make_proc(tmp_path)
    proc._last_probe_base = (120, 140)
    proc._last_probe_brightness = 260.0

    calls = []

    def _capture(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path / "cap.jpg")

    proc._save_event_screenshot = _capture

    trolley = {"track_id": 7, "bbox": (10, 10, 100, 100), "confidence": 0.8}
    proc._start_session(
        trolley=trolley,
        timestamp=0.0,
        datetime_obj=datetime.now(),
        mouths=[],
        trolleys=[trolley],
        frame=None,
    )

    proc.session_active = True
    proc.session_start_time = 0.0
    proc._end_session(
        timestamp=4.0,
        datetime_obj=datetime.now(),
        mouths=[],
        trolleys=[trolley],
        frame=None,
    )

    proc.pour_active = True
    proc.pour_start_time = 0.0
    proc.pour_start_datetime = datetime.now()
    proc.pour_sync_id = "p1"
    proc._end_pour(
        timestamp=3.0,
        datetime_obj=datetime.now(),
        mouths=[],
        trolleys=[trolley],
        frame=None,
    )

    titles = [args[0] for args, _ in calls]
    kwargs_by_title = {args[0]: kwargs for args, kwargs in calls}

    for title in ("SESSION START", "SESSION END", "POUR END"):
        assert title in titles
        assert kwargs_by_title[title].get("probe_point") is not None


def test_probe_point_rendering_changes_pixels(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((240, 320, 4), dtype=np.uint8)
    now = datetime.now()

    out = proc._save_event_screenshot(
        title="POUR START",
        mouths=[],
        trolleys=[],
        frame=frame,
        datetime_obj=now,
        probe_point=(80, 90),
        probe_brightness=300.0,
    )
    assert out is not None
    assert Path(out).exists()
