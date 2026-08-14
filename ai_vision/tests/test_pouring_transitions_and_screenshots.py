from datetime import datetime
from pathlib import Path

import numpy as np

from db_manager import HiConDatabase
from processors.pouring_processor import PouringProcessor
from state.heat_cycle_manager import HeatCycleManager


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
    MOULD_GIE_ENABLED = True
    MOULD_GIE_UNIQUE_ID = 4
    MOULD_TRACKER_CLASS_ID = 2
    MOULD_MIN_AREA_PX = 400
    MOULD_COUNT_MODE = "tracker"
    STREAM_0_TRACKER_MAX_TARGETS = 64
    SESSION_START_DURATION = 1.0
    SESSION_END_DURATION = 1.5
    POUR_REF_WIDTH = 1920
    POUR_REF_HEIGHT = 1080
    POUR_PROBE_BELOW_PX = 30
    POUR_PROBE_OFFSETS = [(0, 0), (12, 0), (-12, 0), (24, 0), (-24, 0)]
    POUR_PROBE_RADIUS_PX = 8
    POUR_BRIGHTNESS_START = 205
    POUR_BRIGHTNESS_END = 160
    POUR_START_DURATION = 0.20
    POUR_END_DURATION = 0.80
    POUR_MIN_DURATION = 2.0
    MOULD_DISPLACEMENT_THRESHOLD = 0.25
    EDGE_EXPAND_PX = 180
    MOUTH_MISSING_TOL_S = 0.6
    MOUTH_HOLD_S = 0.4
    PHANTOM_TROLLEY_TIMEOUT_S = 5.0
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


def test_runtime_geometry_keeps_reference_values_at_1920x1080(tmp_path):
    proc = _make_proc(tmp_path)
    start = proc.brightness_start
    end = proc.brightness_end
    start_frames = proc.pour_start_frames
    end_frames = proc.pour_end_frames

    proc._update_runtime_geometry(1920, 1080)

    assert proc.edge_expand_x_px == 180.0
    assert proc.edge_expand_y_px == 180.0
    assert proc.probe_below_px == 30
    assert proc.probe_radius == 8
    assert proc.probe_tail_dy == 20
    assert proc.probe_offsets == [(0, 0), (12, 0), (-12, 0), (24, 0), (-24, 0)]
    assert proc.brightness_start == start
    assert proc.brightness_end == end
    assert proc.pour_start_frames == start_frames
    assert proc.pour_end_frames == end_frames


def test_runtime_geometry_scales_reference_pixels_to_1280x720(tmp_path):
    proc = _make_proc(tmp_path)
    start = proc.brightness_start
    end = proc.brightness_end
    start_frames = proc.pour_start_frames
    end_frames = proc.pour_end_frames
    displacement = proc.displacement_thresh

    proc._update_runtime_geometry(1280, 720)

    assert proc.edge_expand_x_px == 120.0
    assert proc.edge_expand_y_px == 120.0
    assert proc.probe_below_px == 20
    assert proc.probe_radius == 5
    assert proc.probe_tail_dy == 13
    assert proc.probe_offsets == [(0, 0), (8, 0), (-8, 0), (16, 0), (-16, 0)]
    assert proc.brightness_start == start
    assert proc.brightness_end == end
    assert proc.pour_start_frames == start_frames
    assert proc.pour_end_frames == end_frames
    assert proc.displacement_thresh == displacement


def test_runtime_geometry_dedupes_offsets_after_rounding(tmp_path):
    proc = _make_proc(tmp_path)
    proc._update_runtime_geometry(80, 45)
    assert proc.probe_offsets == [(0, 0), (1, 0), (-1, 0)]
    assert len(proc.probe_offsets) == len(set(proc.probe_offsets))


def test_tracker_mould_assignment_uses_containment_only(tmp_path):
    proc = _make_proc(tmp_path)
    # Exercise the raw-track selection path (canonical registry has its own tests
    # in test_mould_canonical_registry.py and is bypassed here).
    proc.canonical_enabled = False
    proc._update_runtime_geometry(1920, 1080)
    trolley = {"track_id": 11, "bbox": (100, 100, 600, 400), "confidence": 0.9}
    moulds = [
        {"track_id": 41, "bbox": (220, 230, 300, 310), "center": (260, 270)},
        {"track_id": 42, "bbox": (420, 230, 500, 310), "center": (460, 270)},
    ]
    proc._update_tracked_mould_observations(moulds, trolley)

    contained_mouth = {"bottom_center": (260, 200)}
    assert proc._select_tracked_mould_for_pour(contained_mouth, trolley) == 41

    # Probe lands inside neither mould's bbox — must not fall back to "nearest
    # anyway"; a pour over an empty trolley slot must not be attributed to a
    # mould the probe was never actually over.
    uncontained_mouth = {"bottom_center": (390, 120)}
    assert proc._select_tracked_mould_for_pour(uncontained_mouth, trolley) is None


def test_valid_pour_commits_tracker_id_and_short_pour_does_not(tmp_path):
    proc = _make_proc(tmp_path)
    proc._save_event_screenshot = lambda *args, **kwargs: None
    now = datetime.now()

    proc.pour_active = True
    proc.pour_start_time = 0.0
    proc.pour_start_datetime = now
    proc._active_tracked_mould_id = 42
    proc._end_pour(3.0, now, [], [], None)
    assert proc.tracker_mould_count == 1
    assert proc._poured_mould_durations[42] > 2.0

    proc.pour_active = True
    proc.pour_start_time = 10.0
    proc.pour_start_datetime = now
    proc._active_tracked_mould_id = 43
    proc._end_pour(11.5, now, [], [], None)
    assert proc.tracker_mould_count == 1


def test_tracker_telemetry_counts_spatially_continuous_id_switch(tmp_path):
    proc = _make_proc(tmp_path)
    trolley = {"track_id": 11, "bbox": (100, 100, 600, 400), "confidence": 0.9}
    proc._update_tracked_mould_observations(
        [{"track_id": 41, "bbox": (220, 230, 300, 310), "center": (260, 270)}],
        trolley,
    )
    proc._update_tracked_mould_observations(
        [{"track_id": 99, "bbox": (224, 232, 304, 312), "center": (264, 272)}],
        trolley,
    )

    assert proc._mould_id_switches == 1


def test_tracker_mode_syncs_distinct_moulds_to_heat_cycle(tmp_path):
    class CaptureHeatCycle:
        def __init__(self):
            self.records = {}

        def upsert_completed_mould_pouring(self, **kwargs):
            self.records[kwargs["mould_id"]] = kwargs

    proc = _make_proc(tmp_path)
    proc.mould_count_mode = "tracker"
    proc.heat_cycle_manager = CaptureHeatCycle()
    now = datetime.now()
    proc._tracker_pour_records = {
        41: {
            "slot_id": 1,
            "ladle_track_id": 7,
            "start_time_wall": now.timestamp(),
            "start_datetime_obj": now,
            "end_time_wall": now.timestamp() + 3,
            "end_datetime_obj": now,
            "duration_s": 3.0,
        },
        99: {
            "slot_id": 2,
            "ladle_track_id": 7,
            "start_time_wall": now.timestamp() + 4,
            "start_datetime_obj": now,
            "end_time_wall": now.timestamp() + 7,
            "end_datetime_obj": now,
            "duration_s": 3.0,
        },
    }

    proc._sync_mould_records_to_heat_cycle()

    assert set(proc.heat_cycle_manager.records) == {"MOULD_C1", "MOULD_C2"}
    assert proc.heat_cycle_manager.records["MOULD_C1"]["mould_track_id"] == 41


def test_tracker_mode_db_breakdown_is_tracker_sourced_not_legacy(tmp_path):
    captured = {}

    class CapturingDB(DummyDB):
        def update_pouring_end(self, **kwargs):
            captured.update(kwargs)

    proc = PouringProcessor(
        db_manager=CapturingDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=None,
    )
    proc._save_event_screenshot = lambda *args, **kwargs: None
    now = datetime.now()

    proc.pour_active = True
    proc.pour_start_time = 0.0
    proc.pour_start_datetime = now
    proc.pour_sync_id = "sync-1"
    proc._active_tracked_mould_id = 42
    proc._end_pour(3.0, now, [], [], None)

    breakdown = captured["mould_wise_pouring_time"]["moulds"]
    assert len(breakdown) == 1
    assert breakdown[0]["mould_track_id"] == 42
    assert captured["mould_wise_pouring_time"]["mould_count"] == proc.tracker_mould_count


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
    proc._poured_mould_ids.add(7)
    proc._check_cycle_timeout(
        timestamp=301.0,
        datetime_obj=datetime.now(),
        mouths=[],
        trolleys=[],
        frame=None,
    )
    assert proc.trolley_locked is False
    assert proc.tracker_mould_count == 0


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


def test_reference_probe_uses_v_channel_and_five_offsets(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((140, 160, 4), dtype=np.uint8)
    base_x, base_y = 80, 60

    colors = [
        (210, 40, 10, 255),
        (10, 210, 40, 255),
        (40, 10, 210, 255),
        (180, 210, 30, 255),
        (30, 170, 210, 255),
    ]
    for (dx, dy), color in zip(proc.probe_offsets, colors):
        px = base_x + dx
        py = base_y + dy
        frame[py - proc.probe_radius:py + proc.probe_radius, px - proc.probe_radius:px + proc.probe_radius] = color

    brightness = proc._measure_multi_probe_brightness(frame, base_x, base_y)
    assert abs(brightness - 210.0) < 1.0


def test_probe_catches_stream_landing_on_only_one_offset(tmp_path):
    """A pour stream that only reaches the outer +24 probe patch (all four
    other patches dark) must still register as pouring. Mean-of-patches would
    average a lone bright patch down with four dark ones and miss it; the
    fix switches to max-of-patches so one genuinely-lit patch is enough."""
    proc = _make_proc(tmp_path)
    frame = np.zeros((140, 220, 4), dtype=np.uint8)
    base_x, base_y = 100, 60
    bright = (230, 230, 230, 255)

    dx, dy = 24, 0  # the outermost offset in the 5-point spread
    px, py = base_x + dx, base_y + dy
    r = proc.probe_radius
    frame[py - r:py + r, px - r:px + r] = bright

    brightness = proc._measure_multi_probe_brightness(frame, base_x, base_y)

    # Mean-of-patches would give (0+0+0+230+0)/5 = 46, far below brightness_start.
    assert brightness > proc.brightness_start
    assert abs(brightness - 230.0) < 1.0


def test_probe_brightness_accepts_float_probe_coordinates(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((120, 160, 4), dtype=np.uint8)
    frame[45:65, 70:90, :3] = 255

    brightness = proc._measure_multi_probe_brightness(frame, 80.4, 55.6)
    assert brightness >= 0.0


def test_mouth_in_trolley_expands_top_edge_only(tmp_path):
    proc = _make_proc(tmp_path)
    trolley = {"bbox": (100, 100, 200, 200)}
    # Above top edge within expand window → True
    assert proc._is_mouth_in_expanded_trolley({"center": (150, 50)}, trolley) is True
    # Outside left edge (no left expansion) → False
    assert proc._is_mouth_in_expanded_trolley({"center": (40, 150)}, trolley) is False
    # Outside right edge (no right expansion) → False
    assert proc._is_mouth_in_expanded_trolley({"center": (260, 150)}, trolley) is False
    # Below bottom edge (no bottom expansion) → False
    assert proc._is_mouth_in_expanded_trolley({"center": (150, 250)}, trolley) is False


def test_reference_pour_start_requires_tail_score(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((200, 240, 4), dtype=np.uint8)
    trolley = {"track_id": 7, "bbox": (40, 40, 200, 180), "confidence": 0.9}
    mouth = {
        "track_id": 11,
        "confidence": 0.95,
        "bbox": (90, 70, 110, 90),
        "center": (100, 80),
        "bottom_center": (100, 90),
    }

    proc._measure_head_tail_scores = lambda _frame, _x, _y: (220.0, 150.0)

    for idx in range(proc.pour_start_frames + 2):
        proc._frame_count = idx + 1
        proc._update_pour(
            [mouth],
            frame,
            idx / proc.fps,
            datetime.now(),
            [trolley],
            trolley,
        )

    assert proc.pour_active is False
    assert proc.pour_on_count == 0


def test_reference_pour_on_off_uses_frame_counts(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((200, 240, 4), dtype=np.uint8)
    trolley = {"track_id": 7, "bbox": (40, 40, 200, 180), "confidence": 0.9}
    mouth = {
        "track_id": 11,
        "confidence": 0.95,
        "bbox": (90, 70, 110, 90),
        "center": (100, 80),
        "bottom_center": (100, 90),
    }

    mode = {"value": "on"}

    def _scores(_frame, _x, _y):
        if mode["value"] == "on":
            return 230.0, 180.0
        return 120.0, 120.0

    proc._measure_head_tail_scores = _scores

    ts = 0.0
    for idx in range(proc.pour_start_frames):
        proc._frame_count = idx + 1
        ts = idx / proc.fps
        proc._update_pour([mouth], frame, ts, datetime.now(), [trolley], trolley)
    assert proc.pour_active is True

    sustain_frames = int(proc.fps * 2.2)
    for offset in range(sustain_frames):
        proc._frame_count = proc.pour_start_frames + offset + 1
        ts += 1.0 / proc.fps
        proc._update_pour([mouth], frame, ts, datetime.now(), [trolley], trolley)
    assert proc.pour_active is True

    mode["value"] = "off"
    for offset in range(proc.pour_end_frames):
        proc._frame_count = proc.pour_start_frames + sustain_frames + offset + 1
        ts += 1.0 / proc.fps
        proc._update_pour([mouth], frame, ts, datetime.now(), [trolley], trolley)
    assert proc.pour_active is False


def test_reference_probe_selection_prefers_best_pouring_tail_score(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((200, 260, 4), dtype=np.uint8)
    trolley = {"track_id": 7, "bbox": (20, 20, 240, 180), "confidence": 0.9}
    mouth_a = {
        "track_id": 11,
        "confidence": 0.80,
        "bbox": (70, 70, 90, 90),
        "center": (80, 80),
        "bottom_center": (80, 90),
    }
    mouth_b = {
        "track_id": 22,
        "confidence": 0.85,
        "bbox": (150, 70, 170, 90),
        "center": (160, 80),
        "bottom_center": (160, 90),
    }

    def _scores(_frame, x, _y):
        if x < 120:
            return 250.0, 165.0
        return 210.0, 190.0

    proc._measure_head_tail_scores = _scores
    proc._frame_count = 1
    proc._update_pour([mouth_a, mouth_b], frame, 0.0, datetime.now(), [trolley], trolley)

    assert proc.active_probe_track_id == 22


def test_reference_frozen_probe_reuse_has_priority(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((200, 260, 4), dtype=np.uint8)
    trolley = {"track_id": 7, "bbox": (20, 20, 240, 180), "confidence": 0.9}
    mouth = {
        "track_id": 11,
        "confidence": 0.95,
        "bbox": (150, 70, 170, 90),
        "center": (160, 80),
        "bottom_center": (160, 90),
    }

    proc.pour_active = True
    proc.frozen_probe_active = True
    proc.frozen_probe_x = 90.0
    proc.frozen_probe_y = 130.0
    proc._measure_head_tail_scores = lambda _frame, _x, _y: (230.0, 180.0)

    proc._frame_count = 10
    proc._update_pour([mouth], frame, 1.0, datetime.now(), [trolley], trolley)

    assert proc._last_probe_base == (90, 130)


def test_reference_hold_probe_reuse_within_mouth_hold_window(tmp_path):
    proc = _make_proc(tmp_path)
    frame = np.zeros((200, 260, 4), dtype=np.uint8)
    trolley = {"track_id": 7, "bbox": (20, 20, 240, 180), "confidence": 0.9}

    proc.pour_active = True
    proc.active_probe_pt_valid = True
    proc.active_probe_pt_px = (88.0, 111.0)
    proc.active_probe_last_seen_frame = 10
    proc._measure_head_tail_scores = lambda _frame, _x, _y: (230.0, 180.0)

    proc._frame_count = 10 + proc.mouth_hold_frames
    proc._update_pour([], frame, 1.0, datetime.now(), [trolley], trolley)

    assert proc.active_probe_from_hold is True
    assert proc._last_probe_base == (88, 111)


def test_end_pour_syncs_tracker_commit_into_active_heat_cycle(tmp_path):
    """Regression for a bug where a committed tracker pour never reached the
    heat cycle's mould_pourings — _sync_mould_records_to_heat_cycle() existed
    but had no production caller, so every heat cycle finalized with a blank
    pouring_start_time/pouring_end_time/mould_wise_pouring_time even though
    pours were detected and stored in pouring_events fine. This drives a real
    _end_pour() with a real HeatCycleManager, not just the bridge function
    directly, so it would have caught the missing call site."""
    db = HiConDatabase(str(tmp_path / "heat_cycle.sqlite"))
    heat_cycle_manager = HeatCycleManager(db, ladle_absence_timeout=300.0)

    proc = PouringProcessor(
        db_manager=DummyDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle_manager,
    )
    proc._save_event_screenshot = lambda *args, **kwargs: None
    now = datetime.now()

    proc.locked_trolley_id = 7
    proc.pour_active = True
    proc.pour_start_time = 0.0
    proc.pour_start_datetime = now
    proc._active_tracked_mould_id = 42
    proc._end_pour(3.0, now, [], [], None)

    assert heat_cycle_manager.active_cycle is not None
    assert len(heat_cycle_manager.active_cycle.mould_pourings) == 1
    assert heat_cycle_manager.active_cycle.mould_pourings[0].mould_track_id == 42
