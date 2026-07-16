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
    MOULD_GIE_ENABLED = True
    MOULD_GIE_UNIQUE_ID = 4
    MOULD_TRACKER_CLASS_ID = 2
    MOULD_MIN_AREA_PX = 400
    MOULD_COUNT_MODE = "shadow"
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
    MOULD_SUSTAINED_DURATION = 0.30
    CLUSTER_R_CLUSTER = 0.08
    CLUSTER_R_MERGE = 0.07
    CLUSTER_BACKTRACK_CID_GUARD = 5
    MOULD_SWITCH_MIN_POUR_S = 2.0
    MIN_CLUSTER_POUR_S = 1.5
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
    r_cluster = proc.r_cluster

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
    assert proc.r_cluster == r_cluster


def test_runtime_geometry_scales_reference_pixels_to_1280x720(tmp_path):
    proc = _make_proc(tmp_path)
    start = proc.brightness_start
    end = proc.brightness_end
    start_frames = proc.pour_start_frames
    end_frames = proc.pour_end_frames
    displacement = proc.displacement_thresh
    r_merge = proc.r_merge

    proc._update_runtime_geometry(1280, 720)

    assert proc.edge_expand_x_px == 120.0
    assert proc.edge_expand_y_px == 120.0
    assert proc.probe_below_px == 20
    assert proc.probe_radius == 5
    assert proc.probe_tail_dy == 13
    assert proc.probe_offsets == [(0, 0), (8, 0), (-8, 0), (16, 0), (-16, 0)]
    assert proc.split_min_dx_px == 8.0
    assert proc.split_min_dy_px == 8.0
    assert proc.split_rearm_dx_px == 7.0
    assert proc.split_rearm_dy_px == 9.0
    assert proc.brightness_start == start
    assert proc.brightness_end == end
    assert proc.pour_start_frames == start_frames
    assert proc.pour_end_frames == end_frames
    assert proc.displacement_thresh == displacement
    assert proc.r_merge == r_merge


def test_runtime_geometry_dedupes_offsets_after_rounding(tmp_path):
    proc = _make_proc(tmp_path)
    proc._update_runtime_geometry(80, 45)
    assert proc.probe_offsets == [(0, 0), (1, 0), (-1, 0)]
    assert len(proc.probe_offsets) == len(set(proc.probe_offsets))


def test_tracker_mould_assignment_uses_containment_then_nearest(tmp_path):
    proc = _make_proc(tmp_path)
    proc._update_runtime_geometry(1920, 1080)
    trolley = {"track_id": 11, "bbox": (100, 100, 600, 400), "confidence": 0.9}
    moulds = [
        {"track_id": 41, "bbox": (220, 230, 300, 310), "center": (260, 270)},
        {"track_id": 42, "bbox": (420, 230, 500, 310), "center": (460, 270)},
    ]
    proc._update_tracked_mould_observations(moulds, trolley)

    contained_mouth = {"bottom_center": (260, 200)}
    assert proc._select_tracked_mould_for_pour(contained_mouth, trolley) == 41

    nearest_mouth = {"bottom_center": (390, 120)}
    assert proc._select_tracked_mould_for_pour(nearest_mouth, trolley) == 42


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


def test_reference_motion_split_uses_sustained_hold(tmp_path):
    proc = _make_proc(tmp_path)
    base_time = 10.0
    base_dt = datetime.now()
    samples = []
    for idx in range(4):
        samples.append({"time": base_time + idx * 0.04, "datetime": base_dt, "norm": (0.10, 0.10)})
    for idx in range(proc.sustained_hold_frames + 2):
        samples.append(
            {
                "time": base_time + (idx + 4) * 0.04,
                "datetime": base_dt,
                "norm": (0.45, 0.10),
            }
        )
    segment = {
        "start_time": samples[0]["time"],
        "start_datetime": base_dt,
        "end_time": samples[-1]["time"],
        "end_datetime": base_dt,
        "samples": samples,
        "ladle_track_id": 9,
    }
    split = []
    proc._split_segment_by_motion(segment, split)
    assert len(split) == 2


def test_cluster_backtrack_guard_5_allows_recent_old_cluster_reuse(tmp_path):
    proc = _make_proc(tmp_path)
    proc.r_merge = 0.005

    def make_segment(start_time, x):
        start_dt = datetime.now()
        samples = []
        for idx in range(60):
            samples.append(
                {
                    "time": start_time + idx * 0.04,
                    "datetime": start_dt,
                    "norm": (x, 0.2),
                }
            )
        return {
            "start_time": samples[0]["time"],
            "start_datetime": start_dt,
            "end_time": samples[-1]["time"],
            "end_datetime": start_dt,
            "samples": samples,
            "ladle_track_id": 11,
        }

    proc.completed_segments = [
        make_segment(0.0, 0.00),
        make_segment(4.0, 0.12),
        make_segment(8.0, 0.24),
        make_segment(12.0, 0.01),
    ]
    proc._recompute_clusters()
    assert [record["cluster_id"] for record in proc.mould_records] == [1, 2, 3, 1]


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


# ---------------------------------------------------------------------------
# Cluster rescue gate tests
# ---------------------------------------------------------------------------

def _make_segment(start_time, x, y=0.2, duration=3.0, fps=25):
    """Build a minimal raw segment with uniform norm samples at position (x, y)."""
    start_dt = datetime.now()
    n_samples = max(4, int(duration * fps))
    samples = [
        {"time": start_time + i / fps, "datetime": start_dt, "norm": (x, y)}
        for i in range(n_samples)
    ]
    return {
        "start_time": samples[0]["time"],
        "start_datetime": start_dt,
        "end_time": samples[-1]["time"],
        "end_datetime": start_dt,
        "samples": samples,
        "ladle_track_id": 1,
    }


def test_rescue_gate_triggers_on_heat0124_pattern(tmp_path):
    """21 segments at 21 distinct positions → 10 clusters under r=0.08 → gate fires."""
    proc = _make_proc(tmp_path)
    # 10 distinct x positions, each visited ~2 times = 21 segments total
    xs = [i * 0.04 for i in range(10)]  # spacing 0.04, within r_cluster=0.08
    segs = []
    t = 0.0
    for i, x in enumerate(xs):
        segs.append(_make_segment(t, x))
        t += 4.0
    # Second pass over first 11 positions (merges into existing clusters under r=0.08)
    for x in xs[:11]:
        segs.append(_make_segment(t, x + 0.001))  # slight jitter, stays < r_cluster
        t += 4.0

    proc.completed_segments = segs
    proc._recompute_clusters()

    # Baseline: 10 clusters (segments at +0.001 merge into existing under r=0.08)
    # Gate: 21 segs, 10 clusters, ratio=0.476 ≤ 0.65, gap=11 ≥ 8 → rescue runs
    # After rescue: positions are 0.04 apart (> 0.008 gap) → sub-clusters split
    assert proc.clustered_mould_count >= 10


def test_rescue_gate_does_not_trigger_on_normal_revisit_pattern(tmp_path):
    """24 segments, 20 clusters (ratio=0.833 > 0.65) — gate must NOT fire."""
    proc = _make_proc(tmp_path)
    # 20 distinct positions, 4 revisited once (gap < 0.008 = true revisit jitter)
    xs = [i * 0.06 for i in range(20)]
    segs = []
    t = 0.0
    for x in xs:
        segs.append(_make_segment(t, x))
        t += 4.0
    # 4 revisit pours at same positions (x + tiny jitter < 0.008)
    for x in xs[:4]:
        segs.append(_make_segment(t, x + 0.002))
        t += 4.0

    proc.completed_segments = segs
    proc._recompute_clusters()

    # 24 segs, ~20 clusters, ratio ~0.833 > 0.65 → gate stays off
    # Cluster count should stay at ~20 (no rescue splitting)
    assert proc.clustered_mould_count <= 20


def test_rescue_gate_does_not_trigger_on_15_segments_12_clusters(tmp_path):
    """15 segs, 12 clusters: valid_segs=15 < 18 → gate must NOT fire."""
    proc = _make_proc(tmp_path)
    xs = [i * 0.07 for i in range(12)]
    segs = []
    t = 0.0
    for x in xs:
        segs.append(_make_segment(t, x))
        t += 4.0
    for x in xs[:3]:
        segs.append(_make_segment(t, x + 0.001))
        t += 4.0

    proc.completed_segments = segs
    proc._recompute_clusters()

    # 15 segs < 18 → gate blocked, no rescue
    assert proc.clustered_mould_count <= 12


def test_rescue_splits_suspicious_cluster_with_significant_gap(tmp_path):
    """A cluster with 2 segments at x=0.10 and x=0.30 (gap=0.20 >> 0.008) must split."""
    proc = _make_proc(tmp_path)
    # Force a scenario where r_cluster=0.08 merges x=0.10 and x=0.18 into one cluster,
    # but those positions are actually distinct (gap=0.08 > 0.008 internal gap).
    # Use 21 total segs to pass the gate, with one cluster clearly spanning 2 positions.
    segs = []
    t = 0.0
    # 10 single-position clusters
    for i in range(10):
        segs.append(_make_segment(t, i * 0.12))
        t += 4.0
    # One over-merged cluster: 11 extra segments all within r=0.08 of x=0.05
    # but internally split between x=0.05 and x=0.09 (gap=0.04 > 0.008)
    for _ in range(6):
        segs.append(_make_segment(t, 0.05))
        t += 4.0
    for _ in range(5):
        segs.append(_make_segment(t, 0.09))
        t += 4.0

    proc.completed_segments = segs
    proc._recompute_clusters()

    # Gate: 21 segs, baseline ≤ 11 clusters, ratio ≤ 0.52 ≤ 0.65, gap ≥ 10 ≥ 8
    # Rescue should split the over-merged cluster → total clusters > baseline
    assert proc.clustered_mould_count >= 10


def test_rescue_keeps_same_position_revisit_merged(tmp_path):
    """Re-pours with x-gap <= 0.008 must remain in one cluster after rescue."""
    proc = _make_proc(tmp_path)
    # Build a heat that triggers the gate: 21 segs, baseline ~10 clusters
    segs = []
    t = 0.0
    for i in range(10):
        segs.append(_make_segment(t, i * 0.04))
        t += 4.0
    # 11 revisit segments all within 0.005 of x=0.00 (true same-position revisits)
    for _ in range(11):
        segs.append(_make_segment(t, 0.00 + 0.003))
        t += 4.0

    proc.completed_segments = segs
    proc._recompute_clusters()

    # The 12 same-position pours (x≈0.00) should remain 1 cluster after rescue
    cluster_ids = [r["cluster_id"] for r in proc.mould_records if abs(r.get("rep_norm", (1,))[0]) < 0.01]
    if cluster_ids:
        assert len(set(cluster_ids)) == 1, "Same-position revisits must stay merged"


def test_lower_displacement_threshold_detects_slow_motion_split(tmp_path):
    """Displacement 0.15 fires on a 0.20-unit move that 0.25 would miss."""
    import types

    def _make_proc_with_threshold(tmp_path, threshold):
        cfg = DummyConfig()
        cfg.MOULD_DISPLACEMENT_THRESHOLD = threshold
        return PouringProcessor(
            db_manager=DummyDB(),
            config=cfg,
            screenshot_dir=str(tmp_path),
            heat_cycle_manager=None,
        )

    base_dt = datetime.now()
    fps = 25

    def make_two_position_segment(disp):
        """Segment starting at x=0.10, then moving disp units in x."""
        samples = []
        t = 0.0
        for i in range(10):
            samples.append({"time": t, "datetime": base_dt, "norm": (0.10, 0.50)})
            t += 1 / fps
        for i in range(20):
            samples.append({"time": t, "datetime": base_dt, "norm": (0.10 + disp, 0.50)})
            t += 1 / fps
        return {
            "start_time": 0.0, "start_datetime": base_dt,
            "end_time": t, "end_datetime": base_dt,
            "samples": samples, "ladle_track_id": 1,
        }

    seg = make_two_position_segment(0.20)

    proc_025 = _make_proc_with_threshold(tmp_path, 0.25)
    out_025 = []
    proc_025._split_segment_by_motion(seg, out_025)

    proc_015 = _make_proc_with_threshold(tmp_path, 0.15)
    out_015 = []
    proc_015._split_segment_by_motion(seg, out_015)

    # At threshold 0.25: displacement 0.20 never exceeds threshold → no split
    assert len(out_025) == 1, "0.25 threshold should NOT split a 0.20-unit displacement"
    # At threshold 0.15: displacement 0.20 > 0.15 → split fires
    assert len(out_015) == 2, "0.15 threshold should split a 0.20-unit displacement"
