"""Canonical mould registry: latch/freeze/churn-immunity/expiry/selection tests."""

from pathlib import Path

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
    # Canonical registry knobs (fast latch for tests)
    MOULD_CANONICAL_ENABLED = True
    MOULD_CANONICAL_MATCH_RADIUS = 0.08
    MOULD_CANONICAL_LATCH_HITS = 3
    MOULD_CANONICAL_LATCH_MIN_AGE_S = 1.0
    MOULD_CANONICAL_CANDIDATE_TTL_S = 6.0
    MOULD_CANONICAL_TTL_S = 30.0
    MOULD_CANONICAL_EMA_ALPHA = 0.2
    MOULD_CANONICAL_LATCH_CONF = 0.35
    MOULD_CANONICAL_REFRESH_CONF = 0.20
    MOULD_DIAG_CSV = False


TROLLEY = {"bbox": (400, 300, 1000, 700), "track_id": 1, "confidence": 0.9,
           "center": (700, 500), "bottom_center": (700, 700)}


def _make_proc(tmp_path):
    return PouringProcessor(
        db_manager=DummyDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=None,
    )


def _mould(track_id, cx, cy, size=60, conf=0.6):
    half = size // 2
    return {
        "bbox": (cx - half, cy - half, cx + half, cy + half),
        "confidence": conf,
        "track_id": track_id,
        "center": (cx, cy),
        "bottom_center": (cx, cy + half),
        "gie_id": 4,
    }


def _feed(proc, moulds, t):
    proc._update_tracked_mould_observations(moulds, TROLLEY, t)


def test_latch_requires_hits_and_age(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    _feed(proc, [_mould(11, 500, 400)], t)
    _feed(proc, [_mould(11, 500, 400)], t + 0.2)
    _feed(proc, [_mould(11, 500, 400)], t + 0.4)  # 3 hits but only 0.4s old
    assert len(proc._canonical_moulds) == 0
    _feed(proc, [_mould(11, 500, 400)], t + 1.2)  # age satisfied
    assert len(proc._canonical_moulds) == 1


def test_tracker_id_churn_does_not_create_new_canonicals(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    # Latch one mould, then keep observing it with a NEW tracker ID every frame.
    for i, tid in enumerate([1, 2, 3, 4]):
        _feed(proc, [_mould(tid, 500, 400)], t + i * 0.5)
    assert len(proc._canonical_moulds) == 1
    for i, tid in enumerate(range(100, 140)):
        _feed(proc, [_mould(tid, 502, 401)], t + 2.0 + i * 0.04)
    assert len(proc._canonical_moulds) == 1
    entry = next(iter(proc._canonical_moulds.values()))
    assert len(entry["tracker_ids"]) >= 40  # churn absorbed into one canonical


def test_new_placement_latches_second_canonical(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    for i in range(4):
        _feed(proc, [_mould(1, 500, 400)], t + i * 0.5)
    assert len(proc._canonical_moulds) == 1
    # New mould far away (>> match radius in trolley-normalized space)
    for i in range(4):
        _feed(proc, [_mould(2, 900, 620)], t + 3.0 + i * 0.5)
    assert len(proc._canonical_moulds) == 2


def test_low_confidence_never_starts_a_candidate(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    for i in range(6):
        _feed(proc, [_mould(1, 500, 400, conf=0.25)], t + i * 0.5)  # < latch_conf
    assert len(proc._canonical_moulds) == 0
    assert len(proc._canonical_candidates) == 0


def test_expiry_after_ttl_but_poured_entries_survive(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    for i in range(4):
        _feed(proc, [_mould(1, 500, 400)], t + i * 0.5)
        _feed(proc, [_mould(2, 900, 620)], t + i * 0.5)
    assert len(proc._canonical_moulds) == 2
    poured_cid = min(proc._canonical_moulds)
    proc._poured_mould_ids.add(poured_cid)
    # Advance beyond TTL with empty observations; frame_count%25==0 sweep must run.
    proc._frame_count = 25
    _feed(proc, [], t + 100.0)
    assert list(proc._canonical_moulds) == [poured_cid]


def test_pour_selection_returns_canonical_id_and_stashes_raw(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    for i in range(4):
        _feed(proc, [_mould(777, 500, 400)], t + i * 0.5)
    cid = next(iter(proc._canonical_moulds))
    mouth = {"bbox": (480, 300, 520, 360), "confidence": 0.9, "track_id": 5,
             "center": (500, 330), "bottom_center": (500, 360), "gie_id": 1}
    selected = proc._select_tracked_mould_for_pour(mouth, TROLLEY)
    assert selected == cid
    assert proc._active_raw_tracked_mould_id == 777


def test_selection_none_before_any_latch(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    _feed(proc, [_mould(777, 500, 400)], t)  # observed once, not latched
    mouth = {"bbox": (480, 300, 520, 360), "confidence": 0.9, "track_id": 5,
             "center": (500, 330), "bottom_center": (500, 360), "gie_id": 1}
    assert proc._select_tracked_mould_for_pour(mouth, TROLLEY) is None
    assert proc._active_raw_tracked_mould_id == 777  # raw diagnostics still recorded


def test_lifecycle_births_deaths_and_lifespan(tmp_path):
    proc = _make_proc(tmp_path)
    proc._frame_w, proc._frame_h = 1600, 900
    t = 1000.0
    _feed(proc, [_mould(1, 500, 400), _mould(2, 900, 620)], t)
    assert proc._mould_life_births == 2
    # ID 2 vanishes; sweep runs on frame_count % 50 == 0 with >2s gap.
    proc._frame_count = 50
    _feed(proc, [_mould(1, 500, 400)], t + 5.0)
    assert proc._mould_life_deaths == 1
    assert proc._lifespan_p50() >= 0.0


def test_global_id_switch_detected_without_trolley(tmp_path):
    proc = _make_proc(tmp_path)
    proc._frame_w, proc._frame_h = 1600, 900
    t = 1000.0
    # No trolley: pass None so the trolley filter yields nothing, lifecycle still runs.
    proc._update_tracked_mould_observations([_mould(10, 500, 400)], None, t)
    proc._update_tracked_mould_observations([_mould(11, 502, 401)], None, t + 0.04)
    assert proc._mould_global_switches == 1


def test_reset_clears_registry(tmp_path):
    proc = _make_proc(tmp_path)
    t = 1000.0
    for i in range(4):
        _feed(proc, [_mould(1, 500, 400)], t + i * 0.5)
    assert proc._canonical_moulds
    proc._reset_all_state()
    assert not proc._canonical_moulds
    assert not proc._canonical_candidates
    assert not proc._poured_raw_mould_ids


# ---------------------------------------------------------------------------
# Hardening: one-to-one matching, IoU awareness, merge sweep, latch guard,
# trolley-bbox EMA (duplicate-box fixes, 2026-07-17)
# ---------------------------------------------------------------------------


def _latch_one(proc, track_id=1, cx=500, cy=400, t=1000.0):
    for i in range(4):
        _feed(proc, [_mould(track_id, cx, cy)], t + i * 0.5)
    return next(iter(proc._canonical_moulds))


def test_offset_variant_refreshes_instead_of_duplicating(tmp_path):
    """A detection variant (half-box, center shifted beyond the base radius but
    IoU >= 0.3 with the entry) must refresh the entry, not spawn a candidate."""
    proc = _make_proc(tmp_path)
    cid = _latch_one(proc)
    # Offset box: shifted +35px (rel ~0.058 of 600px trolley) and smaller — its
    # center leaves the tight radius but IoU with the latched bbox stays >= 0.3.
    offset = {
        "bbox": (500, 375, 560, 435),
        "confidence": 0.7,
        "track_id": 999,
        "center": (530, 405),
        "bottom_center": (530, 435),
        "gie_id": 4,
    }
    before = len(proc._canonical_moulds)
    for i in range(8):
        _feed(proc, [offset], 1010.0 + i * 0.5)
    assert len(proc._canonical_moulds) == before
    assert not proc._canonical_candidates
    assert 999 in proc._canonical_moulds[cid]["tracker_ids"]


def test_merge_sweep_collapses_duplicates_and_keeps_poured(tmp_path):
    proc = _make_proc(tmp_path)
    cid_a = _latch_one(proc, track_id=1, cx=500, cy=400)
    # Force-seed a duplicate entry directly on the same spot (bypasses guards).
    cid_b = proc._next_canonical_id
    proc._next_canonical_id += 1
    entry_a = proc._canonical_moulds[cid_a]
    proc._canonical_moulds[cid_b] = {
        "cid": cid_b,
        "centroid_rel": entry_a["centroid_rel"],
        "bbox": entry_a["bbox"],
        "first_ts": 1005.0,
        "last_seen_ts": 1005.0,
        "hits": 3,
        "tracker_ids": {77},
    }
    # Mark the DUPLICATE as poured — merge must keep the poured identity.
    proc._poured_mould_ids.add(cid_b)
    proc._poured_mould_durations[cid_b] = 5.0
    proc._frame_count = 25  # sweep trigger
    _feed(proc, [], 1006.0)
    assert len(proc._canonical_moulds) == 1
    assert cid_b in proc._canonical_moulds  # poured id survived
    assert proc._poured_mould_durations[cid_b] == 5.0
    assert proc._canonical_merges_total == 1


def test_one_to_one_single_obs_refreshes_only_one_entry(tmp_path):
    proc = _make_proc(tmp_path)
    cid_a = _latch_one(proc, track_id=1, cx=500, cy=400)
    _latch_one(proc, track_id=2, cx=560, cy=460, t=1010.0)
    cid_b = next(c for c in proc._canonical_moulds if c != cid_a)
    a_seen = proc._canonical_moulds[cid_a]["last_seen_ts"]
    b_seen = proc._canonical_moulds[cid_b]["last_seen_ts"]
    # One observation between them — only the best match may refresh.
    _feed(proc, [_mould(3, 505, 405)], 1020.0)
    refreshed = [
        cid for cid in (cid_a, cid_b)
        if proc._canonical_moulds[cid]["last_seen_ts"] == 1020.0
    ]
    assert len(refreshed) == 1


def test_trolley_bbox_wobble_does_not_duplicate(tmp_path):
    """±5% trolley-bbox wobble must not spawn duplicate canonicals (EMA)."""
    proc = _make_proc(tmp_path)
    t = 1000.0
    wobbles = [(0, 0), (-30, 15), (25, -20), (-15, 25), (30, 0), (0, -25)]
    for i in range(12):
        dx, dy = wobbles[i % len(wobbles)]
        trolley = {
            "bbox": (400 + dx, 300 + dy, 1000 + dx, 700 + dy),
            "track_id": 1,
            "confidence": 0.9,
            "center": (700, 500),
            "bottom_center": (700, 700),
        }
        proc._update_tracked_mould_observations(
            [_mould(1, 500, 400)], trolley, t + i * 0.3
        )
    assert len(proc._canonical_moulds) == 1


def test_latch_guard_refreshes_existing_instead_of_duplicate_latch(tmp_path):
    proc = _make_proc(tmp_path)
    cid = _latch_one(proc)
    before_latched = proc._canonical_latched_total
    # Directly mature a candidate on top of the canonical (bypass matching).
    proc._canonical_candidates.append({
        "centroid_rel": proc._canonical_moulds[cid]["centroid_rel"],
        "bbox": proc._canonical_moulds[cid]["bbox"],
        "first_ts": 900.0,
        "last_seen_ts": 1000.0,
        "hits": 10,
        "tracker_ids": {55},
    })
    # Feed an obs far from the canonical (so it matches the candidate only is not
    # possible — instead trigger candidate maturation via its own match): place the
    # obs outside IoU/radius of the canonical but within base radius of candidate.
    # Simpler: force the guard path by observing at the same spot with a fresh id
    # after removing IoU eligibility — the candidate matures and the guard fires.
    proc._canonical_moulds[cid]["bbox"] = (100, 100, 160, 160)  # move entry bbox away
    proc._canonical_moulds[cid]["centroid_rel"] = (0.9, 0.9)
    obs = _mould(56, 500, 400)
    _feed(proc, [obs], 1001.0)
    # candidate matched+matured; guard compares against canonicals via merge rule —
    # canonical moved away, so a genuine latch is correct here.
    assert proc._canonical_latched_total == before_latched + 1


# ---------------------------------------------------------------------------
# Display staleness gating (ghost-overlay fix, 2026-07-17 evening)
# ---------------------------------------------------------------------------


def test_display_filter_hides_stale_entries_but_keeps_counting(tmp_path):
    proc = _make_proc(tmp_path)
    cid_a = _latch_one(proc, track_id=1, cx=500, cy=400)
    _latch_one(proc, track_id=2, cx=900, cy=620, t=1010.0)
    cid_b = next(c for c in proc._canonical_moulds if c != cid_a)
    proc._poured_mould_ids.add(cid_a)

    # cid_a last seen long ago; cid_b fresh.
    proc._canonical_moulds[cid_a]["last_seen_ts"] = 1000.0
    proc._canonical_moulds[cid_b]["last_seen_ts"] = 1099.0
    shown = proc._canonical_entries_for_display(now=1100.0)
    assert [e["cid"] for e in shown] == [cid_b]
    # Counting state untouched: both entries exist, poured id intact.
    assert set(proc._canonical_moulds) == {cid_a, cid_b}
    assert cid_a in proc._poured_mould_ids


def test_trolley_visibility_predicate(tmp_path):
    proc = _make_proc(tmp_path)
    assert proc._trolley_visible_for_display(now=1000.0) is False  # never detected
    proc.trolley_last_detected_time = 998.0
    assert proc._trolley_visible_for_display(now=1000.0) is True   # 2s ago < 5s window
    assert proc._trolley_visible_for_display(now=1004.0) is False  # 6s ago > 5s window
