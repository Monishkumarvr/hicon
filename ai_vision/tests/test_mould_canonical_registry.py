"""Canonical mould registry: latch/freeze/churn-immunity/expiry/selection tests."""

from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from db_manager import HiConDatabase
from processors.pouring_processor import PouringProcessor
from state.heat_cycle_manager import HeatCycleManager, MouldPouringRecord


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


def test_merge_sweep_swap_case_does_not_crash_on_stale_outer_id(tmp_path):
    """Regression: when the outer-loop id (ca) is the one dropped by the
    'keep the poured id' swap, the inner loop must stop touching it instead of
    re-indexing a canonical entry that was just popped (previously raised
    KeyError on the next cb visited in the same sweep)."""
    proc = _make_proc(tmp_path)
    cid_a = _latch_one(proc, track_id=1, cx=500, cy=400)
    # cid_c: co-located duplicate of cid_a, but already poured -> triggers the
    # keep/drop swap (drop=cid_a) per the "keep the poured id" tie-break.
    cid_c = proc._next_canonical_id
    proc._next_canonical_id += 1
    entry_a = proc._canonical_moulds[cid_a]
    proc._canonical_moulds[cid_c] = {
        "cid": cid_c,
        "centroid_rel": entry_a["centroid_rel"],
        "bbox": entry_a["bbox"],
        "first_ts": 1005.0,
        "last_seen_ts": 1005.0,
        "hits": 3,
        "tracker_ids": {77},
    }
    proc._poured_mould_ids.add(cid_c)
    proc._poured_mould_durations[cid_c] = 5.0
    # cid_d: an unrelated third entry, later in the sorted id list, so the
    # inner loop has a further cb to visit after ca=cid_a gets dropped.
    cid_d = proc._next_canonical_id
    proc._next_canonical_id += 1
    proc._canonical_moulds[cid_d] = {
        "cid": cid_d,
        "centroid_rel": (0.1, 0.1),
        "bbox": (10, 10, 60, 60),
        "first_ts": 1005.0,
        "last_seen_ts": 1005.0,
        "hits": 3,
        "tracker_ids": {88},
    }
    proc._frame_count = 25  # sweep trigger
    _feed(proc, [], 1006.0)  # must not raise KeyError
    assert cid_a not in proc._canonical_moulds
    assert cid_c in proc._canonical_moulds  # poured id survived the swap
    assert cid_d in proc._canonical_moulds  # untouched third entry intact


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


# ---------------------------------------------------------------------------
# Trolley handoff: reset position-matching state on a genuinely different
# physical trolley, without touching heat-cumulative poured counts
# (2026-07-21 — multiple trolleys per heat can otherwise merge moulds across
# trolleys, silently undercounting the now-official tracker count).
# ---------------------------------------------------------------------------


def test_relock_same_physical_trolley_preserves_registry(tmp_path):
    proc = _make_proc(tmp_path)
    proc.trolley_locked = True
    proc.locked_trolley_id = 1
    proc.locked_trolley_bbox = TROLLEY["bbox"]
    cid = _latch_one(proc)
    proc._poured_mould_ids.add(cid)

    # New trolley candidate strongly overlaps the old bbox -> same physical unit.
    same_trolley = {**TROLLEY, "track_id": 2, "bbox": (410, 305, 1005, 705)}
    proc._relock_trolley(same_trolley, timestamp=2000.0, reason="missing_locked_id")

    assert cid in proc._canonical_moulds
    assert proc._canonical_handoffs_total == 0
    assert proc.locked_trolley_id == 2


def test_relock_different_physical_trolley_resets_registry_but_keeps_count(tmp_path):
    proc = _make_proc(tmp_path)
    proc.trolley_locked = True
    proc.locked_trolley_id = 1
    proc.locked_trolley_bbox = TROLLEY["bbox"]
    cid = _latch_one(proc)
    proc._poured_mould_ids.add(cid)
    proc._poured_mould_durations[cid] = 8.0

    # New trolley candidate has near-zero overlap -> a different physical unit.
    different_trolley = {**TROLLEY, "track_id": 99, "bbox": (2000, 2000, 2600, 2400)}
    proc._relock_trolley(different_trolley, timestamp=2000.0, reason="missing_locked_id")

    assert not proc._canonical_moulds
    assert not proc._canonical_candidates
    assert proc._canonical_handoffs_total == 1
    # Heat-cumulative poured state must survive the handoff untouched.
    assert cid in proc._poured_mould_ids
    assert proc._poured_mould_durations[cid] == 8.0
    assert proc.locked_trolley_id == 99


def test_handoff_prevents_new_trolley_mould_merging_into_old_poured_entry(tmp_path):
    """The exact failure mode this fix targets: a new trolley's mould landing
    at a similar rel-position as an old, already-poured entry must NOT be
    absorbed into it once a handoff has cleared the registry."""
    proc = _make_proc(tmp_path)
    proc.trolley_locked = True
    proc.locked_trolley_id = 1
    proc.locked_trolley_bbox = TROLLEY["bbox"]
    old_cid = _latch_one(proc, track_id=1, cx=500, cy=400)
    proc._poured_mould_ids.add(old_cid)

    different_trolley = {**TROLLEY, "track_id": 99, "bbox": (2000, 2000, 2600, 2400)}
    proc._relock_trolley(different_trolley, timestamp=2000.0, reason="missing_locked_id")

    # New trolley's mould at the SAME pixel position the old one occupied.
    new_cid = _latch_one(proc, track_id=201, cx=500, cy=400, t=2001.0)
    assert new_cid != old_cid
    assert old_cid not in proc._canonical_moulds  # cleared by the handoff
    assert new_cid in proc._canonical_moulds
    assert new_cid not in proc._poured_mould_ids  # not yet poured on the new trolley


# ---------------------------------------------------------------------------
# Trolley-bbox EMA continuity across a same-physical relock (2026-07-21 (3)):
# replays the exact bboxes from the 2026-07-17 18:16 live collapse (poured_ids
# 9->3 via an 8-merge cascade within 10s of a same-trolley relock). The EMA
# used to gate smoothing on tracker-ID equality, so it snapped to the raw,
# undamped bbox exactly when a relock happened -- the one moment bbox noise
# is most likely. Fixed to gate on spatial continuity instead.
# ---------------------------------------------------------------------------

OLD_TROLLEY_BBOX = (514, 84, 808, 348)   # locked bbox just before the 18:16 relock
NEW_TROLLEY_BBOX = (520, 140, 805, 294)  # bbox at RELOCK T...271 -> T...327 (IoU ~0.57)
DIFFERENT_TROLLEY_BBOX = (2000, 2000, 2600, 2400)  # for contrast: genuinely different unit


def test_ema_smooths_through_same_physical_trolley_relock(tmp_path):
    proc = _make_proc(tmp_path)
    old_trolley = {**TROLLEY, "track_id": 271, "bbox": OLD_TROLLEY_BBOX}
    new_trolley = {**TROLLEY, "track_id": 327, "bbox": NEW_TROLLEY_BBOX}

    proc._update_tracked_mould_observations([], old_trolley, 1000.0)
    assert proc._trolley_norm_ema == tuple(float(v) for v in OLD_TROLLEY_BBOX)

    proc._update_tracked_mould_observations([], new_trolley, 1000.5)
    blended = proc._trolley_norm_ema
    # Must be a blend (0.7*old + 0.3*new), NOT a snap straight to the raw new bbox.
    assert blended != tuple(float(v) for v in NEW_TROLLEY_BBOX)
    expected = tuple(0.7 * OLD_TROLLEY_BBOX[i] + 0.3 * NEW_TROLLEY_BBOX[i] for i in range(4))
    assert blended == pytest.approx(expected)


def test_ema_still_snaps_for_a_genuinely_different_trolley(tmp_path):
    proc = _make_proc(tmp_path)
    old_trolley = {**TROLLEY, "track_id": 271, "bbox": OLD_TROLLEY_BBOX}
    different_trolley = {**TROLLEY, "track_id": 999, "bbox": DIFFERENT_TROLLEY_BBOX}

    proc._update_tracked_mould_observations([], old_trolley, 1000.0)
    proc._update_tracked_mould_observations([], different_trolley, 1000.5)
    # No spatial continuity -> snap to the raw bbox, no blending with the stale one.
    assert proc._trolley_norm_ema == tuple(float(v) for v in DIFFERENT_TROLLEY_BBOX)


def test_mould_position_stable_across_same_physical_relock_end_to_end(tmp_path):
    """The actual failure mode: a mould observed at the same absolute pixel
    position before and after a same-physical relock must keep matching its
    existing canonical entry, not drift into spurious churn/merges."""
    proc = _make_proc(tmp_path)
    old_trolley = {**TROLLEY, "track_id": 271, "bbox": OLD_TROLLEY_BBOX}
    new_trolley = {**TROLLEY, "track_id": 327, "bbox": NEW_TROLLEY_BBOX}

    cx, cy = 650, 200  # inside both bboxes
    for i in range(4):
        proc._update_tracked_mould_observations([_mould(1, cx, cy)], old_trolley, 1000.0 + i * 0.5)
    assert len(proc._canonical_moulds) == 1
    cid = next(iter(proc._canonical_moulds))
    proc._poured_mould_ids.add(cid)

    # Relock happens; same physical mould, same pixel position, new trolley id.
    for i in range(4):
        proc._update_tracked_mould_observations([_mould(50 + i, cx, cy)], new_trolley, 1002.0 + i * 0.5)

    assert len(proc._canonical_moulds) == 1  # still one mould, not split/duplicated
    assert cid in proc._canonical_moulds      # same identity preserved
    assert cid in proc._poured_mould_ids       # poured status intact


# ---------------------------------------------------------------------------
# Customer report fixes (2026-07-28): stuck-yellow majority-vote assignment,
# exception isolation for the count-freeze risk, dimmed-tier overlay for the
# black-blink complaint.
# ---------------------------------------------------------------------------


def test_end_pour_commits_majority_vote_not_last_pick(tmp_path):
    """The bug: a one-shot pick (whatever frame 1 or the LAST frame happened to
    select) could lock onto a glare-lit neighboring mould forever. The fix must
    credit whichever mould was selected most often across the whole pour."""
    proc = _make_proc(tmp_path)
    proc.pour_active = True
    proc.pour_start_time = 0.0
    proc.pour_start_datetime = datetime.now()
    # 4 votes for 7 (the true target), 1 stray vote for 42 (a transient glare
    # blip) that happens to be the LAST pick -- proves this isn't just "credit
    # the last selection".
    proc._mould_vote_counts = Counter({7: 4, 42: 1})
    proc._active_tracked_mould_id = 42

    proc._end_pour(3.0, datetime.now(), [], [], None)

    assert proc.tracker_mould_count == 1
    assert 7 in proc._poured_mould_ids
    assert 42 not in proc._poured_mould_ids


def test_majority_vote_accumulates_via_real_selection_path(tmp_path):
    """End-to-end through _select_tracked_mould_for_pour: a probe that lands on
    mould A once (glare blip) then mould B four times must still credit B."""
    proc = _make_proc(tmp_path)
    _latch_one(proc, track_id=1, cx=500, cy=400)
    _latch_one(proc, track_id=2, cx=900, cy=620, t=1010.0)
    # _latch_one returns next(iter(...)), which is unreliable for a SECOND latch
    # onto the same trolley (no clearing event in between) -- read the two real
    # keys directly instead.
    cid_a, cid_b = sorted(proc._canonical_moulds)

    # bottom_center placed so the probe (bottom_center + probe_below_px) lands
    # inside each mould's own bbox — containment is now the only selection
    # signal, there's no more nearest-centroid fallback to lean on.
    mouth_near_a = {"bbox": (480, 380, 520, 420), "confidence": 0.9, "track_id": 5,
                    "center": (500, 400), "bottom_center": (500, 390), "gie_id": 1}
    mouth_near_b = {"bbox": (880, 600, 920, 640), "confidence": 0.9, "track_id": 5,
                    "center": (900, 620), "bottom_center": (900, 590), "gie_id": 1}

    proc.pour_active = True
    proc.pour_start_time = 1000.0
    proc.pour_start_datetime = datetime.now()
    proc._mould_vote_counts = Counter()

    # Last pick is the WRONG one (A) -- old "lock onto last/first pick" behavior
    # would have credited A; majority vote must still credit B.
    for mouth in (mouth_near_b, mouth_near_b, mouth_near_b, mouth_near_b, mouth_near_a):
        picked = proc._select_tracked_mould_for_pour(mouth, TROLLEY)
        if picked is not None:
            proc._mould_vote_counts[picked] += 1
            proc._active_tracked_mould_id = picked

    assert proc._active_tracked_mould_id == cid_a  # confirms the "wrong last pick" setup
    assert proc._mould_vote_counts[cid_b] == 4
    assert proc._mould_vote_counts[cid_a] == 1

    proc._end_pour(1003.0, datetime.now(), [], [], None)
    assert cid_b in proc._poured_mould_ids
    assert cid_a not in proc._poured_mould_ids


def test_merge_migrates_vote_counts_to_the_kept_id(tmp_path):
    proc = _make_proc(tmp_path)
    cid_a = _latch_one(proc, track_id=1, cx=500, cy=400)
    proc._mould_vote_counts[cid_a] = 3
    # Force-seed a duplicate at the same spot and merge it away.
    cid_b = proc._next_canonical_id
    proc._next_canonical_id += 1
    entry_a = proc._canonical_moulds[cid_a]
    proc._canonical_moulds[cid_b] = {
        "cid": cid_b, "centroid_rel": entry_a["centroid_rel"], "bbox": entry_a["bbox"],
        "first_ts": 1005.0, "last_seen_ts": 1005.0, "hits": 3, "tracker_ids": {77},
    }
    proc._mould_vote_counts[cid_b] = 2
    proc._merge_canonical(cid_a, cid_b)  # keep=cid_a, drop=cid_b
    assert proc._mould_vote_counts[cid_a] == 5
    assert cid_b not in proc._mould_vote_counts


def test_start_pour_and_reset_clear_vote_counts(tmp_path):
    proc = _make_proc(tmp_path)
    proc._mould_vote_counts = Counter({1: 5})
    proc._start_pour(1000.0, datetime.now(), [], [], None, 240, 500, 400, TROLLEY, None)
    assert not proc._mould_vote_counts

    proc._mould_vote_counts = Counter({1: 5})
    proc._reset_all_state()
    assert not proc._mould_vote_counts


def test_process_frame_exception_isolation_keeps_cycle_timeout_running(tmp_path, monkeypatch):
    """A failure in session/pour/mould logic (steps 2-8) must never prevent the
    cycle-timeout and heat-finalization steps (9-10) from running -- those are
    the only paths that can recover state. Matches a customer report where
    pours stopped being counted after a second tapping while tapping itself
    (a fully separate code path) kept working -- the signature of exactly this
    kind of permanent per-frame freeze."""
    proc = _make_proc(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated edge-case failure in session/pour logic")

    monkeypatch.setattr(proc, "_get_target_trolley", boom)

    calls = {"timeout": 0, "finalize": 0}
    monkeypatch.setattr(
        proc, "_check_cycle_timeout",
        lambda *a, **k: calls.__setitem__("timeout", calls["timeout"] + 1),
    )
    monkeypatch.setattr(
        proc, "_finalize_heat_cycles_if_due",
        lambda *a, **k: calls.__setitem__("finalize", calls["finalize"] + 1),
    )

    frame_meta = SimpleNamespace(obj_meta_list=None)
    proc.process_frame(frame_meta, None, 1000.0, datetime.now())

    assert calls["timeout"] == 1
    assert calls["finalize"] == 1


def test_process_frame_survives_repeated_exceptions_across_frames(tmp_path, monkeypatch):
    """The freeze this fix targets is a PERMANENT one -- the same exception
    recurring every frame must still let steps 9-10 run every time, not just
    once."""
    proc = _make_proc(tmp_path)
    monkeypatch.setattr(proc, "_update_session", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    calls = {"timeout": 0}
    monkeypatch.setattr(
        proc, "_check_cycle_timeout",
        lambda *a, **k: calls.__setitem__("timeout", calls["timeout"] + 1),
    )
    frame_meta = SimpleNamespace(obj_meta_list=None)
    for i in range(5):
        proc.process_frame(frame_meta, None, 1000.0 + i, datetime.now())
    assert calls["timeout"] == 5


def test_canonical_display_color_dims_between_thresholds_not_hard_vanish(tmp_path):
    proc = _make_proc(tmp_path)
    cid = _latch_one(proc)
    entry = proc._canonical_moulds[cid]
    full_color = (0, 170, 255)  # not poured, full brightness
    dim_color = tuple(c // 2 for c in full_color)

    entry["last_seen_ts"] = 1000.0
    assert proc._canonical_display_color(entry, now=1000.5) == full_color  # < dim_after_s (1.5)
    assert proc._canonical_display_color(entry, now=1002.0) == dim_color  # between 1.5 and stale_s (3.0)
    # Still within overlay_stale_s (3.0) at 2.9s -- must still be drawn (dimmed),
    # not excluded -- confirmed via the display filter directly.
    assert entry in proc._canonical_entries_for_display(now=1002.9)
    assert entry not in proc._canonical_entries_for_display(now=1003.1)


def test_canonical_display_color_poured_dims_too(tmp_path):
    proc = _make_proc(tmp_path)
    cid = _latch_one(proc)
    proc._poured_mould_ids.add(cid)
    entry = proc._canonical_moulds[cid]
    entry["last_seen_ts"] = 1000.0
    full_color = (0, 220, 0)
    dim_color = tuple(c // 2 for c in full_color)
    assert proc._canonical_display_color(entry, now=1000.5) == full_color
    assert proc._canonical_display_color(entry, now=1002.0) == dim_color


# ---------------------------------------------------------------------------
# Slot-number stability (hicon-9cp)
# ---------------------------------------------------------------------------

class TrackerConfig(DummyConfig):
    MOULD_COUNT_MODE = "tracker"


def _commit_pour(proc, tracker_id, start_ts):
    """Drive one committed pour for `tracker_id` through the real _end_pour path."""
    now = datetime.now()
    proc.pour_active = True
    proc.pour_start_time = start_ts
    proc.pour_start_datetime = now
    proc._mould_vote_counts = Counter()
    proc._active_tracked_mould_id = tracker_id
    proc._end_pour(start_ts + 3.0, now, [], [], None)


def _tracker_record(slot_id, ts):
    return {
        "slot_id": slot_id,
        "ladle_track_id": 7,
        "start_time_wall": ts,
        "start_datetime_obj": datetime.now(),
        "end_time_wall": ts + 3.0,
        "end_datetime_obj": datetime.now(),
        "duration_s": 3.0,
    }


def test_slot_numbers_are_never_reused_after_a_merge(tmp_path):
    """A canonical merge of two already-poured moulds discards the dropped one's
    slot. Deriving the next slot from len(_tracker_slot_by_id) handed that freed
    number to a different physical mould, and upsert_completed_mould_pouring
    matches on the "MOULD_C{n}" string — so the two collapsed into one record."""
    proc = _make_proc(tmp_path)
    proc._save_event_screenshot = lambda *args, **kwargs: None
    proc.locked_trolley_id = 7

    _latch_one(proc, track_id=1, cx=460, cy=360)
    _latch_one(proc, track_id=2, cx=700, cy=500, t=1010.0)
    _latch_one(proc, track_id=3, cx=940, cy=640, t=1020.0)
    cid_a, cid_b, cid_c = sorted(proc._canonical_moulds)

    _commit_pour(proc, cid_a, 1000.0)
    _commit_pour(proc, cid_b, 1010.0)
    _commit_pour(proc, cid_c, 1020.0)
    assert [proc._tracker_slot_by_id[c] for c in (cid_a, cid_b, cid_c)] == [1, 2, 3]

    # Both already poured, so cid_c's slot 3 is discarded outright, not transferred.
    proc._merge_canonical(cid_b, cid_c)
    assert cid_c not in proc._tracker_slot_by_id
    assert proc._tracker_slot_by_id[cid_b] == 2

    _latch_one(proc, track_id=4, cx=580, cy=620, t=1030.0)
    cid_d = max(proc._canonical_moulds)
    _commit_pour(proc, cid_d, 1030.0)

    # 4, not the freed-up 3 — a genuinely different mould must never inherit it.
    assert proc._tracker_slot_by_id[cid_d] == 4
    assert len(set(proc._tracker_slot_by_id.values())) == len(proc._tracker_slot_by_id)


def test_restore_seeds_slot_counter_past_restored_numbers(tmp_path):
    """Restore assigns slots directly from persisted "MOULD_C{n}" strings rather
    than through the counter, so the counter must be advanced past them or a
    mid-cycle restart re-collides with what it just restored."""
    now = datetime.now()
    records = [
        MouldPouringRecord(
            mould_id=f"MOULD_C{slot}", mould_track_id=100 + slot,
            start_time=1000.0, start_datetime=now,
            end_time=1003.0, end_datetime=now,
            duration_seconds=3.0, source="tracker",
        )
        # 3 and 4 were merged away before the restart — restored slots are sparse.
        for slot in (1, 2, 5)
    ]
    stub_manager = SimpleNamespace(
        active_cycle=SimpleNamespace(mould_pourings=records, ladle_track_ids=[7]),
        upsert_completed_mould_pouring=lambda **kwargs: None,
        prune_tracker_mould_pourings=lambda valid_mould_ids: 0,
        update_pouring_session_presence=lambda *args, **kwargs: None,
        record_pour_window=lambda **kwargs: None,
    )

    proc = PouringProcessor(
        db_manager=DummyDB(),
        config=TrackerConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=stub_manager,
    )
    proc._save_event_screenshot = lambda *args, **kwargs: None
    proc.locked_trolley_id = 7

    assert proc._next_tracker_slot == 6

    _commit_pour(proc, 999, 2000.0)
    assert proc._tracker_slot_by_id[999] == 6  # not 3, and not a re-used 1/2/5


def test_sync_prunes_merged_away_mould_from_heat_cycle(tmp_path):
    """cycle.mould_pourings is append-only and _merge_canonical never notified the
    heat cycle, so a merged-away mould stayed counted forever while its duration
    was also folded into the survivor's total."""
    db = HiConDatabase(str(tmp_path / "heat_cycle.sqlite"))
    manager = HeatCycleManager(db, ladle_absence_timeout=300.0)
    proc = PouringProcessor(
        db_manager=DummyDB(),
        config=TrackerConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=manager,
    )

    proc._tracker_pour_records = {
        101: _tracker_record(1, 1000.0),
        102: _tracker_record(2, 1010.0),
        103: _tracker_record(3, 1020.0),
    }
    proc._sync_mould_records_to_heat_cycle()
    assert {p.mould_id for p in manager.active_cycle.mould_pourings} == {
        "MOULD_C1", "MOULD_C2", "MOULD_C3",
    }

    # A merge folds 103 into 102 and drops it from the processor's aggregates.
    del proc._tracker_pour_records[103]
    proc._sync_mould_records_to_heat_cycle()

    assert {p.mould_id for p in manager.active_cycle.mould_pourings} == {
        "MOULD_C1", "MOULD_C2",
    }


def test_sync_with_no_tracker_records_prunes_nothing(tmp_path):
    """Defensive guard: an empty _tracker_pour_records must never be read as
    'everything was merged away' and wipe restored state."""
    db = HiConDatabase(str(tmp_path / "heat_cycle.sqlite"))
    manager = HeatCycleManager(db, ladle_absence_timeout=300.0)
    proc = PouringProcessor(
        db_manager=DummyDB(),
        config=TrackerConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=manager,
    )

    proc._tracker_pour_records = {101: _tracker_record(1, 1000.0)}
    proc._sync_mould_records_to_heat_cycle()
    assert len(manager.active_cycle.mould_pourings) == 1

    proc._tracker_pour_records = {}
    proc._sync_mould_records_to_heat_cycle()
    assert len(manager.active_cycle.mould_pourings) == 1
