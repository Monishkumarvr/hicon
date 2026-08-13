from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from db_manager import HiConDatabase
from processors.melting_analysis_controller import MeltingAnalysisController
from processors.melting_meta_reader import DecodedMeltingState, DecodedMeltingZoneState
from processors.pouring_processor import PouringProcessor
from state.heat_cycle_manager import HeatCycleManager


class DummyDB:
    def __init__(self):
        self.inserted_heat_cycles = []
        self.inserted_pouring_events = []
        self.inserted_melting_events = []

    def insert_pouring_event(self, **kwargs):
        self.inserted_pouring_events.append(kwargs)
        return 1

    def update_pouring_end(self, **kwargs):
        return None

    def insert_heat_cycle(self, **kwargs):
        self.inserted_heat_cycles.append(kwargs)
        return "0001"

    def insert_melting_event(self, **kwargs):
        self.inserted_melting_events.append(kwargs)
        return "0002"


class DummyConfig:
    CUSTOMER_ID = "C1"
    LOCATION = "Loc"
    CAMERA_ID_STREAM_0 = "Cam-0"
    MOUTH_CONFIDENCE = 0.4
    TROLLEY_CONFIDENCE = 0.25
    SESSION_START_DURATION = 1.0
    SESSION_END_DURATION = 1.5
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


class CapturePresenceHeatCycleManager:
    def __init__(self):
        self.calls = []

    def update_pouring_session_presence(self, track_id, current_time, current_datetime):
        self.calls.append((track_id, current_time, current_datetime))

    def check_and_finalize_cycles(self, current_time, current_datetime):
        return []


class CaptureTappingHeatCycleManager:
    def __init__(self):
        self.calls = []

    def add_tapping_event(self, **kwargs):
        self.calls.append(kwargs)


def _make_db(tmp_path):
    return HiConDatabase(str(tmp_path / "heat_cycle.sqlite"))


def _make_processor(tmp_path, heat_cycle_manager):
    return PouringProcessor(
        db_manager=DummyDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle_manager,
    )


def _make_melting_controller(tmp_path, db_manager, heat_cycle_manager=None):
    zones = {
        "metadata": {"ref_width": 40, "ref_height": 40},
        "tapping": {
            "zones": {
                "tap-1": {"roi_points": [[5, 5], [20, 5], [20, 20], [5, 20]]}
            },
            "abs_brightness_threshold": 210,
            "start_white_ratio": 0.25,
            "start_frame_count": 20,
            "end_white_ratio": 0.1,
            "end_frame_count": 25,
        },
        "deslagging": {"zones": {}},
        "spectro": {"zones": {}},
    }
    return MeltingAnalysisController(
        zones_config=zones,
        db_manager=db_manager,
        config=DummyConfig,
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle_manager,
        enable_display_meta=False,
    )


def _native_melting_state(*, tapping_active):
    return DecodedMeltingState(
        version=1,
        debug_code=0,
        blackout_active=False,
        frame_num=1,
        ntp_timestamp=0,
        tapping=[
            DecodedMeltingZoneState(
                valid=True,
                active=tapping_active,
                raw_count=1 if tapping_active else 0,
                filtered_count=1 if tapping_active else 0,
                white_ratio=0.4 if tapping_active else 0.0,
                max_blob_area=0.0,
                max_blob_brightness=0.0,
            )
        ],
        deslagging=[],
        spectro=[],
    )


def test_pouring_cycle_finalizes_at_last_valid_presence_and_backfills_next_start(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(db, ladle_absence_timeout=300.0)

    first_seen = datetime(2026, 3, 31, 10, 0, 0)
    last_seen = first_seen + timedelta(seconds=40)
    manager.update_pouring_session_presence(11, 100.0, first_seen)
    manager.update_pouring_session_presence(11, 140.0, last_seen)

    assert manager.check_and_finalize_cycles(439.0, last_seen + timedelta(seconds=299)) == []

    finalized = manager.check_and_finalize_cycles(440.0, last_seen + timedelta(seconds=300))
    assert len(finalized) == 1
    cycle = finalized[0]
    assert cycle.cycle_end_time == 140.0
    assert cycle.cycle_end_datetime == last_seen

    next_event_dt = last_seen + timedelta(seconds=500)
    manager.update_pouring_session_presence(22, 640.0, next_event_dt)
    assert manager.active_cycle is not None
    assert manager.active_cycle.cycle_start_time == 140.0
    assert manager.active_cycle.cycle_start_datetime == last_seen


def test_tapping_only_cycle_finalizes_at_last_tapping_end(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(
        db,
        ladle_absence_timeout=300.0,
        tapping_only_timeout=300.0,
    )

    tap1_start = datetime(2026, 3, 31, 11, 0, 0)
    tap1_end = tap1_start + timedelta(seconds=15)
    tap2_start = tap1_start + timedelta(seconds=30)
    tap2_end = tap2_start + timedelta(seconds=20)

    manager.add_tapping_event(100.0, tap1_start, 115.0, tap1_end, 15.0)
    manager.add_tapping_event(130.0, tap2_start, 150.0, tap2_end, 20.0)

    finalized = manager.check_and_finalize_cycles(450.0, tap2_end + timedelta(seconds=300))
    assert len(finalized) == 1

    cycle = finalized[0]
    assert cycle.tapping_start_time == 100.0
    assert cycle.tapping_end_time == 150.0
    assert cycle.cycle_end_time == 150.0
    assert cycle.cycle_end_datetime == tap2_end
    assert cycle.total_pouring_time == 0
    assert cycle.mould_wise_pouring_time == []


def test_tapping_zone_sets_cycle_furnace_label(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(
        db,
        ladle_absence_timeout=300.0,
        tapping_only_timeout=300.0,
    )

    tap_start = datetime(2026, 3, 31, 11, 0, 0)
    tap_end = tap_start + timedelta(seconds=15)

    manager.add_tapping_event(
        100.0,
        tap_start,
        115.0,
        tap_end,
        15.0,
        zone_name="tap-2",
    )

    assert manager.active_cycle is not None
    assert manager.active_cycle.furnace_label == "Furnace2"


def test_tracker_mould_upsert_keeps_one_api_record_per_distinct_id(tmp_path):
    manager = HeatCycleManager(_make_db(tmp_path), ladle_absence_timeout=300.0)
    start = datetime(2026, 7, 16, 10, 0, 0)

    manager.upsert_completed_mould_pouring(
        ladle_track_id=7,
        mould_id="MOULD_C1",
        mould_track_id=41,
        start_time=100.0,
        start_datetime=start,
        end_time=103.0,
        end_datetime=start + timedelta(seconds=3),
        duration_seconds=3.0,
    )
    manager.upsert_completed_mould_pouring(
        ladle_track_id=7,
        mould_id="MOULD_C1",
        mould_track_id=41,
        start_time=100.0,
        start_datetime=start,
        end_time=108.0,
        end_datetime=start + timedelta(seconds=8),
        duration_seconds=6.0,
    )
    manager.upsert_completed_mould_pouring(
        ladle_track_id=7,
        mould_id="MOULD_C2",
        mould_track_id=99,
        start_time=109.0,
        start_datetime=start + timedelta(seconds=9),
        end_time=112.0,
        end_datetime=start + timedelta(seconds=12),
        duration_seconds=3.0,
    )

    cycle = manager.active_cycle
    assert cycle is not None
    assert len(cycle.mould_pourings) == 2
    assert cycle.mould_pourings[0].duration_seconds == 6.0
    manager._finalize_cycle(cycle, 112.0, start + timedelta(seconds=12))
    assert len(cycle.mould_wise_pouring_time) == 2


def test_backfill_preserves_zone_name_and_sets_cycle_furnace(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(
        db,
        ladle_absence_timeout=300.0,
        tapping_only_timeout=300.0,
    )

    pre_start = datetime(2026, 3, 31, 10, 55, 0)
    pre_end = pre_start + timedelta(seconds=8)
    db.insert_melting_event(
        sync_id="deslag-1",
        customer_id="C1",
        event_type="deslagging",
        start_time=pre_start.isoformat(),
        end_time=pre_end.isoformat(),
        duration_sec=8.0,
        camera_id="Cam-0",
        location="Loc",
        zone_name="zone-1",
    )

    tap_start = datetime(2026, 3, 31, 11, 0, 0)
    tap_end = tap_start + timedelta(seconds=15)
    manager.add_tapping_event(
        100.0,
        tap_start,
        115.0,
        tap_end,
        15.0,
        zone_name="tap-1",
    )

    assert manager.active_cycle is not None
    assert manager.active_cycle.furnace_label == "Furnace1"
    assert len(manager.active_cycle.deslagging_events) == 1
    assert manager.active_cycle.deslagging_events[0]["zone_name"] == "zone-1"


def test_conflicting_furnace_events_do_not_flip_cycle_furnace(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(
        db,
        ladle_absence_timeout=300.0,
        tapping_only_timeout=300.0,
    )

    tap_start = datetime(2026, 3, 31, 11, 0, 0)
    tap_end = tap_start + timedelta(seconds=15)
    manager.add_tapping_event(
        100.0,
        tap_start,
        115.0,
        tap_end,
        15.0,
        zone_name="tap-1",
    )

    pyro_start = tap_end + timedelta(seconds=10)
    pyro_end = pyro_start + timedelta(seconds=5)
    manager.add_pyrometer_event(
        125.0,
        pyro_start,
        130.0,
        pyro_end,
        5.0,
        zone_name="furnace-2",
    )

    assert manager.active_cycle is not None
    assert manager.active_cycle.furnace_label == "Furnace1"


def test_cycle_furnace_updates_existing_pouring_locations(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(
        db,
        ladle_absence_timeout=300.0,
        tapping_only_timeout=300.0,
        base_location="Loc",
    )

    cycle_start_dt = datetime(2026, 3, 31, 11, 0, 0)
    manager.update_pouring_session_presence(11, 100.0, cycle_start_dt)
    assert manager.active_cycle is not None

    db.insert_pouring_event(
        sync_id="pour-1",
        customer_id="C1",
        date="2026-03-31",
        shift="DAY",
        heat_no=manager.active_cycle.heat_no,
        ladle_number="",
        location="Loc",
        camera_id="Cam-0",
        pouring_start_time=cycle_start_dt.isoformat(),
    )

    pyro_start = cycle_start_dt + timedelta(seconds=30)
    pyro_end = pyro_start + timedelta(seconds=5)
    manager.add_pyrometer_event(
        130.0,
        pyro_start,
        135.0,
        pyro_end,
        5.0,
        zone_name="furnace-1",
    )

    conn = db._get_connection()
    row = conn.execute(
        "SELECT location FROM pouring_events WHERE sync_id = ?",
        ("pour-1",),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "Loc Furnace1"


def test_non_creator_events_do_not_create_cycle(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(db, ladle_absence_timeout=300.0)
    now = datetime(2026, 3, 31, 12, 0, 0)

    manager.add_deslagging_event(1.0, now, 2.0, now + timedelta(seconds=1), 1.0)
    manager.add_spectro_event(3.0, now, 4.0, now + timedelta(seconds=1), 1.0)
    manager.add_pyrometer_event(5.0, now, 6.0, now + timedelta(seconds=1), 1.0)

    assert manager.active_cycle is None


def test_pouring_processor_refreshes_heat_cycle_only_after_session_is_active(tmp_path):
    heat_cycle_manager = CapturePresenceHeatCycleManager()
    processor = _make_processor(tmp_path, heat_cycle_manager)

    mouth = {
        "track_id": 5,
        "confidence": 0.95,
        "bbox": (120, 120, 150, 150),
        "bottom_center": (135, 150),
    }
    trolley = {"track_id": 7, "bbox": (100, 100, 200, 220), "confidence": 0.9}
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)

    processor._extract_detections = lambda _frame_meta, _ts=None: ([mouth], [trolley])
    processor._get_target_trolley = lambda _trolleys, _timestamp=None, _mouths=None: trolley
    processor._select_best_mouth_for_trolley = lambda _mouths, _trolley: mouth
    processor._check_cycle_timeout = lambda *args, **kwargs: None

    def _keep_session_inactive(*args, **kwargs):
        processor.session_active = False

    processor._update_session = _keep_session_inactive
    processor.process_frame(frame_meta, None, 10.0, datetime(2026, 3, 31, 13, 0, 0))
    assert heat_cycle_manager.calls == []

    def _activate_session(*args, **kwargs):
        processor.session_active = True

    processor._update_session = _activate_session
    processor.process_frame(
        frame_meta,
        np.zeros((8, 8, 4), dtype=np.uint8),
        11.0,
        datetime(2026, 3, 31, 13, 0, 1),
    )
    assert len(heat_cycle_manager.calls) == 1
    assert heat_cycle_manager.calls[0][0] == 5


def test_melting_controller_emits_tapping_event_and_updates_heat_cycle(tmp_path):
    db = DummyDB()
    heat_cycle_manager = CaptureTappingHeatCycleManager()
    controller = _make_melting_controller(tmp_path, db, heat_cycle_manager)
    frame = np.zeros((40, 40, 4), dtype=np.uint8)

    start_state = _native_melting_state(tapping_active=True)
    end_state = _native_melting_state(tapping_active=False)

    assert controller.needs_frame(start_state) is True
    controller.process_native_state(
        native_state=start_state,
        frame_meta=None,
        frame=frame,
        timestamp=10.0,
    )
    assert controller.needs_frame(end_state) is True
    controller.process_native_state(
        native_state=end_state,
        frame_meta=None,
        frame=frame,
        timestamp=14.5,
    )

    assert len(db.inserted_melting_events) == 1
    assert db.inserted_melting_events[0]["event_type"] == "tapping"
    assert db.inserted_melting_events[0]["duration_sec"] == 4.5
    assert db.inserted_melting_events[0]["zone_name"] == "tap-1"
    assert db.inserted_melting_events[0]["screenshot_path"]
    assert len(heat_cycle_manager.calls) == 1
    assert heat_cycle_manager.calls[0]["zone_name"] == "tap-1"


def test_pouring_processor_inserts_heat_cycle_with_empty_ladle_number(tmp_path):
    db = DummyDB()
    furnace_helper = SimpleNamespace(
        location_with_furnace=lambda base, furnace: f"{base} {furnace}".strip()
    )
    processor = PouringProcessor(
        db_manager=db,
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=furnace_helper,
    )

    cycle = SimpleNamespace(
        heat_no="HEAT_0001",
        mould_pourings=[],
        tapping_events=[{"start": "2026-03-31T14:00:00", "end": "2026-03-31T14:00:10"}],
        cycle_start_datetime=datetime(2026, 3, 31, 14, 0, 0),
        cycle_end_datetime=None,
        pouring_start_time=None,
        pouring_end_time=None,
        total_pouring_time=0,
        mould_wise_pouring_time=[],
        tapping_start_datetime=datetime(2026, 3, 31, 14, 0, 0),
        tapping_end_datetime=datetime(2026, 3, 31, 14, 0, 10),
        deslagging_events=[],
        spectro_events=[],
        pyrometer_events=[],
        furnace_label="Furnace1",
        has_pouring_session=False,
    )

    processor._insert_heat_cycle_to_db(cycle)

    assert len(db.inserted_heat_cycles) == 1
    inserted = db.inserted_heat_cycles[0]
    assert inserted["ladle_number"] == ""
    assert inserted["cycle_end_time"]
    assert inserted["location"] == "Loc Furnace1"
    assert inserted["pouring_start_time"] == ""
    assert inserted["pouring_end_time"] == ""

# ---------------------------------------------------------------------------
# Pour window fallback when no mould gets attributed (hicon-7ha)
# ---------------------------------------------------------------------------

def test_unattributed_pours_still_populate_pouring_window(tmp_path):
    """A pour whose probe never lands inside a detected mould bbox is left
    unattributed on purpose (containment-only rule) -- but it still happened,
    and _finalize_cycle used to derive pouring_start_time/end/total exclusively
    from mould_pourings, so an all-unattributed cycle reported no pouring at
    all. record_pour_window() tracks the window independently and this must
    be what _finalize_cycle falls back to."""
    manager = HeatCycleManager(_make_db(tmp_path), ladle_absence_timeout=300.0)
    start = datetime(2026, 8, 8, 15, 32, 43)
    manager.update_pouring_session_presence(7, start.timestamp(), start)

    manager.record_pour_window(start_datetime=start, end_datetime=start + timedelta(seconds=9),
                                duration_seconds=9.0)
    manager.record_pour_window(start_datetime=start + timedelta(seconds=24),
                                end_datetime=start + timedelta(seconds=28), duration_seconds=4.0)

    cycle = manager.active_cycle
    assert cycle is not None
    assert cycle.mould_pourings == []  # nothing attributed

    manager._finalize_cycle(cycle, (start + timedelta(seconds=28)).timestamp(),
                             start + timedelta(seconds=28))

    assert cycle.pouring_start_time == start
    assert cycle.pouring_end_time == start + timedelta(seconds=28)
    assert cycle.total_pouring_time == 13  # 9 + 4
    assert cycle.mould_wise_pouring_time == []


def test_attributed_cycle_ignores_pour_window(tmp_path):
    """Regression guard: when moulds ARE attributed, timing still comes from
    mould_pourings as before -- the window is only a fallback."""
    manager = HeatCycleManager(_make_db(tmp_path), ladle_absence_timeout=300.0)
    start = datetime(2026, 8, 8, 10, 0, 0)

    # A wider, wrong window that must NOT be what gets reported.
    manager.record_pour_window(start_datetime=start - timedelta(seconds=100),
                                end_datetime=start + timedelta(seconds=100), duration_seconds=200.0)
    manager.upsert_completed_mould_pouring(
        ladle_track_id=7, mould_id="MOULD_C1", mould_track_id=41,
        start_time=start.timestamp(), start_datetime=start,
        end_time=(start + timedelta(seconds=3)).timestamp(),
        end_datetime=start + timedelta(seconds=3), duration_seconds=3.0,
    )

    cycle = manager.active_cycle
    manager._finalize_cycle(cycle, (start + timedelta(seconds=3)).timestamp(),
                             start + timedelta(seconds=3))

    assert cycle.pouring_start_time == start
    assert cycle.pouring_end_time == start + timedelta(seconds=3)
    assert cycle.total_pouring_time == 3
    assert len(cycle.mould_wise_pouring_time) == 1


def test_unattributed_pours_with_no_tapping_are_not_dropped(tmp_path):
    """The early-return guard used to fire on 'no moulds AND no tapping',
    which would have swallowed a pours-but-no-tapping cycle entirely."""
    manager = HeatCycleManager(_make_db(tmp_path), ladle_absence_timeout=300.0)
    start = datetime(2026, 8, 8, 11, 6, 3)
    manager.update_pouring_session_presence(7, start.timestamp(), start)

    manager.record_pour_window(start_datetime=start, end_datetime=start + timedelta(seconds=7),
                                duration_seconds=7.0)
    cycle = manager.active_cycle
    assert not cycle.tapping_events

    manager._finalize_cycle(cycle, (start + timedelta(seconds=7)).timestamp(),
                             start + timedelta(seconds=7))

    assert cycle.pouring_start_time == start
    assert cycle.total_pouring_time == 7


def test_pour_window_survives_checkpoint_round_trip(tmp_path):
    manager = HeatCycleManager(_make_db(tmp_path), ladle_absence_timeout=300.0)
    start = datetime(2026, 8, 8, 9, 0, 0)
    manager.update_pouring_session_presence(7, start.timestamp(), start)
    manager.record_pour_window(start_datetime=start, end_datetime=start + timedelta(seconds=5),
                                duration_seconds=5.0)

    as_dict = manager._cycle_to_dict(manager.active_cycle)
    restored = HeatCycleManager._cycle_from_dict(as_dict)
    assert restored.pour_window_start_datetime == start
    assert restored.pour_window_end_datetime == start + timedelta(seconds=5)
    assert restored.pour_total_seconds == 5.0
    assert restored.pour_count == 1

    # A checkpoint written before these fields existed must still load.
    stale = dict(as_dict)
    for key in ("pour_window_start_datetime", "pour_window_end_datetime",
                "pour_total_seconds", "pour_count"):
        stale.pop(key, None)
    restored_stale = HeatCycleManager._cycle_from_dict(stale)
    assert restored_stale.pour_window_start_datetime is None
    assert restored_stale.pour_total_seconds == 0.0
    assert restored_stale.pour_count == 0
