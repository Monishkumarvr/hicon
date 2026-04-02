from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from db_manager import HiConDatabase
from processors.pouring_analysis_controller import HybridPouringController
from processors.pouring_meta_reader import DecodedPouringState, EVENT_SESSION_START
from processors.pouring_processor import PouringProcessor
from state.heat_cycle_manager import HeatCycleManager


class DummyDB:
    def __init__(self):
        self.inserted_heat_cycles = []
        self.inserted_pouring_events = []

    def insert_pouring_event(self, **kwargs):
        self.inserted_pouring_events.append(kwargs)
        return 1

    def update_pouring_end(self, **kwargs):
        return None

    def insert_heat_cycle(self, **kwargs):
        self.inserted_heat_cycles.append(kwargs)
        return "0001"


class DummyConfig:
    CUSTOMER_ID = "C1"
    LOCATION = "Loc"
    CAMERA_ID = "Cam-0"
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
    POURING_CYCLE_TIMEOUT_S = 300.0
    ENABLE_INFERENCE_VIDEO = False
    VIDEO_DIR = Path("/tmp")


class CaptureHeatCycleManager:
    def __init__(self):
        self.calls = []

    def update_pouring_session_presence(self, track_id, current_time, current_datetime):
        self.calls.append((track_id, current_time, current_datetime))

    def check_and_finalize_cycles(self, current_time, current_datetime):
        return []


class HeatBootstrapManager(CaptureHeatCycleManager):
    def __init__(self, heat_no="HEAT_TEST"):
        super().__init__()
        self.active_cycle = None
        self.heat_no = heat_no

    def update_pouring_session_presence(self, track_id, current_time, current_datetime):
        super().update_pouring_session_presence(track_id, current_time, current_datetime)
        self.active_cycle = SimpleNamespace(heat_no=self.heat_no)


class FinalizingHeatCycleManager:
    def __init__(self, cycle):
        self.cycle = cycle
        self.calls = 0

    def check_and_finalize_cycles(self, current_time, current_datetime):
        self.calls += 1
        if self.calls == 1:
            return [self.cycle]
        return []


def _make_db(tmp_path):
    return HiConDatabase(str(tmp_path / "heat_cycle.sqlite"))


def _make_processor(tmp_path, heat_cycle_manager):
    return PouringProcessor(
        db_manager=DummyDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle_manager,
    )


def _make_hybrid_controller(tmp_path, heat_cycle_manager):
    return HybridPouringController(
        db_manager=DummyDB(),
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle_manager,
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
    manager = HeatCycleManager(db, ladle_absence_timeout=300.0)

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
    assert cycle.pouring_start_time is None
    assert cycle.pouring_end_time is None


def test_non_creator_events_do_not_create_cycle(tmp_path):
    db = _make_db(tmp_path)
    manager = HeatCycleManager(db, ladle_absence_timeout=300.0)
    now = datetime(2026, 3, 31, 12, 0, 0)

    manager.add_deslagging_event(1.0, now, 2.0, now + timedelta(seconds=1), 1.0)
    manager.add_spectro_event(3.0, now, 4.0, now + timedelta(seconds=1), 1.0)
    manager.add_pyrometer_event(5.0, now, 6.0, now + timedelta(seconds=1), 1.0)

    assert manager.active_cycle is None


def test_pouring_processor_refreshes_heat_cycle_only_after_session_is_active(tmp_path):
    heat_cycle_manager = CaptureHeatCycleManager()
    processor = _make_processor(tmp_path, heat_cycle_manager)

    mouth = {
        "track_id": 5,
        "confidence": 0.95,
        "bbox": (120, 120, 150, 150),
        "bottom_center": (135, 150),
    }
    trolley = {"track_id": 7, "bbox": (100, 100, 200, 220), "confidence": 0.9}
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)

    processor._extract_detections = lambda _frame_meta: ([mouth], [trolley])
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
    processor.process_frame(frame_meta, object(), 11.0, datetime(2026, 3, 31, 13, 0, 1))
    assert len(heat_cycle_manager.calls) == 1
    assert heat_cycle_manager.calls[0][0] == 5


def test_hybrid_controller_refreshes_heat_cycle_only_for_mouth_in_trolley(tmp_path):
    heat_cycle_manager = CaptureHeatCycleManager()
    controller = _make_hybrid_controller(tmp_path, heat_cycle_manager)
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)

    active_state = DecodedPouringState(
        version=2,
        session_active=True,
        mouth_present_in_trolley=True,
        probe_valid=True,
        event=EVENT_SESSION_START,
        trolley_track_id=7,
        mouth_track_id=5,
        trolley_bbox=(100.0, 100.0, 200.0, 220.0),
        mouth_bbox=(120.0, 120.0, 150.0, 150.0),
        probe_x_px=135.0,
        probe_y_px=200.0,
        mouth_norm_x=0.35,
        mouth_norm_y=0.40,
    )
    controller.process_native_state(
        frame_meta=frame_meta,
        native_state=active_state,
        frame=None,
        timestamp=10.0,
        datetime_obj=datetime(2026, 3, 31, 13, 0, 0),
    )
    assert len(heat_cycle_manager.calls) == 1
    assert heat_cycle_manager.calls[0][0] == 5

    heat_cycle_manager.calls.clear()
    no_presence_state = DecodedPouringState(
        version=2,
        session_active=True,
        mouth_present_in_trolley=False,
        probe_valid=True,
        event=0,
        trolley_track_id=7,
        mouth_track_id=5,
        trolley_bbox=(100.0, 100.0, 200.0, 220.0),
        mouth_bbox=(120.0, 120.0, 150.0, 150.0),
        probe_x_px=135.0,
        probe_y_px=200.0,
        mouth_norm_x=0.35,
        mouth_norm_y=0.40,
    )
    controller.process_native_state(
        frame_meta=frame_meta,
        native_state=no_presence_state,
        frame=None,
        timestamp=11.0,
        datetime_obj=datetime(2026, 3, 31, 13, 0, 1),
    )
    assert heat_cycle_manager.calls == []


def test_hybrid_controller_needs_frame_for_active_native_session(tmp_path):
    controller = _make_hybrid_controller(tmp_path, heat_cycle_manager=None)
    idle_state = DecodedPouringState(
        version=2,
        session_active=False,
        mouth_present_in_trolley=False,
        probe_valid=False,
        event=0,
        trolley_track_id=0,
        mouth_track_id=0,
        trolley_bbox=(0.0, 0.0, 0.0, 0.0),
        mouth_bbox=(0.0, 0.0, 0.0, 0.0),
        probe_x_px=0.0,
        probe_y_px=0.0,
        mouth_norm_x=-1.0,
        mouth_norm_y=-1.0,
    )
    active_state = DecodedPouringState(
        version=2,
        session_active=True,
        mouth_present_in_trolley=True,
        probe_valid=True,
        event=EVENT_SESSION_START,
        trolley_track_id=7,
        mouth_track_id=5,
        trolley_bbox=(100.0, 100.0, 200.0, 220.0),
        mouth_bbox=(120.0, 120.0, 150.0, 150.0),
        probe_x_px=135.0,
        probe_y_px=200.0,
        mouth_norm_x=0.35,
        mouth_norm_y=0.40,
    )

    assert controller.needs_frame(idle_state) is False
    assert controller.needs_frame(active_state) is True


def test_hybrid_controller_inserts_heat_cycle_without_ladle_number_or_fallback_name_error(tmp_path):
    db = DummyDB()
    controller = HybridPouringController(
        db_manager=db,
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=None,
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
    )

    controller._insert_heat_cycle_to_db(cycle)

    assert len(db.inserted_heat_cycles) == 1
    inserted = db.inserted_heat_cycles[0]
    assert inserted["ladle_number"] == ""
    assert inserted["cycle_end_time"]
    assert inserted["pouring_start_time"] == ""
    assert inserted["pouring_end_time"] == ""


def test_pouring_processor_refreshes_heat_before_pour_logic_runs(tmp_path):
    heat_cycle_manager = HeatBootstrapManager("HEAT_1234")
    db = DummyDB()
    processor = PouringProcessor(
        db_manager=db,
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle_manager,
    )

    mouth = {
        "track_id": 5,
        "confidence": 0.95,
        "bbox": (120, 120, 150, 150),
        "bottom_center": (135, 150),
    }
    trolley = {"track_id": 7, "bbox": (100, 100, 200, 220), "confidence": 0.9}
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)

    processor._extract_detections = lambda _frame_meta: ([mouth], [trolley])
    processor._get_target_trolley = lambda _trolleys, _timestamp=None, _mouths=None: trolley
    processor._select_best_mouth_for_trolley = lambda _mouths, _trolley: mouth
    processor._check_cycle_timeout = lambda *args, **kwargs: None
    processor._update_session = lambda *args, **kwargs: setattr(processor, "session_active", True)

    seen = {}

    def _capture_update_pour(*args, **kwargs):
        seen["heat_no"] = getattr(
            getattr(heat_cycle_manager, "active_cycle", None),
            "heat_no",
            None,
        )

    processor._update_pour = _capture_update_pour
    processor.process_frame(
        frame_meta,
        np.zeros((8, 8, 4), dtype=np.uint8),
        11.0,
        datetime(2026, 3, 31, 13, 0, 1),
    )

    assert seen["heat_no"] == "HEAT_1234"


def test_hybrid_controller_refreshes_heat_before_pour_logic_runs(tmp_path):
    heat_cycle_manager = HeatBootstrapManager("HEAT_5678")
    controller = _make_hybrid_controller(tmp_path, heat_cycle_manager)
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)

    active_state = DecodedPouringState(
        version=2,
        session_active=True,
        mouth_present_in_trolley=True,
        probe_valid=True,
        event=EVENT_SESSION_START,
        trolley_track_id=7,
        mouth_track_id=5,
        trolley_bbox=(100.0, 100.0, 200.0, 220.0),
        mouth_bbox=(120.0, 120.0, 150.0, 150.0),
        probe_x_px=135.0,
        probe_y_px=200.0,
        mouth_norm_x=0.35,
        mouth_norm_y=0.40,
    )

    seen = {}

    def _capture_update_pour(*args, **kwargs):
        seen["heat_no"] = getattr(
            getattr(heat_cycle_manager, "active_cycle", None),
            "heat_no",
            None,
        )

    controller._update_pour = _capture_update_pour
    controller._check_cycle_timeout = lambda *args, **kwargs: None

    controller.process_native_state(
        frame_meta=frame_meta,
        native_state=active_state,
        frame=np.zeros((8, 8, 4), dtype=np.uint8),
        timestamp=10.0,
        datetime_obj=datetime(2026, 3, 31, 13, 0, 0),
    )

    assert seen["heat_no"] == "HEAT_5678"


def test_hybrid_controller_uses_actual_frame_detections_for_pour_logic(tmp_path):
    controller = _make_hybrid_controller(tmp_path, heat_cycle_manager=None)
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)
    actual_trolley = {"track_id": 7, "bbox": (100, 100, 200, 220), "confidence": 0.9}
    actual_mouth = {
        "track_id": 5,
        "confidence": 0.95,
        "bbox": (120, 120, 150, 150),
        "center": (135, 135),
        "bottom_center": (135, 150),
    }
    controller._extract_detections = lambda _frame_meta: ([actual_mouth], [actual_trolley])

    seen = {}

    def _capture_update_pour(mouths, _frame, _timestamp, _datetime_obj, _trolleys, target_trolley):
        seen["mouths"] = mouths
        seen["target_trolley"] = target_trolley
        return actual_mouth

    controller._update_pour = _capture_update_pour
    controller._check_cycle_timeout = lambda *args, **kwargs: None

    active_state = DecodedPouringState(
        version=2,
        session_active=True,
        mouth_present_in_trolley=True,
        probe_valid=True,
        event=EVENT_SESSION_START,
        trolley_track_id=7,
        mouth_track_id=5,
        trolley_bbox=(100.0, 100.0, 200.0, 220.0),
        mouth_bbox=(120.0, 120.0, 150.0, 150.0),
        probe_x_px=135.0,
        probe_y_px=180.0,
        mouth_norm_x=0.35,
        mouth_norm_y=0.40,
    )

    controller.process_native_state(
        frame_meta=frame_meta,
        native_state=active_state,
        frame=np.zeros((8, 8, 4), dtype=np.uint8),
        timestamp=10.0,
        datetime_obj=datetime(2026, 3, 31, 13, 0, 0),
    )

    assert seen["mouths"][0]["track_id"] == 5
    assert seen["target_trolley"]["track_id"] == 7


def test_hybrid_controller_finalizes_tapping_only_cycle_without_native_meta(tmp_path):
    db = DummyDB()
    controller = HybridPouringController(
        db_manager=db,
        config=DummyConfig(),
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=None,
    )

    cycle = SimpleNamespace(
        heat_no="HEAT_9001",
        mould_pourings=[],
        tapping_events=[{"start": "2026-03-31T14:00:00", "end": "2026-03-31T14:00:10"}],
        cycle_start_datetime=datetime(2026, 3, 31, 14, 0, 0),
        cycle_end_datetime=datetime(2026, 3, 31, 14, 0, 10),
        pouring_start_time=None,
        pouring_end_time=None,
        total_pouring_time=0,
        mould_wise_pouring_time=[],
        tapping_start_datetime=datetime(2026, 3, 31, 14, 0, 0),
        tapping_end_datetime=datetime(2026, 3, 31, 14, 0, 10),
        deslagging_events=[],
        spectro_events=[],
        pyrometer_events=[],
    )
    manager = FinalizingHeatCycleManager(cycle)
    controller.heat_cycle_manager = manager
    frame_meta = SimpleNamespace(source_frame_width=1280, source_frame_height=720)
    controller._check_cycle_timeout = lambda *args, **kwargs: None

    for idx in range(10):
        controller.process_native_state(
            frame_meta=frame_meta,
            native_state=None,
            frame=None,
            timestamp=400.0 + idx,
            datetime_obj=datetime(2026, 3, 31, 14, 5, idx),
        )

    assert len(db.inserted_heat_cycles) == 1
    inserted = db.inserted_heat_cycles[0]
    assert inserted["heat_no"] == "HEAT_9001"
    assert inserted["pouring_start_time"] == ""
    assert inserted["pouring_end_time"] == ""
