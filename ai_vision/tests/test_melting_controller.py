from pathlib import Path
from types import SimpleNamespace

import numpy as np

from processors.melting_analysis_controller import MeltingAnalysisController
from processors.melting_meta_reader import DecodedMeltingState, DecodedMeltingZoneState


class FakeDB:
    def __init__(self):
        self.events = []

    def insert_melting_event(self, **kwargs):
        self.events.append(kwargs)


class FakeHeatCycleManager:
    def __init__(self):
        self.calls = []
        self.active_cycle = SimpleNamespace(furnace_label="Furnace1")

    def add_tapping_event(self, **kwargs):
        self.calls.append(("tapping", kwargs))

    def add_deslagging_event(self, **kwargs):
        self.calls.append(("deslagging", kwargs))

    def add_spectro_event(self, **kwargs):
        self.calls.append(("spectro", kwargs))


class DummyConfig:
    CUSTOMER_ID = "cust"
    CAMERA_ID_STREAM_0 = "cam0"
    LOCATION = "plant"


def _zone_state(active=False, white_ratio=0.0, raw_count=0, filtered_count=0):
    return DecodedMeltingZoneState(
        valid=True,
        active=active,
        raw_count=raw_count,
        filtered_count=filtered_count,
        white_ratio=white_ratio,
        max_blob_area=0.0,
        max_blob_brightness=0.0,
    )


def _native_state(
    tapping_active=False,
    deslagging_active=False,
    spectro_active=False,
):
    return DecodedMeltingState(
        version=1,
        debug_code=0,
        blackout_active=False,
        frame_num=1,
        ntp_timestamp=0,
        tapping=[_zone_state(active=tapping_active, white_ratio=0.4 if tapping_active else 0.0)],
        deslagging=[_zone_state(active=deslagging_active, raw_count=1 if deslagging_active else 0)],
        spectro=[_zone_state(active=spectro_active, raw_count=1 if spectro_active else 0)],
    )


def test_melting_controller_emits_tapping_event_and_heat_cycle(tmp_path):
    db = FakeDB()
    heat_cycle = FakeHeatCycleManager()
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
    controller = MeltingAnalysisController(
        zones_config=zones,
        db_manager=db,
        config=DummyConfig,
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle,
        enable_display_meta=False,
    )

    start_state = _native_state(tapping_active=True)
    end_state = _native_state(tapping_active=False)
    frame = np.zeros((40, 40, 4), dtype=np.uint8)

    assert controller.needs_frame(start_state) is True
    controller.process_native_state(
        native_state=start_state,
        frame_meta=None,
        frame=frame,
        timestamp=10.0,
    )
    assert db.events == []
    assert controller.needs_frame(end_state) is True
    controller.process_native_state(
        native_state=end_state,
        frame_meta=None,
        frame=frame,
        timestamp=14.5,
    )

    assert len(db.events) == 1
    event = db.events[0]
    assert event["event_type"] == "tapping"
    assert event["duration_sec"] == 4.5
    assert event["zone_name"] == "tap-1"
    assert event["screenshot_path"]
    assert Path(event["screenshot_path"]).exists()
    assert heat_cycle.calls[0][0] == "tapping"
    assert heat_cycle.calls[0][1]["zone_name"] == "tap-1"


def _melting_zones():
    return {
        "metadata": {"ref_width": 40, "ref_height": 40},
        "tapping": {
            "zones": {
                "tap-1": {"roi_points": [[5, 5], [20, 5], [20, 20], [5, 20]]}
            },
            "abs_brightness_threshold": 230,
            "start_white_ratio": 0.18,
            "start_frame_count": 20,
            "end_white_ratio": 0.1,
            "end_frame_count": 25,
        },
        "deslagging": {
            "zones": {
                "deslag-1": {"roi_points": [[5, 5], [20, 5], [20, 20], [5, 20]]}
            },
            "min_blob_area": 500,
            "brightness_thresh": 200,
        },
        "spectro": {
            "zones": {
                "spectro-zone": {
                    "roi_points": [[5, 5], [20, 5], [20, 20], [5, 20]],
                    "on_frames": 0,
                    "max_aspect_ratio": None,
                    "max_coverage": None,
                }
            },
            "min_blob_area": 50,
            "brightness_thresh": 180,
        },
    }


def test_melting_controller_rejects_spectro_until_after_deslag_window(tmp_path):
    db = FakeDB()
    heat_cycle = FakeHeatCycleManager()
    controller = MeltingAnalysisController(
        zones_config=_melting_zones(),
        db_manager=db,
        config=DummyConfig,
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle,
        enable_display_meta=False,
    )

    frame = np.zeros((40, 40, 4), dtype=np.uint8)
    controller.process_native_state(_native_state(spectro_active=True), None, frame, timestamp=8.0)
    controller.process_native_state(_native_state(), None, frame, timestamp=8.5)
    assert db.events == []

    controller.process_native_state(_native_state(deslagging_active=True), None, frame, timestamp=10.0)
    controller.process_native_state(_native_state(spectro_active=True), None, frame, timestamp=12.0)
    controller.process_native_state(_native_state(), None, frame, timestamp=12.5)
    assert [event["event_type"] for event in db.events] == ["deslagging"]

    controller.process_native_state(_native_state(spectro_active=True), None, frame, timestamp=16.1)
    controller.process_native_state(_native_state(), None, frame, timestamp=17.0)
    assert [event["event_type"] for event in db.events] == ["deslagging", "spectro"]
    assert db.events[-1]["zone_name"] == "Furnace1"


def test_melting_controller_tapping_disables_deslagging_and_spectro(tmp_path):
    db = FakeDB()
    heat_cycle = FakeHeatCycleManager()
    controller = MeltingAnalysisController(
        zones_config=_melting_zones(),
        db_manager=db,
        config=DummyConfig,
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle,
        enable_display_meta=False,
    )

    frame = np.zeros((40, 40, 4), dtype=np.uint8)
    controller.process_native_state(_native_state(tapping_active=True), None, frame, timestamp=1.0)
    controller.process_native_state(_native_state(deslagging_active=True), None, frame, timestamp=2.0)
    controller.process_native_state(_native_state(spectro_active=True), None, frame, timestamp=3.0)
    controller.process_native_state(_native_state(), None, frame, timestamp=4.0)

    assert [event["event_type"] for event in db.events] == ["tapping"]
    assert [call[0] for call in heat_cycle.calls] == ["tapping"]
