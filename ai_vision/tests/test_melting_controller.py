from pathlib import Path

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


def _native_state(tapping_active=False):
    return DecodedMeltingState(
        version=1,
        debug_code=0,
        blackout_active=False,
        frame_num=1,
        ntp_timestamp=0,
        tapping=[_zone_state(active=tapping_active, white_ratio=0.4 if tapping_active else 0.0)],
        deslagging=[],
        spectro=[],
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
    assert event["screenshot_path"]
    assert Path(event["screenshot_path"]).exists()
    assert heat_cycle.calls[0][0] == "tapping"
