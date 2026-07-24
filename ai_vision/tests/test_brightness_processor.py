from pathlib import Path
from types import SimpleNamespace

import numpy as np

from processors.brightness_processor import BrightnessProcessor


class FakeDB:
    def __init__(self):
        self.events = []

    def insert_melting_event(self, **kwargs):
        self.events.append(kwargs)


class FakeHeatCycleManager:
    def __init__(self):
        self.calls = []
        self.active_cycle = SimpleNamespace(furnace_label=None, locked_trolley_id=None)

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


def _zones_config():
    return {
        "metadata": {"ref_width": 40, "ref_height": 40},
        "tapping": {
            "zones": {
                "tap-2": {"roi_points": [[5, 5], [35, 5], [35, 35], [5, 35]]}
            },
            "abs_brightness_threshold": 200,
            "start_white_ratio": 0.5,
            "start_frame_count": 2,
            "end_white_ratio": 0.5,
            "end_frame_count": 2,
        },
        "deslagging": {"zones": {}},
        "spectro": {"zones": {}},
    }


def test_brightness_processor_tags_tapping_event_with_zone_name(tmp_path):
    db = FakeDB()
    heat_cycle = FakeHeatCycleManager()
    processor = BrightnessProcessor(
        zones_config=_zones_config(),
        db_manager=db,
        config=DummyConfig,
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=heat_cycle,
        enable_display_meta=False,
    )

    bright_frame = np.zeros((40, 40, 4), dtype=np.uint8)
    bright_frame[5:35, 5:35, :3] = 255
    dark_frame = np.zeros((40, 40, 4), dtype=np.uint8)

    for _ in range(2):
        processor.process_frame_with_array(bright_frame, frame_meta=None)
    assert db.events == []

    for _ in range(2):
        processor.process_frame_with_array(dark_frame, frame_meta=None)

    assert len(db.events) == 1
    event = db.events[0]
    assert event["event_type"] == "tapping"
    assert event["zone_name"] == "tap-2"
    assert event["screenshot_path"]
    assert Path(event["screenshot_path"]).exists()
    assert heat_cycle.calls[0][0] == "tapping"
    assert heat_cycle.calls[0][1]["zone_name"] == "tap-2"
