import pytest

pytest.importorskip("pyds")

from processors.pyrometer_processor import PyrometerProcessor


def test_pyrometer_zone_check_uses_top_left_and_bottom_left():
    processor = object.__new__(PyrometerProcessor)
    zone = [(0, 0), (100, 0), (100, 100), (0, 100)]

    assert processor._any_detection_in_zone(
        [{"top_left": (10, 10), "bottom_left": (10, 90)}],
        zone,
    )
    assert not processor._any_detection_in_zone(
        [{"top_left": (10, 10), "bottom_left": (-1, 90)}],
        zone,
    )
