from state.brightness_tracker import BrightnessTracker


def test_brightness_tracker_emits_event_when_max_not_exceeded():
    tracker = BrightnessTracker(
        name="spectro",
        brightness_threshold=250,
        start_white_ratio=0.03,
        start_frame_count=3,
        end_white_ratio=0.03,
        end_frame_count=3,
        max_white_ratio=0.20,
    )

    for _ in range(3):
        tracker.update(0.05)
    assert tracker.is_active

    event = None
    for _ in range(3):
        event = tracker.update(0.01)

    assert event is not None
    assert event["type"] == "spectro"
    assert tracker.state == "IDLE"


def test_brightness_tracker_discards_event_when_max_exceeded():
    tracker = BrightnessTracker(
        name="spectro",
        brightness_threshold=250,
        start_white_ratio=0.03,
        start_frame_count=3,
        end_white_ratio=0.03,
        end_frame_count=3,
        max_white_ratio=0.20,
    )

    for _ in range(3):
        tracker.update(0.05)
    assert tracker.is_active

    # exceed max ratio while ACTIVE
    tracker.update(0.25)

    event = None
    for _ in range(3):
        event = tracker.update(0.01)

    assert event is None
    assert tracker.state == "IDLE"
