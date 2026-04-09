import queue
from datetime import datetime
from pathlib import Path

import numpy as np

from db_manager import AsyncDBWriter
from utils.screenshot import AsyncScreenshotWriter


class FakeDB:
    def __init__(self):
        self.calls = []

    def insert_pouring_event(self, **kwargs):
        self.calls.append(("insert_pouring_event", kwargs))
        return 11

    def update_pouring_end(self, **kwargs):
        self.calls.append(("update_pouring_end", kwargs))
        return None

    def insert_heat_cycle(self, **kwargs):
        self.calls.append(("insert_heat_cycle", kwargs))
        return 22

    def insert_melting_event(self, **kwargs):
        self.calls.append(("insert_melting_event", kwargs))
        return 33

    def delete_pouring_event(self, **kwargs):
        self.calls.append(("delete_pouring_event", kwargs))
        return None


def test_async_db_writer_drains_pending_writes_on_stop():
    db = FakeDB()
    writer = AsyncDBWriter(db, maxsize=4)
    try:
        assert writer.insert_pouring_event(sync_id="pour-1") is None
        writer.update_pouring_end(sync_id="pour-1")
        writer.insert_heat_cycle(sync_id="heat-1")
        writer.stop(timeout=5.0)
    finally:
        writer.stop(timeout=0.1, drain=False)

    assert [name for name, _ in db.calls] == [
        "insert_pouring_event",
        "update_pouring_end",
        "insert_heat_cycle",
    ]


def test_async_db_writer_falls_back_to_sync_when_queue_is_full(monkeypatch):
    db = FakeDB()
    writer = AsyncDBWriter(db, maxsize=1)
    try:
        monkeypatch.setattr(
            writer._queue,
            "put_nowait",
            lambda item: (_ for _ in ()).throw(queue.Full),
        )
        assert writer.insert_pouring_event(sync_id="pour-2") == 11
    finally:
        writer.stop(timeout=0.1, drain=False)

    assert db.calls == [("insert_pouring_event", {"sync_id": "pour-2"})]


def test_async_screenshot_writer_returns_path_and_flushes_on_stop(tmp_path):
    writer = AsyncScreenshotWriter(maxsize=4)
    try:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        timestamp = datetime(2026, 4, 9, 18, 30, 0)
        output_path = writer.save(
            frame,
            prefix="cam0",
            tag="event",
            timestamp=timestamp,
            screenshot_dir=Path(tmp_path),
        )
        assert output_path is not None
        assert output_path.endswith("cam0_event_20260409_183000.jpg")
        writer.stop(timeout=5.0)
    finally:
        writer.stop(timeout=0.1, drain=False)

    assert Path(output_path).exists()


def test_async_screenshot_writer_drops_new_jobs_when_queue_is_full(monkeypatch, tmp_path):
    writer = AsyncScreenshotWriter(maxsize=1)
    try:
        monkeypatch.setattr(
            writer._queue,
            "put_nowait",
            lambda item: (_ for _ in ()).throw(queue.Full),
        )
        output_path = writer.save(
            np.zeros((8, 8, 3), dtype=np.uint8),
            prefix="cam0",
            tag="event",
            timestamp=datetime(2026, 4, 9, 18, 31, 0),
            screenshot_dir=Path(tmp_path),
        )
    finally:
        writer.stop(timeout=0.1, drain=False)

    assert output_path is None
