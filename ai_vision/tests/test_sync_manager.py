import json

from sync.sync_manager import SyncManager


class FakeDB:
    def __init__(self, cycles):
        self.cycles = cycles
        self.synced_ids = []
        self.sync_errors = []
        self.synced_windows = []

    def get_unsynced_heat_cycles(self, limit=50):
        return self.cycles[:limit]

    def mark_heat_cycles_synced(self, sync_ids):
        self.synced_ids.extend(sync_ids)

    def update_heat_cycle_sync_status(self, sync_id, error):
        self.sync_errors.append((sync_id, error))

    def mark_melting_events_synced_by_window(self, start_time, end_time):
        self.synced_windows.append((start_time, end_time))


class FakeAPI:
    def __init__(self):
        self.melting_batches = []
        self.pouring_batches = []

    def send_melting_data(self, items):
        self.melting_batches.append(items)
        return {
            "results": [
                {"sync_id": item["sync_id"], "success": True}
                for item in items
            ]
        }

    def send_pouring_data(self, items):
        self.pouring_batches.append(items)
        return {
            "results": [
                {"sync_id": item["sync_id"], "success": True}
                for item in items
            ]
        }


def test_sync_manager_uses_cycle_furnace_for_agni_location():
    cycle = {
        "sync_id": "cycle-1",
        "heat_no": "HEAT_0001",
        "customer_id": "cust-1",
        "date": "2026-04-11",
        "location": "Casting Section Furnace1",
        "camera_id": "Cam-Process",
        "cycle_start_time": "2026-04-11T10:00:00",
        "cycle_end_time": "2026-04-11T10:20:00",
        "pouring_start_time": "2026-04-11T10:05:00",
        "pouring_end_time": "2026-04-11T10:10:00",
        "total_pouring_time": "300",
        "mould_wise_pouring_time": json.dumps([]),
        "tapping_start_time": "2026-04-11T10:06:00",
        "tapping_end_time": "2026-04-11T10:09:00",
        "tapping_events": json.dumps([
            {
                "start": "2026-04-11T10:06:00",
                "end": "2026-04-11T10:07:00",
                "duration_sec": 60.0,
                "zone_name": "tap-1",
            },
        ]),
        "deslagging_events": json.dumps([
            {
                "start": "2026-04-11T10:02:00",
                "end": "2026-04-11T10:03:00",
                "duration_sec": 60.0,
                "zone_name": "zone-1",
            }
        ]),
        "spectro_events": json.dumps([
            {
                "start": "2026-04-11T10:04:00",
                "end": "2026-04-11T10:04:30",
                "duration_sec": 30.0,
                "zone_name": "zone-2",
            }
        ]),
        "pyrometer_events": json.dumps([
            {
                "start": "2026-04-11T10:01:00",
                "end": "2026-04-11T10:01:10",
                "duration_sec": 10.0,
                "zone_name": "furnace-1",
            },
        ]),
    }
    db = FakeDB([cycle])
    api = FakeAPI()
    manager = SyncManager(
        database=db,
        api_client=api,
        customer_id="cust-1",
        camera_id="Cam-Process",
        location="Casting Section",
        furnace_id="",
    )

    melting_synced, pouring_synced, finalized_synced = manager._sync_heat_cycles()

    assert melting_synced == 1
    assert pouring_synced == 1
    assert finalized_synced == 1
    assert db.synced_ids == ["cycle-1"]

    assert len(api.melting_batches) == 1
    melting_items = api.melting_batches[0]
    assert len(melting_items) == 1
    assert melting_items[0]["location"] == "Casting Section Furnace1"
    assert melting_items[0]["furnace"] == "Furnace1"
    assert melting_items[0]["tapping_start_time"] == "2026-04-11 10:06:00"
    assert melting_items[0]["tapping_end_time"] == "2026-04-11 10:07:00"
    assert melting_items[0]["deslagging"] is True
    assert melting_items[0]["spectro"] is True
    assert melting_items[0]["pyrometer"] is True


def _incomplete_pouring_cycle(has_pouring_session):
    return {
        "sync_id": "cycle-2",
        "heat_no": "HEAT_0002",
        "customer_id": "cust-1",
        "date": "2026-08-07",
        "location": "Casting Section",
        "camera_id": "Cam-Process",
        "cycle_start_time": "2026-08-07T10:00:00",
        "cycle_end_time": "2026-08-07T10:20:00",
        "pouring_start_time": "",
        "pouring_end_time": "",
        "total_pouring_time": "0",
        "mould_wise_pouring_time": json.dumps([]),
        "tapping_start_time": None,
        "tapping_end_time": None,
        "tapping_events": json.dumps([]),
        "deslagging_events": json.dumps([]),
        "spectro_events": json.dumps([]),
        "pyrometer_events": json.dumps([]),
        "has_pouring_session": has_pouring_session,
    }


def test_sync_manager_errors_loudly_when_pouring_session_missing_aggregate(caplog):
    """Regression for hicon-3q4: a heat cycle that had a pouring session but
    whose pouring aggregate came out empty (the detected-pour -> heat-cycle
    bridge bug) must be logged as an error, not silently skipped — this exact
    combination went unnoticed for over a week because it only ever hit
    logger.debug."""
    db = FakeDB([_incomplete_pouring_cycle(has_pouring_session=1)])
    api = FakeAPI()
    manager = SyncManager(
        database=db, api_client=api, customer_id="cust-1",
        camera_id="Cam-Process", location="Casting Section", furnace_id="",
    )

    with caplog.at_level("ERROR", logger="sync.sync_manager"):
        manager._sync_heat_cycles()

    assert len(api.pouring_batches) == 0  # never attempts the incomplete payload
    assert any(
        "HEAT_0002" in record.message and "pouring session" in record.message
        for record in caplog.records if record.levelname == "ERROR"
    )


def test_sync_manager_stays_silent_for_a_genuine_no_pouring_cycle(caplog):
    """A tapping-only cycle that never had a pouring session at all is the
    legitimate case — no pouring data is expected, so no error."""
    db = FakeDB([_incomplete_pouring_cycle(has_pouring_session=0)])
    api = FakeAPI()
    manager = SyncManager(
        database=db, api_client=api, customer_id="cust-1",
        camera_id="Cam-Process", location="Casting Section", furnace_id="",
    )

    with caplog.at_level("DEBUG", logger="sync.sync_manager"):
        manager._sync_heat_cycles()

    assert len(api.pouring_batches) == 0
    assert not any(record.levelname == "ERROR" for record in caplog.records)
