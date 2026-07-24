import json
import sqlite3
from pathlib import Path

import pytest

from tools.replicate_demo_data import (
    ImportValidationError,
    execute_manifest,
    select_latest_paired_cycles,
    submit_with_failed_only_retries,
    validate_manifest,
)


def _create_source_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE heat_cycles (
            sync_id TEXT UNIQUE NOT NULL,
            heat_no TEXT UNIQUE NOT NULL,
            customer_id TEXT NOT NULL,
            date TEXT NOT NULL,
            location TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            cycle_start_time TEXT NOT NULL,
            cycle_end_time TEXT NOT NULL,
            pouring_start_time TEXT NOT NULL,
            pouring_end_time TEXT NOT NULL,
            total_pouring_time TEXT NOT NULL,
            mould_wise_pouring_time TEXT NOT NULL,
            synced INTEGER NOT NULL,
            tapping_start_time TEXT,
            tapping_end_time TEXT,
            tapping_events TEXT,
            deslagging_events TEXT,
            spectro_events TEXT,
            pyrometer_events TEXT
        )
        """
    )
    for index in range(35):
        day = index + 1
        date = f"2026-07-{day:02d}" if day <= 31 else "2026-08-01"
        hour = index % 24
        start = f"{date}T{hour:02d}:00:00"
        end = f"{date}T{hour:02d}:30:00"
        pour_start = f"{date}T{hour:02d}:20:00"
        pour_end = f"{date}T{hour:02d}:25:00"
        tapping_events = json.dumps(
            [
                {
                    "start": f"{date}T{hour:02d}:10:00",
                    "end": f"{date}T{hour:02d}:12:00",
                    "duration_sec": 120,
                }
            ]
        )
        mould_times = json.dumps(
            [
                {
                    "mould_id": "MOULD_1",
                    "start": pour_start,
                    "end": pour_end,
                    "duration": "300",
                }
            ]
        )
        connection.execute(
            """
            INSERT INTO heat_cycles VALUES (
                ?, ?, '1157', ?, 'Casting Section', 'Cam-Process',
                ?, ?, ?, ?, '300', ?, 1, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                f"heat_cycle-{index}",
                f"HEAT_{index:04d}",
                date,
                start,
                end,
                pour_start,
                pour_end,
                mould_times,
                f"{date}T{hour:02d}:10:00",
                f"{date}T{hour:02d}:12:00",
                tapping_events,
                json.dumps([{"event": "deslag"}]) if index % 2 else "[]",
                json.dumps([{"event": "spectro"}]),
                json.dumps([{"event": "pyro"}]) if index % 3 else "[]",
            ),
        )

    # Newer rows that must not displace valid paired source records.
    connection.execute(
        """
        UPDATE heat_cycles
        SET synced = 0
        WHERE sync_id = 'heat_cycle-34'
        """
    )
    connection.execute(
        """
        UPDATE heat_cycles
        SET pouring_end_time = ''
        WHERE sync_id = 'heat_cycle-33'
        """
    )
    connection.commit()
    connection.close()


def test_selection_freezes_exact_paired_records_oldest_first(tmp_path):
    database = tmp_path / "source.db"
    _create_source_db(database)

    manifest = select_latest_paired_cycles(database, "1157", "1256", 30)

    assert manifest["count"] == 30
    assert len(manifest["agni_items"]) == 30
    assert len(manifest["pouring_items"]) == 30
    assert [item["heat_no"] for item in manifest["agni_items"]] == [
        item["heat_no"] for item in manifest["pouring_items"]
    ]
    starts = [item["heat_start_time"] for item in manifest["agni_items"]]
    assert starts == sorted(starts)

    agni = manifest["agni_items"][-1]
    pouring = manifest["pouring_items"][-1]
    assert agni["customer_id"] == "1256"
    assert agni["furnace"] == "Casting Section"
    assert agni["sync_id"].startswith("demo-1256-heat_cycle-")
    assert agni["sync_id"].endswith("-a")
    assert agni["tapping_events"][0]["tapping_start_time"].endswith("10:00")
    assert pouring["sync_id"].endswith("-p")
    assert pouring["mould_wise_pouring_time"][0]["mould_id"] == "MOULD_1"
    assert not {
        "screenshot",
        "image",
        "annotations",
        "tags",
    }.intersection(agni | pouring)
    validate_manifest(manifest)


def test_selection_rejects_request_larger_than_available(tmp_path):
    database = tmp_path / "source.db"
    _create_source_db(database)

    with pytest.raises(ImportValidationError, match="Requested 34"):
        select_latest_paired_cycles(database, "1157", "1256", 34)


def test_failed_only_retry_and_duplicate_acceptance():
    calls = []

    def send(items):
        calls.append([item["sync_id"] for item in items])
        if len(calls) == 1:
            return {
                "results": [
                    {"sync_id": "one", "success": True, "error": None},
                    {"sync_id": "two", "success": False, "error": "temporary"},
                    {"sync_id": "three", "success": False, "error": "Duplicate"},
                ]
            }
        return {
            "results": [
                {"sync_id": "two", "success": True, "error": None},
            ]
        }

    result = submit_with_failed_only_retries(
        send,
        [{"sync_id": "one"}, {"sync_id": "two"}, {"sync_id": "three"}],
    )

    assert calls == [["one", "two", "three"], ["two"]]
    assert result["complete"] is True
    assert result["accepted_sync_ids"] == ["one", "three", "two"]
    assert result["failures"] == {}


class _FakeAPI:
    def __init__(self, fail_agni=False):
        self.fail_agni = fail_agni
        self.agni_calls = []
        self.pouring_calls = []

    def send_melting_data(self, items):
        self.agni_calls.append(items)
        return {
            "results": [
                {
                    "sync_id": item["sync_id"],
                    "success": not self.fail_agni,
                    "error": "blocked" if self.fail_agni else None,
                }
                for item in items
            ]
        }

    def send_pouring_data(self, items):
        self.pouring_calls.append(items)
        return {
            "results": [
                {"sync_id": item["sync_id"], "success": True, "error": None}
                for item in items
            ]
        }


def _minimal_manifest():
    return {
        "manifest_version": 1,
        "destination_customer_id": "1256",
        "count": 1,
        "agni_items": [
            {
                "sync_id": "demo-1256-source-a",
                "customer_id": "1256",
                "heat_no": "HEAT_1",
            }
        ],
        "pouring_items": [
            {
                "sync_id": "demo-1256-source-p",
                "customer_id": "1256",
                "heat_no": "HEAT_1",
            }
        ],
        "execution": {
            "status": "not_started",
            "agni": {"accepted_sync_ids": [], "failures": {}},
            "pouring": {"accepted_sync_ids": [], "failures": {}},
        },
    }


def test_pouring_does_not_start_until_all_agni_records_are_accepted():
    manifest = _minimal_manifest()
    api = _FakeAPI(fail_agni=True)

    execution = execute_manifest(manifest, api)

    assert execution["status"] == "failed"
    assert len(api.agni_calls) == 3
    assert api.pouring_calls == []


def test_completed_manifest_cannot_be_executed_again():
    manifest = _minimal_manifest()
    manifest["execution"]["status"] = "complete"

    with pytest.raises(ImportValidationError, match="already completed"):
        execute_manifest(manifest, _FakeAPI())
