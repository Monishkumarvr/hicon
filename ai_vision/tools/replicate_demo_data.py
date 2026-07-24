#!/usr/bin/env python3
"""Replicate paired AGNI melting and pouring records into a demo customer.

The command is deliberately dry-run-first:

1. Run without ``--execute`` to select records and freeze a JSON manifest.
2. Review the manifest.
3. Run with ``--execute --manifest <path>`` to submit exactly that selection.

No safety or image endpoint is used by this tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


AI_VISION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = AI_VISION_DIR / "data" / "hicon.db"
DEFAULT_OUTPUT_DIR = AI_VISION_DIR / "output" / "demo_imports"
DEFAULT_API_URL = "http://ai-bakend-v2.ap-south-1.elasticbeanstalk.com/api/v1"
MANIFEST_VERSION = 1


class ImportValidationError(ValueError):
    """Raised when source data or a frozen manifest is unsafe to import."""


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    parsed = value
    for _ in range(2):
        if not isinstance(parsed, str):
            break
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ImportValidationError(f"Invalid JSON list: {exc}") from exc
    if not isinstance(parsed, list):
        raise ImportValidationError("Expected a JSON list")
    return [item for item in parsed if isinstance(item, dict)]


def _format_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ImportValidationError(f"Invalid timestamp {value!r}") from exc


def _duration_hhmmss(start: str, end: str) -> str:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    seconds = int((end_dt - start_dt).total_seconds())
    if seconds < 0:
        raise ImportValidationError(f"Negative duration: {start} -> {end}")
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_tapping_events(raw_value: Any) -> list[dict[str, Any]] | None:
    events = []
    for event in _parse_json_list(raw_value):
        start = event.get("start")
        end = event.get("end")
        if not start or not end:
            continue
        duration = event.get("duration")
        if duration is None:
            duration = _duration_hhmmss(start, end)
        events.append(
            {
                "tapping_start_time": _format_timestamp(start),
                "tapping_end_time": _format_timestamp(end),
                "duration": str(duration),
                "inoculation_check": event.get("inoculation_check"),
            }
        )
    return events or None


def _normalize_mould_times(raw_value: Any) -> list[dict[str, str]]:
    mould_times = []
    for item in _parse_json_list(raw_value):
        required = ("mould_id", "start", "end", "duration")
        if any(item.get(field) in (None, "") for field in required):
            raise ImportValidationError(f"Incomplete mould timing item: {item}")
        mould_times.append(
            {
                "mould_id": str(item["mould_id"]),
                "start": _format_timestamp(str(item["start"])),
                "end": _format_timestamp(str(item["end"])),
                "duration": str(item["duration"]),
            }
        )
    if not mould_times:
        raise ImportValidationError("Completed pouring cycle has no mould timings")
    return mould_times


def _demo_sync_id(destination_customer_id: str, source_sync_id: str, suffix: str) -> str:
    base = source_sync_id.removesuffix("-a").removesuffix("-p")
    return f"demo-{destination_customer_id}-{base}-{suffix}"


def _row_to_payloads(
    row: sqlite3.Row,
    destination_customer_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cycle_start = row["cycle_start_time"]
    cycle_end = row["cycle_end_time"]
    tapping_events = _format_tapping_events(row["tapping_events"])
    tapping_start = _format_timestamp(row["tapping_start_time"])
    tapping_end = _format_timestamp(row["tapping_end_time"])

    if tapping_events:
        tapping_start = min(event["tapping_start_time"] for event in tapping_events)
        tapping_end = max(event["tapping_end_time"] for event in tapping_events)

    location = row["location"]
    agni_item = {
        "sync_id": _demo_sync_id(
            destination_customer_id, row["sync_id"], "a"
        ),
        "customer_id": destination_customer_id,
        "date": row["date"],
        "camera_id": row["camera_id"],
        "location": location,
        # Preserve the source value exactly, per the demo-data request.
        "furnace": location,
        "heat_no": row["heat_no"],
        "heat_start_time": _format_timestamp(cycle_start),
        "heat_end_time": _format_timestamp(cycle_end),
        "heat_duration": _duration_hhmmss(cycle_start, cycle_end),
        "tapping_start_time": tapping_start,
        "tapping_end_time": tapping_end,
        "tapping_events": tapping_events,
        "deslagging": bool(_parse_json_list(row["deslagging_events"])),
        "pyrometer": bool(_parse_json_list(row["pyrometer_events"])),
        "spectro": bool(_parse_json_list(row["spectro_events"])),
    }
    pouring_item = {
        "sync_id": _demo_sync_id(
            destination_customer_id, row["sync_id"], "p"
        ),
        "customer_id": destination_customer_id,
        "date": row["date"],
        "camera_id": row["camera_id"],
        "location": location,
        "pouring_start_time": _format_timestamp(row["pouring_start_time"]),
        "pouring_end_time": _format_timestamp(row["pouring_end_time"]),
        "total_pouring_time": str(row["total_pouring_time"]),
        "mould_wise_pouring_time": _normalize_mould_times(
            row["mould_wise_pouring_time"]
        ),
        "heat_no": row["heat_no"],
    }
    return agni_item, pouring_item


def select_latest_paired_cycles(
    database_path: Path,
    source_customer_id: str,
    destination_customer_id: str,
    count: int,
) -> dict[str, Any]:
    """Select and transform the newest completed, synced cycles.

    The one-month window is anchored to the latest successfully synced source
    date, not the host clock. Selected rows are returned oldest-first so AGNI
    can calculate tap-to-tap values in chronological order.
    """
    if count < 1:
        raise ImportValidationError("Count must be positive")
    if not database_path.exists():
        raise ImportValidationError(f"Source database not found: {database_path}")

    connection = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        latest_row = connection.execute(
            """
            SELECT MAX(date) AS latest_date
            FROM heat_cycles
            WHERE customer_id = ? AND synced = 1
            """,
            (source_customer_id,),
        ).fetchone()
        latest_date = latest_row["latest_date"] if latest_row else None
        if not latest_date:
            raise ImportValidationError("No successfully synced source cycles found")

        cutoff_row = connection.execute(
            "SELECT date(?, '-1 month') AS cutoff_date",
            (latest_date,),
        ).fetchone()
        cutoff_date = cutoff_row["cutoff_date"]
        rows = connection.execute(
            """
            SELECT *
            FROM heat_cycles
            WHERE customer_id = ?
              AND synced = 1
              AND date >= ?
              AND cycle_start_time IS NOT NULL
              AND cycle_start_time != ''
              AND cycle_end_time IS NOT NULL
              AND cycle_end_time != ''
              AND pouring_start_time IS NOT NULL
              AND pouring_start_time != ''
              AND pouring_end_time IS NOT NULL
              AND pouring_end_time != ''
            ORDER BY cycle_start_time DESC
            LIMIT ?
            """,
            (source_customer_id, cutoff_date, count),
        ).fetchall()
    finally:
        connection.close()

    if len(rows) != count:
        raise ImportValidationError(
            f"Requested {count} paired cycles but found {len(rows)} "
            f"between {cutoff_date} and {latest_date}"
        )

    selected_rows = list(reversed(rows))
    agni_items: list[dict[str, Any]] = []
    pouring_items: list[dict[str, Any]] = []
    source_sync_ids: list[str] = []
    for row in selected_rows:
        agni_item, pouring_item = _row_to_payloads(
            row, destination_customer_id
        )
        agni_items.append(agni_item)
        pouring_items.append(pouring_item)
        source_sync_ids.append(row["sync_id"])

    _validate_pairing(agni_items, pouring_items, count)
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "customer_id": source_customer_id,
            "database": str(database_path.resolve()),
            "latest_date": latest_date,
            "cutoff_date": cutoff_date,
            "source_sync_ids": source_sync_ids,
        },
        "destination_customer_id": destination_customer_id,
        "count": count,
        "selection_order": "oldest_to_newest",
        "agni_items": agni_items,
        "pouring_items": pouring_items,
        "execution": {
            "status": "not_started",
            "agni": {"accepted_sync_ids": [], "failures": {}},
            "pouring": {"accepted_sync_ids": [], "failures": {}},
        },
    }


def _validate_pairing(
    agni_items: list[dict[str, Any]],
    pouring_items: list[dict[str, Any]],
    expected_count: int,
) -> None:
    if len(agni_items) != expected_count or len(pouring_items) != expected_count:
        raise ImportValidationError(
            "Manifest does not contain the expected number of paired records"
        )
    agni_heats = [item["heat_no"] for item in agni_items]
    pouring_heats = [item["heat_no"] for item in pouring_items]
    if agni_heats != pouring_heats:
        raise ImportValidationError("AGNI and pouring heat sets/order do not match")
    if len(set(agni_heats)) != expected_count:
        raise ImportValidationError("Manifest contains duplicate heat numbers")
    for item in [*agni_items, *pouring_items]:
        forbidden = {"screenshot", "image", "images", "annotations", "tags", "tagging"}
        if forbidden.intersection(item):
            raise ImportValidationError("Manifest contains forbidden image/tag fields")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ImportValidationError("Unsupported manifest version")
    count = manifest.get("count")
    if not isinstance(count, int) or count < 1:
        raise ImportValidationError("Manifest count is invalid")
    destination = str(manifest.get("destination_customer_id", ""))
    if not destination:
        raise ImportValidationError("Manifest destination customer is missing")
    _validate_pairing(
        manifest.get("agni_items", []),
        manifest.get("pouring_items", []),
        count,
    )
    for item in [*manifest["agni_items"], *manifest["pouring_items"]]:
        if item.get("customer_id") != destination:
            raise ImportValidationError("Payload customer does not match destination")
        if not str(item.get("sync_id", "")).startswith(f"demo-{destination}-"):
            raise ImportValidationError("Payload has a non-demo sync ID")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportValidationError(f"Cannot load manifest {path}: {exc}") from exc
    validate_manifest(manifest)
    return manifest


def _accepted_result(result: dict[str, Any]) -> bool:
    error = str(result.get("error") or "")
    return bool(result.get("success")) or "duplicate" in error.lower()


def submit_with_failed_only_retries(
    send: Callable[[list[dict[str, Any]]], dict[str, Any]],
    items: list[dict[str, Any]],
    already_accepted: Iterable[str] = (),
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Submit a batch, retrying only items that did not succeed."""
    accepted = set(already_accepted)
    item_by_id = {item["sync_id"]: item for item in items}
    unknown_accepted = accepted.difference(item_by_id)
    if unknown_accepted:
        raise ImportValidationError(
            f"Execution state contains unknown accepted IDs: {unknown_accepted}"
        )
    failures: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []

    for round_number in range(1, max_rounds + 1):
        pending = [
            item for item in items if item["sync_id"] not in accepted
        ]
        if not pending:
            break
        pending_ids = {item["sync_id"] for item in pending}
        try:
            response = send(pending)
            results = response.get("results", [])
        except Exception as exc:  # transport errors are recorded and retried
            results = []
            response = {"exception": f"{type(exc).__name__}: {exc}"}

        result_by_id = {
            result.get("sync_id"): result
            for result in results
            if result.get("sync_id") in pending_ids
        }
        attempt = {
            "round": round_number,
            "submitted_sync_ids": sorted(pending_ids),
            "response": response,
        }
        attempts.append(attempt)
        for sync_id in pending_ids:
            result = result_by_id.get(sync_id)
            if result and _accepted_result(result):
                accepted.add(sync_id)
                failures.pop(sync_id, None)
            elif result:
                failures[sync_id] = str(result.get("error") or "Unknown error")
            else:
                failures[sync_id] = str(
                    response.get("exception") or "Missing per-item result"
                )

    pending_ids = [
        item["sync_id"] for item in items if item["sync_id"] not in accepted
    ]
    return {
        "accepted_sync_ids": sorted(accepted),
        "failures": {sync_id: failures[sync_id] for sync_id in pending_ids},
        "attempts": attempts,
        "complete": not pending_ids,
    }


def execute_manifest(
    manifest: dict[str, Any],
    api_client: Any,
    persist: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute a frozen manifest, persisting after each endpoint phase."""
    validate_manifest(manifest)
    execution = manifest.setdefault("execution", {})
    if execution.get("status") == "complete":
        raise ImportValidationError("Manifest has already completed")

    agni_previous = execution.get("agni", {}).get("accepted_sync_ids", [])
    agni_result = submit_with_failed_only_retries(
        api_client.send_melting_data,
        manifest["agni_items"],
        already_accepted=agni_previous,
    )
    execution["agni"] = agni_result
    execution["status"] = "agni_complete" if agni_result["complete"] else "failed"
    execution["updated_at"] = datetime.now(timezone.utc).isoformat()
    if persist:
        persist(manifest)
    if not agni_result["complete"]:
        return execution

    pouring_previous = execution.get("pouring", {}).get(
        "accepted_sync_ids", []
    )
    pouring_result = submit_with_failed_only_retries(
        api_client.send_pouring_data,
        manifest["pouring_items"],
        already_accepted=pouring_previous,
    )
    execution["pouring"] = pouring_result
    execution["status"] = "complete" if pouring_result["complete"] else "failed"
    execution["updated_at"] = datetime.now(timezone.utc).isoformat()
    if persist:
        persist(manifest)
    return execution


def _default_manifest_path(destination_customer_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"customer_{destination_customer_id}_{timestamp}.json"


def _load_runtime_config() -> tuple[str, str]:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportValidationError("python-dotenv is required for execution") from exc
    load_dotenv(AI_VISION_DIR / ".env")
    api_url = os.getenv("HICON_API_URL", DEFAULT_API_URL).rstrip("/")
    secret = os.getenv("HICON_HMAC_SECRET")
    if not secret:
        raise ImportValidationError(
            "HICON_HMAC_SECRET must be configured for execution"
        )
    return api_url, secret


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Read-only source SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument("--source-customer-id", default="1157")
    parser.add_argument("--destination-customer-id", default="1256")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        help="Dry-run manifest path (default: ignored output directory)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Frozen manifest to execute",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform writes from --manifest; omitted means dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if not args.execute:
            if args.manifest:
                raise ImportValidationError(
                    "--manifest is only valid together with --execute"
                )
            manifest = select_latest_paired_cycles(
                args.source_db,
                args.source_customer_id,
                args.destination_customer_id,
                args.count,
            )
            output = args.output or _default_manifest_path(
                args.destination_customer_id
            )
            write_manifest(output, manifest)
            heat_numbers = [item["heat_no"] for item in manifest["agni_items"]]
            print("DRY RUN ONLY - no API writes performed")
            print(f"Manifest: {output.resolve()}")
            print(
                f"Selected {len(heat_numbers)} paired heats "
                f"from {manifest['source']['cutoff_date']} "
                f"through {manifest['source']['latest_date']}"
            )
            print(f"Oldest/newest heat: {heat_numbers[0]} / {heat_numbers[-1]}")
            print(f"Destination customer: {args.destination_customer_id}")
            return 0

        if not args.manifest:
            raise ImportValidationError("--execute requires --manifest")
        manifest = load_manifest(args.manifest)
        if manifest["destination_customer_id"] != args.destination_customer_id:
            raise ImportValidationError(
                "CLI destination does not match frozen manifest"
            )

        api_url, secret = _load_runtime_config()
        if str(AI_VISION_DIR) not in sys.path:
            sys.path.insert(0, str(AI_VISION_DIR))
        from sync.api_client import APIClient

        client = APIClient(
            api_url,
            secret,
            manifest["destination_customer_id"],
        )
        execution = execute_manifest(
            manifest,
            client,
            persist=lambda value: write_manifest(args.manifest, value),
        )
        print(f"Execution status: {execution['status']}")
        print(
            "AGNI accepted: "
            f"{len(execution['agni']['accepted_sync_ids'])}/{manifest['count']}"
        )
        pouring = execution.get("pouring", {})
        print(
            "Pouring accepted: "
            f"{len(pouring.get('accepted_sync_ids', []))}/{manifest['count']}"
        )
        print(f"Audit manifest: {args.manifest.resolve()}")
        return 0 if execution["status"] == "complete" else 1
    except ImportValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
