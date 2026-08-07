#!/usr/bin/env python3
"""
Backfill heat_cycles.pouring_start_time / pouring_end_time / total_pouring_time /
mould_wise_pouring_time for cycles finalized while the detected-pour -> heat-cycle
bridge was disconnected in production (hicon-3q4), and send the reconstructed
pouring payload to the cloud /pouring endpoint.

Background: pouring_processor.py detected and stored every pour correctly in
pouring_events the whole time, but the function that rolls a committed pour up
into the active heat cycle (PouringProcessor._sync_mould_records_to_heat_cycle)
was never called from production code, only from a unit test. Every heat cycle
since 2026-07-30 19:44:43 finalized with blank pouring aggregates and got marked
synced=1 anyway (melting/tapping synced fine for the same row), so sync_manager
never attempted /pouring for these rows and never will on its own.

This script reconstructs the missing aggregate directly from pouring_events
(which has heat_no + per-pour timing + a per-pour mould breakdown) and:
  1. UPDATEs the heat_cycles row in place (pouring_start_time/pouring_end_time/
     total_pouring_time/mould_wise_pouring_time only -- leaves `synced` alone,
     since melting/tapping for that row already synced correctly and touching
     the general synced flag would redundantly re-send /agni too).
  2. POSTs the reconstructed payload to /pouring via the same api_client used
     by the live sync loop, using the same "<sync_id>-p" convention so the
     backend's own sync_id-based dedup applies if this is ever re-run.

DRY-RUN BY DEFAULT. This writes to the production DB and calls a live external
API for historical data -- review the printed output for a few cycles before
passing --apply. Nothing is written or sent without --apply.

Usage:
    python3 tools/backfill_heat_cycle_pouring.py                  # dry run, all affected
    python3 tools/backfill_heat_cycle_pouring.py --heat-no HEAT_1481   # dry run, one cycle
    python3 tools/backfill_heat_cycle_pouring.py --limit 5        # dry run, first 5 only
    python3 tools/backfill_heat_cycle_pouring.py --apply          # actually write + send
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
from sync.api_client import APIClient  # noqa: E402
from sync.sync_manager import format_timestamp_for_api  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/hicon.db")


def _parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _find_affected_heat_nos(conn, heat_no_filter=None, limit=None):
    c = conn.cursor()
    query = """
        SELECT sync_id, heat_no, customer_id, date, location, camera_id
        FROM heat_cycles
        WHERE (pouring_start_time = '' OR pouring_start_time IS NULL)
    """
    params = []
    if heat_no_filter:
        query += " AND heat_no = ?"
        params.append(heat_no_filter)
    query += " ORDER BY created_at ASC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    c.execute(query, params)
    return [dict(row) for row in c.fetchall()]


def _reconstruct_from_pouring_events(conn, heat_no):
    """Returns None if this heat_no has no pouring_events at all (a genuine
    tapping-only cycle -- leave it alone), else a reconstructed aggregate.

    Important: each pouring_events row's mould_wise_pouring_time.moulds list is
    a CUMULATIVE snapshot (PouringProcessor._tracker_pour_records accumulates
    every mould poured so far in the current heat cycle, not just this one
    pour) -- so the correct per-cycle mould breakdown is just the LAST row's
    breakdown as-is, never merged/summed across rows. Only pouring_start_time
    (min) / pouring_end_time (max) / total_pouring_time (sum of each row's own
    individual pour duration) are genuinely per-row values worth aggregating.
    """
    c = conn.cursor()
    c.execute(
        """
        SELECT pouring_start_time, pouring_end_time, total_pouring_time,
               mould_wise_pouring_time
        FROM pouring_events
        WHERE heat_no = ?
        ORDER BY id ASC
        """,
        (heat_no,),
    )
    rows = c.fetchall()
    if not rows:
        return None

    starts, ends = [], []
    total_seconds = 0
    last_moulds_raw = None

    for row in rows:
        start_dt = _parse_iso(row["pouring_start_time"])
        end_dt = _parse_iso(row["pouring_end_time"])
        if start_dt:
            starts.append(start_dt)
        if end_dt:
            ends.append(end_dt)
        try:
            total_seconds += int(float(row["total_pouring_time"] or 0))
        except (TypeError, ValueError):
            pass
        if row["mould_wise_pouring_time"]:
            last_moulds_raw = row["mould_wise_pouring_time"]

    if not starts or not last_moulds_raw:
        return None

    try:
        parsed = json.loads(last_moulds_raw)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    mould_wise = []
    for rec in (parsed or {}).get("moulds", []) or []:
        rec_start = _parse_iso(rec.get("start"))
        rec_end = _parse_iso(rec.get("end"))
        if not rec_start or not rec_end:
            continue
        slot_id = rec.get("mould_slot_id", rec.get("mould_track_id"))
        mould_wise.append({
            "mould_id": f"MOULD_C{slot_id}",
            "start": rec_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": rec_end.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(int(float(rec.get("duration_s", 0.0)))),
        })

    if not mould_wise:
        return None

    return {
        "pouring_start_time": min(starts),
        "pouring_end_time": max(ends) if ends else min(starts),
        "total_pouring_time": total_seconds,
        "mould_wise_pouring_time": mould_wise,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heat-no", help="Only backfill this specific heat_no")
    parser.add_argument("--limit", type=int, help="Only backfill the first N affected cycles")
    parser.add_argument("--apply", action="store_true", help="Actually write to the DB and call /pouring (default: dry run)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    affected = _find_affected_heat_nos(conn, heat_no_filter=args.heat_no, limit=args.limit)
    if not affected:
        print("No affected heat cycles found (nothing with a blank pouring_start_time).")
        return

    api = None
    if args.apply:
        api = APIClient(base_url=config.API_URL, secret=config.HMAC_SECRET, customer_id=config.CUSTOMER_ID)

    updated, sent, skipped_no_pours, failed = 0, 0, 0, 0

    for cycle in affected:
        heat_no = cycle["heat_no"]
        recon = _reconstruct_from_pouring_events(conn, heat_no)
        if recon is None:
            print(f"[skip] {heat_no}: no pouring_events rows -- legitimate no-pouring cycle, leaving alone")
            skipped_no_pours += 1
            continue

        pouring_start_iso = recon["pouring_start_time"].isoformat()
        pouring_end_iso = recon["pouring_end_time"].isoformat()
        total_pouring_time = str(recon["total_pouring_time"])
        mould_wise = recon["mould_wise_pouring_time"]

        print(f"\n[{'APPLY' if args.apply else 'DRY-RUN'}] {heat_no} (sync_id={cycle['sync_id']})")
        print(f"  pouring_start_time : {pouring_start_iso}")
        print(f"  pouring_end_time   : {pouring_end_iso}")
        print(f"  total_pouring_time : {total_pouring_time}s")
        print(f"  mould_wise_pouring_time ({len(mould_wise)} moulds):")
        for m in mould_wise:
            print(f"    - {m['mould_id']}: {m['start']} -> {m['end']} ({m['duration']}s)")

        if not args.apply:
            continue

        try:
            conn.execute(
                """
                UPDATE heat_cycles
                SET pouring_start_time = ?, pouring_end_time = ?,
                    total_pouring_time = ?, mould_wise_pouring_time = ?
                WHERE sync_id = ?
                """,
                (pouring_start_iso, pouring_end_iso, total_pouring_time,
                 json.dumps(mould_wise), cycle["sync_id"]),
            )
            conn.commit()
            updated += 1
        except sqlite3.Error as exc:
            print(f"  [FAILED db update] {exc}")
            failed += 1
            continue

        pouring_item = {
            "sync_id": cycle["sync_id"] + "-p",
            "customer_id": cycle["customer_id"],
            "date": cycle["date"],
            "heat_no": heat_no,
            "location": cycle["location"],
            "camera_id": cycle["camera_id"],
            "mould_count": len(mould_wise),
            "pouring_start_time": format_timestamp_for_api(pouring_start_iso),
            "pouring_end_time": format_timestamp_for_api(pouring_end_iso),
            "total_pouring_time": total_pouring_time,
            "mould_wise_pouring_time": mould_wise,
        }
        try:
            result = api.send_pouring_data([pouring_item])
            print(f"  [sent] {result}")
            sent += 1
        except Exception as exc:  # noqa: BLE001 -- report and keep going, don't abort the batch
            print(f"  [FAILED /pouring send] {exc}")
            failed += 1

    conn.close()

    print(
        f"\n{'Applied' if args.apply else 'Would apply'}: "
        f"{updated if args.apply else len(affected) - skipped_no_pours} cycles reconstructed, "
        f"{sent} sent to /pouring, {skipped_no_pours} skipped (no pouring_events), "
        f"{failed} failed."
    )
    if not args.apply:
        print("Dry run only -- re-run with --apply to actually write and send.")


if __name__ == "__main__":
    main()
