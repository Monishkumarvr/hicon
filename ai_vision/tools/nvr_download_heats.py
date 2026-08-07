#!/usr/bin/env python3
"""
Download NVR clips for heat cycles from hicon.db.

Queries heat_cycles for Cam-Process from May 14, downloads each clip
(tapping_start - 2min  →  pouring_end + 1min) from the recorder via RTSP playback.

Usage:
    python3 tools/nvr_download_heats.py
    python3 tools/nvr_download_heats.py --limit 5        # first 5 only
    python3 tools/nvr_download_heats.py --from 2026-05-15
    python3 tools/nvr_download_heats.py --dry-run         # show timestamps, no download
"""
import argparse
import os
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timedelta

# ─── NVR config ──────────────────────────────────────────────────────────────
# 192.168.28.8 (NVR-1) does NOT hold HiCon cameras — its track 3401 is a different
# camera ("EP Area"). The only recorder with our footage is 192.168.28.6
# (Hikvision DS-7716NXI-K4). See ai_vision/docs/nvr_backfill_feasibility_2026-07-23.md.
NVR_IP    = "192.168.28.6"
NVR_USER  = "admin"
NVR_PASS  = "NVR@321#"
TRACK_ID  = "1201"          # Stream 0 / Process — ch12 "POURING" on 192.168.28.6

# ─── DB / output ─────────────────────────────────────────────────────────────
DB_PATH   = os.path.join(os.path.dirname(__file__), "../data/hicon.db")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "../output/nvr_downloads")

# Buffer around the clip
LEAD_SECS  = 120   # 2 min before tapping start
TRAIL_SECS = 60    # 1 min after pouring end


def fmt_nvr(dt: datetime) -> str:
    """Format datetime as NVR compact time: 20260514T083000Z
    NVR clock is IST — it interprets these timestamps as IST despite the Z suffix.
    Pass DB timestamps (IST) directly without any UTC conversion.
    """
    return dt.strftime("%Y%m%dT%H%M%SZ")


def fmt_ist(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M IST")


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def probe_duration(path: str):
    """Return the media duration in seconds, or None if ffprobe can't read it."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        return None


MAX_RETRIES = 3

def download_clip(heat_no: str, start_dt: datetime, end_dt: datetime, out_path: str) -> bool:
    # This NVR's firmware (V4.76.015) rejects ISAPI ContentMgmt/download's XML body
    # (statusCode 6 / badXmlContent) — RTSP playback works cleanly instead.
    nvr_pass_enc = urllib.parse.quote(NVR_PASS, safe="")
    url = (
        f"rtsp://{NVR_USER}:{nvr_pass_enc}@{NVR_IP}/Streaming/tracks/{TRACK_ID}/"
        f"?starttime={fmt_nvr(start_dt)}&endtime={fmt_nvr(end_dt)}"
    )
    tmp_path = out_path + ".tmp"
    clip_secs = (end_dt - start_dt).total_seconds()
    # The NVR doesn't cut off cleanly at `endtime` on its own (observed ~40% overshoot) —
    # `-t` forces ffmpeg to stop at the exact requested duration regardless.
    # proc_timeout is a safety net, not the primary stop mechanism: measured ~24s fixed
    # RTSP/seek overhead plus roughly real-time transfer once flowing; 1.5x + buffer covers
    # a slower NVR/network without depending on exact throughput.
    proc_timeout = max(300, clip_secs * 1.5 + 180)
    # The Jetson<->camera-segment link this NVR sits behind drops for 30-110s every ~300s
    # (see camera_lan_periodic_outage_2026_07 memory). Without a socket-level timeout, a
    # blackout mid-download stalls the TCP read and ffmpeg just hangs until proc_timeout
    # kills it — wasting the whole attempt instead of failing fast so the retry loop below
    # can land on a clean window. 90s covers the observed blackout durations with margin.
    stall_timeout_us = 90_000_000

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"  Retry {attempt}/{MAX_RETRIES} ...", end="", flush=True)
            time.sleep(5)

        if attempt == 1:
            print(f"  Downloading ...", end="", flush=True)

        t0 = time.time()
        try:
            result = subprocess.run(
                # -f matroska: the ".tmp" suffix on tmp_path defeats ffmpeg's
                # extension-based format sniffing otherwise.
                ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-stimeout", str(stall_timeout_us),
                 "-i", url, "-c", "copy", "-t", f"{clip_secs:.0f}", "-f", "matroska", tmp_path],
                capture_output=True, text=True, timeout=proc_timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"\n  ✗ ffmpeg timed out after {proc_timeout:.0f}s (attempt {attempt})")
            continue

        if result.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"\n  ✗ ffmpeg failed (attempt {attempt}): {result.stderr.strip()[-300:]}")
            continue

        # A stall (e.g. the ~300s network blackout hitting mid-download) makes ffmpeg's
        # -stimeout end the input early -- ffmpeg then exits 0 with a valid but truncated
        # file. Reject anything meaningfully short of the requested duration instead of
        # silently keeping partial footage.
        actual_secs = probe_duration(tmp_path)
        if actual_secs is None or actual_secs < clip_secs - 5:
            os.remove(tmp_path)
            got = f"{actual_secs:.1f}s" if actual_secs is not None else "unknown"
            print(f"\n  ✗ truncated (attempt {attempt}): got {got} of {clip_secs:.0f}s requested -- likely a network stall")
            continue

        elapsed = time.time() - t0
        written = os.path.getsize(tmp_path)
        speed   = written / elapsed / 1024 / 1024
        os.rename(tmp_path, out_path)
        print(f" done  {human_size(written)}  ({speed:.1f} MB/s)  [{elapsed:.0f}s]")
        return True

    print(f"  ✗ Failed after {MAX_RETRIES} attempts — skipping")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download NVR clips for heat cycles")
    parser.add_argument("--limit",   type=int, default=20,           help="Max clips (default 20)")
    parser.add_argument("--from",    dest="from_date", default="2026-05-14", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--to",      dest="to_date",   default="2026-05-21", help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--random",  action="store_true", default=True,      help="Random sample (default on)")
    parser.add_argument("--no-random", dest="random", action="store_false",  help="Take first N chronologically")
    parser.add_argument("--heats",   nargs="+", metavar="HEAT_XXXX",         help="Explicit list of heat_no to download")
    parser.add_argument("--exclude", nargs="+", metavar="HEAT_XXXX",         help="Heat numbers to exclude from random selection")
    parser.add_argument("--dry-run", action="store_true",                    help="Print timestamps only, no download")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    if args.heats:
        placeholders = ",".join("?" for _ in args.heats)
        cur.execute(
            f"""SELECT heat_no, tapping_start_time, pouring_end_time
               FROM heat_cycles
               WHERE heat_no IN ({placeholders})
                 AND tapping_start_time IS NOT NULL AND tapping_start_time != ''
                 AND pouring_end_time   IS NOT NULL AND pouring_end_time   != ''
               ORDER BY tapping_start_time ASC""",
            args.heats,
        )
    else:
        order = "ORDER BY RANDOM()" if args.random else "ORDER BY cycle_start_time ASC"
        exclude = args.exclude or []
        excl_clause = ""
        excl_params = []
        if exclude:
            excl_clause = "AND heat_no NOT IN ({})".format(",".join("?" for _ in exclude))
            excl_params = exclude
        cur.execute(
            f"""SELECT heat_no, tapping_start_time, pouring_end_time
               FROM heat_cycles
               WHERE camera_id = 'Cam-Process'
                 AND cycle_start_time >= ?
                 AND cycle_start_time <  ?
                 AND tapping_start_time IS NOT NULL AND tapping_start_time != ''
                 AND pouring_end_time   IS NOT NULL AND pouring_end_time   != ''
                 {excl_clause}
               {order}
               LIMIT ?""",
            (args.from_date, args.to_date, *excl_params, args.limit),
        )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No heat cycles found for given criteria.")
        sys.exit(1)

    print(f"Found {len(rows)} heat cycles to download\n")
    print(f"{'#':<3}  {'Heat':<12}  {'Start (IST)':<20}  {'End (IST)':<20}  {'Duration':>8}  {'File'}")
    print("-" * 100)

    clips = []
    for heat_no, tap_start_str, pour_end_str in rows:
        tap_start = datetime.fromisoformat(tap_start_str)
        pour_end  = datetime.fromisoformat(pour_end_str)

        clip_start = tap_start - timedelta(seconds=LEAD_SECS)
        clip_end   = pour_end  + timedelta(seconds=TRAIL_SECS)
        duration_m = (clip_end - clip_start).total_seconds() / 60

        filename = f"{heat_no}_{tap_start.strftime('%Y%m%d_%H%M')}_IST.mkv"
        out_path = os.path.join(OUT_DIR, filename)
        clips.append((heat_no, clip_start, clip_end, out_path, duration_m))

    for i, (heat_no, clip_start, clip_end, out_path, dur_m) in enumerate(clips, 1):
        exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        flag   = " [exists]" if exists else ""
        print(f"{i:<3}  {heat_no:<12}  {fmt_ist(clip_start):<20}  {fmt_ist(clip_end):<20}  {dur_m:>6.1f}m  {os.path.basename(out_path)}{flag}")

    if args.dry_run:
        print("\nDry run — no downloads performed.")
        return

    print()
    ok_count = 0
    for i, (heat_no, clip_start, clip_end, out_path, dur_m) in enumerate(clips, 1):
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{i}/{len(clips)}] {heat_no}  SKIP (already exists)")
            ok_count += 1
            continue

        print(f"[{i}/{len(clips)}] {heat_no}  {fmt_ist(clip_start)} → {fmt_ist(clip_end)}  ({dur_m:.1f} min)")
        success = download_clip(heat_no, clip_start, clip_end, out_path)
        if success:
            ok_count += 1
        else:
            # Remove empty/partial file
            if os.path.exists(out_path):
                os.remove(out_path)
        time.sleep(2)   # brief pause between downloads to not flood NVR

    print(f"\nDone: {ok_count}/{len(clips)} clips downloaded to {OUT_DIR}/")


if __name__ == "__main__":
    main()
