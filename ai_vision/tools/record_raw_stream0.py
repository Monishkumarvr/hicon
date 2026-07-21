#!/usr/bin/env python3
"""Record raw (no annotations) HEVC video from any stream for offline analysis.

Uses a dual-ffmpeg pipe architecture:
  - Reader:    ffmpeg -rtsp_transport tcp -i URL -c:v copy -f hevc pipe:1
               Zero disk I/O — camera TCP session never stalls.
               Restarts silently on drop.
  - Segmenter: ffmpeg -f hevc -r 25 -i pipe:0 -c:v copy -f segment seg_%06d.hevc
               Reads from pipe, writes 90s segments to disk.
               Disk I/O fully isolated from camera TCP.

After 40 segments (~1 hour), concatenates into a single MKV file.
Per-stream staging dirs allow recording multiple streams simultaneously.

Usage:
    python3 tools/record_raw_stream0.py                # stream 0 (default)
    python3 tools/record_raw_stream0.py --stream 1     # pyrometer cam
    python3 tools/record_raw_stream0.py --stream 2     # pouring2 cam
    python3 tools/record_raw_stream0.py --hours 2
    python3 tools/record_raw_stream0.py --output-dir /path/to/dir
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# --- Defaults ------------------------------------------------------------------

RTSP_URLS = {
    0: "rtsp://admin:india%40789@192.168.28.119:554/Streaming/Channels/102",
    1: "rtsp://admin:india%40789@192.168.28.172:554/Streaming/Channels/102",
    2: "rtsp://admin:india%40789@192.168.28.174:554/Streaming/Channels/102",
}

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "output" / "recordings"
STAGING_BASE = Path("/tmp/hicon_raw_staging")

SEGMENT_SECS = 90
SEGS_PER_HOUR = 40      # 40 × 90s = 3600s = 1 hour
FPS = 25
CODEC = "hevc"          # Hikvision cameras output H.265 (HEVC)
SEG_EXT = ".hevc"

# stimeout in microseconds (10 seconds) — exits FIN-WAIT-1 promptly on camera drop
RTSP_STIMEOUT_US = 10_000_000

PIPE_SIZE = 1024 * 1024  # 1 MB — absorbs brief segmenter stalls

# -------------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _reader_loop(rtsp_url: str, w_fd: int, stop_event: threading.Event) -> None:
    """Continuously run the RTSP reader, restarting on every exit."""
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-stimeout", str(RTSP_STIMEOUT_US),
        "-i", rtsp_url,
        "-c:v", "copy",
        "-f", CODEC,
        "pipe:1",
    ]
    while not stop_event.is_set():
        proc = subprocess.Popen(
            cmd,
            stdout=w_fd,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        if stop_event.is_set():
            break
        print(f"[{_ts()}] Reader exited (code={proc.returncode}), restarting in 1s ...")
        time.sleep(1)


def _complete_segments(staging: Path, after_idx: int) -> list[Path]:
    """Return complete segment files (all except the last, which may still be written)."""
    segs = sorted(staging.glob(f"seg_*{SEG_EXT}"), key=lambda p: int(p.stem.split("_")[1]))
    if len(segs) < 2:
        return []
    complete = [s for s in segs[:-1] if int(s.stem.split("_")[1]) > after_idx]
    return complete


def _concat(seg_files: list[Path], output_path: Path) -> bool:
    """Concatenate HEVC segment files into a single MKV using ffmpeg concat."""
    filelist = output_path.parent / f"_concat_{output_path.stem}.txt"
    try:
        with open(filelist, "w") as f:
            for seg in seg_files:
                f.write(f"file '{seg}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(filelist),
            "-c", "copy",
            str(output_path),
        ]
        print(f"[{_ts()}] Assembling {len(seg_files)} segments → {output_path.name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{_ts()}] ERROR: concat failed:\n{result.stderr[-600:]}", file=sys.stderr)
            return False
        size_mb = output_path.stat().st_size / 1_000_000
        print(f"[{_ts()}] Saved: {output_path.name} ({size_mb:.1f} MB)")
        return True
    finally:
        filelist.unlink(missing_ok=True)


def _save_partial(staged: list[Path], output_dir: Path, stream_id: int) -> None:
    if not staged:
        print(f"[{_ts()}] No staged segments to save.")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"stream{stream_id}_{ts}_partial.mkv"
    _concat(staged, out)
    for s in staged:
        s.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record raw stream video for analysis.")
    parser.add_argument("--stream", type=int, choices=[0, 1, 2], default=0,
                        help="Stream ID (0=process, 1=pyrometer, 2=pouring2). Default: 0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--hours", type=int, default=0,
                        help="Number of 1-hour files to record. 0 = run until Ctrl+C.")
    args = parser.parse_args()

    rtsp_url = RTSP_URLS[args.stream]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-stream staging dir (allows simultaneous recording of multiple streams)
    staging_dir = STAGING_BASE / f"stream{args.stream}"
    if staging_dir.exists():
        for f in staging_dir.glob(f"seg_*{SEG_EXT}"):
            f.unlink(missing_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{_ts()}] Stream {args.stream} → {rtsp_url}")
    print(f"[{_ts()}] Output: {output_dir}")
    print(f"[{_ts()}] Codec: {CODEC.upper()} (copy, no re-encode)")
    print(f"[{_ts()}] Segments: {SEGMENT_SECS}s × {SEGS_PER_HOUR} = 1 hour per file")
    if args.hours:
        print(f"[{_ts()}] Recording {args.hours} hour(s) then stopping.")
    else:
        print(f"[{_ts()}] Recording until Ctrl+C.")

    # Set up pipe: reader stdout → w_fd → pipe → r_fd → segmenter stdin
    r_fd, w_fd = os.pipe()
    try:
        fcntl.fcntl(w_fd, fcntl.F_SETPIPE_SZ, PIPE_SIZE)
    except OSError:
        pass  # best-effort; Jetson allows up to 1MB

    # Start segmenter (long-running)
    seg_cmd = [
        "ffmpeg",
        "-f", CODEC,
        "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "copy",
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECS),
        "-reset_timestamps", "1",
        "-y",
        str(staging_dir / f"seg_%06d{SEG_EXT}"),
    ]
    segmenter = subprocess.Popen(
        seg_cmd,
        stdin=r_fd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.close(r_fd)  # segmenter owns r_fd now

    # Start reader thread
    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=_reader_loop,
        args=(rtsp_url, w_fd, stop_event),
        daemon=True,
    )
    reader_thread.start()

    staged: list[Path] = []
    last_seg_idx = -1
    hour_n = 0
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while running:
            new_complete = _complete_segments(staging_dir, last_seg_idx)
            for seg_path in new_complete:
                idx = int(seg_path.stem.split("_")[1])
                size_kb = seg_path.stat().st_size // 1024
                staged.append(seg_path)
                last_seg_idx = idx
                total_secs = len(staged) * SEGMENT_SECS
                print(
                    f"[{_ts()}] Seg {idx:04d}  "
                    f"size={size_kb}KB  "
                    f"staged={len(staged)}/{SEGS_PER_HOUR}  "
                    f"hour_total={total_secs//60}m{total_secs%60}s"
                )

                if len(staged) >= SEGS_PER_HOUR:
                    batch = staged[:SEGS_PER_HOUR]
                    staged = staged[SEGS_PER_HOUR:]
                    hour_n += 1
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = output_dir / f"stream{args.stream}_{ts}_h{hour_n:02d}.mkv"
                    ok = _concat(batch, out)
                    if ok:
                        for s in batch:
                            s.unlink(missing_ok=True)
                    if args.hours > 0 and hour_n >= args.hours:
                        running = False
                        break

            time.sleep(0.5)

    finally:
        print(f"\n[{_ts()}] Shutting down ...")
        stop_event.set()
        os.close(w_fd)
        segmenter.terminate()
        try:
            segmenter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            segmenter.kill()
        reader_thread.join(timeout=5)

        if staged:
            print(f"[{_ts()}] Saving partial hour ({len(staged)} segments) ...")
            _save_partial(staged, output_dir, args.stream)

        for leftover in staging_dir.glob(f"seg_*{SEG_EXT}"):
            leftover.unlink(missing_ok=True)
        print(f"[{_ts()}] Done.")


if __name__ == "__main__":
    main()
