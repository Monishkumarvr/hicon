#!/usr/bin/env python3
"""NVR-backed rolling-window timeline helper — gap-free delayed playback.

Requests sequential NVR RTSP-playback windows for one camera track and paces
them into a local FIFO for downstream re-publishing (see timeline_service.py).
Reuses the segment-file / FIFO / pacing conventions proven in
pipeline/segment_buffer_helper.py, but instead of restarting the SAME live URL
on a drop, this walks forward through capture TIME — and does so with several
CONCURRENT lanes, because this NVR's RTSP *playback* (not live) delivery is
structurally slower than real-time per session (measured ~0.45-0.5x, confirmed
independent of decode/probe overhead and independent of concurrent-session
contention — two concurrent fetches of the same track each still ran at
~0.46x, meaning the cap is per-session, not shared bandwidth). A single
sequential fetcher can therefore never keep up: lag would grow unboundedly.
Running N lanes in parallel, each independently fetching a different window,
recovers aggregate throughput close to or above 1x.

Window schedule is a fixed, deterministic timeline divided into windows of
`chunk_seconds`, numbered k=0,1,2,... from a startup epoch t0. Lane `i` claims
whichever window index is next unclaimed (self-balancing — a lane that
finishes early claims the next one, so lanes don't need round-robin pinning).
This means multiple lanes may be mid-fetch on DIFFERENT windows at once, but
the FEEDER must still deliver them in strict k order — the shared
`_window_status` table (protected by `_window_lock`) is what the feeder
polls: it blocks/rebuffers on whichever k is next until that lane's fetch
resolves to "done" (serve it) or "failed" (that window is unrecoverable
after `max_delay_seconds` — skip forward and log the gap; this is the only
path that can create an actual gap, and only after real, budgeted retries).

Each NVR window is written as exactly ONE segment file (no internal
segment-time splitting): the requested [start, end) duration is a lower
bound — Hikvision playback rounds up to the next keyframe past `endtime`, so
actual content is typically a little longer than requested (never shorter).
Real per-window duration is measured via packet count (NOT frame count/full
decode — decoding was the original bottleneck before lanes were added: ~40x
slower than demux-level counting for the same, exact count) and used for
playback pacing and the capture-clock anchor, so anchor drift can't
accumulate from the NVR's keyframe rounding or from concurrent lanes
resolving windows out of submission order.

The feeder publishes a capture-clock anchor (capture_epoch, host_monotonic,
rate) to anchor.json as it paces bytes out, so a downstream consumer can map
"now" to "capture time" without needing PTS:
    capture_now = capture_epoch + (monotonic_now - host_monotonic) * rate
The anchor is clamped to be non-decreasing at publish time as a defensive
invariant — belt-and-suspenders against any residual scheduling edge case
producing a value earlier than what's already been published.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

FIFO_NAME = "stream.fifo"
STATE_FILE_NAME = "state.json"
ANCHOR_FILE_NAME = "anchor.json"
SEGMENTS_DIR_NAME = "segments"
POLL_INTERVAL_SEC = 0.25
ANCHOR_PUBLISH_INTERVAL_SEC = 1.0
NVR_TIME_FMT = "%Y%m%dT%H%M%SZ"  # NVR quirk: 'Z' suffix but digits are local IST wall-clock
DEFAULT_LANE_COUNT = 3


@dataclass(frozen=True, order=True)
class SegmentRef:
    epoch: int
    index: int
    path: Path


def parse_segment_ref(path: Path) -> SegmentRef | None:
    if path.suffix not in (".h264", ".h265"):
        return None
    try:
        epoch = int(path.parent.name.split("_", 1)[1])
        index = int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return SegmentRef(epoch=epoch, index=index, path=path)


def build_playback_url(nvr_host: str, nvr_user: str, nvr_pass: str, track_id: int,
                        window_start: datetime, window_end: datetime) -> str:
    """rtsp://user:pass@host/Streaming/tracks/{track}/?starttime=..&endtime=..

    NVR quirk (confirmed in the feasibility spike): the 'Z' suffix in start/endtime
    is NOT UTC — pass local IST wall-clock digits directly, matching
    tools/nvr_download_heats.py's fmt_nvr().
    """
    user = urllib.parse.quote(nvr_user, safe="")
    pw = urllib.parse.quote(nvr_pass, safe="")
    start_s = window_start.strftime(NVR_TIME_FMT)
    end_s = window_end.strftime(NVR_TIME_FMT)
    return (f"rtsp://{user}:{pw}@{nvr_host}/Streaming/tracks/{track_id}/"
            f"?starttime={start_s}&endtime={end_s}")


class NvrTimelineHelper:
    """Owns the concurrent-lane NVR pull (writer) and paced FIFO feed (feeder)
    for one stream."""

    def __init__(self, *, stream_id: int, nvr_host: str, nvr_user: str, nvr_pass: str,
                 track_id: int, codec: str, fps: float, buffer_dir: str,
                 chunk_seconds: int, initial_delay_seconds: int,
                 min_delay_seconds: int, max_delay_seconds: int,
                 retention_seconds: int, lane_count: int = DEFAULT_LANE_COUNT):
        self.stream_id = stream_id
        self.nvr_host = nvr_host
        self.nvr_user = nvr_user
        self.nvr_pass = nvr_pass
        self.track_id = track_id
        self.codec = codec.lower()
        self.fps = max(1.0, float(fps))
        self.buffer_dir = Path(buffer_dir)
        self.segments_root = self.buffer_dir / SEGMENTS_DIR_NAME
        self.fifo_path = self.buffer_dir / FIFO_NAME
        self.state_path = self.buffer_dir / STATE_FILE_NAME
        self.anchor_path = self.buffer_dir / ANCHOR_FILE_NAME
        self.chunk_seconds = max(5, int(chunk_seconds))
        self.initial_delay_seconds = max(self.chunk_seconds, int(initial_delay_seconds))
        self.min_delay_seconds = max(self.chunk_seconds, int(min_delay_seconds))
        self.max_delay_seconds = max(self.min_delay_seconds, int(max_delay_seconds))
        self.retention_seconds = max(self.max_delay_seconds, int(retention_seconds))
        self.lane_count = max(1, int(lane_count))
        self.target_windows = max(1, -(-self.initial_delay_seconds // self.chunk_seconds))
        self.low_watermark_windows = max(1, self.target_windows // 4)
        # Lanes may claim ahead of what the feeder has served; cap it so a
        # stalled feeder/downstream doesn't let disk usage grow unbounded.
        self.lookahead_cap = self.target_windows + self.lane_count * 3

        self._stop_event = threading.Event()
        self._window_lock = threading.Lock()
        self._t0: datetime | None = None
        self._next_unassigned_k = 0
        self._feeder_next_k = 0  # feeder's current position — bounds lane lookahead
        self._window_status: dict[int, str] = {}    # k -> "done" | "failed"
        self._window_duration: dict[int, float] = {}  # k -> real duration (only "done")
        self._lane_reader_procs: dict[int, subprocess.Popen] = {}  # lane_id -> active reader
        self._lane_threads: list[threading.Thread] = []
        self._feeder_thread: threading.Thread | None = None
        self._last_published_capture_epoch: float | None = None

    def run(self) -> int:
        self._install_signal_handlers()
        self._prepare_buffer_dir()
        self._t0 = datetime.now() - timedelta(seconds=self.initial_delay_seconds)

        self._feeder_thread = threading.Thread(target=self._feeder_loop, name="nvr-feeder", daemon=True)
        self._feeder_thread.start()
        for lane_id in range(self.lane_count):
            t = threading.Thread(target=self._lane_loop, args=(lane_id,),
                                  name=f"nvr-lane-{lane_id}", daemon=True)
            self._lane_threads.append(t)
            t.start()

        logger.info("Stream %s: %s lanes started (chunk=%ss, target_delay=%ss)",
                    self.stream_id, self.lane_count, self.chunk_seconds, self.initial_delay_seconds)
        try:
            while not self._stop_event.is_set():
                if not self._feeder_thread.is_alive():
                    logger.error("Stream %s: feeder thread exited unexpectedly", self.stream_id)
                    self._stop_event.set()
                    break
                dead_lanes = [t.name for t in self._lane_threads if not t.is_alive()]
                if dead_lanes:
                    logger.error("Stream %s: lane thread(s) exited unexpectedly: %s",
                                 self.stream_id, dead_lanes)
                    self._stop_event.set()
                    break
                time.sleep(0.5)
        finally:
            self.close()
        return 0

    def close(self) -> None:
        self._stop_event.set()
        for proc in list(self._lane_reader_procs.values()):
            self._terminate_proc(proc)
        threads = list(self._lane_threads) + ([self._feeder_thread] if self._feeder_thread else [])
        for t in threads:
            if t.is_alive():
                t.join(timeout=5)
        self._publish_state("stopped", pending_windows=0)

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            logger.info("Stream %s: helper received signal %s, shutting down", self.stream_id, signum)
            self._stop_event.set()
            for proc in list(self._lane_reader_procs.values()):
                self._terminate_proc(proc)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, _handler)

    def _prepare_buffer_dir(self) -> None:
        shutil.rmtree(self.buffer_dir, ignore_errors=True)
        self.segments_root.mkdir(parents=True, exist_ok=True)
        if self.fifo_path.exists():
            self.fifo_path.unlink()
        os.mkfifo(self.fifo_path, 0o644)
        self._publish_state("buffering", pending_windows=0)
        logger.info(
            "Stream %s: NVR timeline buffer ready (dir=%s, track=%s, target_delay=%ss, chunk=%ss)",
            self.stream_id, self.buffer_dir, self.track_id, self.initial_delay_seconds, self.chunk_seconds,
        )

    def _publish_state(self, mode: str, *, pending_windows: int) -> None:
        state = {
            "mode": mode,
            "pending_segments": max(0, int(pending_windows)),
            "target_segments": self.target_windows,
            "lane_count": self.lane_count,
            "updated_at": time.time(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _publish_anchor(self, capture_epoch: float, rate: float) -> None:
        # Defensive invariant: never publish a capture time earlier than one
        # already published. Lane concurrency means windows can resolve
        # slightly out of the order they'll ultimately be fed in; the feeder
        # already enforces strict k-order delivery, but this is cheap
        # belt-and-suspenders against any edge case in that reasoning.
        if self._last_published_capture_epoch is not None:
            capture_epoch = max(capture_epoch, self._last_published_capture_epoch)
        self._last_published_capture_epoch = capture_epoch
        anchor = {
            "stream_id": self.stream_id,
            "capture_epoch": capture_epoch,
            "host_monotonic": time.monotonic(),
            "rate": rate,
            "updated_at": time.time(),
        }
        tmp = self.anchor_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(anchor, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.anchor_path)

    def _window_bounds(self, k: int) -> tuple[datetime, datetime]:
        start = self._t0 + timedelta(seconds=k * self.chunk_seconds)
        end = start + timedelta(seconds=self.chunk_seconds)
        return start, end

    def _epoch_dir_for_k(self, k: int) -> Path:
        window_start, _ = self._window_bounds(k)
        epoch = int(window_start.timestamp())
        return self.segments_root / f"epoch_{epoch:012d}"

    # ---- lanes: concurrent NVR pulls -------------------------------------------------

    def _claim_next_window(self) -> int | None:
        with self._window_lock:
            # Don't let lanes race indefinitely far ahead of the feeder's
            # actual position — bounds disk usage if downstream consumption
            # stalls for a long time (lanes would otherwise keep resolving
            # windows regardless of whether anything is draining them).
            if self._next_unassigned_k >= self._feeder_next_k + self.lookahead_cap:
                return None
            k = self._next_unassigned_k
            self._next_unassigned_k += 1
            return k

    def _lane_loop(self, lane_id: int) -> None:
        while not self._stop_event.is_set():
            k = self._claim_next_window()
            if k is None:
                self._stop_event.wait(POLL_INTERVAL_SEC)
                continue

            window_start, window_end = self._window_bounds(k)

            # Don't chase the live edge — the NVR needs a moment to finalize
            # a segment before it's queryable/playable (measured ~100-120s).
            while not self._stop_event.is_set():
                now = datetime.now()
                if window_end <= now - timedelta(seconds=self.min_delay_seconds):
                    break
                self._stop_event.wait(2.0)
            if self._stop_event.is_set():
                return

            epoch_dir = self._epoch_dir_for_k(k)
            epoch_dir.mkdir(parents=True, exist_ok=True)

            consecutive_failures = 0
            while not self._stop_event.is_set():
                url = build_playback_url(
                    self.nvr_host, self.nvr_user, self.nvr_pass, self.track_id, window_start, window_end
                )
                ok = self._pull_window(lane_id, k, url, epoch_dir)
                if ok:
                    with self._window_lock:
                        self._window_status[k] = "done"
                    break

                consecutive_failures += 1
                stuck_for = (datetime.now() - window_start).total_seconds()
                if stuck_for > self.max_delay_seconds:
                    logger.error(
                        "Stream %s: lane %s window k=%s (%s) unrecoverable after %.0fs "
                        "(max_delay=%ss) — skipping forward, gap logged",
                        self.stream_id, lane_id, k, window_start.isoformat(), stuck_for,
                        self.max_delay_seconds,
                    )
                    with self._window_lock:
                        self._window_status[k] = "failed"
                    break

                backoff = min(10.0, 2.0 * consecutive_failures)
                logger.warning(
                    "Stream %s: lane %s NVR pull failed for window k=%s (%s), attempt %s, retrying in %.0fs",
                    self.stream_id, lane_id, k, window_start.isoformat(), consecutive_failures, backoff,
                )
                self._stop_event.wait(backoff)

    def _pull_window(self, lane_id: int, k: int, url: str, epoch_dir: Path) -> bool:
        """Pull one finite NVR playback window into a single raw elementary-stream
        file. Returns True if a non-empty file with at least one decodable
        frame was produced; records its real duration in self._window_duration."""
        fmt = "hevc" if self.codec == "h265" else "h264"
        ext = ".h265" if self.codec == "h265" else ".h264"
        out_path = epoch_dir / f"seg_000000{ext}"
        reader_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-rtsp_transport", "tcp", "-stimeout", "15000000",
            "-i", url,
            "-map", "0:v:0", "-c:v", "copy", "-an",
            "-f", fmt, str(out_path),
        ]
        reader = subprocess.Popen(reader_cmd, stderr=subprocess.PIPE, close_fds=True)
        self._lane_reader_procs[lane_id] = reader
        threading.Thread(target=self._drain_stderr, args=(lane_id, reader), daemon=True).start()

        # Generous: measured single-session fetch of a 30s window took ~60s wall
        # (NVR playback runs ~0.5x real-time), and concurrent lanes add further
        # slack — chunk_seconds*5 comfortably covers that with margin rather
        # than killing a fetch that would have succeeded.
        deadline = max(150, self.chunk_seconds * 5)
        try:
            reader.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            reader.kill()
        self._lane_reader_procs.pop(lane_id, None)

        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.warning(
                "Stream %s: lane %s: no data produced for window %s (reader exit=%s)",
                self.stream_id, lane_id, epoch_dir.name, reader.returncode,
            )
            shutil.rmtree(epoch_dir, ignore_errors=True)
            return False

        real_duration = self._probe_duration(out_path, fmt)
        if real_duration is None or real_duration <= 0:
            logger.warning("Stream %s: lane %s: could not determine duration for %s, discarding",
                           self.stream_id, lane_id, epoch_dir.name)
            shutil.rmtree(epoch_dir, ignore_errors=True)
            return False

        with self._window_lock:
            self._window_duration[k] = real_duration
        return True

    def _probe_duration(self, path: Path, fmt: str) -> float | None:
        """Real content duration via packet count / fps — robust to the NVR
        rounding actual footage to slightly more than the requested window.

        -count_packets (demux-level NAL/access-unit parsing) gives an
        identical count to -count_frames (full software decode) for these
        raw H.264/H.265 elementary streams, but ~40x faster (measured: 0.3s
        vs 13.4s for a 32s stream-0 clip) — decoding every window here was
        the dominant client-side cost; even after fixing it, NVR-side
        playback throughput itself is the real bottleneck (~0.5x real-time
        per session), which is why this helper now runs concurrent lanes."""
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-f", fmt, "-i", str(path),
                 "-select_streams", "v:0", "-count_packets",
                 "-show_entries", "stream=nb_read_packets",
                 "-of", "default=noprint_wrappers=1:nokey=1"],
                capture_output=True, text=True, timeout=self.chunk_seconds + 15,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Stream %s: ffprobe failed for %s: %s", self.stream_id, path, exc)
            return None
        try:
            nb_frames = int(out.stdout.strip())
        except ValueError:
            return None
        if nb_frames <= 0:
            return None
        return nb_frames / self.fps

    def _drain_stderr(self, lane_id: int, proc: subprocess.Popen) -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            msg = line.decode(errors="replace").rstrip()
            if msg:
                logger.debug("Stream %s: lane %s ffmpeg: %s", self.stream_id, lane_id, msg)
        proc.stderr.close()

    def _terminate_proc(self, proc: subprocess.Popen) -> None:
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ---- feeder: strict k-order paced FIFO output + capture-clock anchor -------------

    def _feeder_loop(self) -> None:
        fifo_fd = os.open(self.fifo_path, os.O_WRONLY)
        for sz in (1024 * 1024, 512 * 1024, 256 * 1024):
            try:
                fcntl.fcntl(fifo_fd, fcntl.F_SETPIPE_SZ, sz)
                break
            except OSError:
                continue

        next_k = 0
        primed = False
        while not self._stop_event.is_set():
            resolved_ahead = self._count_resolved_from(next_k)

            if self._should_rebuffer(resolved_ahead, primed):
                self._publish_state("rebuffering" if primed else "buffering", pending_windows=resolved_ahead)
                if primed:
                    self._publish_anchor(self._pending_capture_epoch(next_k), rate=0.0)
                primed = False
                self._stop_event.wait(POLL_INTERVAL_SEC)
                continue
            if not primed:
                logger.info("Stream %s: timeline primed (%s windows resolved ahead)", self.stream_id, resolved_ahead)
                primed = True

            self._publish_state("playing", pending_windows=resolved_ahead)

            with self._window_lock:
                status = self._window_status.get(next_k)
            if status == "failed":
                logger.error("Stream %s: window k=%s permanently unavailable — skipping (gap)",
                             self.stream_id, next_k)
                next_k += 1
                with self._window_lock:
                    self._feeder_next_k = next_k
                self._prune_old_windows(next_k)
                continue

            # status == "done"
            epoch_dir = self._epoch_dir_for_k(next_k)
            ext = ".h265" if self.codec == "h265" else ".h264"
            seg_path = epoch_dir / f"seg_000000{ext}"
            with self._window_lock:
                real_duration = self._window_duration.get(next_k, self.chunk_seconds)
            window_start, _ = self._window_bounds(next_k)
            capture_start = int(window_start.timestamp())
            self._write_segment(fifo_fd, seg_path, capture_start, real_duration)
            next_k += 1
            with self._window_lock:
                self._feeder_next_k = next_k
            self._prune_old_windows(next_k)

    def _count_resolved_from(self, start_k: int) -> int:
        with self._window_lock:
            count = 0
            k = start_k
            while k in self._window_status:
                count += 1
                k += 1
            return count

    def _should_rebuffer(self, resolved_ahead: int, primed: bool) -> bool:
        if primed:
            return resolved_ahead < self.low_watermark_windows
        return resolved_ahead < self.target_windows

    def _pending_capture_epoch(self, next_k: int) -> float:
        window_start, _ = self._window_bounds(next_k)
        return int(window_start.timestamp())

    def _write_segment(self, fifo_fd: int, path: Path, capture_start: int, real_duration: float) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size == 0:
            return
        bytes_per_sec = size / real_duration
        chunk = 65536
        start_mono = time.monotonic()
        last_anchor_pub = 0.0
        written = 0
        self._publish_anchor(capture_start, rate=1.0)
        with path.open("rb") as fh:
            while not self._stop_event.is_set():
                data = fh.read(chunk)
                if not data:
                    break
                self._write_all(fifo_fd, data)
                written += len(data)
                elapsed = time.monotonic() - start_mono
                if elapsed - last_anchor_pub >= ANCHOR_PUBLISH_INTERVAL_SEC:
                    self._publish_anchor(capture_start + min(elapsed, real_duration), rate=1.0)
                    last_anchor_pub = elapsed
                target = written / bytes_per_sec
                slack = target - elapsed
                if slack > 0.005:
                    self._stop_event.wait(slack)
        self._publish_anchor(capture_start + real_duration, rate=1.0)

    def _write_all(self, fifo_fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data) and not self._stop_event.is_set():
            n = os.write(fifo_fd, data[offset:])
            if n <= 0:
                raise RuntimeError("FIFO write returned no progress")
            offset += n

    def _prune_old_windows(self, fed_up_to_k: int) -> None:
        cutoff = time.time() - self.retention_seconds
        with self._window_lock:
            done_ks = list(self._window_status.keys())
        for k in done_ks:
            if k >= fed_up_to_k:
                continue
            window_start, _ = self._window_bounds(k)
            if window_start.timestamp() >= cutoff:
                continue
            epoch_dir = self._epoch_dir_for_k(k)
            shutil.rmtree(epoch_dir, ignore_errors=True)
            with self._window_lock:
                self._window_status.pop(k, None)
                self._window_duration.pop(k, None)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream-id", type=int, required=True)
    p.add_argument("--nvr-host", required=True)
    p.add_argument("--nvr-user", required=True)
    p.add_argument("--nvr-pass", required=True)
    p.add_argument("--track-id", type=int, required=True)
    p.add_argument("--codec", default="h265")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--buffer-dir", required=True)
    p.add_argument("--chunk-seconds", type=int, default=30)
    p.add_argument("--initial-delay-seconds", type=int, default=205)
    p.add_argument("--min-delay-seconds", type=int, default=180)
    p.add_argument("--max-delay-seconds", type=int, default=300)
    p.add_argument("--retention-seconds", type=int, default=900)
    p.add_argument("--lane-count", type=int, default=DEFAULT_LANE_COUNT)
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    helper = NvrTimelineHelper(
        stream_id=args.stream_id, nvr_host=args.nvr_host, nvr_user=args.nvr_user,
        nvr_pass=args.nvr_pass, track_id=args.track_id, codec=args.codec, fps=args.fps,
        buffer_dir=args.buffer_dir, chunk_seconds=args.chunk_seconds,
        initial_delay_seconds=args.initial_delay_seconds, min_delay_seconds=args.min_delay_seconds,
        max_delay_seconds=args.max_delay_seconds, retention_seconds=args.retention_seconds,
        lane_count=args.lane_count,
    )
    return helper.run()


if __name__ == "__main__":
    raise SystemExit(main())
