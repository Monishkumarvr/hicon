#!/usr/bin/env python3
"""NVR-backed rolling-window timeline helper — gap-free delayed playback.

Continuously requests sequential NVR RTSP-playback windows for one camera track
and paces them into a local FIFO for downstream re-publishing (see
timeline_service.py). Reuses the segment-file / FIFO / pacing conventions proven
in pipeline/segment_buffer_helper.py, but instead of restarting the SAME live URL
on a drop, this walks forward through capture TIME: each cycle requests
[read_head, read_head + chunk) from the NVR and advances read_head regardless of
transient fetch failures (retried within the delay budget).

Feasibility basis: docs/nvr_backfill_feasibility_2026-07-23.md confirmed NVR
192.168.28.6 records all 3 HiCon cameras continuously (segment- and frame-level,
25fps, 0 gaps) through the network outages this exists to cover — so there is no
live/backfill switchover state machine: the NVR is the only source here, always
played back at a lag behind wall-clock, and that lag absorbs transient NVR
retrieval hiccups instead of needing them to succeed in real time.

Each NVR window is written as exactly ONE segment file per epoch (no internal
segment-time splitting): the requested [start, end) duration is a lower bound —
Hikvision playback rounds up to the next keyframe past `endtime`, so actual
content is typically a little longer than requested (never shorter — the safe
direction: a little overlap between consecutive windows, never a gap). The
segment "epoch" doubles as capture-time metadata: epoch = the unix second of
that window's absolute start, so capture_time(segment) = epoch exactly. Real
per-window duration (frame_count / fps, not the nominal chunk length) is tracked
separately and used for playback pacing and the capture-clock anchor, so anchor
drift can't accumulate from the NVR's keyframe rounding.

The feeder publishes a capture-clock anchor (capture_epoch, host_monotonic, rate)
to anchor.json as it paces bytes out, so a downstream consumer can map "now" to
"capture time" without needing PTS:
    capture_now = capture_epoch + (monotonic_now - host_monotonic) * rate
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


def list_complete_segments(segments_root: Path, active_epoch: int | None) -> list[SegmentRef]:
    """Segments in playback order; the active (still-being-written) epoch's
    latest segment is withheld since ffmpeg may still be writing it."""
    refs = []
    for path in segments_root.glob("epoch_*/seg_*.*"):
        ref = parse_segment_ref(path)
        if ref is not None:
            refs.append(ref)
    refs.sort()
    if active_epoch is None:
        return refs
    latest_active_index = None
    for ref in refs:
        if ref.epoch == active_epoch:
            latest_active_index = ref.index
    if latest_active_index is None:
        return refs
    return [r for r in refs if not (r.epoch == active_epoch and r.index == latest_active_index)]


def should_rebuffer(pending_count: int, primed: bool, target_segments: int, low_watermark: int) -> bool:
    if primed:
        return pending_count < low_watermark
    return pending_count < target_segments


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
    """Owns the rolling NVR pull (writer) and paced FIFO feed (feeder) for one stream."""

    def __init__(self, *, stream_id: int, nvr_host: str, nvr_user: str, nvr_pass: str,
                 track_id: int, codec: str, fps: float, buffer_dir: str,
                 chunk_seconds: int, initial_delay_seconds: int,
                 min_delay_seconds: int, max_delay_seconds: int,
                 retention_seconds: int):
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
        self.target_segments = max(1, -(-self.initial_delay_seconds // self.chunk_seconds))
        self.low_watermark_segments = max(1, self.target_segments // 4)

        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._active_epoch: int | None = None
        self._read_head: datetime | None = None
        self._ffmpeg_reader_proc: subprocess.Popen | None = None
        self._last_fed: SegmentRef | None = None
        self._writer_thread: threading.Thread | None = None
        self._feeder_thread: threading.Thread | None = None
        # epoch -> actual content duration (frame_count / fps), NOT the nominal
        # chunk length — NVR playback rounds up to the next keyframe past
        # `endtime`, so real duration is typically a little longer than requested.
        self._segment_durations: dict[int, float] = {}

    def run(self) -> int:
        self._install_signal_handlers()
        self._prepare_buffer_dir()
        self._writer_thread = threading.Thread(target=self._writer_loop, name="nvr-writer", daemon=True)
        self._feeder_thread = threading.Thread(target=self._feeder_loop, name="nvr-feeder", daemon=True)
        self._writer_thread.start()
        self._feeder_thread.start()
        try:
            while not self._stop_event.is_set():
                if self._writer_thread.is_alive() and self._feeder_thread.is_alive():
                    time.sleep(0.5)
                    continue
                if not self._writer_thread.is_alive():
                    logger.error("Stream %s: writer thread exited unexpectedly", self.stream_id)
                if not self._feeder_thread.is_alive():
                    logger.error("Stream %s: feeder thread exited unexpectedly", self.stream_id)
                self._stop_event.set()
        finally:
            self.close()
        return 0

    def close(self) -> None:
        self._stop_event.set()
        self._terminate_ffmpeg()
        for t in (self._writer_thread, self._feeder_thread):
            if t and t.is_alive():
                t.join(timeout=5)
        self._publish_state("stopped", pending_segments=0)

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            logger.info("Stream %s: helper received signal %s, shutting down", self.stream_id, signum)
            self._stop_event.set()
            self._terminate_ffmpeg()

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, _handler)

    def _prepare_buffer_dir(self) -> None:
        shutil.rmtree(self.buffer_dir, ignore_errors=True)
        self.segments_root.mkdir(parents=True, exist_ok=True)
        if self.fifo_path.exists():
            self.fifo_path.unlink()
        os.mkfifo(self.fifo_path, 0o644)
        self._publish_state("buffering", pending_segments=0)
        logger.info(
            "Stream %s: NVR timeline buffer ready (dir=%s, track=%s, target_delay=%ss, chunk=%ss)",
            self.stream_id, self.buffer_dir, self.track_id, self.initial_delay_seconds, self.chunk_seconds,
        )

    def _publish_state(self, mode: str, *, pending_segments: int) -> None:
        state = {
            "mode": mode,
            "pending_segments": max(0, int(pending_segments)),
            "target_segments": self.target_segments,
            "updated_at": time.time(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _publish_anchor(self, capture_epoch: float, rate: float) -> None:
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

    # ---- writer: rolling NVR pulls -------------------------------------------------

    def _writer_loop(self) -> None:
        self._read_head = datetime.now() - timedelta(seconds=self.initial_delay_seconds)
        consecutive_failures = 0
        while not self._stop_event.is_set():
            window_start = self._read_head
            window_end = window_start + timedelta(seconds=self.chunk_seconds)

            # Don't chase the live edge — the NVR needs a moment to finalize a
            # segment before it's queryable/playable.
            now = datetime.now()
            if window_end > now - timedelta(seconds=self.min_delay_seconds):
                self._stop_event.wait(2.0)
                continue

            epoch = int(window_start.timestamp())
            epoch_dir = self.segments_root / f"epoch_{epoch:012d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            with self._state_lock:
                self._active_epoch = epoch

            url = build_playback_url(
                self.nvr_host, self.nvr_user, self.nvr_pass, self.track_id, window_start, window_end
            )
            ok = self._pull_window(epoch, url, epoch_dir)

            with self._state_lock:
                self._active_epoch = None

            if ok:
                consecutive_failures = 0
                # Advance by the MEASURED real duration, not the nominal chunk
                # length — the NVR rounds up to the next keyframe past `endtime`
                # (real_duration >= chunk_seconds), so advancing by the nominal
                # length would make consecutive epochs' claimed capture-time
                # ranges overlap, producing a small backward jump in the anchor
                # at every segment boundary. Advancing by real duration keeps
                # segments exactly contiguous: epoch(N+1) == epoch(N) + real_duration(N).
                real_duration = self._segment_duration(epoch)
                self._read_head = window_start + timedelta(seconds=real_duration)
                continue

            consecutive_failures += 1
            stuck_for = (datetime.now() - window_start).total_seconds()
            if stuck_for > self.max_delay_seconds:
                logger.error(
                    "Stream %s: window %s unrecoverable after %.0fs (max_delay=%ss) — "
                    "skipping forward, gap logged",
                    self.stream_id, window_start.isoformat(), stuck_for, self.max_delay_seconds,
                )
                self._read_head = window_end
                consecutive_failures = 0
                continue

            backoff = min(10.0, 2.0 * consecutive_failures)
            logger.warning(
                "Stream %s: NVR pull failed for window %s (attempt %s), retrying in %.0fs",
                self.stream_id, window_start.isoformat(), consecutive_failures, backoff,
            )
            self._stop_event.wait(backoff)

    def _pull_window(self, epoch: int, url: str, epoch_dir: Path) -> bool:
        """Pull one finite NVR playback window into a single raw elementary-stream
        file (no internal segment-time splitting — see module docstring for why).
        Returns True if a non-empty file with at least one decodable frame was
        produced; records its real duration in self._segment_durations[epoch]."""
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
        self._ffmpeg_reader_proc = reader
        threading.Thread(target=self._drain_stderr, args=(reader, "reader"), daemon=True).start()

        deadline = self.chunk_seconds + 20  # generous: connect + auth + playback + teardown
        try:
            reader.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            reader.kill()
        self._ffmpeg_reader_proc = None

        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.warning(
                "Stream %s: no data produced for window starting %s (reader exit=%s)",
                self.stream_id, epoch_dir.name, reader.returncode,
            )
            shutil.rmtree(epoch_dir, ignore_errors=True)
            return False

        real_duration = self._probe_duration(out_path, fmt)
        if real_duration is None or real_duration <= 0:
            logger.warning("Stream %s: could not determine duration for %s, discarding",
                           self.stream_id, epoch_dir.name)
            shutil.rmtree(epoch_dir, ignore_errors=True)
            return False

        with self._state_lock:
            self._segment_durations[epoch] = real_duration
        return True

    def _probe_duration(self, path: Path, fmt: str) -> float | None:
        """Real content duration via frame count / fps — robust to the NVR
        rounding actual footage to slightly more than the requested window."""
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-f", fmt, "-i", str(path),
                 "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames",
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

    def _drain_stderr(self, proc: subprocess.Popen, label: str) -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            msg = line.decode(errors="replace").rstrip()
            if msg:
                logger.debug("Stream %s: ffmpeg-%s: %s", self.stream_id, label, msg)
        proc.stderr.close()

    def _terminate_ffmpeg(self) -> None:
        proc = self._ffmpeg_reader_proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ---- feeder: paced FIFO output + capture-clock anchor ---------------------------

    def _feeder_loop(self) -> None:
        fifo_fd = os.open(self.fifo_path, os.O_WRONLY)
        for sz in (1024 * 1024, 512 * 1024, 256 * 1024):
            try:
                fcntl.fcntl(fifo_fd, fcntl.F_SETPIPE_SZ, sz)
                break
            except OSError:
                continue

        primed = False
        while not self._stop_event.is_set():
            with self._state_lock:
                active_epoch = self._active_epoch
            complete = list_complete_segments(self.segments_root, active_epoch)
            pending = complete if self._last_fed is None else [s for s in complete if s > self._last_fed]

            if should_rebuffer(len(pending), primed, self.target_segments, self.low_watermark_segments):
                self._publish_state("rebuffering" if primed else "buffering", pending_segments=len(pending))
                if primed:
                    self._publish_anchor(self._current_capture_epoch(), rate=0.0)
                primed = False
                self._stop_event.wait(POLL_INTERVAL_SEC)
                continue
            if not primed:
                logger.info("Stream %s: timeline primed (%s pending segments)", self.stream_id, len(pending))
                primed = True

            self._publish_state("playing", pending_segments=len(pending))
            next_seg = pending[0]
            self._write_segment(fifo_fd, next_seg)
            self._last_fed = next_seg
            self._prune_old_segments()

    def _segment_duration(self, epoch: int) -> float:
        with self._state_lock:
            return self._segment_durations.get(epoch, self.chunk_seconds)

    def _current_capture_epoch(self) -> float:
        if self._last_fed is None:
            return time.time() - self.initial_delay_seconds
        return self._last_fed.epoch + self._segment_duration(self._last_fed.epoch)

    def _write_segment(self, fifo_fd: int, ref: SegmentRef) -> None:
        try:
            size = ref.path.stat().st_size
        except OSError:
            return
        if size == 0:
            return
        capture_start = ref.epoch  # single file per epoch — the epoch IS the capture start
        real_duration = self._segment_duration(ref.epoch)
        bytes_per_sec = size / real_duration
        chunk = 65536
        start_mono = time.monotonic()
        last_anchor_pub = 0.0
        written = 0
        self._publish_anchor(capture_start, rate=1.0)
        with ref.path.open("rb") as fh:
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
        with self._state_lock:
            self._segment_durations.pop(ref.epoch, None)

    def _write_all(self, fifo_fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data) and not self._stop_event.is_set():
            n = os.write(fifo_fd, data[offset:])
            if n <= 0:
                raise RuntimeError("FIFO write returned no progress")
            offset += n

    def _prune_old_segments(self) -> None:
        cutoff = time.time() - self.retention_seconds
        for path in sorted(self.segments_root.glob("epoch_*/seg_*.*")):
            ref = parse_segment_ref(path)
            if ref is None:
                continue
            if ref.epoch < cutoff and (self._last_fed is None or ref < self._last_fed):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue
        for d in sorted(self.segments_root.glob("epoch_*")):
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                except OSError:
                    pass


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
    )
    return helper.run()


if __name__ == "__main__":
    raise SystemExit(main())
