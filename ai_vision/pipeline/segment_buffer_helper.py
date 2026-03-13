"""Delayed MPEG-TS segment spooler for Stream 0 no-loss playback."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)

FIFO_NAME = "stream.fifo"
STATE_FILE_NAME = "state.json"
SEGMENTS_DIR_NAME = "segments"
READ_CHUNK_SIZE = 1024 * 1024
POLL_INTERVAL_SEC = 0.25
RESTART_DELAY_SEC = 2.0


@dataclass(frozen=True, order=True)
class SegmentRef:
    """Stable ordering for segment files across reconnect epochs."""

    epoch: int
    index: int
    path: Path


def parse_segment_ref(path: Path) -> SegmentRef | None:
    """Parse a segment path like epoch_000001/seg_000123.ts."""
    if path.suffix not in (".ts", ".h264"):
        return None
    try:
        epoch = int(path.parent.name.split("_", 1)[1])
        index = int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return SegmentRef(epoch=epoch, index=index, path=path)


def list_complete_segments(
    segments_root: Path,
    active_epoch: int | None,
    finalized_epochs: Iterable[int],
) -> list[SegmentRef]:
    """Return complete segment files in playback order.

    The latest segment in the active epoch is withheld while ffmpeg may still be
    writing to it. Once the epoch is finalized, all of its segments are eligible.
    """
    finalized = set(finalized_epochs)
    refs: list[SegmentRef] = []
    for path in segments_root.glob("epoch_*/seg_*.h264"):
        ref = parse_segment_ref(path)
        if ref is not None:
            refs.append(ref)

    refs.sort()
    if active_epoch is None or active_epoch in finalized:
        return refs

    latest_active_index = None
    for ref in refs:
        if ref.epoch == active_epoch:
            latest_active_index = ref.index

    if latest_active_index is None:
        return refs

    return [
        ref
        for ref in refs
        if not (ref.epoch == active_epoch and ref.index == latest_active_index)
    ]


def should_rebuffer(
    pending_count: int,
    primed: bool,
    target_segments: int,
    low_watermark_segments: int,
) -> bool:
    """Decide whether playback should pause to rebuild backlog."""
    if primed:
        return pending_count < low_watermark_segments
    return pending_count < target_segments


class SegmentBufferHelper:
    """Own the writer, spool, and FIFO feed for delayed playback."""

    def __init__(
        self,
        *,
        stream_id: int,
        rtsp_url: str,
        codec: str,
        fps: float = 25.0,
        buffer_dir: str,
        segment_seconds: int,
        delay_seconds: int,
        retention_seconds: int,
    ):
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.codec = codec.lower()
        self.fps = max(1.0, float(fps))
        self.buffer_dir = Path(buffer_dir)
        self.segments_root = self.buffer_dir / SEGMENTS_DIR_NAME
        self.fifo_path = self.buffer_dir / FIFO_NAME
        self.state_path = self.buffer_dir / STATE_FILE_NAME
        self.segment_seconds = max(1, int(segment_seconds))
        self.delay_seconds = max(self.segment_seconds, int(delay_seconds))
        self.retention_seconds = max(self.delay_seconds, int(retention_seconds))
        self.target_segments = max(1, math.ceil(self.delay_seconds / self.segment_seconds))
        self.low_watermark_segments = max(1, self.target_segments // 4)

        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._active_epoch: int | None = None
        self._finalized_epochs: set[int] = set()
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._ffmpeg_segmenter_proc: subprocess.Popen | None = None
        self._last_fed: SegmentRef | None = None
        self._fed_history: deque[tuple[float, SegmentRef]] = deque()
        self._fifo_guard_read_fd: int | None = None
        self._fifo_guard_write_fd: int | None = None
        self._fifo_data_write_fd: int | None = None
        self._writer_thread: threading.Thread | None = None
        self._feeder_thread: threading.Thread | None = None
        self._last_published_state: dict | None = None

    def run(self) -> int:
        """Start the helper and block until shutdown."""
        self._install_signal_handlers()
        self._prepare_buffer_dir()
        self._open_fifo_guard_fds()

        self._writer_thread = threading.Thread(target=self._writer_loop, name="segment-writer", daemon=True)
        self._feeder_thread = threading.Thread(target=self._feeder_loop, name="segment-feeder", daemon=True)
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
        """Stop worker threads and close resources."""
        self._stop_event.set()
        self._terminate_ffmpeg()
        for thread in (self._writer_thread, self._feeder_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5)
        self._publish_state("stopped", pending_segments=0, active_epoch=None)
        for attr in ("_fifo_data_write_fd", "_fifo_guard_write_fd", "_fifo_guard_read_fd"):
            fd = getattr(self, attr)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)

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
        self._publish_state("buffering", pending_segments=0, active_epoch=None)
        logger.info(
            "Stream %s: segment buffer ready (dir=%s, fifo=%s, target=%ss, retention=%ss)",
            self.stream_id,
            self.segments_root,
            self.fifo_path,
            self.delay_seconds,
            self.retention_seconds,
        )

    def _publish_state(self, mode: str, *, pending_segments: int, active_epoch: int | None) -> None:
        state = {
            "mode": mode,
            "pending_segments": max(0, int(pending_segments)),
            "target_segments": self.target_segments,
            "updated_at": time.time(),
            "active_epoch": active_epoch,
        }
        if self._last_published_state is not None:
            previous = dict(self._last_published_state)
            previous_updated = previous.pop("updated_at", None)
            current = dict(state)
            current_updated = current.pop("updated_at", None)
            if previous == current and previous_updated is not None and current_updated is not None:
                if (current_updated - previous_updated) < 1.0:
                    return

        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.state_path)
        self._last_published_state = state

    def _build_ffmpeg_reader_cmd(self) -> list[str]:
        """RTSP reader: camera → raw H264 bitstream → stdout pipe.

        Raw H264 (not MPEGTS) avoids timestamp discontinuity issues: when the camera
        briefly resets its RTSP session, MPEGTS timestamps jump and confuse the
        segment muxer. Raw H264 carries no timestamps so the segmenter assigns its
        own frame-rate-based timestamps and splits cleanly regardless of camera events.

        -stimeout 10000000: exit after 10s of no data so the writer_loop can start a
        new epoch promptly. Without this the reader can get stuck in TCP FIN-WAIT-1
        (camera stops RTP but doesn't close TCP) and block the pipe indefinitely.
        """
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "10000000",
            "-i",
            self.rtsp_url,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-f",
            "h264",
            "pipe:1",
        ]

    def _build_ffmpeg_segmenter_cmd(self, epoch_dir: Path) -> list[str]:
        """Segmenter: raw H264 bitstream from stdin pipe → H264 segment files.

        -f h264 -r {fps}: demux as raw H264 and assign timestamps at the camera's
        native framerate. The segment muxer then splits every segment_seconds based
        on these assigned timestamps rather than on RTSP PTS values, making it immune
        to camera timestamp discontinuities. NO -nostdin: stdin IS the input pipe.
        """
        output_pattern = str(epoch_dir / "seg_%06d.h264")
        fps = str(self.fps)
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-r",
            fps,
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_seconds),
            "-reset_timestamps",
            "1",
            output_pattern,
        ]

    def _open_fifo_guard_fds(self) -> None:
        self._fifo_guard_read_fd = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        self._fifo_guard_write_fd = os.open(self.fifo_path, os.O_WRONLY | os.O_NONBLOCK)

    def _writer_loop(self) -> None:
        epoch = 0
        while not self._stop_event.is_set():
            epoch_dir = self.segments_root / f"epoch_{epoch:06d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            with self._state_lock:
                self._active_epoch = epoch
            proc = self._spawn_ffmpeg(epoch_dir)
            self._ffmpeg_proc = proc
            exit_code = proc.wait()
            # Reader exited — terminate the segmenter (its stdin pipe is now broken).
            seg = self._ffmpeg_segmenter_proc
            if seg and seg.poll() is None:
                seg.terminate()
                try:
                    seg.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    seg.kill()
            with self._state_lock:
                self._finalized_epochs.add(epoch)
                self._active_epoch = None
                self._ffmpeg_proc = None
                self._ffmpeg_segmenter_proc = None

            if self._stop_event.is_set():
                break

            logger.warning(
                "Stream %s: ffmpeg reader exited (code=%s), restarting in %.0fs",
                self.stream_id,
                exit_code,
                RESTART_DELAY_SEC,
            )
            epoch += 1
            self._stop_event.wait(RESTART_DELAY_SEC)

    def _spawn_ffmpeg(self, epoch_dir: Path) -> subprocess.Popen:
        """Spawn reader + segmenter connected by a pipe. Return the reader process.

        The reader reads RTSP and writes MPEGTS to the pipe — zero disk I/O, no
        -stimeout, emulating the standalone /dev/null test that survived 27+ min.
        The segmenter reads MPEGTS from the pipe and writes segment files — disk I/O
        is completely decoupled from the camera session. A 1MB pipe absorbs brief
        segmenter stalls without blocking the reader.
        """
        pipe_read_fd, pipe_write_fd = os.pipe()
        for sz in (1024 * 1024, 512 * 1024, 256 * 1024):
            try:
                fcntl.fcntl(pipe_write_fd, fcntl.F_SETPIPE_SZ, sz)
                logger.info(
                    "Stream %s: inter-ffmpeg pipe size set to %d KB",
                    self.stream_id,
                    sz // 1024,
                )
                break
            except OSError:
                continue

        reader_cmd = self._build_ffmpeg_reader_cmd()
        segmenter_cmd = self._build_ffmpeg_segmenter_cmd(epoch_dir)
        logger.warning(
            "Stream %s: dual-ffmpeg starting (segment=%ss, epoch_dir=%s)",
            self.stream_id,
            self.segment_seconds,
            epoch_dir,
        )
        reader_proc = subprocess.Popen(
            reader_cmd,
            stdout=pipe_write_fd,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        segmenter_proc = subprocess.Popen(
            segmenter_cmd,
            stdin=pipe_read_fd,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        os.close(pipe_read_fd)
        os.close(pipe_write_fd)
        self._ffmpeg_segmenter_proc = segmenter_proc
        threading.Thread(
            target=self._drain_stderr, args=(reader_proc, "reader"), daemon=True
        ).start()
        threading.Thread(
            target=self._drain_stderr, args=(segmenter_proc, "segmenter"), daemon=True
        ).start()
        return reader_proc

    def _drain_stderr(self, proc: subprocess.Popen, label: str) -> None:
        """Drain a subprocess's stderr to the logger to prevent pipe blocking."""
        if proc.stderr is None:
            return
        for line in proc.stderr:
            msg = line.decode(errors="replace").rstrip()
            if msg:
                logger.warning(
                    "Stream %s: ffmpeg-%s: %s", self.stream_id, label, msg
                )
        proc.stderr.close()

    def _feeder_loop(self) -> None:
        self._fifo_data_write_fd = os.open(self.fifo_path, os.O_WRONLY)
        # Increase pipe buffer so segment writes complete without blocking.
        # Default 64 KB < one 2s segment (~200 KB at sub-stream bitrate), so os.write
        # blocks for ~segment_seconds, halving effective FPS. System pipe-max-size
        # limits non-root processes to 1 MB; fall back to smaller sizes if needed.
        for _pipe_sz in (1024 * 1024, 512 * 1024, 256 * 1024):
            try:
                fcntl.fcntl(self._fifo_data_write_fd, fcntl.F_SETPIPE_SZ, _pipe_sz)
                logger.info("Stream %s: FIFO pipe size set to %d KB", self.stream_id, _pipe_sz // 1024)
                break
            except OSError:
                continue
        primed = False
        next_feed_deadline: float | None = None
        while not self._stop_event.is_set():
            active_epoch, finalized_epochs = self._snapshot_state()
            complete_segments = list_complete_segments(self.segments_root, active_epoch, finalized_epochs)
            pending_segments = self._pending_segments(complete_segments)
            pending_count = len(pending_segments)

            if should_rebuffer(
                pending_count,
                primed,
                self.target_segments,
                self.low_watermark_segments,
            ):
                self._publish_state(
                    "rebuffering" if primed else "buffering",
                    pending_segments=pending_count,
                    active_epoch=active_epoch,
                )
                if primed:
                    logger.warning(
                        "Stream %s: buffer depth dropped to %s segments, pausing to rebuffer to %ss",
                        self.stream_id,
                        pending_count,
                        self.delay_seconds,
                    )
                primed = False
                next_feed_deadline = None
                self._stop_event.wait(POLL_INTERVAL_SEC)
                continue

            if not primed:
                logger.info(
                    "Stream %s: buffer primed (%s pending segments, target=%s)",
                    self.stream_id,
                    pending_count,
                    self.target_segments,
                )
                primed = True
                next_feed_deadline = None

            self._publish_state(
                "playing",
                pending_segments=pending_count,
                active_epoch=active_epoch,
            )
            if not self._wait_for_feed_slot(next_feed_deadline):
                break

            next_segment = pending_segments[0]
            write_start = time.monotonic()
            self._write_segment(next_segment.path)
            self._last_fed = next_segment
            self._fed_history.append((time.time(), next_segment))
            self._prune_fed_segments()
            # Use write_start (not write-end) so the deadline advances from when
            # the segment started playing, not after the rate-limited write finishes.
            next_feed_deadline = self._advance_feed_deadline(next_feed_deadline, write_start)

    def _snapshot_state(self) -> tuple[int | None, set[int]]:
        with self._state_lock:
            return self._active_epoch, set(self._finalized_epochs)

    def _pending_segments(self, complete_segments: list[SegmentRef]) -> list[SegmentRef]:
        if self._last_fed is None:
            return complete_segments
        return [segment for segment in complete_segments if segment > self._last_fed]

    def _advance_feed_deadline(self, previous_deadline: float | None, handed_off_at: float) -> float:
        base = handed_off_at if previous_deadline is None else max(previous_deadline, handed_off_at)
        return base + self.segment_seconds

    def _wait_for_feed_slot(self, next_feed_deadline: float | None) -> bool:
        if next_feed_deadline is None:
            return True
        while not self._stop_event.is_set():
            remaining = next_feed_deadline - time.monotonic()
            if remaining <= 0:
                return True
            self._stop_event.wait(min(remaining, POLL_INTERVAL_SEC))
        return False

    def _write_segment(self, path: Path) -> None:
        """Write segment to FIFO, paced to the segment's natural bitrate.

        Without rate-limiting the full segment (~530 KB) is consumed by the
        decoder in <100 ms, producing a 2 s decode burst then a 2 s idle gap
        (GPU swings 33% ↔ 1%) that halves effective pipeline FPS to ~14 fps.
        Writing in 64 KB chunks at the stream's natural byte-rate keeps the
        decoder fed at a steady 25 fps.
        """
        logger.debug("Stream %s: feeding %s", self.stream_id, path.relative_to(self.buffer_dir))
        try:
            file_size = path.stat().st_size
        except OSError:
            return
        if file_size == 0:
            return

        # bytes/sec matching the segment's natural bitrate
        bytes_per_sec = file_size / self.segment_seconds
        chunk_size = 65536  # match fdsrc blocksize for natural per-chunk pacing
        start = time.monotonic()
        bytes_written = 0
        with path.open("rb") as fh:
            while not self._stop_event.is_set():
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                self._write_all(chunk)
                bytes_written += len(chunk)
                elapsed = time.monotonic() - start
                target_time = bytes_written / bytes_per_sec
                slack = target_time - elapsed
                if slack > 0.005:
                    self._stop_event.wait(slack)

    def _write_all(self, data: bytes) -> None:
        assert self._fifo_data_write_fd is not None
        offset = 0
        while offset < len(data) and not self._stop_event.is_set():
            written = os.write(self._fifo_data_write_fd, data[offset:])
            if written <= 0:
                raise RuntimeError("FIFO write returned no progress")
            offset += written

    def _prune_fed_segments(self) -> None:
        now = time.time()
        while self._fed_history and (now - self._fed_history[0][0]) >= self.retention_seconds:
            _, segment = self._fed_history.popleft()
            try:
                segment.path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Stream %s: prune failed for %s: %s", self.stream_id, segment.path, exc)
                continue
            try:
                segment.path.parent.rmdir()
            except OSError:
                pass

    def _terminate_ffmpeg(self) -> None:
        for proc in (self._ffmpeg_proc, self._ffmpeg_segmenter_proc):
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-id", type=int, required=True)
    parser.add_argument("--rtsp-url", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--buffer-dir", required=True)
    parser.add_argument("--segment-seconds", type=int, required=True)
    parser.add_argument("--delay-seconds", type=int, required=True)
    parser.add_argument("--retention-seconds", type=int, required=True)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args()
    helper = SegmentBufferHelper(
        stream_id=args.stream_id,
        rtsp_url=args.rtsp_url,
        codec=args.codec,
        fps=args.fps,
        buffer_dir=args.buffer_dir,
        segment_seconds=args.segment_seconds,
        delay_seconds=args.delay_seconds,
        retention_seconds=args.retention_seconds,
    )
    return helper.run()


if __name__ == "__main__":
    raise SystemExit(main())
