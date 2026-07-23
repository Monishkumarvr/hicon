#!/usr/bin/env python3
"""hicon-timeline — additive, standalone service.

Publishes a continuous, gap-free, delayed re-stream of each HiCon camera into a
dedicated local MediaMTX instance, sourced entirely from NVR 192.168.28.6
(confirmed continuous through the site's recurring L2 network outages — see
docs/nvr_backfill_feasibility_2026-07-23.md).

This process is completely independent of hicon_pipeline.py / hicon-vision.service,
which keeps running unmodified against the live cameras. Nothing here changes the
existing pipeline; pointing it at this timeline's output is a separate, later,
explicit config choice (HICON_RTSP_STREAM_x -> rtsp://127.0.0.1:{port}/timelineN).

Per stream, spawns two child processes:
  - pipeline/nvr_timeline_helper.py — rolls forward through NVR playback windows,
    writes a paced FIFO + a capture-clock anchor (anchor.json).
  - a publisher ffmpeg that reads that FIFO and RTSP-pushes it into the timeline
    MediaMTX instance, the same rtspclientsink-via-ffmpeg idiom already used by
    tools/stream0_ffmpeg_publisher.sh.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("timeline_service")

HELPER_PATH = BASE_DIR / "pipeline" / "nvr_timeline_helper.py"
STREAMS = (0, 1, 2)
DEFAULT_TRACKS = {0: "1201", 1: "901", 2: "1301"}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _load_config() -> dict:
    cfg = {
        "nvr_host": _env("HICON_TIMELINE_NVR_HOST", "192.168.28.6"),
        "nvr_user": _env("HICON_TIMELINE_NVR_USER", "admin"),
        "nvr_pass": _env("HICON_TIMELINE_NVR_PASS", ""),
        "chunk_seconds": int(_env("HICON_TIMELINE_CHUNK_SEC", "30")),
        "delay_seconds": int(_env("HICON_TIMELINE_DELAY_SEC", "205")),
        "min_delay_seconds": int(_env("HICON_TIMELINE_MIN_DELAY_SEC", "180")),
        "max_delay_seconds": int(_env("HICON_TIMELINE_MAX_DELAY_SEC", "300")),
        "retention_seconds": int(_env("HICON_TIMELINE_RETENTION_SEC", "900")),
        "buffer_dir": _env("HICON_TIMELINE_BUFFER_DIR", str(BASE_DIR / "output" / "timeline_buffer")),
        "mediamtx_port": _env("HICON_TIMELINE_MEDIAMTX_PORT", "8555"),
    }
    for sid in STREAMS:
        cfg[f"track_{sid}"] = _env(f"HICON_TIMELINE_TRACK_{sid}", DEFAULT_TRACKS[sid])
        cfg[f"codec_{sid}"] = _env(f"HICON_TIMELINE_CODEC_{sid}", "h265")
        cfg[f"fps_{sid}"] = _env(f"HICON_TIMELINE_FPS_{sid}", "25")
    if not cfg["nvr_pass"]:
        raise SystemExit("HICON_TIMELINE_NVR_PASS not set — add it to .env before running timeline_service.py")
    return cfg


def _build_helper_cmd(stream_id: int, cfg: dict) -> list[str]:
    return [
        sys.executable, str(HELPER_PATH),
        "--stream-id", str(stream_id),
        "--nvr-host", cfg["nvr_host"],
        "--nvr-user", cfg["nvr_user"],
        "--nvr-pass", cfg["nvr_pass"],
        "--track-id", str(cfg[f"track_{stream_id}"]),
        "--codec", cfg[f"codec_{stream_id}"],
        "--fps", str(cfg[f"fps_{stream_id}"]),
        "--buffer-dir", str(_stream_buffer_dir(stream_id, cfg)),
        "--chunk-seconds", str(cfg["chunk_seconds"]),
        "--initial-delay-seconds", str(cfg["delay_seconds"]),
        "--min-delay-seconds", str(cfg["min_delay_seconds"]),
        "--max-delay-seconds", str(cfg["max_delay_seconds"]),
        "--retention-seconds", str(cfg["retention_seconds"]),
    ]


def _stream_buffer_dir(stream_id: int, cfg: dict) -> Path:
    return Path(cfg["buffer_dir"]) / f"stream{stream_id}"


def _build_publisher_cmd(stream_id: int, cfg: dict) -> list[str]:
    fifo_path = _stream_buffer_dir(stream_id, cfg) / "stream.fifo"
    fmt = "hevc" if cfg[f"codec_{stream_id}"] == "h265" else "h264"
    fps = str(cfg[f"fps_{stream_id}"])
    port = cfg["mediamtx_port"]
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-f", fmt, "-r", fps, "-i", str(fifo_path),
        "-map", "0:v:0", "-c:v", "copy", "-an",
        "-f", "rtsp", "-rtsp_transport", "tcp",
        f"rtsp://127.0.0.1:{port}/timeline{stream_id}",
    ]


class TimelineService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._procs: dict[str, subprocess.Popen] = {}
        self._stop_event = threading.Event()

    def run(self) -> int:
        self._install_signal_handlers()
        for sid in STREAMS:
            self._start_helper(sid)
        for sid in STREAMS:
            self._wait_for_fifo(sid)
            self._start_publisher(sid)

        logger.info("hicon-timeline running (streams=%s, mediamtx_port=%s)", STREAMS, self.cfg["mediamtx_port"])
        try:
            while not self._stop_event.is_set():
                self._check_children()
                self._stop_event.wait(2.0)
        finally:
            self._shutdown()
        return 0

    def _wait_for_fifo(self, sid: int, timeout: float = 30.0) -> None:
        fifo_path = _stream_buffer_dir(sid, self.cfg) / "stream.fifo"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if fifo_path.exists():
                return
            time.sleep(0.2)
        logger.warning("Stream %s: FIFO %s not ready after %.0fs, starting publisher anyway",
                       sid, fifo_path, timeout)

    def _start_helper(self, sid: int) -> None:
        cmd = _build_helper_cmd(sid, self.cfg)
        logger.info("Stream %s: starting NVR timeline helper (track=%s)", sid, self.cfg[f"track_{sid}"])
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._procs[f"helper{sid}"] = proc

    def _start_publisher(self, sid: int) -> None:
        cmd = _build_publisher_cmd(sid, self.cfg)
        logger.info("Stream %s: starting publisher -> rtsp://127.0.0.1:%s/timeline%s",
                    sid, self.cfg["mediamtx_port"], sid)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._procs[f"publisher{sid}"] = proc
        threading.Thread(target=self._drain_stderr, args=(sid, proc), daemon=True).start()

    def _drain_stderr(self, sid: int, proc: subprocess.Popen) -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            msg = line.decode(errors="replace").rstrip()
            if msg:
                logger.debug("Stream %s: publisher: %s", sid, msg)

    def _check_children(self) -> None:
        if self._stop_event.is_set():
            return
        for name, proc in list(self._procs.items()):
            if proc.poll() is None:
                continue
            logger.error("%s exited (code=%s), restarting", name, proc.returncode)
            sid = int(name[-1])
            if name.startswith("helper"):
                self._start_helper(sid)
            else:
                self._wait_for_fifo(sid, timeout=10.0)
                self._start_publisher(sid)

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            logger.info("Received signal %s, shutting down", signum)
            self._stop_event.set()

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, _handler)

    def _shutdown(self) -> None:
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.terminate()
        for name, proc in self._procs.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        logger.info("hicon-timeline stopped")


def main() -> int:
    cfg = _load_config()
    Path(cfg["buffer_dir"]).mkdir(parents=True, exist_ok=True)
    service = TimelineService(cfg)
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())
