#!/usr/bin/env python3
"""Push the local Stream 0 overlay RTSP path to the remote RTSP destination."""
from __future__ import annotations

import os
import sys


def build_ffmpeg_command(rtsp_port: str, mtx_path: str, remote_url: str) -> list[str]:
    local_input = f"rtsp://127.0.0.1:{rtsp_port}/{mtx_path}"
    return [
        "/usr/bin/ffmpeg",
        "-nostdin",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-i",
        local_input,
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        remote_url,
    ]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) != 3:
        print("Usage: stream0_overlay_remote_relay.py <rtsp_port> <mtx_path>", file=sys.stderr)
        return 1

    rtsp_port = argv[1]
    mtx_path = argv[2]
    remote_url = os.environ.get("HICON_STREAM0_REMOTE_RELAY_URL", "").strip()
    if not remote_url:
        print("HICON_STREAM0_REMOTE_RELAY_URL is not set", file=sys.stderr)
        return 1

    cmd = build_ffmpeg_command(rtsp_port, mtx_path, remote_url)
    os.execv(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
