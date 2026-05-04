#!/usr/bin/env python3
"""Capture a repeatable Stream 0 RTSP soak evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _now_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(command: list[str], output_path: Path | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if output_path is not None:
        output_path.write_text(
            f"$ {' '.join(command)}\n\n"
            f"exit_code={result.returncode}\n\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n",
            encoding="utf-8",
        )
    return result


def _best_effort_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _detect_iface(camera_ip: str) -> str | None:
    if shutil.which("ip") is None:
        return None
    result = _run(["ip", "route", "get", camera_ip])
    if result.returncode != 0:
        return None
    match = re.search(r"\bdev\s+(\S+)", result.stdout)
    return match.group(1) if match else None


def _read_netdev_stats(iface: str | None) -> dict[str, int]:
    if not iface:
        return {}
    stats_dir = Path("/sys/class/net") / iface / "statistics"
    stats = {}
    if not stats_dir.exists():
        return stats
    for stat_path in sorted(stats_dir.iterdir()):
        try:
            stats[stat_path.name] = int(stat_path.read_text(encoding="utf-8").strip())
        except Exception:
            continue
    return stats


def _capture_env_subset() -> dict[str, str]:
    prefixes = (
        "HICON_RTSP_",
        "HICON_USE_NVURISRCBIN_",
        "HICON_STREAM_0_",
        "HICON_ENABLE_RTSP_STREAM_",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }


def _capture_env_file_subset() -> dict[str, str]:
    env_path = Path("ai_vision/.env")
    if not env_path.exists():
        return {}
    subset = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.startswith((
            "HICON_RTSP_",
            "HICON_USE_NVURISRCBIN_",
            "HICON_STREAM_0_",
            "HICON_ENABLE_RTSP_STREAM_",
        )):
            subset[key] = value
    return subset


def _parse_outage_summary(log_text: str, stream_id: int) -> dict:
    starts = []
    recoveries = []
    for line in log_text.splitlines():
        if f"[RTSP-OUTAGE] stream={stream_id} phase=start" in line:
            starts.append(line)
        elif f"[RTSP-OUTAGE] stream={stream_id} phase=recovered" in line:
            recoveries.append(line)
    return {
        "stream_id": stream_id,
        "start_count": len(starts),
        "recovery_count": len(recoveries),
        "start_lines": starts,
        "recovery_lines": recoveries,
    }


def _start_tegrastats(output_path: Path, interval_ms: int) -> subprocess.Popen | None:
    tegrastats = shutil.which("tegrastats")
    if tegrastats is None:
        return None
    out_handle = output_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [tegrastats, "--interval", str(interval_ms)],
        stdout=out_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._hicon_output_handle = out_handle  # type: ignore[attr-defined]
    return proc


def _stop_process(proc: subprocess.Popen | None):
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)
    finally:
        output_handle = getattr(proc, "_hicon_output_handle", None)
        if output_handle is not None:
            output_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Short soak label, e.g. baseline or force-tcp")
    parser.add_argument("--duration-sec", type=int, default=1800, help="Soak duration in seconds")
    parser.add_argument("--stream-id", type=int, default=0, help="Stream id to summarize")
    parser.add_argument("--camera-ip", default="192.168.28.119", help="Camera IP for route/NIC detection")
    parser.add_argument("--service", default="hicon-vision", help="systemd service name")
    parser.add_argument(
        "--output-root",
        default="ai_vision/output/stream0_soaks",
        help="Directory that will receive the soak bundle",
    )
    parser.add_argument(
        "--tegrastats-interval-ms",
        type=int,
        default=1000,
        help="tegrastats sampling interval in milliseconds",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.output_root) / f"{_now_label()}_{args.label}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()
    start_iso = datetime.fromtimestamp(start_wall).strftime("%Y-%m-%d %H:%M:%S")
    iface = _detect_iface(args.camera_ip)
    start_net_stats = _read_netdev_stats(iface)

    _best_effort_json(
        bundle_dir / "metadata.json",
        {
            "label": args.label,
            "duration_sec": args.duration_sec,
            "stream_id": args.stream_id,
            "camera_ip": args.camera_ip,
            "service": args.service,
            "started_at": start_iso,
            "detected_iface": iface,
            "environment": _capture_env_subset(),
            "env_file": _capture_env_file_subset(),
        },
    )
    _best_effort_json(bundle_dir / "netdev_start.json", start_net_stats)

    _run(["uname", "-a"], bundle_dir / "uname.txt")
    _run(["sysctl", "net.core.rmem_default", "net.core.rmem_max"], bundle_dir / "sysctl_core.txt")
    _run(
        ["sysctl", "net.ipv4.tcp_rmem", "net.ipv4.udp_rmem_min", "net.core.netdev_max_backlog"],
        bundle_dir / "sysctl_net.txt",
    )
    _run(["ss", "-tin"], bundle_dir / "ss_start.txt")
    if iface:
        _run(["ethtool", "-S", iface], bundle_dir / "ethtool_start.txt")

    tegrastats_proc = _start_tegrastats(bundle_dir / "tegrastats.log", args.tegrastats_interval_ms)

    try:
        time.sleep(args.duration_sec)
    finally:
        _stop_process(tegrastats_proc)

    end_wall = time.time()
    end_iso = datetime.fromtimestamp(end_wall).strftime("%Y-%m-%d %H:%M:%S")

    _run(["ss", "-tin"], bundle_dir / "ss_end.txt")
    if iface:
        _run(["ethtool", "-S", iface], bundle_dir / "ethtool_end.txt")
    _best_effort_json(bundle_dir / "netdev_end.json", _read_netdev_stats(iface))

    journal = _run(
        ["journalctl", "-u", args.service, "--since", start_iso, "--until", end_iso, "--no-pager"],
        bundle_dir / "journal.txt",
    )
    journal_text = journal.stdout if journal.returncode == 0 else ""
    _best_effort_json(
        bundle_dir / "summary.json",
        {
            "label": args.label,
            "stream_id": args.stream_id,
            "camera_ip": args.camera_ip,
            "service": args.service,
            "started_at": start_iso,
            "ended_at": end_iso,
            "duration_sec": round(end_wall - start_wall, 1),
            "detected_iface": iface,
            "outage_summary": _parse_outage_summary(journal_text, args.stream_id),
        },
    )
    print(bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
