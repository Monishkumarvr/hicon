"""Allocation-light aggregate telemetry (Edge_Optimization_Plan.md Phase 1).

Design notes:
  - Latency samples are pushed by `utils.perf.timed_section` (already wired at the
    3 per-stream probe call sites) into a bounded per-name deque; no new hot-path
    instrumentation was added to collect them.
  - Queue depths / counters are pulled via zero-arg gauge callables registered once
    at startup (e.g. `AsyncDBWriter._queue.qsize`), not pushed per-frame.
  - GPU/NVDEC/VIC/EMC/power are NOT sampled here: tegrastats is root-only on this
    kernel and the app process is correctly unprivileged. That coverage comes from
    the separate `hicon-tegrastats.service` sidecar. This module covers what an
    unprivileged process can read cheaply: probe latency, process RSS/threads,
    thermal zones, zram, and this process's own cgroup memory.
  - This kernel has no CONFIG_PSI (verified) — cgroup `memory.current`/`memory.stat`
    are used instead of `/proc/pressure/*`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("metrics")

_LATENCY_WINDOW = 2000  # samples per metric name before oldest is dropped


class MetricsRegistry:
    """Thread-safe latency/counter/gauge store with periodic percentile snapshots."""

    def __init__(self, window: int = _LATENCY_WINDOW):
        self._window = window
        self._lock = threading.Lock()
        self._latencies: Dict[str, deque] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, Callable[[], Optional[float]]] = {}

    def record_latency(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            dq = self._latencies.get(name)
            if dq is None:
                dq = deque(maxlen=self._window)
                self._latencies[name] = dq
            dq.append(elapsed_ms)

    def increment(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def register_gauge(self, name: str, fn: Callable[[], Optional[float]]) -> None:
        """Register a zero-arg callable read at snapshot time. Idempotent by name."""
        with self._lock:
            self._gauges[name] = fn

    @staticmethod
    def _percentiles(samples) -> dict:
        if not samples:
            return {}
        ordered = sorted(samples)
        n = len(ordered)

        def pct(p):
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return round(ordered[idx], 2)

        return {
            "count": n,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": round(ordered[-1], 2),
        }

    def snapshot(self, reset_latencies: bool = True) -> dict:
        with self._lock:
            latency_snap = {
                name: self._percentiles(list(dq))
                for name, dq in self._latencies.items()
                if dq
            }
            counters_snap = dict(self._counters)
            if reset_latencies:
                for dq in self._latencies.values():
                    dq.clear()

        gauges_snap = {}
        # Gauge callables run outside the lock — they may call into other subsystems
        # (queue.qsize) and must never be able to deadlock against this registry.
        for name, fn in list(self._gauges.items()):
            try:
                gauges_snap[name] = fn()
            except Exception:
                gauges_snap[name] = None

        return {"latency_ms": latency_snap, "counters": counters_snap, "gauges": gauges_snap}


REGISTRY = MetricsRegistry()


def _read_process_stats() -> dict:
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_kb"] = int(line.split()[1])
                elif line.startswith("Threads:"):
                    out["threads"] = int(line.split()[1])
    except OSError:
        pass
    return out


def _read_thermal_zones() -> dict:
    out = {}
    try:
        base = "/sys/class/thermal"
        for entry in os.listdir(base):
            if not entry.startswith("thermal_zone"):
                continue
            try:
                with open(f"{base}/{entry}/type") as f:
                    zone_type = f.read().strip()
                with open(f"{base}/{entry}/temp") as f:
                    milli_c = int(f.read().strip())
                out[zone_type] = round(milli_c / 1000.0, 1)
            except (OSError, ValueError, TypeError):
                # Some tegra thermal nodes intermittently fail reads with odd
                # errors (TypeError from codecs on a None raw read) — skip zone.
                continue
    except OSError:
        pass
    return out


def _read_zram() -> dict:
    """Aggregate mm_stat across all zram devices. Columns per kernel docs:
    orig_data_size compr_data_size mem_used_total mem_limit mem_used_max
    same_pages pages_compacted huge_pages huge_pages_since.
    """
    orig_total = 0
    compr_total = 0
    used_total = 0
    found = False
    try:
        for entry in os.listdir("/sys/block"):
            if not entry.startswith("zram"):
                continue
            try:
                with open(f"/sys/block/{entry}/mm_stat") as f:
                    fields = f.read().split()
                orig_total += int(fields[0])
                compr_total += int(fields[1])
                used_total += int(fields[2])
                found = True
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass
    if not found:
        return {}
    return {
        "orig_mb": round(orig_total / (1024 * 1024), 1),
        "compressed_mb": round(compr_total / (1024 * 1024), 1),
        "used_mb": round(used_total / (1024 * 1024), 1),
    }


def _read_own_cgroup_memory() -> dict:
    """Read memory.current + a few memory.stat fields for this process's own
    cgroup (path resolved from /proc/self/cgroup — works regardless of unit name).
    """
    out = {}
    try:
        with open("/proc/self/cgroup") as f:
            content = f.read().strip()
        # cgroup v2: single line "0::/system.slice/hicon-vision.service"
        cgroup_path = content.split(":")[-1]
        base = f"/sys/fs/cgroup{cgroup_path}"
        with open(f"{base}/memory.current") as f:
            out["current_mb"] = round(int(f.read().strip()) / (1024 * 1024), 1)
        with open(f"{base}/memory.stat") as f:
            for line in f:
                key, _, value = line.strip().partition(" ")
                if key in ("anon", "file", "kernel_stack"):
                    out[f"stat_{key}_mb"] = round(int(value) / (1024 * 1024), 1)
        events_path = f"{base}/memory.events"
        if os.path.exists(events_path):
            with open(events_path) as f:
                for line in f:
                    key, _, value = line.strip().partition(" ")
                    if key in ("high", "max", "oom", "oom_kill"):
                        out[f"events_{key}"] = int(value)
    except (OSError, ValueError, IndexError):
        pass
    return out


class MetricsReporter(threading.Thread):
    """Background thread: every `interval_sec`, snapshot the registry + cheap
    unprivileged system stats, and emit one structured INFO line. Never touches
    the GStreamer probe hot path.
    """

    def __init__(self, interval_sec: float = 60.0):
        super().__init__(name="hicon-metrics-reporter", daemon=True)
        self._interval_sec = max(5.0, float(interval_sec))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("MetricsReporter started (interval=%.0fs)", self._interval_sec)
        while not self._stop_event.wait(self._interval_sec):
            try:
                payload = {
                    "probes": REGISTRY.snapshot(),
                    "process": _read_process_stats(),
                    "thermal_c": _read_thermal_zones(),
                    "zram": _read_zram(),
                    "cgroup_memory": _read_own_cgroup_memory(),
                }
                metrics_logger.info("[METRICS] %s", json.dumps(payload, sort_keys=True))
            except Exception:
                logger.exception("MetricsReporter: snapshot failed")
        logger.info("MetricsReporter stopped")
