"""
HiCon Pipeline - Main entry point for 3-camera DeepStream pipeline.

Stream 0 (Process Camera):   pouring (nvinfer GIE-1 + tracker) + brightness (tapping, deslagging, spectro)
Stream 1 (Pyrometer Camera): rod detection (nvinfer GIE-2)
Stream 2 (Pouring2 Camera):  pouring only (nvinfer GIE-3 + tracker, no brightness)
All cameras decode H.265/HEVC via nvv4l2decoder.
"""
import sys
import os
import logging
import time
import json
import configparser
import signal
import threading
from pathlib import Path
from datetime import datetime
from io import StringIO

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds

# Ensure ai_vision is on path
sys.path.insert(0, str(Path(__file__).parent))

import config
from db_manager import HiConDatabase, AsyncDBWriter
from pipeline.gst_builder import DeepStreamPipelineBuilder
from pipeline.bus_handler import BusHandler
from pipeline.recording import RecordingManager
from pipeline.stream0_local_relay import Stream0LocalRelayManager
from processors.brightness_processor import BrightnessProcessor
from processors.melting_analysis_controller import MeltingAnalysisController
from processors.melting_meta_reader import MeltingMetaReader
from processors.pyrometer_processor import PyrometerProcessor
from state.heat_cycle_manager import HeatCycleManager
from sync.api_client import APIClient
from sync.sync_manager import SyncManager
from utils.perf import timed_section
from utils.screenshot import AsyncScreenshotWriter
from utils.zone_loader import load_zones_config
from streaming.mjpeg_server import MJPEGServer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = config.LOG_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'pipeline.log'),
    ],
)
logger = logging.getLogger('hicon')


# ---------------------------------------------------------------------------
# Globals (set during init)
# ---------------------------------------------------------------------------
pouring_processor = None
pouring_processor_2 = None       # Optional Stream 2 pouring processor
heat_cycle_manager_2 = None      # Backward-compatible alias for the shared heat cycle manager
brightness_processor = None
brightness_processor_2 = None
melting_controller = None
melting_meta_reader = None
melting_controller_2 = None
melting_meta_reader_2 = None
pyrometer_processor = None
bus_handler = None
sync_manager = None
recording_manager = None
stream0_local_relay_manager = None
mjpeg_server = None
async_db_writer = None
screenshot_writer = None
_live_stream_last_extract = {}
_live_stream_warmup_deadline = 0.0
_RECORDING_STARTUP_WARMUP_SEC = 5


def _read_int_file(path: Path):
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _resolve_self_cgroup_memory():
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                rel_path = parts[2].strip().lstrip("/")
                cgroup_root = Path("/sys/fs/cgroup")
                memory_path = cgroup_root / rel_path / "memory.current" if rel_path else cgroup_root / "memory.current"
                return parts[2].strip() or "/", _read_int_file(memory_path)
    except Exception:
        pass
    return None, None


def _read_meminfo_snapshot():
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if fields:
                meminfo[key] = int(fields[0]) * 1024
    except Exception:
        return None, None

    mem_available = meminfo.get("MemAvailable")
    swap_total = meminfo.get("SwapTotal")
    swap_free = meminfo.get("SwapFree")
    swap_used = None
    if swap_total is not None and swap_free is not None:
        swap_used = max(0, swap_total - swap_free)
    return mem_available, swap_used


def _format_bytes(value):
    if value is None:
        return "n/a"
    return f"{value / (1024 * 1024):.1f}MiB"


def _log_memory_snapshot(reason: str):
    cgroup_path, cgroup_memory = _resolve_self_cgroup_memory()
    mem_available, swap_used = _read_meminfo_snapshot()
    logger.info(
        "[MEMORY] reason=%s cgroup_path=%s cgroup_current=%s mem_available=%s swap_used=%s",
        reason,
        cgroup_path or "n/a",
        _format_bytes(cgroup_memory),
        _format_bytes(mem_available),
        _format_bytes(swap_used),
    )


class _ProbeTimestampResolver:
    """Map per-stream PTS into stable wall-clock datetimes."""

    def __init__(self):
        self._state = {}
        self._lock = threading.Lock()

    def resolve(self, gst_buffer, stream_id):
        wall_now = time.time()
        pts_ns = getattr(gst_buffer, "pts", Gst.CLOCK_TIME_NONE) if gst_buffer else Gst.CLOCK_TIME_NONE
        if pts_ns in (None, Gst.CLOCK_TIME_NONE) or pts_ns < 0:
            return wall_now, datetime.fromtimestamp(wall_now)

        pts_sec = float(pts_ns) / float(Gst.SECOND)
        with self._lock:
            state = self._state.get(stream_id)
            if state is None or pts_sec + 1e-6 < state["last_pts_sec"]:
                state = {
                    "anchor_wall": wall_now - pts_sec,
                    "last_pts_sec": pts_sec,
                }
                self._state[stream_id] = state
            else:
                state["last_pts_sec"] = pts_sec
            timestamp = state["anchor_wall"] + pts_sec
        return timestamp, datetime.fromtimestamp(timestamp)


_probe_timestamp_resolver = _ProbeTimestampResolver()


def _bbox_from_points(points):
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _stream_zone_config(zones_config, stream_id):
    """Return a legacy-compatible zone config for one stream."""
    stream_key = f"stream_{stream_id}"
    stream_zones = zones_config.get("streams", {}).get(stream_key)
    if not stream_zones:
        stream_zones = zones_config.get("streams", {}).get(f"stream{stream_id}")
    if not stream_zones:
        return zones_config

    scoped = {"metadata": zones_config.get("metadata", {})}
    for section_name in ("tapping", "deslagging", "spectro"):
        scoped[section_name] = stream_zones.get(section_name, {"zones": {}})
    return scoped


def _source_size_for_section(section, metadata):
    size = section.get("annotation_size") or section.get("source_size")
    if size and len(size) == 2:
        return float(size[0]), float(size[1])
    return (
        float(metadata.get("source_width", metadata.get("ref_width", 1920))),
        float(metadata.get("source_height", metadata.get("ref_height", 1080))),
    )


def _scale_points(points, source_w, source_h, target_w, target_h):
    sx = float(target_w) / float(source_w) if source_w else 1.0
    sy = float(target_h) / float(source_h) if source_h else 1.0
    return [[float(x) * sx, float(y) * sy] for x, y in points]


def _scale_zone_config(zone_config, target_w, target_h):
    """Scale one detector zone config from its annotation space into mux space."""
    if not zone_config:
        return {}
    metadata = zone_config.get("metadata", {})
    scaled = {}
    for key, value in zone_config.items():
        if key == "zones":
            continue
        scaled[key] = value
    source_w, source_h = _source_size_for_section(zone_config, metadata)
    scaled["annotation_size"] = [int(target_w), int(target_h)]
    scaled_zones = {}
    for zone_name, zone in zone_config.get("zones", {}).items():
        if not zone.get("enabled", True):
            continue
        zone_copy = dict(zone)
        zone_copy["roi_points"] = _scale_points(
            zone.get("roi_points", []),
            source_w,
            source_h,
            target_w,
            target_h,
        )
        scaled_zones[zone_name] = zone_copy
    scaled["zones"] = scaled_zones
    return scaled


def _scale_detector_config(zones_config, target_w, target_h):
    metadata = zones_config.get("metadata", {})
    scaled = {"metadata": {"ref_width": int(target_w), "ref_height": int(target_h)}}
    for section_name in ("tapping", "deslagging", "spectro", "pyrometer"):
        section = zones_config.get(section_name)
        if not section:
            continue
        section_with_meta = dict(section)
        section_with_meta["metadata"] = metadata
        scaled[section_name] = _scale_zone_config(section_with_meta, target_w, target_h)
    return scaled


def _serialize_melting_plugin_config(zones_config, *, target_width=1280, target_height=720):
    """Serialize a stream's melting config into a GLib key-file string."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser["global"] = {
        "fps": str(float(getattr(config, "STREAM_FPS", 25.0))),
    }
    metadata = zones_config.get("metadata", {})

    for section_name in ("tapping", "deslagging", "spectro"):
        section = zones_config.get(section_name, {})
        zones = section.get("zones", {})
        source_w, source_h = _source_size_for_section(section, metadata)
        section_values = {}
        if section_name == "tapping":
            section_values.update({
                "abs_brightness_threshold": str(
                    section.get("abs_brightness_threshold", section.get("brightness_threshold", 210))
                ),
                "start_white_ratio": str(section.get("start_white_ratio", 0.25)),
                "start_frame_count": str(section.get("start_frame_count", 20)),
                "end_white_ratio": str(section.get("end_white_ratio", 0.10)),
                "end_frame_count": str(section.get("end_frame_count", 25)),
            })
        else:
            section_values.update({
                "min_blob_area": str(section.get("min_blob_area", 0)),
                "brightness_thresh": str(section.get("brightness_thresh", 180)),
            })

        zone_names = []
        for zone_idx, (zone_name, zone_cfg) in enumerate(zones.items()):
            if not zone_cfg.get("enabled", True):
                continue
            points = zone_cfg.get("roi_points", [])
            if not points:
                continue
            scaled_points = _scale_points(points, source_w, source_h, target_width, target_height)
            x1, y1, x2, y2 = _bbox_from_points(scaled_points)
            zone_names.append(zone_name)
            section_values[f"zone_name.{zone_idx}"] = zone_name
            section_values[f"bbox.{zone_idx}"] = f"{x1},{y1},{x2},{y2}"
            if section_name == "spectro":
                section_values[f"on_frames.{zone_idx}"] = str(zone_cfg.get("on_frames", 0))
                if zone_cfg.get("max_aspect_ratio") is not None:
                    section_values[f"max_aspect_ratio.{zone_idx}"] = str(
                        zone_cfg["max_aspect_ratio"]
                    )
                if zone_cfg.get("max_coverage") is not None:
                    section_values[f"max_coverage.{zone_idx}"] = str(zone_cfg["max_coverage"])
        section_values["zone_count"] = str(len(zone_names))
        parser[section_name] = section_values

    buffer = StringIO()
    parser.write(buffer)
    return buffer.getvalue()


def _resolve_frame_timestamp(gst_buffer, frame_meta, fallback_stream_id):
    stream_id = int(getattr(frame_meta, "source_id", fallback_stream_id))
    return _probe_timestamp_resolver.resolve(gst_buffer, stream_id)


# ---------------------------------------------------------------------------
# Pad probe callbacks
# ---------------------------------------------------------------------------
def _process_stream0_cpu_analysis(info, update_main_path=False, update_analysis_path=False):
    """Run Stream 0 CPU frame extraction and processors on the provided buffer."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    with timed_section("probe.stream0.cpu_analysis", logger=logger):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            native_melting_state = None
            if bus_handler:
                if update_main_path:
                    bus_handler.update_frame_time(0)
                if update_analysis_path:
                    bus_handler.update_stream0_analysis_time()

            if melting_meta_reader is not None:
                try:
                    native_melting_state = melting_meta_reader.decode_frame_meta(frame_meta)
                except Exception as e:
                    logger.error(f"Melting meta reader error: {e}", exc_info=True)

            run_python_pouring = pouring_processor is not None and update_main_path
            run_hybrid_melting = melting_controller is not None and native_melting_state is not None

        # CUDA brightness operates directly on GPU NvBufSurface — no CPU copy needed.
        # CPU brightness still needs frame extraction like before.
            cuda_brightness = (
                brightness_processor is not None
                and hasattr(brightness_processor, "process_frame_cuda")
            )
            cpu_brightness = (
                brightness_processor is not None
                and not cuda_brightness
                and not run_hybrid_melting
            )
            melting_needs_frame = (
                run_hybrid_melting
                and config.ENABLE_FRAME_PROCESSING
                and melting_controller.needs_frame(native_melting_state)
            )

            stream0_needs_frame = (
                config.ENABLE_FRAME_PROCESSING
                and (
                    run_python_pouring
                    or cpu_brightness
                    or melting_needs_frame
                )
            )

            frame = None
            if stream0_needs_frame:
                try:
                    import numpy as np
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    frame = np.array(n_frame, copy=True, order='C')
                except Exception as e:
                    logger.error(f"Frame extraction error: {e}", exc_info=True)

            frame_timestamp, frame_datetime = _resolve_frame_timestamp(gst_buffer, frame_meta, 0)

            try:
                # Pouring path first so brightness suppression sees the newest pouring state.
                if run_python_pouring:
                    try:
                        pouring_processor.process_frame(
                            frame_meta=frame_meta,
                            frame=frame,
                            batch_meta=batch_meta,
                            timestamp=frame_timestamp,
                            datetime_obj=frame_datetime,
                        )
                    except Exception as e:
                        logger.error(f"Pouring processor error: {e}", exc_info=True)

                if run_hybrid_melting:
                    try:
                        melting_controller.process_native_state(
                            native_state=native_melting_state,
                            frame_meta=frame_meta,
                            frame=frame if melting_needs_frame else None,
                            batch_meta=batch_meta,
                            timestamp=frame_timestamp,
                            datetime_obj=frame_datetime,
                        )
                    except Exception as e:
                        logger.error(f"Hybrid melting controller error: {e}", exc_info=True)

                if run_hybrid_melting and brightness_processor is not None:
                    try:
                        native_tapping_active = False
                        native_tapping_ratio = 0.0
                        for zone_state in getattr(native_melting_state, "tapping", []):
                            if not getattr(zone_state, "valid", 0):
                                continue
                            native_tapping_ratio = max(
                                native_tapping_ratio,
                                float(getattr(zone_state, "white_ratio", 0.0) or 0.0),
                            )
                            if bool(getattr(zone_state, "active", 0)):
                                native_tapping_active = True
                        brightness_processor.tapping_tracker.state = (
                            "ACTIVE" if native_tapping_active else "IDLE"
                        )
                        if not native_tapping_active:
                            brightness_processor.tapping_tracker.start_counter = 0
                            brightness_processor.tapping_tracker.end_counter = 0
                        brightness_processor._last_white_ratios["tapping"] = native_tapping_ratio
                    except Exception as e:
                        logger.error(f"Native tapping state sync error: {e}", exc_info=True)

                # CUDA brightness: runs on GPU NvBufSurface directly (no CPU frame)
                if cuda_brightness:
                    try:
                        brightness_processor.process_frame_cuda(
                            gst_buffer, frame_meta, batch_meta,
                        )
                    except Exception as e:
                        logger.error(f"CUDA brightness processor error: {e}", exc_info=True)

                # CPU brightness: legacy path using NumPy frame
                elif cpu_brightness and frame is not None:
                    try:
                        brightness_processor.process_frame_with_array(frame, frame_meta)
                        # display_meta intentionally NOT written here — overlays are
                        # drawn via CPU/OpenCV in post_osd_probe_stream0_for_streaming.
                        # Writing NvDsDisplayMeta in the analysis branch shares batch_meta
                        # with the main path, causing nvosd GPU accumulation → crash.
                    except Exception as e:
                        logger.error(f"Brightness processor error: {e}", exc_info=True)
            finally:
                if frame is not None:
                    try:
                        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    except Exception:
                        pass

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

    return Gst.PadProbeReturn.OK


def osd_sink_pad_probe_stream0(pad, info):
    """
    Probe on nvosd_0 sink pad (Stream 0 — Process Camera).
    Handles:
      1. Pouring detection (nvinfer object meta + CPU brightness)
      2. Brightness analysis (tapping + deslagging + spectro via CPU frame)

    Frame is extracted once and shared by both processors.
    CRITICAL: unmap_nvds_buf_surface() MUST be called on Jetson.
    """
    return _process_stream0_cpu_analysis(
        info,
        update_main_path=True,
        update_analysis_path=False,
    )


def osd_sink_pad_probe_stream0_heartbeat_main(pad, info):
    """Minimal Stream 0 main-path probe used to preserve FPS/watchdog visibility."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    if bus_handler:
        bus_handler.update_frame_time(0)
    return Gst.PadProbeReturn.OK


def osd_sink_pad_probe_stream0_heartbeat(pad, info):
    """Backward-compatible Stream 0 heartbeat probe used by diagnostic modes."""
    return osd_sink_pad_probe_stream0_heartbeat_main(pad, info)


def osd_sink_pad_probe_stream0_display_meta(pad, info):
    """Main-path display meta writer for decoupled analysis mode + active recording.

    Reads cached overlay state from analysis processors (populated by the analysis branch
    probe) and writes NvDsDisplayMeta so nvosd_0 renders detections, probe circles, and
    brightness status into the recording branch (tee_0 → rec-valve → …).

    Zero analysis computation — purely struct writes (~200 µs/frame).
    Also ticks the per-stream watchdog (replaces heartbeat_main).
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Overlays are now drawn via CPU/OpenCV in post_osd_probe_stream0_for_streaming.
    # nvosd receives no display_meta and acts as a pure pass-through — eliminating
    # the CUDA GPU state accumulation that caused ~70-min crashes.
    if bus_handler:
        bus_handler.update_frame_time(0)
    return Gst.PadProbeReturn.OK


def analysis_pad_probe_stream0_cpu(pad, info):
    """Run Stream 0 CPU analysis on the decoupled NV12 side branch."""
    return _process_stream0_cpu_analysis(
        info,
        update_main_path=False,
        update_analysis_path=True,
    )


def _mark_stream0_stage(stage_name, info):
    """Record Stream 0 liveness for a specific boundary if a buffer is present."""
    gst_buffer = info.get_buffer()
    if bus_handler and gst_buffer:
        bus_handler.update_stream0_stage_sample(stage_name, gst_buffer.pts)
    return Gst.PadProbeReturn.OK


def stream0_stage_probe_decoder_src(pad, info):
    """Track Stream 0 liveness at decoder0.src."""
    return _mark_stream0_stage("decoder_src", info)


def stream0_stage_probe_nvvidconv_src(pad, info):
    """Track Stream 0 liveness at nvvidconv0.src."""
    return _mark_stream0_stage("nvvidconv_src", info)


def stream0_stage_probe_caps_src(pad, info):
    """Track Stream 0 liveness at caps0.src."""
    return _mark_stream0_stage("caps_src", info)


def stream0_stage_probe_premuxq_src(pad, info):
    """Track Stream 0 liveness at premuxq0.src."""
    return _mark_stream0_stage("premuxq_src", info)


def stream0_stage_probe_mux_src(pad, info):
    """Track Stream 0 liveness at mux_0.src."""
    return _mark_stream0_stage("mux_src", info)


def stream0_stage_probe_postmuxq_src(pad, info):
    """Track Stream 0 liveness at postmuxq0.src."""
    return _mark_stream0_stage("postmuxq_src", info)


def stream0_stage_probe_pgie_sink(pad, info):
    """Track Stream 0 liveness at pgie_pouring.sink."""
    return _mark_stream0_stage("pgie_sink", info)


def stream0_stage_probe_pgie_src(pad, info):
    """Track Stream 0 liveness at pgie_pouring.src and sample raw object output."""
    _mark_stream0_stage("pgie_src", info)

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    frame_count = getattr(stream0_stage_probe_pgie_src, "_frames_seen", 0)

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_count += 1
        should_log = frame_count in {10, 25, 50, 100} or frame_count % 250 == 0
        if should_log:
            raw_obj_count = 0
            untracked_count = 0
            class_hist = {}
            class_conf_max = {}
            mouth_confs = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                raw_obj_count += 1
                cid = int(obj_meta.class_id)
                conf = float(obj_meta.confidence)
                class_hist[cid] = class_hist.get(cid, 0) + 1
                class_conf_max[cid] = max(class_conf_max.get(cid, 0.0), conf)
                if cid == 0:
                    mouth_confs.append(conf)
                if int(obj_meta.object_id) == ((1 << 64) - 1):
                    untracked_count += 1

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            classes = "none"
            if class_hist:
                classes = ",".join(
                    f"{cid}:{class_hist[cid]}@{class_conf_max.get(cid, 0.0):.3f}"
                    for cid in sorted(class_hist.keys())
                )
            mouth_confs.sort(reverse=True)
            mouth_top = (
                ",".join(f"{conf:.3f}" for conf in mouth_confs[:3])
                if mouth_confs
                else "none"
            )

            logger.info(
                "[STREAM0-PGIE] frame=%d src=%s raw_objs=%d untracked=%d "
                "classes=%s mouth_count=%d mouth_top=%s",
                frame_count,
                frame_meta.source_id,
                raw_obj_count,
                untracked_count,
                classes,
                len(mouth_confs),
                mouth_top,
            )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    stream0_stage_probe_pgie_src._frames_seen = frame_count
    return Gst.PadProbeReturn.OK


def stream0_stage_probe_tracker_sink(pad, info):
    """Track Stream 0 liveness at tracker_0.sink."""
    return _mark_stream0_stage("tracker_sink", info)


def stream0_stage_probe_tracker_src(pad, info):
    """Track Stream 0 liveness at tracker_0.src and sample tracked mouth output."""
    _mark_stream0_stage("tracker_src", info)

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    frame_count = getattr(stream0_stage_probe_tracker_src, "_frames_seen", 0)

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_count += 1
        should_log = frame_count in {10, 25, 50, 100} or frame_count % 250 == 0
        if should_log:
            tracked_mouth_ids = []
            tracked_mouth_confs = []
            tracked_trolley_ids = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                oid = int(obj_meta.object_id)
                if oid != ((1 << 64) - 1):
                    cid = int(obj_meta.class_id)
                    if cid == 0:
                        tracked_mouth_ids.append(oid)
                        tracked_mouth_confs.append(float(obj_meta.confidence))
                    elif cid == 1:
                        tracked_trolley_ids.append(oid)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            tracked_mouth_confs.sort(reverse=True)
            tracked_mouth_top = (
                ",".join(f"{conf:.3f}" for conf in tracked_mouth_confs[:3])
                if tracked_mouth_confs
                else "none"
            )
            mouth_ids = (
                ",".join(str(oid) for oid in tracked_mouth_ids[:3])
                if tracked_mouth_ids
                else "none"
            )
            trolley_ids = (
                ",".join(str(oid) for oid in tracked_trolley_ids[:3])
                if tracked_trolley_ids
                else "none"
            )

            logger.info(
                "[STREAM0-TRACKER] frame=%d src=%s tracked_mouth=%d mouth_ids=%s "
                "mouth_top=%s tracked_trolley=%d trolley_ids=%s",
                frame_count,
                frame_meta.source_id,
                len(tracked_mouth_ids),
                mouth_ids,
                tracked_mouth_top,
                len(tracked_trolley_ids),
                trolley_ids,
            )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    stream0_stage_probe_tracker_src._frames_seen = frame_count
    return Gst.PadProbeReturn.OK


def _should_extract_live_frame(stream_id: int) -> bool:
    """Throttle MJPEG extraction before expensive GPU-to-CPU frame copies."""
    global _live_stream_warmup_deadline

    target_fps = max(int(getattr(config, "LIVE_STREAM_FPS", 0) or 0), 0)
    if target_fps <= 0:
        return True

    now = time.monotonic()
    if now < _live_stream_warmup_deadline:
        return False

    min_interval = 1.0 / float(target_fps)
    last_extract = _live_stream_last_extract.get(stream_id, 0.0)
    if last_extract and (now - last_extract) < min_interval:
        return False

    _live_stream_last_extract[stream_id] = now
    return True


def post_osd_probe_stream0_for_streaming(pad, info):
    """
    Probe on post-OSD path (after nvosd rendering) for live streaming.
    Extracts frames WITH overlays for MJPEG server.
    This runs AFTER all display_meta has been rendered by nvosd.
    """
    if not mjpeg_server:
        return Gst.PadProbeReturn.OK

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    if not _should_extract_live_frame(0):
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Extract frame and draw CPU overlays (replaces nvosd GPU rendering)
        try:
            import numpy as np
            import cv2
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_rgba = np.array(n_frame, copy=True, order='C')
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            if brightness_processor is not None:
                brightness_processor.draw_cpu_overlay(frame_bgr)
            elif melting_controller is not None:
                melting_controller.draw_cpu_overlay(frame_bgr)
            if pouring_processor is not None:
                pouring_processor.draw_cpu_overlay(frame_bgr)
            mjpeg_server.update_frame(stream_id=0, frame_bgr=frame_bgr)
        except Exception as e:
            logger.error(f"Post-OSD frame extraction error: {e}", exc_info=True)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def nvinfer_src_pad_probe_stream1(pad, info):
    """
    Probe on nvosd_1 sink pad (Stream 1 — Pyrometer Camera).
    Buffer is RGBA at this point (after nvvideoconvert + capsfilter).
    Handles pyrometer rod detection with zone check + temporal filtering.
    Extracts frame for annotated event screenshots.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    with timed_section("probe.stream1.pyrometer", logger=logger):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if bus_handler:
                bus_handler.update_frame_time(1)

            frame_timestamp, frame_datetime = _resolve_frame_timestamp(gst_buffer, frame_meta, 1)

            # Extract CPU frame lazily for event screenshots only when detections exist or
            # an active pyrometer event may soon need a start/end snapshot.
            frame = None
            need_frame = False
            if config.ENABLE_FRAME_PROCESSING and pyrometer_processor:
                try:
                    obj_count = int(getattr(frame_meta, "num_obj_meta", 0) or 0)
                    need_frame = pyrometer_processor.needs_frame(obj_count)
                except Exception:
                    need_frame = False

            if need_frame:
                try:
                    import numpy as np
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    frame = np.array(n_frame, copy=True, order='C')
                except Exception as e:
                    logger.error(f"Stream 1 frame extraction error: {e}", exc_info=True)

            try:
                if pyrometer_processor:
                    try:
                        pyrometer_processor.process_frame(
                            frame_meta,
                            frame=frame,
                            timestamp=frame_timestamp,
                            datetime_obj=frame_datetime,
                        )
                        # Add DS-native overlay for pyrometer zone + status
                        if config.ENABLE_INFERENCE_VIDEO and batch_meta is not None:
                            pyrometer_processor.add_inference_display_meta(
                                batch_meta=batch_meta,
                                frame_meta=frame_meta,
                            )
                    except Exception as e:
                        logger.error(f"Pyrometer processor error: {e}", exc_info=True)
            finally:
                if frame is not None:
                    try:
                        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    except Exception:
                        pass

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


def post_osd_probe_stream1_for_streaming(pad, info):
    """
    Probe on Stream 1 sink (after nvosd rendering) for live streaming.
    Extracts frames WITH nvosd-rendered bounding boxes for MJPEG server.
    """
    if not mjpeg_server:
        return Gst.PadProbeReturn.OK

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    if not _should_extract_live_frame(1):
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Extract annotated frame (post-OSD, WITH bounding boxes)
        try:
            import numpy as np
            import cv2
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_rgba = np.array(n_frame, copy=True, order='C')
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            mjpeg_server.update_frame(stream_id=1, frame_bgr=frame_bgr)
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except Exception as e:
            logger.error(f"Stream 1 post-OSD frame extraction error: {e}", exc_info=True)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def post_osd_probe_stream2_for_streaming(pad, info):
    """
    Probe on Stream 2 post-OSD path for live streaming.
    Extracts frames for MJPEG and draws CPU melting overlays when native display
    meta is disabled.
    """
    if not mjpeg_server:
        return Gst.PadProbeReturn.OK

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    if not _should_extract_live_frame(2):
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        try:
            import numpy as np
            import cv2
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_rgba = np.array(n_frame, copy=True, order='C')
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            if brightness_processor_2 is not None:
                brightness_processor_2.draw_cpu_overlay(frame_bgr)
            elif melting_controller_2 is not None:
                melting_controller_2.draw_cpu_overlay(frame_bgr)
            if pouring_processor_2 is not None:
                pouring_processor_2.draw_cpu_overlay(frame_bgr)
            mjpeg_server.update_frame(stream_id=2, frame_bgr=frame_bgr)
        except Exception as e:
            logger.error(f"Stream 2 post-OSD frame extraction error: {e}", exc_info=True)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def pgie_src_pad_probe_stream2_diag(pad, info):
    """Low-rate diagnostic probe on Stream 2 pgie source to separate infer vs tracker issues."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    frame_count = getattr(pgie_src_pad_probe_stream2_diag, "_frames_seen", 0)

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_count += 1
        if frame_count % 250 == 0:
            raw_obj_count = 0
            untracked_count = 0
            class_hist = {}

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                raw_obj_count += 1
                class_hist[obj_meta.class_id] = class_hist.get(obj_meta.class_id, 0) + 1
                if int(obj_meta.object_id) == ((1 << 64) - 1):
                    untracked_count += 1

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            classes = "none"
            if class_hist:
                classes = ",".join(f"{cid}:{count}" for cid, count in sorted(class_hist.items()))

            logger.info(
                "[STREAM2-PGIE] frame=%d src=%s raw_objs=%d untracked=%d classes=%s",
                frame_count,
                frame_meta.source_id,
                raw_obj_count,
                untracked_count,
                classes,
            )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    pgie_src_pad_probe_stream2_diag._frames_seen = frame_count
    return Gst.PadProbeReturn.OK


def osd_sink_pad_probe_stream2(pad, info):
    """
    Probe on nvosd_2 sink pad.
    Stream 2 now carries Furnace 1 tapping/deslagging plus shared spectro; optional
    pouring remains available behind HICON_ENABLE_STREAM_2_POURING_PROCESSOR.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    with timed_section("probe.stream2.pouring", logger=logger):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        run_python_pouring = pouring_processor_2 is not None

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if bus_handler:
                bus_handler.update_frame_time(2)

            native_melting_state = None
            if melting_meta_reader_2 is not None:
                try:
                    native_melting_state = melting_meta_reader_2.decode_frame_meta(frame_meta)
                except Exception as e:
                    logger.error(f"Stream 2 melting meta reader error: {e}", exc_info=True)

            frame_timestamp, frame_datetime = _resolve_frame_timestamp(gst_buffer, frame_meta, 2)

            run_hybrid_melting = (
                melting_controller_2 is not None and native_melting_state is not None
            )
            run_cpu_melting = brightness_processor_2 is not None
            melting_needs_frame = (
                run_hybrid_melting
                and config.ENABLE_FRAME_PROCESSING
                and melting_controller_2.needs_frame(native_melting_state)
            )
            frame = None
            if config.ENABLE_FRAME_PROCESSING and (run_python_pouring or run_cpu_melting or melting_needs_frame):
                try:
                    obj_count = int(getattr(frame_meta, "num_obj_meta", 0) or 0)
                except Exception:
                    obj_count = 0
                needs_pouring_frame = (
                    run_python_pouring and pouring_processor_2.needs_frame(obj_count)
                )
                if needs_pouring_frame or run_cpu_melting or melting_needs_frame:
                    try:
                        import numpy as np
                        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        frame = np.array(n_frame, copy=True, order='C')
                    except Exception as e:
                        logger.error(f"Stream 2 frame extraction error: {e}", exc_info=True)

            try:
                if run_python_pouring:
                    try:
                        pouring_processor_2.process_frame(
                            frame_meta=frame_meta,
                            frame=frame,
                            batch_meta=batch_meta,
                            timestamp=frame_timestamp,
                            datetime_obj=frame_datetime,
                        )
                    except Exception as e:
                        logger.error(f"Stream 2 pouring processor error: {e}", exc_info=True)
                if run_hybrid_melting:
                    try:
                        melting_controller_2.process_native_state(
                            native_state=native_melting_state,
                            frame_meta=frame_meta,
                            frame=frame if melting_needs_frame else None,
                            batch_meta=batch_meta,
                            timestamp=frame_timestamp,
                            datetime_obj=frame_datetime,
                        )
                    except Exception as e:
                        logger.error(f"Stream 2 hybrid melting controller error: {e}", exc_info=True)
                elif run_cpu_melting and frame is not None:
                    try:
                        brightness_processor_2.process_frame_with_array(frame, frame_meta)
                    except Exception as e:
                        logger.error(f"Stream 2 CPU melting processor error: {e}", exc_info=True)
            finally:
                if frame is not None:
                    try:
                        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    except Exception:
                        pass

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Sync thread
# ---------------------------------------------------------------------------
def sync_thread_func(stop_event):
    """Background thread for periodic cloud sync."""
    logger.info("Sync thread started")
    while not stop_event.is_set():
        try:
            if sync_manager:
                sync_manager.sync_all()
        except Exception as e:
            logger.error(f"Sync thread error: {e}", exc_info=True)
        stop_event.wait(timeout=config.SYNC_INTERVAL)
    logger.info("Sync thread stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global pouring_processor, pouring_processor_2
    global heat_cycle_manager_2
    global brightness_processor, brightness_processor_2, pyrometer_processor
    global melting_controller, melting_meta_reader
    global melting_controller_2, melting_meta_reader_2
    global bus_handler, sync_manager, recording_manager, stream0_local_relay_manager, mjpeg_server
    global async_db_writer, screenshot_writer

    logger.info("=" * 60)
    logger.info("HiCon Pipeline Starting")
    logger.info("=" * 60)

    # Initialize GStreamer
    Gst.init(None)

    cpp_melting_plugin_path = str(
        Path(__file__).parent / 'custom_plugins' / 'hicon_melting' / 'libgsthiconmelting.so'
    )
    requested_cuda_brightness = bool(config.USE_CUDA_BRIGHTNESS)
    safe_cuda_topology_ready = (
        config.STREAM_0_DECOUPLED_ANALYSIS_MODE
        and config.STREAM_0_ANALYSIS_BRANCH_ENABLED
        and config.STREAM_0_ANALYSIS_PROBE_ENABLED
    )
    use_safe_cuda_brightness = False
    if requested_cuda_brightness:
        if not safe_cuda_topology_ready:
            logger.warning(
                "CUDA brightness requested, but the safe NV12 analysis topology is unavailable; "
                "falling back to CPU brightness"
            )
        else:
            try:
                Gst.Plugin.load_file(cpp_melting_plugin_path)
                use_safe_cuda_brightness = True
                logger.info("C++ melting plugin loaded: %s", cpp_melting_plugin_path)
            except Exception as e:
                logger.warning(
                    "Safe CUDA brightness plugin not available (%s), falling back to CPU brightness",
                    e,
                )

    # Create output directories
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    db = HiConDatabase(str(config.DB_PATH))
    async_db_writer = AsyncDBWriter(db, maxsize=64)
    screenshot_writer = AsyncScreenshotWriter(maxsize=20)

    # Load zone configuration
    zones_path = str(config.CONFIG_DIR / 'zones.json')
    zones_config = load_zones_config(zones_path)
    logger.info(f"Loaded zones config from {zones_path}")
    stream0_zones_config = _stream_zone_config(zones_config, 0)
    stream2_zones_config = _stream_zone_config(zones_config, 2)
    stream1_detector_config = _scale_detector_config(
        {"metadata": zones_config.get("metadata", {}),
         "pyrometer": zones_config.get("pyrometer", {})},
        int(getattr(config, "STREAM_1_MUX_WIDTH", 1280)),
        int(getattr(config, "STREAM_1_MUX_HEIGHT", 720)),
    )
    melting_plugin_config_ini = _serialize_melting_plugin_config(
        stream0_zones_config,
        target_width=int(getattr(config, "STREAM_0_MUX_WIDTH", 1280)),
        target_height=int(getattr(config, "STREAM_0_MUX_HEIGHT", 720)),
    )
    melting_plugin_config_ini_2 = _serialize_melting_plugin_config(
        stream2_zones_config,
        target_width=int(getattr(config, "STREAM_2_MUX_WIDTH", 1280)),
        target_height=int(getattr(config, "STREAM_2_MUX_HEIGHT", 720)),
    )

    # Create shared HeatCycleManager for Stream 0 (owned by pipeline, shared by processors)
    heat_cycle_manager = HeatCycleManager(
        db_manager=db,
        ladle_absence_timeout=config.POURING_CYCLE_TIMEOUT_S,
        tapping_only_timeout=config.TAPPING_ONLY_CYCLE_TIMEOUT_S,
        base_location=config.LOCATION,
    )

    # Stream 2 melting events aggregate into the same active heat cycle as Stream 0 pouring.
    heat_cycle_manager_2 = heat_cycle_manager

    # Initialize processors
    # Enable display_meta when recording is active so overlay state is computed on the
    # analysis branch and then re-written onto the main path by the recording display_meta
    # probe.  In recording-off + decoupled mode there is no benefit to generating overlay
    # data that would be discarded at analysis_sink0, so keep it disabled.
    stream0_enable_display_meta = (
        config.ENABLE_INFERENCE_VIDEO or not config.STREAM_0_DECOUPLED_ANALYSIS_MODE
    )
    if config.STREAM_0_DECOUPLED_ANALYSIS_MODE:
        if config.ENABLE_INFERENCE_VIDEO:
            logger.info(
                "Stream 0: Display meta enabled on analysis branch for recording overlay "
                "(decoupled mode + ENABLE_INFERENCE_VIDEO)"
            )
        else:
            logger.info(
                "Stream 0: CPU-generated display meta disabled (decoupled mode, recording off)"
            )

    use_cuda_brightness = use_safe_cuda_brightness

    if config.ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR:
        if use_cuda_brightness:
            melting_meta_reader = MeltingMetaReader(stream_label="stream0", heartbeat_every=25)
            melting_controller = MeltingAnalysisController(
                zones_config=stream0_zones_config,
                db_manager=async_db_writer,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=stream0_enable_display_meta,
                screenshot_writer=screenshot_writer,
            )
            brightness_processor = melting_controller
            logger.info(
                "Stream 0: Safe CUDA tapping path initialized "
                "(native tapping/deslagging/spectro)"
            )
        else:
            brightness_processor = BrightnessProcessor(
                zones_config=stream0_zones_config,
                db_manager=async_db_writer,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=stream0_enable_display_meta,
                screenshot_writer=screenshot_writer,
            )
            logger.info("Stream 0: CPU brightness processor initialized (NumPy)")
            melting_meta_reader = None
            melting_controller = None
    else:
        brightness_processor = None
        melting_meta_reader = None
        melting_controller = None
        logger.warning("Stream 0: Brightness processor disabled for diagnostics")

    pyrometer_processor = PyrometerProcessor(
        zone_config=stream1_detector_config.get('pyrometer', {}),
        db_manager=async_db_writer,
        config=config,
        screenshot_dir=str(config.SCREENSHOT_DIR),
        heat_cycle_manager=heat_cycle_manager,
        screenshot_writer=screenshot_writer,
    )

    # Stream 0 pouring processor
    if config.ENABLE_STREAM_0_POURING_PROCESSOR:
        try:
            from processors.pouring_processor import PouringProcessor
            pouring_processor = PouringProcessor(
                db_manager=async_db_writer,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=stream0_enable_display_meta,
                screenshot_writer=screenshot_writer,
            )
            logger.info("Stream 0: Pouring processor initialized")
        except Exception as e:
            logger.warning(f"Stream 0: Pouring processor not available: {e}")
            pouring_processor = None
    else:
        pouring_processor = None
        logger.warning("Stream 0: Pouring processor disabled for diagnostics")

    if config.ENABLE_BRIGHTNESS_STREAM_2:
        if use_cuda_brightness:
            melting_meta_reader_2 = MeltingMetaReader(stream_label="stream2", heartbeat_every=25)
            melting_controller_2 = MeltingAnalysisController(
                zones_config=stream2_zones_config,
                db_manager=async_db_writer,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=False,
                screenshot_writer=screenshot_writer,
                camera_id_override=config.CAMERA_ID_STREAM_2,
            )
            brightness_processor_2 = None
            logger.info("Stream 2: Native CUDA melting initialized (tapping/deslagging/spectro)")
        else:
            melting_meta_reader_2 = None
            melting_controller_2 = None
            brightness_processor_2 = BrightnessProcessor(
                zones_config=stream2_zones_config,
                db_manager=async_db_writer,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=False,
                screenshot_writer=screenshot_writer,
                camera_id_override=config.CAMERA_ID_STREAM_2,
            )
            logger.info("Stream 2: CPU melting fallback initialized")
    else:
        melting_meta_reader_2 = None
        melting_controller_2 = None
        brightness_processor_2 = None
        logger.warning("Stream 2: Melting processor disabled")

    # Optional Stream 2 pouring processor (disabled by default in the 3-stream routing).
    if config.ENABLE_STREAM_2_POURING_PROCESSOR:
        try:
            from processors.pouring_processor import PouringProcessor
            pouring_processor_2 = PouringProcessor(
                db_manager=async_db_writer,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                camera_id_override=config.CAMERA_ID_STREAM_2,
                screenshot_writer=screenshot_writer,
            )
            logger.info("Stream 2: Pouring processor initialized")
        except Exception as e:
            logger.warning(f"Stream 2: Pouring processor not available: {e}")
            pouring_processor_2 = None
    else:
        pouring_processor_2 = None
        logger.info("Stream 2: Pouring processor disabled by routing")

    # Initialize sync manager
    if config.ENABLE_SYNC and config.HMAC_SECRET:
        api_client = APIClient(
            base_url=config.API_URL,
            secret=config.HMAC_SECRET,
            customer_id=config.CUSTOMER_ID,
        )
        sync_manager = SyncManager(
            database=db,
            api_client=api_client,
            customer_id=config.CUSTOMER_ID,
            camera_id=config.CAMERA_ID_STREAM_0,
            location=config.LOCATION,
            furnace_id=getattr(config, 'FURNACE_ID', ''),
            sync_interval=config.SYNC_INTERVAL,
            batch_size=config.BATCH_SIZE,
        )
    else:
        logger.info("Cloud sync disabled (ENABLE_SYNC=false or no HMAC_SECRET)")

    # Build pipeline configuration
    pipeline_config = {
        'rtsp_stream_0': config.RTSP_STREAM_0,
        'rtsp_stream_1': config.RTSP_STREAM_1,
        'rtsp_stream_2': config.RTSP_STREAM_2,
        'rtsp_codec_0': config.RTSP_CODEC_0,
        'rtsp_codec_1': config.RTSP_CODEC_1,
        'rtsp_codec_2': config.RTSP_CODEC_2,
        'config_pouring': config.CONFIG_POURING,
        'config_pyrometer': config.CONFIG_PYROMETER,
        'config_pouring_2': config.CONFIG_POURING_2,
        'tracker_lib': config.TRACKER_LIB,
        'tracker_config': config.TRACKER_CONFIG,
        'stream_0_tracker_config': config.STREAM_0_TRACKER_CONFIG,
        'rtsp_protocol_0': config.RTSP_PROTOCOL_0,
        'rtsp_protocol_1': config.RTSP_PROTOCOL_1,
        'rtsp_protocol_2': config.RTSP_PROTOCOL_2,
        'rtsp_udp_timeout_us': config.RTSP_UDP_TIMEOUT_US,
        'rtsp_tcp_timeout_us': config.RTSP_TCP_TIMEOUT_US,
        'rtsp_port_retry': config.RTSP_PORT_RETRY,
        'rtsp_do_retransmission': config.RTSP_DO_RETRANSMISSION,
        'rtsp_latency_ms': config.RTSP_LATENCY_MS,
        'stream_0_bypass_tracker': config.STREAM_0_BYPASS_TRACKER,
        'stream_0_bypass_pgie': config.STREAM_0_BYPASS_PGIE,
        'stream_0_decode_only_mode': config.STREAM_0_DECODE_ONLY_MODE,
        'stream_0_postmux_only_mode': config.STREAM_0_POSTMUX_ONLY_MODE,
        'stream_0_postconv_only_mode': config.STREAM_0_POSTCONV_ONLY_MODE,
        'stream_0_preosd_only_mode': config.STREAM_0_PREOSD_ONLY_MODE,
        'stream_0_decoupled_analysis_mode': config.STREAM_0_DECOUPLED_ANALYSIS_MODE,
        'stream_0_analysis_branch_enabled': config.STREAM_0_ANALYSIS_BRANCH_ENABLED,
        'stream_0_analysis_rgba_enabled': config.STREAM_0_ANALYSIS_RGBA_ENABLED,
        'stream_0_analysis_probe_enabled': config.STREAM_0_ANALYSIS_PROBE_ENABLED,
        'stream_0_mux_width': config.STREAM_0_MUX_WIDTH,
        'stream_0_mux_height': config.STREAM_0_MUX_HEIGHT,
        'stream_1_mux_width': config.STREAM_1_MUX_WIDTH,
        'stream_1_mux_height': config.STREAM_1_MUX_HEIGHT,
        'stream_2_mux_width': config.STREAM_2_MUX_WIDTH,
        'stream_2_mux_height': config.STREAM_2_MUX_HEIGHT,
        'stream_0_tracker_width': config.STREAM_0_TRACKER_WIDTH,
        'stream_0_tracker_height': config.STREAM_0_TRACKER_HEIGHT,
        'enable_stream_2_pouring_processor': config.ENABLE_STREAM_2_POURING_PROCESSOR,
        'enable_brightness_stream_2': config.ENABLE_BRIGHTNESS_STREAM_2,
        'enable_inference_video': config.ENABLE_INFERENCE_VIDEO,
        'enable_inference_video_stream_0': config.ENABLE_INFERENCE_VIDEO_STREAM_0,
        'enable_inference_video_stream_1': config.ENABLE_INFERENCE_VIDEO_STREAM_1,
        'enable_inference_video_stream_2': config.ENABLE_INFERENCE_VIDEO_STREAM_2,
        'enable_live_stream_0': bool(config.ENABLE_LIVE_STREAM and config.ENABLE_LIVE_STREAM_0),
        'enable_live_stream_1': bool(config.ENABLE_LIVE_STREAM and config.ENABLE_LIVE_STREAM_1),
        'enable_live_stream_2': bool(config.ENABLE_LIVE_STREAM and config.ENABLE_LIVE_STREAM_2),
        'live_stream_timestamp_overlay': config.LIVE_STREAM_TIMESTAMP_OVERLAY,
        'enable_stream0_local_relay': config.ENABLE_STREAM0_LOCAL_RELAY,
        'use_nvurisrcbin_0': config.USE_NVURISRCBIN_0,
        'use_nvurisrcbin_1': config.USE_NVURISRCBIN_1,
        'use_nvurisrcbin_2': config.USE_NVURISRCBIN_2,
        'use_segment_buffer_0': config.USE_SEGMENT_BUFFER_0,
        'segment_buffer_dir_0': config.SEGMENT_BUFFER_DIR_0,
        'segment_buffer_segment_sec_0': config.SEGMENT_BUFFER_SEGMENT_SEC_0,
        'segment_buffer_delay_sec_0': config.SEGMENT_BUFFER_DELAY_SEC_0,
        'segment_buffer_retention_sec_0': config.SEGMENT_BUFFER_RETENTION_SEC_0,
        'use_segment_buffer_2': config.USE_SEGMENT_BUFFER_2,
        'segment_buffer_dir_2': config.SEGMENT_BUFFER_DIR_2,
        'segment_buffer_segment_sec_2': config.SEGMENT_BUFFER_SEGMENT_SEC_2,
        'segment_buffer_delay_sec_2': config.SEGMENT_BUFFER_DELAY_SEC_2,
        'segment_buffer_retention_sec_2': config.SEGMENT_BUFFER_RETENTION_SEC_2,
        'use_ffmpeg_src_0': config.USE_FFMPEG_SRC_0,
        'use_ffmpeg_src_2': config.USE_FFMPEG_SRC_2,
        'use_udp_loopback_0': config.USE_UDP_LOOPBACK_0,
        'use_udp_loopback_2': config.USE_UDP_LOOPBACK_2,
        'udp_loopback_port_0': config.UDP_LOOPBACK_PORT_0,
        'udp_loopback_port_2': config.UDP_LOOPBACK_PORT_2,
        'use_safe_cuda_brightness': use_safe_cuda_brightness,
        'stream_0_melting_config_ini': melting_plugin_config_ini,
        'stream_2_melting_config_ini': melting_plugin_config_ini_2,
    }

    # Build pipeline (keep builder reference for ffmpeg cleanup on shutdown)
    builder = DeepStreamPipelineBuilder(pipeline_config)
    pipeline, elements = builder.create_pipeline()

    if not pipeline:
        logger.error("Failed to create pipeline")
        sys.exit(1)

    # Create main loop
    loop = GLib.MainLoop()

    # Attach bus handler
    stream_policies = {
        0: config.STREAM_0_ZERO_FPS_POLICY,
        1: config.STREAM_1_ZERO_FPS_POLICY,
        2: config.STREAM_2_ZERO_FPS_POLICY if not config.USE_SEGMENT_BUFFER_2 else 'warn',
    }
    stream0_segment_buffer_state_path = ""
    stream0_startup_grace_sec = 30
    if config.USE_SEGMENT_BUFFER_0:
        stream0_segment_buffer_state_path = str(Path(config.SEGMENT_BUFFER_DIR_0) / "state.json")
        stream0_startup_grace_sec = max(60, int(config.SEGMENT_BUFFER_DELAY_SEC_0) + 30)
    stream_startup_grace_overrides = {1: 150}  # Hikvision: tolerate full network outages
    stream_segment_buffer_state_paths = {}
    if config.USE_SEGMENT_BUFFER_2:
        stream_startup_grace_overrides[2] = max(60, int(config.SEGMENT_BUFFER_DELAY_SEC_2) + 30)
        stream_segment_buffer_state_paths[2] = str(
            Path(config.SEGMENT_BUFFER_DIR_2) / "state.json"
        )
    # nvurisrcbin handles reconnection internally — raise the warn safety cap
    # from 90s to 300s so the watchdog doesn't kill the pipeline during NVR session resets
    any_nvurisrcbin = config.USE_NVURISRCBIN_0 or config.USE_NVURISRCBIN_1 or config.USE_NVURISRCBIN_2
    warn_cap = 300 if any_nvurisrcbin else 90

    bus_handler = BusHandler(
        pipeline,
        loop,
        healthcheck_url=config.HEALTHCHECK_URL,
        stream0_decoupled_analysis_mode=config.STREAM_0_DECOUPLED_ANALYSIS_MODE,
        stream_policies=stream_policies,
        stream0_segment_buffer_mode=config.USE_SEGMENT_BUFFER_0,
        stream0_segment_buffer_state_path=stream0_segment_buffer_state_path,
        stream0_startup_grace_sec=stream0_startup_grace_sec,
        stream_startup_grace_overrides=stream_startup_grace_overrides,
        stream_segment_buffer_state_paths=stream_segment_buffer_state_paths,
        warn_safety_cap_sec=warn_cap,
        rtsp_restart_stale_sec=config.RTSP_RESTART_STALE_SEC,
        rtsp_restart_cooldown_sec=config.RTSP_RESTART_COOLDOWN_SEC,
        rtsp_restart_backoff_sec=config.RTSP_RESTART_BACKOFF_SEC,
        stream_restart_cb=builder.schedule_stream_restart,
        restartable_stream_ids=builder.get_restartable_stream_ids(),
    )

    # Initialize MJPEG live streaming server (if enabled)
    mjpeg_server = None
    if config.ENABLE_LIVE_STREAM:
        mjpeg_server = MJPEGServer(
            host=config.LIVE_STREAM_HOST,
            port=config.LIVE_STREAM_PORT,
            jpeg_quality=config.LIVE_STREAM_QUALITY,
            max_fps=config.LIVE_STREAM_FPS,
            timestamp_overlay=config.LIVE_STREAM_TIMESTAMP_OVERLAY,
        )
        # Only register streams that have pipeline elements and are enabled for live preview.
        live_stream_keys = [
            (0, 'nvosd_0', config.ENABLE_LIVE_STREAM_0),
            (1, 'nvosd_1', config.ENABLE_LIVE_STREAM_1),
            (2, 'nvosd_2', config.ENABLE_LIVE_STREAM_2),
        ]
        for _sid, _key, _enabled in live_stream_keys:
            if _enabled and _key in elements and elements[_key]:
                mjpeg_server.register_stream(_sid)
        mjpeg_server.start()
        logger.info(f"✓ Live streaming enabled: http://{config.LIVE_STREAM_HOST}:{config.LIVE_STREAM_PORT}/")
    else:
        logger.info("Live streaming disabled (ENABLE_LIVE_STREAM=false)")

    # Optional DS-native inference recording branch (post-OSD annotations)
    recording_manager = None
    if config.ENABLE_INFERENCE_VIDEO_STREAM_0:
        tee_0 = elements.get('tee_0')
        if tee_0:
            recording_manager = RecordingManager(
                output_dir=str(config.VIDEO_DIR / 'inference'),
                stream_id=0,
                target_fps=config.INFERENCE_VIDEO_FPS,
                target_width=config.INFERENCE_VIDEO_WIDTH,
                target_height=config.INFERENCE_VIDEO_HEIGHT,
                schedule=config.INFERENCE_VIDEO_SCHEDULE,
                max_duration_s=config.INFERENCE_VIDEO_MAX_DURATION_S,
                retention_days=config.INFERENCE_VIDEO_RETENTION_DAYS,
            )
            if recording_manager.setup_recording_branch(pipeline, tee_0):
                logger.info("Stream 0: DS-native inference recording branch configured")
            else:
                logger.error("Stream 0: failed to configure inference recording branch")
                recording_manager = None
        else:
            logger.warning("Stream 0 inference recording enabled but tee_0 is missing; recording disabled")
    elif config.ENABLE_INFERENCE_VIDEO:
        logger.info("Stream 0 inference recording disabled by HICON_ENABLE_INFERENCE_VIDEO_STREAM_0=false")

    recording_manager_1 = None
    if config.ENABLE_INFERENCE_VIDEO_STREAM_1:
        tee_1 = elements.get('tee_1')
        if tee_1:
            recording_manager_1 = RecordingManager(
                output_dir=str(config.VIDEO_DIR / 'inference'),
                stream_id=1,
                target_fps=config.INFERENCE_VIDEO_FPS,
                target_width=config.INFERENCE_VIDEO_WIDTH,
                target_height=config.INFERENCE_VIDEO_HEIGHT,
                schedule=config.INFERENCE_VIDEO_SCHEDULE,
                max_duration_s=config.INFERENCE_VIDEO_MAX_DURATION_S,
                retention_days=config.INFERENCE_VIDEO_RETENTION_DAYS,
            )
            if recording_manager_1.setup_recording_branch(pipeline, tee_1):
                logger.info("Stream 1: DS-native inference recording branch configured")
            else:
                logger.error("Stream 1: failed to configure inference recording branch")
                recording_manager_1 = None
        else:
            logger.warning("Stream 1 inference recording enabled but tee_1 missing; stream 1 recording disabled")
    elif config.ENABLE_INFERENCE_VIDEO:
        logger.info("Stream 1 inference recording disabled by HICON_ENABLE_INFERENCE_VIDEO_STREAM_1=false")

    recording_manager_2 = None
    if config.ENABLE_INFERENCE_VIDEO_STREAM_2:
        tee_2 = elements.get('tee_2')
        if tee_2:
            recording_manager_2 = RecordingManager(
                output_dir=str(config.VIDEO_DIR / 'inference'),
                stream_id=2,
                target_fps=config.INFERENCE_VIDEO_FPS,
                target_width=config.INFERENCE_VIDEO_WIDTH,
                target_height=config.INFERENCE_VIDEO_HEIGHT,
                schedule=config.INFERENCE_VIDEO_SCHEDULE,
                max_duration_s=config.INFERENCE_VIDEO_MAX_DURATION_S,
                retention_days=config.INFERENCE_VIDEO_RETENTION_DAYS,
            )
            if recording_manager_2.setup_recording_branch(pipeline, tee_2):
                logger.info("Stream 2: DS-native inference recording branch configured")
            else:
                logger.error("Stream 2: failed to configure inference recording branch")
                recording_manager_2 = None
        else:
            logger.warning("Stream 2 inference recording enabled but tee_2 missing; stream 2 recording disabled")
    elif config.ENABLE_INFERENCE_VIDEO:
        logger.info("Stream 2 inference recording disabled by HICON_ENABLE_INFERENCE_VIDEO_STREAM_2=false")

    stream0_local_relay_manager = None
    if config.ENABLE_STREAM0_LOCAL_RELAY:
        tee_0 = elements.get('tee_0')
        if tee_0:
            stream0_local_relay_manager = Stream0LocalRelayManager(
                stream_id=0,
                target_fps=config.INFERENCE_VIDEO_FPS,
                target_width=config.INFERENCE_VIDEO_WIDTH,
                target_height=config.INFERENCE_VIDEO_HEIGHT,
            )
            if stream0_local_relay_manager.setup_relay_branch(pipeline, tee_0):
                logger.info(
                    "Stream 0: local MediaMTX relay branch configured at %s",
                    stream0_local_relay_manager.publish_uri,
                )
            else:
                logger.error("Stream 0: failed to configure local MediaMTX relay branch")
                stream0_local_relay_manager = None
        else:
            logger.warning("Stream 0 local relay enabled but tee_0 is missing; relay disabled")

    # Attach pad probes
    # Stream 0: OSD sink pad probe (pouring + brightness)
    if 'nvosd_0' in elements and elements['nvosd_0']:
        osd_sinkpad = elements['nvosd_0'].get_static_pad("sink")
        if osd_sinkpad:
            if config.STREAM_0_DECOUPLED_ANALYSIS_MODE:
                if config.ENABLE_STREAM_0_PROBE and not config.STREAM_0_ANALYSIS_PROBE_ENABLED:
                    osd_sinkpad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        osd_sink_pad_probe_stream0,
                    )
                    logger.info(
                        "Stream 0: Main-path CPU analysis fallback attached "
                        "(decoupled mode, analysis side probe disabled)"
                    )
                else:
                    if config.ENABLE_INFERENCE_VIDEO:
                        osd_sinkpad.add_probe(
                            Gst.PadProbeType.BUFFER,
                            osd_sink_pad_probe_stream0_display_meta,
                        )
                        logger.info(
                            "Stream 0: Display meta writer probe attached "
                            "(decoupled analysis mode + recording active)"
                        )
                    else:
                        osd_sinkpad.add_probe(
                            Gst.PadProbeType.BUFFER,
                            osd_sink_pad_probe_stream0_heartbeat_main,
                        )
                        logger.info("Stream 0: Main-path heartbeat probe attached (decoupled analysis mode)")
            elif config.ENABLE_STREAM_0_PROBE:
                osd_sinkpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    osd_sink_pad_probe_stream0,
                )
                logger.info("Stream 0: OSD sink pad probe attached (pouring + brightness + spectro)")
            else:
                osd_sinkpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    osd_sink_pad_probe_stream0_heartbeat,
                )
                logger.warning(
                    "Stream 0: OSD sink pad probe disabled for diagnostics "
                    "(heartbeat-only probe attached)"
                )
    if (
        config.STREAM_0_DECOUPLED_ANALYSIS_MODE
        and config.STREAM_0_ANALYSIS_BRANCH_ENABLED
        and config.STREAM_0_ANALYSIS_PROBE_ENABLED
    ):
        # Frames on the analysis branch are NV12 (no nvvideoconvert in decoupled mode).
        # Probe attaches to the analysis branch terminal: after C++ plugin if present, else analysisq0.
        if 'hicon_melting_0' in elements and elements['hicon_melting_0']:
            analysis_probe_pad = elements['hicon_melting_0'].get_static_pad("src")
        elif 'hicon_pouring_0' in elements and elements['hicon_pouring_0']:
            analysis_probe_pad = elements['hicon_pouring_0'].get_static_pad("src")
        elif 'analysisq0' in elements and elements['analysisq0']:
            analysis_probe_pad = elements['analysisq0'].get_static_pad("src")
        else:
            analysis_probe_pad = None
        if analysis_probe_pad:
            if config.ENABLE_STREAM_0_PROBE:
                analysis_probe_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    analysis_pad_probe_stream0_cpu,
                )
                logger.info(
                    "Stream 0: Analysis branch probe attached "
                    "(pouring + brightness + spectro on NV12 analysis branch)"
                )
            else:
                logger.warning(
                    "Stream 0: Analysis branch probe disabled in decoupled mode "
                    "(main-path heartbeat remains attached)"
                )
    elif config.STREAM_0_DECOUPLED_ANALYSIS_MODE and not config.STREAM_0_ANALYSIS_BRANCH_ENABLED:
        logger.info(
            "Stream 0: Analysis branch disabled for isolation; skipping analysis probe "
            "and C++ pouring meta reader on Stream 0"
        )
    elif config.STREAM_0_DECOUPLED_ANALYSIS_MODE and config.STREAM_0_ANALYSIS_BRANCH_ENABLED:
        if config.ENABLE_STREAM_0_PROBE and not config.STREAM_0_ANALYSIS_PROBE_ENABLED:
            logger.info(
                "Stream 0: Analysis side probe disabled; CPU analysis is running on the "
                "main path while the side branch remains shell-only"
            )
        else:
            logger.info(
                "Stream 0: Analysis branch probe disabled for staged isolation "
                "(main-path heartbeat remains attached)"
            )
    elif 'decode_sink_0' in elements and elements['decode_sink_0']:
        decode_sinkpad = elements['decode_sink_0'].get_static_pad("sink")
        if decode_sinkpad:
            decode_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                osd_sink_pad_probe_stream0_heartbeat,
            )
            logger.info("Stream 0: decode-only heartbeat probe attached")
    elif 'postmux_sink_0' in elements and elements['postmux_sink_0']:
        postmux_sinkpad = elements['postmux_sink_0'].get_static_pad("sink")
        if postmux_sinkpad:
            postmux_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                osd_sink_pad_probe_stream0_heartbeat,
            )
            logger.info("Stream 0: post-mux heartbeat probe attached")
    elif 'preosd_sink_0' in elements and elements['preosd_sink_0']:
        preosd_sinkpad = elements['preosd_sink_0'].get_static_pad("sink")
        if preosd_sinkpad:
            preosd_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                osd_sink_pad_probe_stream0_heartbeat,
            )
            logger.info("Stream 0: pre-OSD heartbeat probe attached")
    elif 'postconv_sink_0' in elements and elements['postconv_sink_0']:
        postconv_sinkpad = elements['postconv_sink_0'].get_static_pad("sink")
        if postconv_sinkpad:
            postconv_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                osd_sink_pad_probe_stream0_heartbeat,
            )
            logger.info("Stream 0: post-convert heartbeat probe attached")

    # Stream 0: upstream and shared-path boundary probes for failure localization
    if config.ENABLE_DEBUG_PROBES:
        if 'decoder0' in elements and elements['decoder0']:
            decoder0_srcpad = elements['decoder0'].get_static_pad("src")
            if decoder0_srcpad:
                decoder0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_decoder_src,
                )
                logger.info("Stream 0: stage probe attached at decoder0.src")
        if 'nvvidconv0' in elements and elements['nvvidconv0']:
            nvvidconv0_srcpad = elements['nvvidconv0'].get_static_pad("src")
            if nvvidconv0_srcpad:
                nvvidconv0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_nvvidconv_src,
                )
                logger.info("Stream 0: stage probe attached at nvvidconv0.src")
        if 'caps0' in elements and elements['caps0']:
            caps0_srcpad = elements['caps0'].get_static_pad("src")
            if caps0_srcpad:
                caps0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_caps_src,
                )
                logger.info("Stream 0: stage probe attached at caps0.src")
        if 'premuxq0' in elements and elements['premuxq0']:
            premuxq0_srcpad = elements['premuxq0'].get_static_pad("src")
            if premuxq0_srcpad:
                premuxq0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_premuxq_src,
                )
                logger.info("Stream 0: stage probe attached at premuxq0.src")
        if 'mux_0' in elements and elements['mux_0']:
            mux0_srcpad = elements['mux_0'].get_static_pad("src")
            if mux0_srcpad:
                mux0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_mux_src,
                )
                logger.info("Stream 0: stage probe attached at mux_0.src")
        if 'postmuxq0' in elements and elements['postmuxq0']:
            postmuxq0_srcpad = elements['postmuxq0'].get_static_pad("src")
            if postmuxq0_srcpad:
                postmuxq0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_postmuxq_src,
                )
                logger.info("Stream 0: stage probe attached at postmuxq0.src")
        if 'pgie_pouring' in elements and elements['pgie_pouring']:
            pgie0_sinkpad = elements['pgie_pouring'].get_static_pad("sink")
            if pgie0_sinkpad:
                pgie0_sinkpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_pgie_sink,
                )
                logger.info("Stream 0: stage probe attached at pgie_pouring.sink")
            pgie0_srcpad = elements['pgie_pouring'].get_static_pad("src")
            if pgie0_srcpad:
                pgie0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_pgie_src,
                )
                logger.info("Stream 0: stage probe attached at pgie_pouring.src")
        if 'tracker_0' in elements and elements['tracker_0']:
            tracker0_sinkpad = elements['tracker_0'].get_static_pad("sink")
            if tracker0_sinkpad:
                tracker0_sinkpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_tracker_sink,
                )
                logger.info("Stream 0: stage probe attached at tracker_0.sink")
            tracker0_srcpad = elements['tracker_0'].get_static_pad("src")
            if tracker0_srcpad:
                tracker0_srcpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    stream0_stage_probe_tracker_src,
                )
                logger.info("Stream 0: stage probe attached at tracker_0.src")

    # Stream 0: Post-OSD probe for live streaming (extracts frames WITH overlays)
    if mjpeg_server and 'queue_display_0' in elements and elements['queue_display_0']:
        queue_display_sinkpad = elements['queue_display_0'].get_static_pad("sink")
        if queue_display_sinkpad:
            queue_display_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                post_osd_probe_stream0_for_streaming,
            )
            logger.info("Stream 0: Post-OSD probe attached for live streaming (WITH overlays)")

    # Stream 1: OSD sink pad probe (pyrometer) — must be after RGBA conversion
    if 'nvosd_1' in elements and elements['nvosd_1']:
        osd1_sinkpad = elements['nvosd_1'].get_static_pad("sink")
        if osd1_sinkpad:
            osd1_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                nvinfer_src_pad_probe_stream1,
            )
            logger.info("Stream 1: OSD sink pad probe attached (pyrometer)")

    # Stream 1: Post-OSD probe for live streaming (extracts frames WITH overlays)
    if mjpeg_server and config.ENABLE_LIVE_STREAM_1 and 'sink_1' in elements and elements['sink_1']:
        sink1_sinkpad = elements['sink_1'].get_static_pad("sink")
        if sink1_sinkpad:
            sink1_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                post_osd_probe_stream1_for_streaming,
            )
            logger.info("Stream 1: Post-OSD probe attached for live streaming (WITH bboxes)")
    elif mjpeg_server and not config.ENABLE_LIVE_STREAM_1:
        logger.info("Stream 1: Live streaming disabled (ENABLE_LIVE_STREAM_1=false)")

    # Stream 2: Post-OSD probe for live streaming (melting overlays + optional pouring)
    if mjpeg_server and config.ENABLE_LIVE_STREAM_2 and 'sink_2' in elements and elements['sink_2']:
        sink2_sinkpad = elements['sink_2'].get_static_pad("sink")
        if sink2_sinkpad:
            sink2_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                post_osd_probe_stream2_for_streaming,
            )
            logger.info("Stream 2: Post-OSD probe attached for live streaming (WITH overlays)")
    elif mjpeg_server and not config.ENABLE_LIVE_STREAM_2:
        logger.info("Stream 2: Live streaming disabled (ENABLE_LIVE_STREAM_2=false)")

    # Stream 0: PGIE src diagnostic probe (counts raw inference objects before tracker/plugin)
    if 'pgie_pouring' in elements and elements['pgie_pouring'] and not config.ENABLE_DEBUG_PROBES:
        pgie0_srcpad = elements['pgie_pouring'].get_static_pad("src")
        if pgie0_srcpad:
            pgie0_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                stream0_stage_probe_pgie_src,
            )
            logger.info("Stream 0: PGIE src diagnostic probe attached")
    if 'tracker_0' in elements and elements['tracker_0'] and not config.ENABLE_DEBUG_PROBES:
        tracker0_srcpad = elements['tracker_0'].get_static_pad("src")
        if tracker0_srcpad:
            tracker0_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                stream0_stage_probe_tracker_src,
            )
            logger.info("Stream 0: tracker src diagnostic probe attached")

    # Stream 2: PGIE src diagnostic probe (counts raw inference objects before tracker)
    if 'pgie_pouring_2' in elements and elements['pgie_pouring_2']:
        pgie2_srcpad = elements['pgie_pouring_2'].get_static_pad("src")
        if pgie2_srcpad:
            pgie2_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                pgie_src_pad_probe_stream2_diag,
            )
            logger.info("Stream 2: PGIE src diagnostic probe attached")

    # Stream 2: OSD sink pad probe (Furnace 1 melting + optional pouring)
    if 'nvosd_2' in elements and elements['nvosd_2']:
        osd2_sinkpad = elements['nvosd_2'].get_static_pad("sink")
        if osd2_sinkpad:
            osd2_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                osd_sink_pad_probe_stream2,
            )
            logger.info("Stream 2: OSD sink pad probe attached (melting + optional pouring)")

    # Start sync thread
    sync_stop_event = threading.Event()
    sync_thread = None
    if sync_manager:
        sync_thread = threading.Thread(
            target=sync_thread_func,
            args=(sync_stop_event,),
            daemon=True,
        )
        sync_thread.start()

    # Signal handler for clean shutdown
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, shutting down...")
        loop.quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start pipeline
    logger.info("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("Failed to set pipeline to PLAYING")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)

    logger.info("Pipeline PLAYING — waiting for streams...")
    global _live_stream_warmup_deadline
    _live_stream_warmup_deadline = time.monotonic() + _RECORDING_STARTUP_WARMUP_SEC
    _log_memory_snapshot("startup")

    def _start_recording_branches_after_warmup():
        if recording_manager:
            recording_manager.start_recording(event_prefix="inference_stream0")
        if recording_manager_1:
            recording_manager_1.start_recording(event_prefix="inference_stream1")
        if recording_manager_2:
            recording_manager_2.start_recording(event_prefix="inference_stream2")
        return False

    if any((recording_manager, recording_manager_1, recording_manager_2)):
        logger.info(
            "Deferring inference recording start for %ss warm-up",
            _RECORDING_STARTUP_WARMUP_SEC,
        )
        GLib.timeout_add_seconds(
            _RECORDING_STARTUP_WARMUP_SEC,
            _start_recording_branches_after_warmup,
        )

    # Seed frame times for enabled streams only (disabled streams must not be tracked,
    # otherwise the watchdog fires false stale alerts for streams with no probe)
    for _sid, _key in [(0, 'nvosd_0'), (1, 'nvosd_1'), (2, 'nvosd_2')]:
        if _key in elements and elements[_key]:
            bus_handler.update_frame_time(_sid)
    bus_handler.start_watchdog(interval_sec=60)
    bus_handler.start_fps_logger()
    if config.ENABLE_DEBUG_PROBES:
        def _log_debug_memory_snapshot():
            _log_memory_snapshot("periodic")
            return True

        GLib.timeout_add_seconds(60, _log_debug_memory_snapshot)

    # Log config summary
    summary = config.get_config_summary()
    logger.info(f"Config: {json.dumps(summary, indent=2, default=str)}")

    try:
        loop.run()
    except Exception as e:
        logger.error(f"Main loop error: {e}", exc_info=True)
    finally:
        logger.info("Shutting down pipeline...")
        sync_stop_event.set()
        if sync_thread:
            sync_thread.join(timeout=5)
        if recording_manager:
            try:
                # Important for live RTSP: force EOS so mp4mux can finalize MP4 metadata.
                # Without EOS, files may remain 0 bytes or unplayable on abrupt shutdown.
                pipeline.send_event(Gst.Event.new_eos())
                bus = pipeline.get_bus()
                bus.timed_pop_filtered(
                    5 * Gst.SECOND,
                    Gst.MessageType.EOS | Gst.MessageType.ERROR
                )
                recording_manager.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording manager: {e}", exc_info=True)
        if recording_manager_1:
            try:
                recording_manager_1.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording manager 1: {e}", exc_info=True)
        if recording_manager_2:
            try:
                recording_manager_2.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording manager 2: {e}", exc_info=True)
        if pouring_processor:
            try:
                pouring_processor.close()
            except Exception as e:
                logger.error(f"Error closing pouring processor: {e}", exc_info=True)
        if pouring_processor_2:
            try:
                pouring_processor_2.close()
            except Exception as e:
                logger.error(f"Error closing pouring processor 2: {e}", exc_info=True)
        if screenshot_writer:
            try:
                screenshot_writer.stop(timeout=5.0, drain=True)
            except Exception as e:
                logger.error(f"Error stopping screenshot writer: {e}", exc_info=True)
        if async_db_writer:
            try:
                async_db_writer.stop(timeout=5.0, drain=True)
            except Exception as e:
                logger.error(f"Error stopping async DB writer: {e}", exc_info=True)
        pipeline.set_state(Gst.State.NULL)
        builder.terminate_ffmpeg_procs()
        logger.info("Pipeline stopped")

    if bus_handler.fatal_exit:
        sys.exit(1)


if __name__ == '__main__':
    main()
