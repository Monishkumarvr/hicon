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
import signal
import threading
from pathlib import Path
from datetime import datetime

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds

# Ensure ai_vision is on path
sys.path.insert(0, str(Path(__file__).parent))

import config
from db_manager import HiConDatabase
from pipeline.gst_builder import DeepStreamPipelineBuilder
from pipeline.bus_handler import BusHandler
from pipeline.recording import RecordingManager
from processors.brightness_processor import BrightnessProcessor
from processors.pyrometer_processor import PyrometerProcessor
from state.heat_cycle_manager import HeatCycleManager
from sync.api_client import APIClient
from sync.sync_manager import SyncManager
from utils.zone_loader import load_zones_config
from streaming.mjpeg_server import MJPEGServer
from processors.pouring_analysis_controller import HybridPouringController
from processors.pouring_meta_reader import PouringMetaReader

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
pouring_processor_2 = None       # Stream 2: second pouring camera (pouring-only, no brightness)
pouring_controller = None        # Stream 0: C++ state + Python business logic
pouring_controller_2 = None      # Stream 2: C++ state + Python business logic
pouring_meta_reader = None       # Stream 0 C++ pouring plugin state decoder
pouring_meta_reader_2 = None     # Stream 2 C++ pouring plugin state decoder
heat_cycle_manager_2 = None      # Stream 2: independent heat cycle state
brightness_processor = None
pyrometer_processor = None
bus_handler = None
sync_manager = None
recording_manager = None
mjpeg_server = None
_live_stream_last_extract = {}
_live_stream_warmup_deadline = 0.0
_RECORDING_STARTUP_WARMUP_SEC = 5


# ---------------------------------------------------------------------------
# Pad probe callbacks
# ---------------------------------------------------------------------------
def _process_stream0_cpu_analysis(info, update_main_path=False, update_analysis_path=False):
    """Run Stream 0 CPU frame extraction and processors on the provided buffer."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
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

        if bus_handler:
            if update_main_path:
                bus_handler.update_frame_time(0)
            if update_analysis_path:
                bus_handler.update_stream0_analysis_time()

        meta_decode_enabled = bool(config.STREAM_0_CPP_META_DECODE_ENABLED)
        hybrid_controller_enabled = bool(config.STREAM_0_HYBRID_CONTROLLER_ENABLED)

        native_state = None
        if (
            config.USE_CPP_POURING_PLUGIN
            and pouring_meta_reader is not None
            and meta_decode_enabled
        ):
            try:
                native_state = pouring_meta_reader.decode_frame_meta(frame_meta)
            except Exception as e:
                logger.error(f"Pouring meta reader error: {e}", exc_info=True)

        # Determine which pouring path to use:
        # 1. C++ state plugin + Python hybrid controller
        # 2. Python-only YOLO pouring processor
        run_python_pouring = (
            pouring_processor is not None
            and not config.USE_CPP_POURING_PLUGIN
        )
        run_hybrid_pouring = (
            pouring_controller is not None
            and config.USE_CPP_POURING_PLUGIN
            and hybrid_controller_enabled
        )

        # CUDA brightness operates directly on GPU NvBufSurface — no CPU copy needed.
        # CPU brightness still needs frame extraction like before.
        cuda_brightness = (
            brightness_processor is not None
            and hasattr(brightness_processor, "process_frame_cuda")
        )
        cpu_brightness = brightness_processor is not None and not cuda_brightness

        stream0_needs_frame = (
            config.ENABLE_FRAME_PROCESSING
            and (
                run_python_pouring
                or cpu_brightness
                or (run_hybrid_pouring and pouring_controller.needs_frame(native_state))
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

        try:
            # Pouring path first so brightness suppression sees the newest pouring state.
            if run_python_pouring:
                try:
                    pouring_processor.process_frame(
                        frame_meta=frame_meta,
                        frame=frame,
                        batch_meta=batch_meta,
                        timestamp=time.time(),
                        datetime_obj=datetime.now(),
                    )
                except Exception as e:
                    logger.error(f"Pouring processor error: {e}", exc_info=True)
            elif run_hybrid_pouring:
                try:
                    pouring_controller.process_native_state(
                        frame_meta=frame_meta,
                        native_state=native_state,
                        frame=frame,
                        batch_meta=batch_meta,
                        timestamp=time.time(),
                        datetime_obj=datetime.now(),
                    )
                except Exception as e:
                    logger.error(f"Hybrid pouring controller error: {e}", exc_info=True)

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
                    if (
                        config.ENABLE_INFERENCE_VIDEO
                        and batch_meta is not None
                        and getattr(brightness_processor, "enable_display_meta", True)
                    ):
                        brightness_processor.add_inference_display_meta(
                            batch_meta=batch_meta,
                            frame_meta=frame_meta,
                        )
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
    """Track Stream 0 liveness at pgie_pouring.src."""
    return _mark_stream0_stage("pgie_src", info)


def stream0_stage_probe_tracker_sink(pad, info):
    """Track Stream 0 liveness at tracker_0.sink."""
    return _mark_stream0_stage("tracker_sink", info)


def stream0_stage_probe_tracker_src(pad, info):
    """Track Stream 0 liveness at tracker_0.src."""
    return _mark_stream0_stage("tracker_src", info)


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

        # Extract annotated frame (post-OSD, WITH overlays)
        try:
            import numpy as np
            import cv2
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_rgba = np.array(n_frame, copy=True, order='C')
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            mjpeg_server.update_frame(stream_id=0, frame_bgr=frame_bgr)
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
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

        # Extract CPU frame lazily for event screenshots only when detections exist or
        # an active pyrometer event may soon need a start/end snapshot.
        frame = None
        need_frame = False
        if config.ENABLE_FRAME_PROCESSING and pyrometer_processor:
            try:
                obj_count = int(getattr(frame_meta, "num_obj_meta", 0) or 0)
                need_frame = obj_count > 0 or pyrometer_processor.is_active
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
                    pyrometer_processor.process_frame(frame_meta, frame=frame)
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
    Probe on nvosd_2 sink pad (Stream 2 — Second Pouring Camera).
    When the C++ plugin is enabled, Python consumes the native session/probe
    state and reuses the mature business logic on CPU. Otherwise it falls back
    to the Python-only pouring processor.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    run_python_pouring = (
        pouring_processor_2 is not None
        and not config.USE_CPP_POURING_PLUGIN
    )
    run_hybrid_pouring = (
        pouring_controller_2 is not None
        and config.USE_CPP_POURING_PLUGIN
    )

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        if bus_handler:
            bus_handler.update_frame_time(2)

        native_state = None
        if config.USE_CPP_POURING_PLUGIN and pouring_meta_reader_2 is not None:
            try:
                native_state = pouring_meta_reader_2.decode_frame_meta(frame_meta)
            except Exception as e:
                logger.error(f"Stream 2 pouring meta reader error: {e}", exc_info=True)

        frame = None
        if (
            config.ENABLE_FRAME_PROCESSING
            and (
                run_python_pouring
                or (run_hybrid_pouring and pouring_controller_2.needs_frame(native_state))
            )
        ):
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
                        timestamp=time.time(),
                        datetime_obj=datetime.now(),
                    )
                except Exception as e:
                    logger.error(f"Stream 2 pouring processor error: {e}", exc_info=True)
            elif run_hybrid_pouring:
                try:
                    pouring_controller_2.process_native_state(
                        frame_meta=frame_meta,
                        native_state=native_state,
                        frame=frame,
                        batch_meta=batch_meta,
                        timestamp=time.time(),
                        datetime_obj=datetime.now(),
                    )
                except Exception as e:
                    logger.error(f"Stream 2 hybrid pouring controller error: {e}", exc_info=True)
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
    global pouring_processor, pouring_processor_2, pouring_controller, pouring_controller_2
    global pouring_meta_reader, pouring_meta_reader_2
    global heat_cycle_manager_2
    global brightness_processor, pyrometer_processor
    global bus_handler, sync_manager, recording_manager, mjpeg_server

    logger.info("=" * 60)
    logger.info("HiCon Pipeline Starting")
    logger.info("=" * 60)

    # Initialize GStreamer
    Gst.init(None)

    # Load C++ pouring detection plugin
    cpp_pouring_plugin_path = str(
        Path(__file__).parent / 'custom_plugins' / 'hicon_pouring' / 'libgsthiconpouring.so'
    )
    use_cpp_pouring = config.USE_CPP_POURING_PLUGIN
    if use_cpp_pouring:
        try:
            Gst.Plugin.load_file(cpp_pouring_plugin_path)
            logger.info("C++ pouring plugin loaded: %s", cpp_pouring_plugin_path)
            # Meta decoders/controllers are created after db + heat_cycle_manager init.
        except Exception as e:
            logger.warning("C++ pouring plugin not available (%s), falling back to Python", e)
            use_cpp_pouring = False

    # Create output directories
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    db = HiConDatabase(str(config.DB_PATH))

    # Load zone configuration
    zones_path = str(config.CONFIG_DIR / 'zones.json')
    zones_config = load_zones_config(zones_path)
    logger.info(f"Loaded zones config from {zones_path}")

    # Create shared HeatCycleManager for Stream 0 (owned by pipeline, shared by processors)
    heat_cycle_manager = HeatCycleManager(
        db_manager=db,
        ladle_absence_timeout=config.POURING_CYCLE_TIMEOUT_S,
    )

    # Create independent HeatCycleManager for Stream 2 (physically separate camera)
    heat_cycle_manager_2 = HeatCycleManager(
        db_manager=db,
        ladle_absence_timeout=config.POURING_CYCLE_TIMEOUT_S,
    )

    # Initialize C++ pouring meta readers (after db + heat_cycle_manager)
    if use_cpp_pouring:
        pouring_meta_reader = PouringMetaReader(
            stream_label="stream0",
        )
        pouring_meta_reader_2 = PouringMetaReader(
            stream_label="stream2",
        )
        logger.info("C++ pouring meta readers initialized")
        if not config.STREAM_0_CPP_META_ATTACH_ENABLED:
            logger.warning("Stream 0: C++ user-meta attachment disabled for staged isolation")
        if not config.STREAM_0_CPP_META_DECODE_ENABLED:
            logger.warning("Stream 0: C++ meta decode disabled for staged isolation")
        if not config.STREAM_0_HYBRID_CONTROLLER_ENABLED:
            logger.warning("Stream 0: Hybrid pouring controller disabled for staged isolation")

    # Initialize processors
    stream0_enable_display_meta = not config.STREAM_0_DECOUPLED_ANALYSIS_MODE
    if config.STREAM_0_DECOUPLED_ANALYSIS_MODE:
        logger.info(
            "Stream 0 (CP Plus): CPU-generated live display meta disabled in decoupled mode"
        )

    use_cuda_brightness = bool(config.USE_CUDA_BRIGHTNESS)

    if config.ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR:
        if use_cuda_brightness:
            from processors.cuda_brightness_processor import CudaBrightnessProcessor
            brightness_processor = CudaBrightnessProcessor(
                zones_config=zones_config,
                db_manager=db,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=stream0_enable_display_meta,
            )
            logger.info("Stream 0: CUDA brightness processor initialized (GPU-direct)")
        else:
            brightness_processor = BrightnessProcessor(
                zones_config=zones_config,
                db_manager=db,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=stream0_enable_display_meta,
            )
            logger.info("Stream 0: CPU brightness processor initialized (NumPy)")
    else:
        brightness_processor = None
        logger.warning("Stream 0: Brightness processor disabled for diagnostics")

    pyrometer_processor = PyrometerProcessor(
        zone_config=zones_config.get('pyrometer', {}),
        db_manager=db,
        config=config,
        screenshot_dir=str(config.SCREENSHOT_DIR),
        heat_cycle_manager=heat_cycle_manager,
    )

    # Stream 0 pouring path:
    # - Hybrid controller when the native C++ plugin is enabled
    # - Python-only processor as fallback otherwise
    if config.ENABLE_STREAM_0_POURING_PROCESSOR and use_cpp_pouring:
        try:
            pouring_controller = HybridPouringController(
                db_manager=db,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=stream0_enable_display_meta,
            )
            logger.info("Stream 0: Hybrid pouring controller initialized")
        except Exception as e:
            logger.warning(f"Stream 0: Hybrid pouring controller not available: {e}")
            pouring_controller = None
    else:
        pouring_controller = None

    if config.ENABLE_STREAM_0_POURING_PROCESSOR and not use_cpp_pouring:
        try:
            from processors.pouring_processor import PouringProcessor
            pouring_processor = PouringProcessor(
                db_manager=db,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager,
                enable_display_meta=stream0_enable_display_meta,
            )
            logger.info("Stream 0: Pouring processor initialized")
        except Exception as e:
            logger.warning(f"Stream 0: Pouring processor not available: {e}")
            pouring_processor = None
    else:
        pouring_processor = None
        if not config.ENABLE_STREAM_0_POURING_PROCESSOR:
            logger.warning("Stream 0: Pouring processor disabled for diagnostics")
        elif use_cpp_pouring:
            logger.info("Stream 0: Python-only pouring processor bypassed (hybrid C++ path active)")

    if use_cpp_pouring:
        try:
            pouring_controller_2 = HybridPouringController(
                db_manager=db,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager_2,
                camera_id_override=config.CAMERA_ID_STREAM_2,
                enable_display_meta=False,
            )
            logger.info("Stream 2: Hybrid pouring controller initialized")
        except Exception as e:
            logger.warning(f"Stream 2: Hybrid pouring controller not available: {e}")
            pouring_controller_2 = None
    else:
        pouring_controller_2 = None

    # YOLO-based pouring fallback (Stream 2 — second pouring camera)
    if not use_cpp_pouring:
        try:
            from processors.pouring_processor import PouringProcessor
            pouring_processor_2 = PouringProcessor(
                db_manager=db,
                config=config,
                screenshot_dir=str(config.SCREENSHOT_DIR),
                heat_cycle_manager=heat_cycle_manager_2,
                camera_id_override=config.CAMERA_ID_STREAM_2,
            )
            logger.info("Stream 2: Pouring processor initialized")
        except Exception as e:
            logger.warning(f"Stream 2: Pouring processor not available: {e}")
            pouring_processor_2 = None
    else:
        pouring_processor_2 = None

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
        'stream_0_analysis_cpp_plugin_enabled': config.STREAM_0_ANALYSIS_CPP_PLUGIN_ENABLED,
        'stream_0_analysis_probe_enabled': config.STREAM_0_ANALYSIS_PROBE_ENABLED,
        'enable_inference_video': config.ENABLE_INFERENCE_VIDEO,
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
        'use_cpp_pouring_plugin': use_cpp_pouring,
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
    )

    # Initialize MJPEG live streaming server (if enabled)
    mjpeg_server = None
    if config.ENABLE_LIVE_STREAM:
        mjpeg_server = MJPEGServer(
            host=config.LIVE_STREAM_HOST,
            port=config.LIVE_STREAM_PORT,
            jpeg_quality=config.LIVE_STREAM_QUALITY,
            max_fps=config.LIVE_STREAM_FPS
        )
        # Only register streams that have pipeline elements (i.e. are enabled and linked)
        for _sid, _key in [(0, 'nvosd_0'), (1, 'nvosd_1'), (2, 'nvosd_2')]:
            if _key in elements and elements[_key]:
                mjpeg_server.register_stream(_sid)
        mjpeg_server.start()
        logger.info(f"✓ Live streaming enabled: http://{config.LIVE_STREAM_HOST}:{config.LIVE_STREAM_PORT}/")
    else:
        logger.info("Live streaming disabled (ENABLE_LIVE_STREAM=false)")

    # Optional DS-native inference recording branch (post-OSD annotations)
    recording_manager = None
    if config.ENABLE_INFERENCE_VIDEO:
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
            logger.warning("Inference video enabled but tee_0 is missing; recording disabled")

    recording_manager_1 = None
    if config.ENABLE_INFERENCE_VIDEO:
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
            logger.warning("Inference video enabled but tee_1 missing; stream 1 recording disabled")

    recording_manager_2 = None
    if config.ENABLE_INFERENCE_VIDEO:
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
            logger.warning("Inference video enabled but tee_2 missing; stream 2 recording disabled")

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
        if 'hicon_pouring_0' in elements and elements['hicon_pouring_0']:
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

    # Stream 2: PGIE src diagnostic probe (counts raw inference objects before tracker)
    if 'pgie_pouring_2' in elements and elements['pgie_pouring_2']:
        pgie2_srcpad = elements['pgie_pouring_2'].get_static_pad("src")
        if pgie2_srcpad:
            pgie2_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                pgie_src_pad_probe_stream2_diag,
            )
            logger.info("Stream 2: PGIE src diagnostic probe attached")

    # Stream 2: OSD sink pad probe (second pouring camera — pouring only, no brightness)
    if 'nvosd_2' in elements and elements['nvosd_2']:
        osd2_sinkpad = elements['nvosd_2'].get_static_pad("sink")
        if osd2_sinkpad:
            osd2_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                osd_sink_pad_probe_stream2,
            )
            logger.info("Stream 2: OSD sink pad probe attached (pouring)")

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

    def _start_recording_branches_after_warmup():
        if recording_manager:
            recording_manager.start_recording(event_prefix="inference_stream0")
        if recording_manager_1:
            recording_manager_1.start_recording(event_prefix="inference_stream1")
        if recording_manager_2:
            recording_manager_2.start_recording(event_prefix="inference_stream2")
        return False

    if config.ENABLE_INFERENCE_VIDEO:
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
        if pouring_processor:
            try:
                pouring_processor.close()
            except Exception as e:
                logger.error(f"Error closing pouring processor: {e}", exc_info=True)
        if pouring_controller:
            try:
                pouring_controller.close()
            except Exception as e:
                logger.error(f"Error closing hybrid pouring controller: {e}", exc_info=True)
        if pouring_processor_2:
            try:
                pouring_processor_2.close()
            except Exception as e:
                logger.error(f"Error closing pouring processor 2: {e}", exc_info=True)
        if pouring_controller_2:
            try:
                pouring_controller_2.close()
            except Exception as e:
                logger.error(f"Error closing hybrid pouring controller 2: {e}", exc_info=True)
        pipeline.set_state(Gst.State.NULL)
        builder.terminate_ffmpeg_procs()
        logger.info("Pipeline stopped")

    if bus_handler.fatal_exit:
        sys.exit(1)


if __name__ == '__main__':
    main()
