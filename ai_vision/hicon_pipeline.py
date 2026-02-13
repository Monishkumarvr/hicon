"""
HiCon Pipeline - Main entry point for 2-camera DeepStream pipeline.

Stream 0 (Process Camera): pouring (nvinfer + tracker) + brightness (tapping, deslagging, spectro)
Stream 1 (Pyrometer Camera): rod detection (nvinfer)
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

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = config.LOG_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'pipeline.log'),
    ],
)
logger = logging.getLogger('hicon')


# ---------------------------------------------------------------------------
# Globals (set during init)
# ---------------------------------------------------------------------------
pouring_processor = None
brightness_processor = None
pyrometer_processor = None
bus_handler = None
sync_manager = None
recording_manager = None


# ---------------------------------------------------------------------------
# Pad probe callbacks
# ---------------------------------------------------------------------------
def osd_sink_pad_probe_stream0(pad, info):
    """
    Probe on nvosd_0 sink pad (Stream 0 — Process Camera).
    Handles:
      1. Pouring detection (nvinfer object meta + CPU brightness)
      2. Brightness analysis (tapping + deslagging + spectro via CPU frame)

    Frame is extracted once and shared by both processors.
    CRITICAL: unmap_nvds_buf_surface() MUST be called on Jetson.
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

        # Update bus handler frame counter
        if bus_handler:
            bus_handler.update_frame_time(0)

        # Extract CPU frame once for both processors (RGBA uint8)
        frame = None
        if config.ENABLE_FRAME_PROCESSING:
            try:
                import numpy as np
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame = np.array(n_frame, copy=True, order='C')
            except Exception as e:
                logger.error(f"Frame extraction error: {e}", exc_info=True)

        try:
            # 1. Pouring processor (nvinfer detections + brightness)
            if pouring_processor:
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

            # 2. Brightness processor (tapping + deslagging + spectro)
            if brightness_processor and frame is not None:
                try:
                    brightness_processor.process_frame_with_array(frame, frame_meta)
                    if config.ENABLE_INFERENCE_VIDEO:
                        brightness_processor.add_inference_display_meta(
                            batch_meta=batch_meta,
                            frame_meta=frame_meta,
                        )
                except Exception as e:
                    logger.error(f"Brightness processor error: {e}", exc_info=True)
        finally:
            # MANDATORY on Jetson — prevents memory leak
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

        # Extract CPU frame for event screenshots (RGBA uint8)
        frame = None
        if config.ENABLE_FRAME_PROCESSING:
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
    global pouring_processor, brightness_processor, pyrometer_processor
    global bus_handler, sync_manager, recording_manager

    logger.info("=" * 60)
    logger.info("HiCon Pipeline Starting")
    logger.info("=" * 60)

    # Initialize GStreamer
    Gst.init(None)

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

    # Create shared HeatCycleManager (owned by pipeline, shared by processors)
    heat_cycle_manager = HeatCycleManager(
        db_manager=db,
        ladle_absence_timeout=config.POURING_CYCLE_TIMEOUT_S,
    )

    # Initialize processors
    brightness_processor = BrightnessProcessor(
        zones_config=zones_config,
        db_manager=db,
        config=config,
        screenshot_dir=str(config.SCREENSHOT_DIR),
        heat_cycle_manager=heat_cycle_manager,
    )

    pyrometer_processor = PyrometerProcessor(
        zone_config=zones_config.get('pyrometer', {}),
        db_manager=db,
        config=config,
        screenshot_dir=str(config.SCREENSHOT_DIR),
    )

    # Pouring processor — import conditionally to avoid issues if not adapted yet
    try:
        from processors.pouring_processor import PouringProcessor
        pouring_processor = PouringProcessor(
            db_manager=db,
            config=config,
            screenshot_dir=str(config.SCREENSHOT_DIR),
            heat_cycle_manager=heat_cycle_manager,
        )
        logger.info("Pouring processor initialized")
    except Exception as e:
        logger.warning(f"Pouring processor not available: {e}")
        pouring_processor = None

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
        'config_pouring': config.CONFIG_POURING,
        'config_pyrometer': config.CONFIG_PYROMETER,
        'tracker_lib': config.TRACKER_LIB,
        'tracker_config': config.TRACKER_CONFIG,
        'rtsp_tcp_timeout_us': config.RTSP_TCP_TIMEOUT_US,
        'rtsp_retry': config.RTSP_RETRY,
        'rtsp_timeout_sec': config.RTSP_TIMEOUT_SEC,
        'rtsp_do_retransmission': config.RTSP_DO_RETRANSMISSION,
        'enable_inference_video': config.ENABLE_INFERENCE_VIDEO,
    }

    # Build pipeline
    builder = DeepStreamPipelineBuilder(pipeline_config)
    pipeline, elements = builder.create_pipeline()

    if not pipeline:
        logger.error("Failed to create pipeline")
        sys.exit(1)

    # Create main loop
    loop = GLib.MainLoop()

    # Attach bus handler
    bus_handler = BusHandler(pipeline, loop)

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
            )
            if recording_manager.setup_recording_branch(pipeline, tee_0):
                logger.info("Stream 0: DS-native inference recording branch configured")
            else:
                logger.error("Stream 0: failed to configure inference recording branch")
                recording_manager = None
        else:
            logger.warning("Inference video enabled but tee_0 is missing; recording disabled")

    # Attach pad probes
    # Stream 0: OSD sink pad probe (pouring + brightness)
    if 'nvosd_0' in elements and elements['nvosd_0']:
        osd_sinkpad = elements['nvosd_0'].get_static_pad("sink")
        if osd_sinkpad:
            osd_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                osd_sink_pad_probe_stream0,
            )
            logger.info("Stream 0: OSD sink pad probe attached (pouring + brightness + spectro)")

    # Stream 1: OSD sink pad probe (pyrometer) — must be after RGBA conversion
    if 'nvosd_1' in elements and elements['nvosd_1']:
        osd1_sinkpad = elements['nvosd_1'].get_static_pad("sink")
        if osd1_sinkpad:
            osd1_sinkpad.add_probe(
                Gst.PadProbeType.BUFFER,
                nvinfer_src_pad_probe_stream1,
            )
            logger.info("Stream 1: OSD sink pad probe attached (pyrometer + frame extraction)")

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

    # Configure recording file before PLAYING so filesink location is set in NULL/READY state
    if recording_manager:
        recording_manager.start_recording(event_prefix="inference_stream0")

    # Start pipeline
    logger.info("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("Failed to set pipeline to PLAYING")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)

    logger.info("Pipeline PLAYING — waiting for streams...")

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
        if pouring_processor:
            try:
                pouring_processor.close()
            except Exception as e:
                logger.error(f"Error closing pouring processor: {e}", exc_info=True)
        pipeline.set_state(Gst.State.NULL)
        logger.info("Pipeline stopped")


if __name__ == '__main__':
    main()
