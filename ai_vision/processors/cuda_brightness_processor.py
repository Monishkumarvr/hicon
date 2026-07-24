"""
CUDA Brightness Processor — GPU-accelerated tapping/deslagging/spectro detection.

Replaces CPU NumPy-based BrightnessProcessor with CUDA kernels that operate
directly on the GPU NvBufSurface — no CPU frame copy on the hot path.

Detection methods:
  - Tapping:    CUDA white-ratio extraction (libds_white_ratio.so)
  - Deslagging: CUDA molten blob detection (libds_molten_detect.so)
  - Spectro:    CUDA molten blob detection + per-zone false-positive filters

Ported from deepstream_yolo_hicon/inference_probes.py and adapted for the
ai_vision event handling chain (DB → heat cycle → cloud sync).
"""

import logging
import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyds

from processors.white_ratio_compute import extract_white_ratio
from processors.molten_detect_compute import detect_molten_blobs
from processors.cuda_geometry import (
    Polygon, polygon_bbox, draw_zone_polygon, draw_blob_rect,
    draw_text_label, scale_zones,
)
from utils.utils import generate_sync_id
from utils.screenshot import (prepare_frame, add_header, add_footer,
                               draw_roi_overlay, save as save_screenshot)

logger = logging.getLogger(__name__)


# ============================================================
# SHARED PROBE STATE — Nodulizer blackout coordination
# ============================================================

class SharedProbeState:
    """
    Coordinates nodulizer blackout across all detectors.

    When TappingDetector sees white_ratio > 0.6 in any zone, it triggers a
    10-second global blackout that disables all detectors. This covers the
    period when explosive nodulizers are added to the furnace, during which
    all pixel-based detections are invalid.
    """

    NODULIZER_WHITE_RATIO_THRESHOLD: float = 0.6
    NODULIZER_BLACKOUT_S: float = 10.0

    def __init__(self):
        self.blackout_until_frame: int = -1

    def trigger_blackout(self, current_frame: int, fps: float):
        until = current_frame + int(self.NODULIZER_BLACKOUT_S * fps)
        if until > self.blackout_until_frame:
            self.blackout_until_frame = until
            logger.info(
                f"[Nodulizer] Blackout triggered at frame {current_frame} — "
                f"all probes disabled until frame {until} "
                f"({self.NODULIZER_BLACKOUT_S:.0f}s)"
            )

    def is_blackout(self, current_frame: int) -> bool:
        return current_frame < self.blackout_until_frame


# ============================================================
# TAPPING DETECTOR — CUDA white-ratio
# ============================================================

@dataclass
class TappingZoneState:
    name: str
    zone: Polygon
    bbox: Tuple[float, float, float, float]  # pre-computed AABB
    tapping_active: bool = False
    on_count: int = 0
    off_count: int = 0
    start_frame: int = -1
    end_frame: int = -1
    last_white_ratio: float = 0.0
    event_start_time: float = 0.0
    event_start_datetime: Optional[datetime] = None


class CudaTappingDetector:
    """Multi-zone tapping detection via CUDA white-ratio."""

    def __init__(self, zones: Dict[str, Polygon], config: dict,
                 shared_state: SharedProbeState, fps: float = 25.0):
        self.fps = fps
        self.shared_state = shared_state
        self.on_threshold = config.get('start_white_ratio', 0.25)
        self.off_threshold = config.get('end_white_ratio', 0.1)
        self.on_frames = config.get('start_frame_count', 20)
        self.off_frames = config.get('end_frame_count', 25)
        self.abs_threshold = config.get('abs_brightness_threshold', 210)

        self.zone_states: Dict[str, TappingZoneState] = {}
        for name, poly in zones.items():
            self.zone_states[name] = TappingZoneState(
                name=name, zone=poly, bbox=polygon_bbox(poly),
            )
        self.frame_idx = 0

    def update(self, gst_buffer, frame_meta, capture_ts=None, capture_dt=None) -> List[dict]:
        """Run tapping detection. Returns list of event dicts.

        capture_ts/capture_dt: when supplied (delayed-source capture clock),
        stamp events with these instead of time.time()/datetime.now() — see
        config.HICON_DELAYED_CAPTURE_CLOCK. None (default) reproduces today's
        exact wall-clock behavior.
        """
        self.frame_idx = frame_meta.frame_num
        events = []

        # Always compute white ratios (needed for nodulizer detection)
        for zs in self.zone_states.values():
            ratio = extract_white_ratio(
                gst_buffer, int(frame_meta.batch_id),
                zs.bbox, self.abs_threshold,
            )
            zs.last_white_ratio = float(ratio) if ratio >= 0 else 0.0

        # Nodulizer check: if any zone exceeds threshold, trigger global blackout
        if any(zs.last_white_ratio > SharedProbeState.NODULIZER_WHITE_RATIO_THRESHOLD
               for zs in self.zone_states.values()):
            self.shared_state.trigger_blackout(self.frame_idx, self.fps)

        # Skip state machine during blackout
        if self.shared_state.is_blackout(self.frame_idx):
            return events

        # Update per-zone state machines
        for zs in self.zone_states.values():
            event = self._update_zone(zs, capture_ts, capture_dt)
            if event:
                events.append(event)

        return events

    def _update_zone(self, zs: TappingZoneState, capture_ts=None, capture_dt=None) -> Optional[dict]:
        ratio = zs.last_white_ratio

        if not zs.tapping_active:
            if ratio > self.on_threshold:
                zs.on_count += 1
            else:
                zs.on_count = 0

            if zs.on_count >= self.on_frames:
                zs.tapping_active = True
                zs.start_frame = self.frame_idx - self.on_frames + 1
                zs.on_count = 0
                zs.event_start_time = capture_ts if capture_ts is not None else time.time()
                zs.event_start_datetime = capture_dt if capture_dt is not None else datetime.now()
                logger.info(
                    f"[Tapping] Zone '{zs.name}': STARTED at frame {zs.start_frame} "
                    f"({zs.start_frame / self.fps:.1f}s)"
                )
                return {
                    "type": "tapping",
                    "phase": "start",
                    "zone_name": zs.name,
                    "start": zs.event_start_datetime.isoformat(),
                    "start_wall": zs.event_start_time,
                    "start_datetime": zs.event_start_datetime,
                }
        else:
            if ratio < self.off_threshold:
                zs.off_count += 1
            else:
                zs.off_count = 0

            if zs.off_count >= self.off_frames:
                zs.tapping_active = False
                zs.end_frame = self.frame_idx - self.off_frames + 1
                zs.off_count = 0
                end_time = capture_ts if capture_ts is not None else time.time()
                end_datetime = capture_dt if capture_dt is not None else datetime.now()
                duration = end_time - zs.event_start_time
                logger.info(
                    f"[Tapping] Zone '{zs.name}': ENDED at frame {zs.end_frame} "
                    f"(duration: {duration:.1f}s)"
                )
                return {
                    "type": "tapping",
                    "phase": "end",
                    "zone_name": zs.name,
                    "start": zs.event_start_datetime.isoformat(),
                    "end": end_datetime.isoformat(),
                    "duration_sec": round(duration, 1),
                    "start_wall": zs.event_start_time,
                    "end_wall": end_time,
                    "start_datetime": zs.event_start_datetime,
                    "end_datetime": end_datetime,
                }

        return None

    @property
    def is_active(self) -> bool:
        return any(zs.tapping_active for zs in self.zone_states.values())


# ============================================================
# DESLAGGING DETECTOR — CUDA molten blob detection
# ============================================================

@dataclass
class DeslaggingZoneState:
    name: str
    zone: Polygon
    bbox: Tuple[float, float, float, float]
    deslagging_active: bool = False
    start_frame: int = -1
    end_frame: int = -1
    last_blobs: list = field(default_factory=list)
    event_start_time: float = 0.0
    event_start_datetime: Optional[datetime] = None


class CudaDeslaggingDetector:
    """Multi-zone deslagging detection via CUDA molten blob detection."""

    def __init__(self, zones: Dict[str, Polygon], config: dict,
                 shared_state: SharedProbeState, fps: float = 25.0):
        self.fps = fps
        self.shared_state = shared_state
        self.min_blob_area = config.get('min_blob_area', 500)
        self.brightness_thresh = config.get('brightness_thresh', 180)

        self.zone_states: Dict[str, DeslaggingZoneState] = {}
        for name, poly in zones.items():
            self.zone_states[name] = DeslaggingZoneState(
                name=name, zone=poly, bbox=polygon_bbox(poly),
            )
        self.frame_idx = 0

    def update(self, gst_buffer, frame_meta, capture_ts=None, capture_dt=None) -> List[dict]:
        self.frame_idx = frame_meta.frame_num
        events = []

        if self.shared_state.is_blackout(self.frame_idx):
            return events

        for zs in self.zone_states.values():
            blobs = detect_molten_blobs(
                gst_buffer, int(frame_meta.batch_id), zs.bbox,
                min_blob_area=self.min_blob_area,
                brightness_thresh=self.brightness_thresh,
            )
            zs.last_blobs = blobs
            now_active = len(blobs) > 0

            if now_active and not zs.deslagging_active:
                zs.deslagging_active = True
                zs.start_frame = self.frame_idx
                zs.event_start_time = capture_ts if capture_ts is not None else time.time()
                zs.event_start_datetime = capture_dt if capture_dt is not None else datetime.now()
                logger.info(
                    f"[Deslagging] Zone '{zs.name}': STARTED at frame {self.frame_idx} "
                    f"({self.frame_idx / self.fps:.1f}s)"
                )
                events.append({
                    "type": "deslagging",
                    "phase": "start",
                    "zone_name": zs.name,
                    "start": zs.event_start_datetime.isoformat(),
                    "start_wall": zs.event_start_time,
                    "start_datetime": zs.event_start_datetime,
                })
            elif not now_active and zs.deslagging_active:
                zs.deslagging_active = False
                zs.end_frame = self.frame_idx
                end_time = capture_ts if capture_ts is not None else time.time()
                end_datetime = capture_dt if capture_dt is not None else datetime.now()
                duration = end_time - zs.event_start_time
                logger.info(
                    f"[Deslagging] Zone '{zs.name}': ENDED at frame {self.frame_idx} "
                    f"(duration: {duration:.1f}s)"
                )
                events.append({
                    "type": "deslagging",
                    "phase": "end",
                    "zone_name": zs.name,
                    "start": zs.event_start_datetime.isoformat(),
                    "end": end_datetime.isoformat(),
                    "duration_sec": round(duration, 1),
                    "start_wall": zs.event_start_time,
                    "end_wall": end_time,
                    "start_datetime": zs.event_start_datetime,
                    "end_datetime": end_datetime,
                })

        return events

    @property
    def is_active(self) -> bool:
        return any(zs.deslagging_active for zs in self.zone_states.values())


# ============================================================
# SPECTRO DETECTOR — CUDA molten blob + per-zone filters
# ============================================================

@dataclass
class SpectroZoneState:
    name: str
    zone: Polygon
    bbox: Tuple[float, float, float, float]
    spectro_active: bool = False
    start_frame: int = -1
    end_frame: int = -1
    last_blobs: list = field(default_factory=list)
    on_count: int = 0
    event_start_time: float = 0.0
    event_start_datetime: Optional[datetime] = None


class CudaSpectroDetector:
    """Multi-zone spectro detection via CUDA molten blob + per-zone filters."""

    def __init__(self, zones: Dict[str, Polygon], config: dict,
                 zone_configs: Dict[str, dict],
                 shared_state: SharedProbeState, fps: float = 25.0):
        self.fps = fps
        self.shared_state = shared_state
        self.min_blob_area = config.get('min_blob_area', 50)
        self.brightness_thresh = config.get('brightness_thresh', 180)
        self.zone_configs = zone_configs  # per-zone filter params

        self.zone_states: Dict[str, SpectroZoneState] = {}
        for name, poly in zones.items():
            self.zone_states[name] = SpectroZoneState(
                name=name, zone=poly, bbox=polygon_bbox(poly),
            )
        self.frame_idx = 0

    def update(self, gst_buffer, frame_meta, capture_ts=None, capture_dt=None) -> List[dict]:
        self.frame_idx = frame_meta.frame_num
        events = []

        if self.shared_state.is_blackout(self.frame_idx):
            return events

        for zs in self.zone_states.values():
            blobs = detect_molten_blobs(
                gst_buffer, int(frame_meta.batch_id), zs.bbox,
                min_blob_area=self.min_blob_area,
                brightness_thresh=self.brightness_thresh,
            )
            blobs = self._filter_blobs(blobs, zs.name, zs.bbox)
            zs.last_blobs = blobs

            zone_cfg = self.zone_configs.get(zs.name, {})
            on_frames = zone_cfg.get('on_frames', 0)
            any_blob = len(blobs) > 0

            # ON hysteresis
            if any_blob:
                zs.on_count = min(zs.on_count + 1, on_frames + 1)
            else:
                zs.on_count = 0

            now_active = (zs.on_count >= on_frames) if on_frames > 0 else any_blob

            if now_active and not zs.spectro_active:
                zs.spectro_active = True
                zs.start_frame = self.frame_idx
                zs.event_start_time = capture_ts if capture_ts is not None else time.time()
                zs.event_start_datetime = capture_dt if capture_dt is not None else datetime.now()
                logger.info(
                    f"[Spectro] Zone '{zs.name}': STARTED at frame {self.frame_idx} "
                    f"({self.frame_idx / self.fps:.1f}s)"
                )
                events.append({
                    "type": "spectro",
                    "phase": "start",
                    "zone_name": zs.name,
                    "start": zs.event_start_datetime.isoformat(),
                    "start_wall": zs.event_start_time,
                    "start_datetime": zs.event_start_datetime,
                })
            elif not now_active and zs.spectro_active:
                zs.spectro_active = False
                zs.end_frame = self.frame_idx
                end_time = capture_ts if capture_ts is not None else time.time()
                end_datetime = capture_dt if capture_dt is not None else datetime.now()
                duration = end_time - zs.event_start_time
                logger.info(
                    f"[Spectro] Zone '{zs.name}': ENDED at frame {self.frame_idx} "
                    f"(duration: {duration:.1f}s)"
                )
                events.append({
                    "type": "spectro",
                    "phase": "end",
                    "zone_name": zs.name,
                    "start": zs.event_start_datetime.isoformat(),
                    "end": end_datetime.isoformat(),
                    "duration_sec": round(duration, 1),
                    "start_wall": zs.event_start_time,
                    "end_wall": end_time,
                    "start_datetime": zs.event_start_datetime,
                    "end_datetime": end_datetime,
                })

        return events

    def _filter_blobs(self, blobs, zone_name: str, zone_bbox):
        """Apply per-zone shape filters to discard false-positive blobs."""
        cfg = self.zone_configs.get(zone_name, {})
        max_ar = cfg.get('max_aspect_ratio')
        max_cov = cfg.get('max_coverage')

        if max_ar is None and max_cov is None:
            return blobs

        zone_x1, zone_y1, zone_x2, zone_y2 = zone_bbox
        zone_area = max(1, (zone_x2 - zone_x1) * (zone_y2 - zone_y1))

        filtered = []
        for blob in blobs:
            bw = max(1, blob.x2 - blob.x1)
            bh = max(1, blob.y2 - blob.y1)
            if max_ar is not None and max(bw, bh) / min(bw, bh) > max_ar:
                continue
            if max_cov is not None and blob.area / zone_area > max_cov:
                continue
            filtered.append(blob)
        return filtered

    @property
    def is_active(self) -> bool:
        return any(zs.spectro_active for zs in self.zone_states.values())


# ============================================================
# CUDA BRIGHTNESS PROCESSOR — Main integration class
# ============================================================

class CudaBrightnessProcessor:
    """
    GPU-accelerated tapping/deslagging/spectro detection.

    Drop-in replacement for BrightnessProcessor with CUDA-based detection
    that operates directly on GPU NvBufSurface (no CPU frame copy).

    Uses the same event handling chain: DB insert → heat cycle → cloud sync.
    """

    def __init__(self, zones_config, db_manager, config, screenshot_dir,
                 heat_cycle_manager=None, enable_display_meta=True,
                 screenshot_writer=None):
        self.db_manager = db_manager
        self.config = config
        self.heat_cycle_manager = heat_cycle_manager
        self.enable_display_meta = enable_display_meta
        self.screenshot_writer = screenshot_writer
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.customer_id = config.CUSTOMER_ID
        self.camera_id = config.CAMERA_ID_STREAM_0
        self.location = config.LOCATION
        self.fps = float(getattr(config, 'STREAM_FPS', 25.0))

        # Mux output resolution (zones are scaled to this)
        self.mux_width = 1280
        self.mux_height = 720

        # Shared state for nodulizer blackout
        self.shared_state = SharedProbeState()

        # Parse and scale zones from config
        tapping_cfg = zones_config.get('tapping', {})
        deslagging_cfg = zones_config.get('deslagging', {})
        spectro_cfg = zones_config.get('spectro', {})

        # Build scaled zone polygons
        tapping_zones = self._build_zones(tapping_cfg)
        deslagging_zones = self._build_zones(deslagging_cfg)
        spectro_zones = self._build_zones(spectro_cfg)

        # Build per-zone filter configs for spectro
        spectro_zone_cfgs = {}
        for zname, zcfg in spectro_cfg.get('zones', {}).items():
            spectro_zone_cfgs[zname] = {
                'on_frames': zcfg.get('on_frames', 0),
                'max_aspect_ratio': zcfg.get('max_aspect_ratio'),
                'max_coverage': zcfg.get('max_coverage'),
            }

        # Store raw zone configs for screenshot ROI drawing
        self._tapping_zones = tapping_zones
        self._deslagging_zones = deslagging_zones
        self._spectro_zones = spectro_zones

        # Create detectors
        self.tapping_detector = CudaTappingDetector(
            zones=tapping_zones, config=tapping_cfg,
            shared_state=self.shared_state, fps=self.fps,
        ) if tapping_zones else None

        self.deslagging_detector = CudaDeslaggingDetector(
            zones=deslagging_zones, config=deslagging_cfg,
            shared_state=self.shared_state, fps=self.fps,
        ) if deslagging_zones else None

        self.spectro_detector = CudaSpectroDetector(
            zones=spectro_zones, config=spectro_cfg,
            zone_configs=spectro_zone_cfgs,
            shared_state=self.shared_state, fps=self.fps,
        ) if spectro_zones else None

        logger.info(
            f"CudaBrightnessProcessor initialized — "
            f"tapping: {len(tapping_zones)} zones, "
            f"deslagging: {len(deslagging_zones)} zones, "
            f"spectro: {len(spectro_zones)} zones"
        )

    def _build_zones(self, cfg: dict) -> Dict[str, Polygon]:
        """Parse zones from config and scale to mux output resolution."""
        zones_data = cfg.get('zones', {})
        if not zones_data:
            # Fallback: single-zone legacy format (roi_points at top level)
            roi_pts = cfg.get('roi_points')
            if roi_pts:
                zones_data = {"zone-1": {"roi_points": roi_pts}}
            else:
                return {}

        annotation_size = cfg.get('annotation_size')

        # Convert roi_points lists to {name: [[x,y],...]} format
        raw_zones = {}
        for name, zone_cfg in zones_data.items():
            pts = zone_cfg if isinstance(zone_cfg, list) else zone_cfg.get('roi_points', [])
            if pts:
                raw_zones[name] = pts

        if not raw_zones:
            return {}

        scaled = scale_zones(raw_zones, annotation_size,
                             self.mux_width, self.mux_height)

        if annotation_size:
            logger.info(
                f"  Zone scale: {annotation_size[0]}x{annotation_size[1]} → "
                f"{self.mux_width}x{self.mux_height}"
            )

        return scaled

    def process_frame_cuda(self, gst_buffer, frame_meta, batch_meta, capture_ts=None, capture_dt=None):
        """
        Process a frame using CUDA kernels directly on GPU NvBufSurface.
        No CPU frame extraction needed.

        Args:
            gst_buffer: GStreamer buffer from pad probe
            frame_meta: NvDsFrameMeta
            batch_meta: NvDsBatchMeta
            capture_ts/capture_dt: resolved capture-clock timestamp (delayed-source
                mode only, see config.HICON_DELAYED_CAPTURE_CLOCK); None (default)
                leaves detectors stamping with their own time.time()/datetime.now().
        """
        try:
            events = []

            if self.tapping_detector:
                events.extend(self.tapping_detector.update(gst_buffer, frame_meta, capture_ts, capture_dt))

            if self.deslagging_detector:
                events.extend(self.deslagging_detector.update(gst_buffer, frame_meta, capture_ts, capture_dt))

            if self.spectro_detector:
                events.extend(self.spectro_detector.update(gst_buffer, frame_meta, capture_ts, capture_dt))

            # Handle events (DB insert, heat cycle, screenshots)
            for event in events:
                if event.get("phase") == "start":
                    self._handle_event_start(event, gst_buffer, frame_meta)
                else:
                    self._handle_event(event, gst_buffer, frame_meta)

            # Draw OSD overlays
            if self.enable_display_meta and batch_meta is not None:
                self.add_inference_display_meta(batch_meta, frame_meta)

        except Exception as e:
            logger.error(f"CudaBrightnessProcessor error: {e}", exc_info=True)

    def add_inference_display_meta(self, batch_meta, frame_meta):
        """Draw zone outlines, status text, and blob boxes on OSD."""
        try:
            # Skip drawing during nodulizer blackout (matches source behavior)
            frame_idx = self.tapping_detector.frame_idx if self.tapping_detector else 0
            if self.shared_state.is_blackout(frame_idx):
                return

            # Draw tapping zones
            if self.tapping_detector:
                for zs in self.tapping_detector.zone_states.values():
                    color = (1.0, 0.5, 0.0, 1.0) if zs.tapping_active else (0.0, 1.0, 1.0, 1.0)
                    draw_zone_polygon(batch_meta, frame_meta, zs.zone, color=color)
                    # White-ratio label above zone
                    x_min, y_min, _, _ = zs.bbox
                    draw_text_label(
                        batch_meta, frame_meta,
                        f"{zs.name}: {zs.last_white_ratio:.2f}",
                        int(x_min), max(0, int(y_min) - 20),
                        color=color, font_size=12,
                    )

            # Draw deslagging zones + blob boxes
            if self.deslagging_detector:
                for zs in self.deslagging_detector.zone_states.values():
                    color = (1.0, 0.5, 0.0, 1.0) if zs.deslagging_active else (0.0, 1.0, 1.0, 1.0)
                    draw_zone_polygon(batch_meta, frame_meta, zs.zone, color=color)
                    for blob in zs.last_blobs:
                        draw_blob_rect(batch_meta, frame_meta, blob,
                                       label=f"A:{blob.area} B:{blob.avg_brightness}")

            # Draw spectro zones + blob boxes
            if self.spectro_detector:
                for zs in self.spectro_detector.zone_states.values():
                    color = (1.0, 0.5, 0.0, 1.0) if zs.spectro_active else (0.0, 1.0, 1.0, 1.0)
                    draw_zone_polygon(batch_meta, frame_meta, zs.zone, color=color)
                    for blob in zs.last_blobs:
                        draw_blob_rect(batch_meta, frame_meta, blob,
                                       label=f"A:{blob.area} B:{blob.avg_brightness}")

            # Status text panel
            self._draw_status_text(batch_meta, frame_meta)

        except Exception as e:
            logger.error(f"CudaBrightnessProcessor OSD error: {e}", exc_info=True)

    def _draw_status_text(self, batch_meta, frame_meta):
        """Draw melting events status text panel."""
        dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not dm:
            return

        labels = []

        # Active events summary
        active_events = []
        if self.tapping_detector and self.tapping_detector.is_active:
            active_events.append("TAPPING")
        if self.deslagging_detector and self.deslagging_detector.is_active:
            active_events.append("DESLAG")
        if self.spectro_detector and self.spectro_detector.is_active:
            active_events.append("SPECTRO")

        labels.append(("MELTING EVENTS", (1.0, 1.0, 1.0, 1.0)))
        labels.append((
            "ACTIVE: " + (", ".join(active_events) if active_events else "NONE"),
            (0.0, 1.0, 0.0, 1.0),
        ))

        # Per-detector status
        if self.tapping_detector:
            active = self.tapping_detector.is_active
            labels.append((
                f"TAPPING: {'ON' if active else 'OFF'}",
                (1.0, 0.65, 0.0, 1.0),
            ))
        if self.deslagging_detector:
            active = self.deslagging_detector.is_active
            labels.append((
                f"DESLAG: {'ON' if active else 'OFF'}",
                (1.0, 0.0, 0.0, 1.0),
            ))
        if self.spectro_detector:
            active = self.spectro_detector.is_active
            labels.append((
                f"SPECTRO: {'ON' if active else 'OFF'}",
                (0.0, 1.0, 1.0, 1.0),
            ))

        base_x = 10
        base_y = 45
        line_h = 18
        dm.num_labels = min(len(labels), 16)
        for i in range(dm.num_labels):
            txt = dm.text_params[i]
            txt.display_text = labels[i][0]
            txt.x_offset = base_x
            txt.y_offset = base_y + i * line_h
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 12
            r, g, b, a = labels[i][1]
            txt.font_params.font_color.set(r, g, b, a)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.55)

        pyds.nvds_add_display_meta_to_frame(frame_meta, dm)

    # ── Event handling (same chain as BrightnessProcessor) ──

    def _handle_event_start(self, event, gst_buffer, frame_meta):
        """Handle event start — take screenshot for tapping starts."""
        event_type = event["type"]
        zone_name = event.get("zone_name", "")
        logger.info(f"[{event_type}] Start detected in zone '{zone_name}': {event['start']}")

        if event_type == "tapping":
            frame_rgba = self._extract_cpu_frame(gst_buffer, frame_meta)
            if frame_rgba is not None:
                self._save_annotated_screenshot(frame_rgba, event, phase="start")

    def _handle_event(self, event, gst_buffer, frame_meta):
        """Handle completed event — DB insert, heat cycle, screenshot."""
        event_type = event["type"]
        zone_name = event.get("zone_name", "")
        logger.info(
            f"[{event_type}] Zone '{zone_name}': {event['start']} -> {event['end']} "
            f"({event['duration_sec']}s)"
        )

        # Screenshot
        frame_rgba = self._extract_cpu_frame(gst_buffer, frame_meta)
        screenshot_path = ""
        if frame_rgba is not None:
            screenshot_path = self._save_annotated_screenshot(
                frame_rgba, event, phase="end"
            )

        # Insert melting event into database
        sync_id = generate_sync_id(event_type)
        try:
            self.db_manager.insert_melting_event(
                sync_id=sync_id,
                customer_id=self.customer_id,
                event_type=event_type,
                start_time=event["start"],
                end_time=event["end"],
                duration_sec=event["duration_sec"],
                camera_id=self.camera_id,
                location=self.location,
                screenshot_path=screenshot_path,
                zone_name=zone_name,
            )
        except Exception as e:
            logger.error(f"Failed to insert {event_type} event: {e}")

        # Push to heat cycle manager
        if self.heat_cycle_manager:
            try:
                add_fn = {
                    "tapping": self.heat_cycle_manager.add_tapping_event,
                    "deslagging": self.heat_cycle_manager.add_deslagging_event,
                    "spectro": self.heat_cycle_manager.add_spectro_event,
                }.get(event_type)
                if add_fn:
                    add_fn(
                        start_wall=event["start_wall"],
                        start_dt=event["start_datetime"],
                        end_wall=event["end_wall"],
                        end_dt=event["end_datetime"],
                        duration=event["duration_sec"],
                        zone_name=zone_name,
                    )
            except Exception as e:
                logger.error(f"Failed to push {event_type} to heat cycle manager: {e}")

    def _extract_cpu_frame(self, gst_buffer, frame_meta):
        """Lazy CPU frame extraction — only called on state transitions."""
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame = np.array(n_frame, copy=True, order='C')
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            return frame
        except Exception as e:
            logger.error(f"CPU frame extraction error: {e}", exc_info=True)
            return None

    def _save_annotated_screenshot(self, frame_rgba, event, phase="end"):
        """Save screenshot with ROI overlay and event details."""
        event_type = event["type"]
        zone_name = event.get("zone_name", "")
        try:
            annotated = prepare_frame(frame_rgba)
            screenshot_dt = event.get(
                "end_datetime" if phase == "end" else "start_datetime"
            ) or datetime.now()

            # Pick zone polygon and color for the specific zone that triggered
            roi_pts = self._get_zone_roi_pts(event_type, zone_name)
            if event_type == "tapping":
                roi_color = (0, 165, 255)  # Orange
            elif event_type == "spectro":
                roi_color = (255, 255, 0)  # Cyan
            else:
                roi_color = (0, 0, 255)  # Red

            if roi_pts:
                draw_roi_overlay(annotated, roi_pts, roi_color,
                                 label=f"{event_type.upper()} {zone_name}", alpha=0.2)

            extra_lines = []
            if phase == "end" and "duration_sec" in event:
                extra_lines.append(f"Duration: {event['duration_sec']}s")
            if phase == "end":
                extra_lines.append(f"Start: {event['start']}  End: {event['end']}")
            else:
                extra_lines.append(f"Start: {event['start']}")
            extra_lines.append(f"Zone: {zone_name}")

            title = f"{event_type.upper()} EVENT {phase.upper()}"
            add_header(annotated, title,
                       screenshot_dt.strftime("%Y-%m-%d %H:%M:%S"),
                       extra_lines)
            add_footer(annotated, self.camera_id)

            return save_screenshot(
                annotated,
                event_type,
                phase,
                screenshot_dt,
                self.screenshot_dir,
                writer=self.screenshot_writer,
            )

        except Exception as e:
            logger.error(f"Error saving {event_type} screenshot: {e}")
            return ""

    def _get_zone_roi_pts(self, event_type: str, zone_name: str):
        """Get the polygon vertices as list-of-pairs for a specific zone."""
        zones_dict = {
            "tapping": self._tapping_zones,
            "deslagging": self._deslagging_zones,
            "spectro": self._spectro_zones,
        }.get(event_type, {})

        poly = zones_dict.get(zone_name)
        if poly is None:
            return []

        # Convert flat tuple to list of [x,y] pairs
        return [[poly[i], poly[i + 1]] for i in range(0, len(poly) - 1, 2)]
