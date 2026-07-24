"""
Brightness Processor - Tapping/deslagging/spectro detection via CPU brightness analysis.

Runs as a pad probe on Stream 0 (Process Camera) after OSD.
Uses pyds.get_nvds_buf_surface() → NumPy on CPU (CuPy NOT available on Jetson DeepStream).
CRITICAL: Always call unmap_nvds_buf_surface() after get_nvds_buf_surface() on Jetson.
"""
import logging
import time
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

import pyds

from state.brightness_tracker import BrightnessTracker
from utils.utils import generate_sync_id
from utils.screenshot import (prepare_frame, add_header, add_footer,
                               draw_roi_overlay, save as save_screenshot)

logger = logging.getLogger(__name__)


class BrightnessProcessor:
    """
    Detect tapping, deslagging, and spectro events via brightness analysis in ROI zones.

    Algorithm per zone:
    1. get_nvds_buf_surface() → RGBA frame on CPU
    2. Convert to grayscale
    3. Crop to ROI mask
    4. Threshold: pixels with Y > brightness_threshold → white
    5. Compute white_ratio = count(white) / count(roi_pixels)
    6. Feed white_ratio into BrightnessTracker state machine
    7. unmap_nvds_buf_surface() — MANDATORY on Jetson
    """

    def __init__(self, zones_config, db_manager, config, screenshot_dir,
                 heat_cycle_manager=None, enable_display_meta=True,
                 enable_tapping=True, enable_deslagging=True, enable_spectro=True,
                 screenshot_writer=None, camera_id_override=None):
        """
        Args:
            zones_config: Dict with tapping/deslagging/spectro zone configs from zones.json
            db_manager: HiConDatabase instance
            config: Configuration module
            screenshot_dir: Path for event screenshots
            heat_cycle_manager: Optional shared HeatCycleManager for tapping/deslagging aggregation
            enable_display_meta: Whether to attach CPU-generated live display meta
        """
        self.db_manager = db_manager
        self.config = config
        self.heat_cycle_manager = heat_cycle_manager
        self.enable_display_meta = enable_display_meta
        self.enable_tapping = enable_tapping
        self.enable_deslagging = enable_deslagging
        self.enable_spectro = enable_spectro
        self.screenshot_writer = screenshot_writer
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.customer_id = config.CUSTOMER_ID
        self.camera_id = camera_id_override or config.CAMERA_ID_STREAM_0
        self.location = config.LOCATION

        # Build ROI masks (will be created on first frame when we know dimensions)
        self._tapping_config = zones_config.get('tapping', {})
        self._deslagging_config = zones_config.get('deslagging', {})
        self._spectro_config = zones_config.get('spectro', {})
        # Zone name per event type (e.g. "tap-2"), used to tag emitted events so
        # heat_cycle_manager can infer Furnace1/Furnace2 — see MeltingAnalysisController's
        # _ordered_zone_names for the same pattern on the CUDA path.
        self._zone_name_by_event_type = {
            "tapping": self._first_zone_name(self._tapping_config),
            "deslagging": self._first_zone_name(self._deslagging_config),
            "spectro": self._first_zone_name(self._spectro_config),
        }
        self._masks_built = False
        self._frame_shape = None
        self._tapping_mask = None
        self._deslagging_mask = None
        self._spectro_mask = None
        self._tapping_bbox = None
        self._deslagging_bbox = None
        self._spectro_bbox = None
        self._tapping_pixel_count = 0
        self._deslagging_pixel_count = 0
        self._spectro_pixel_count = 0

        # Coordinate scaling: zones.json ROI points are calibrated at ref_width x ref_height.
        # Scale factors are computed on first frame so overlays and masks match the actual
        # mux output resolution (e.g., if main stream differs from calibration resolution).
        meta = zones_config.get('metadata', {})
        self._ref_w = int(meta.get('ref_width', 1280))
        self._ref_h = int(meta.get('ref_height', 720))
        self._last_white_ratios = {
            "tapping": 0.0,
            "deslagging": 0.0,
            "spectro": 0.0,
        }

        # State machines
        self.tapping_tracker = BrightnessTracker(
            name="tapping",
            brightness_threshold=self._tapping_config.get(
                'abs_brightness_threshold',
                self._tapping_config.get('brightness_threshold', 210),
            ),
            start_white_ratio=self._tapping_config.get('start_white_ratio', 0.25),
            start_frame_count=self._tapping_config.get('start_frame_count', 20),
            end_white_ratio=self._tapping_config.get('end_white_ratio', 0.10),
            end_frame_count=self._tapping_config.get('end_frame_count', 25),
        )

        self.deslagging_tracker = BrightnessTracker(
            name="deslagging",
            brightness_threshold=self._deslagging_config.get(
                'brightness_threshold',
                self._deslagging_config.get('brightness_thresh', 250),
            ),
            start_white_ratio=self._deslagging_config.get('start_white_ratio', 0.01),
            start_frame_count=self._deslagging_config.get('start_frame_count', 10),
            end_white_ratio=self._deslagging_config.get('end_white_ratio', 0.01),
            end_frame_count=self._deslagging_config.get('end_frame_count', 15),
        )

        self.spectro_tracker = BrightnessTracker(
            name="spectro",
            brightness_threshold=self._spectro_config.get(
                'brightness_threshold',
                self._spectro_config.get('brightness_thresh', 250),
            ),
            start_white_ratio=self._spectro_config.get('start_white_ratio', 0.03),
            start_frame_count=self._spectro_config.get('start_frame_count', 10),
            end_white_ratio=self._spectro_config.get('end_white_ratio', 0.03),
            end_frame_count=self._spectro_config.get('end_frame_count', 15),
            max_white_ratio=self._spectro_config.get('max_white_ratio', 0.20),
        )

        enabled_detectors = []
        if self.enable_tapping:
            enabled_detectors.append("tapping")
        if self.enable_deslagging:
            enabled_detectors.append("deslagging")
        if self.enable_spectro:
            enabled_detectors.append("spectro")
        enabled_label = " + ".join(enabled_detectors) if enabled_detectors else "none"
        logger.info("BrightnessProcessor initialized (%s)", enabled_label)

    @staticmethod
    def _first_zone_name(cfg):
        """Return the single configured zone key for an event type, e.g. 'tap-2'."""
        zones = cfg.get('zones', {})
        if zones:
            return next(iter(zones))
        if cfg.get('roi_points'):
            return "zone-1"
        return ""

    def _scale_pts(self, pts, frame_w, frame_h, ref_w, ref_h):
        """Scale zone coordinates from calibration resolution to actual frame resolution."""
        sx = frame_w / float(ref_w) if ref_w > 0 else 1.0
        sy = frame_h / float(ref_h) if ref_h > 0 else 1.0
        if sx == 1.0 and sy == 1.0:
            return pts
        return [[int(round(x * sx)), int(round(y * sy))] for x, y in pts]

    def _get_zone_pts_list(self, cfg, frame_w, frame_h):
        """Return list of polygon arrays from config, handling both flat and multi-zone formats.

        Uses 'annotation_size' from config if present, else falls back to global ref_width/ref_height.
        """
        ref_size = cfg.get('annotation_size')
        if ref_size and len(ref_size) == 2:
            ref_w, ref_h = ref_size
        else:
            ref_w, ref_h = self._ref_w, self._ref_h

        pts_list = []
        zones = cfg.get('zones', {})
        if zones:
            for zone in zones.values():
                if not zone.get('enabled', True):
                    continue
                pts = self._scale_pts(zone.get('roi_points', []), frame_w, frame_h, ref_w, ref_h)
                if pts:
                    pts_list.append(np.array(pts, dtype=np.int32))
        else:
            pts = self._scale_pts(cfg.get('roi_points', []), frame_w, frame_h, ref_w, ref_h)
            if pts:
                pts_list.append(np.array(pts, dtype=np.int32))
        return pts_list

    def _build_masks(self, frame_h, frame_w):
        """Build ROI masks once we know frame dimensions."""
        self._frame_shape = (frame_h, frame_w)
        logger.info(f"Building ROI masks for {frame_w}x{frame_h} frame")

        # Tapping ROI (flat or multi-zone)
        tapping_polys = self._get_zone_pts_list(self._tapping_config, frame_w, frame_h)
        if tapping_polys:
            self._tapping_mask, self._tapping_bbox = self._build_cropped_mask(tapping_polys)
            self._tapping_pixel_count = int(np.sum(self._tapping_mask > 0))
            logger.info(f"Tapping ROI mask: {len(tapping_polys)} zone(s), {self._tapping_pixel_count} pixels, bbox={self._tapping_bbox}")

        # Deslagging ROI (flat or multi-zone)
        deslag_polys = self._get_zone_pts_list(self._deslagging_config, frame_w, frame_h)
        if deslag_polys:
            self._deslagging_mask, self._deslagging_bbox = self._build_cropped_mask(deslag_polys)
            self._deslagging_pixel_count = int(np.sum(self._deslagging_mask > 0))
            logger.info(f"Deslagging ROI mask: {len(deslag_polys)} zone(s), {self._deslagging_pixel_count} pixels")

        # Spectro ROI (flat or multi-zone)
        spectro_polys = self._get_zone_pts_list(self._spectro_config, frame_w, frame_h)
        if spectro_polys:
            self._spectro_mask, self._spectro_bbox = self._build_cropped_mask(spectro_polys)
            self._spectro_pixel_count = int(np.sum(self._spectro_mask > 0))
            logger.info(f"Spectro ROI mask: {len(spectro_polys)} zone(s), {self._spectro_pixel_count} pixels")

        self._masks_built = True

    @staticmethod
    def _build_cropped_mask(polygons):
        """Return an ROI-sized mask and its frame-space bounding rectangle."""
        all_points = np.concatenate(polygons, axis=0)
        x, y, width, height = cv2.boundingRect(all_points)
        shifted = [poly - np.array([x, y], dtype=np.int32) for poly in polygons]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, shifted, 255)
        return mask, (x, y, x + width, y + height)

    @staticmethod
    def _crop_to_bbox(gray, bbox):
        if bbox is None:
            return gray
        x1, y1, x2, y2 = bbox
        return gray[y1:y2, x1:x2]

    def _is_deslagging_suppressed(self):
        """
        Deslagging is suppressed when tapping or pouring cycle is active.
        Molten metal brightness during tapping/pouring causes false deslagging triggers.
        """
        # Suppress during active tapping
        if self.tapping_tracker.is_active or self.tapping_tracker.start_counter > 0:
            return True

        # Suppress during active pouring cycle (trolley locked = pouring in progress)
        if self.heat_cycle_manager and self.heat_cycle_manager.active_cycle:
            if self.heat_cycle_manager.active_cycle.locked_trolley_id is not None:
                return True

        return False

    def process_frame_with_array(self, frame, frame_meta, capture_ts=None, capture_dt=None):
        """
        Process a pre-extracted frame for tapping, deslagging, and spectro detection.

        Called from osd_sink_pad probe on Stream 0 (non-decoupled) or analysis branch probe
        (decoupled mode).  Accepts either RGBA (shape H×W×4) or NV12 (shape H*3/2×W).
        Frame is already extracted and will be unmapped by the caller.

        Args:
            frame: numpy array from get_nvds_buf_surface — RGBA (H,W,4) or NV12 (H*3/2,W)
            frame_meta: NvDsFrameMeta
            capture_ts/capture_dt: resolved capture-clock timestamp (delayed-source
                mode only, see config.HICON_DELAYED_CAPTURE_CLOCK). None (default)
                leaves trackers stamping with their own time.time()/datetime.now().
        """
        try:
            # Detect pixel format and derive actual frame height/width.
            # NV12 semi-planar: shape is (H*3//2, W) — Y plane in rows [0:H].
            # RGBA: shape is (H, W, 4).
            if frame.ndim == 2:
                # NV12: Y plane (luma) is the first 2/3 of rows; use it directly as grayscale.
                frame_h = frame.shape[0] * 2 // 3
                frame_w = frame.shape[1]
                gray = frame[:frame_h, :]
                # For screenshots: wrap Y-plane as 4-channel grayscale (prepare_frame expects RGBA).
                frame_for_ss = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
            else:
                frame_h = frame.shape[0]
                frame_w = frame.shape[1]
                gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
                frame_for_ss = frame

            # Build masks on first frame
            if not self._masks_built:
                self._build_masks(frame_h, frame_w)

            # Process tapping zone
            if self.enable_tapping and self._tapping_mask is not None and self._tapping_pixel_count > 0:
                self._process_zone(
                    self._crop_to_bbox(gray, self._tapping_bbox),
                    self._tapping_mask, self._tapping_pixel_count,
                    self.tapping_tracker, frame_for_ss, capture_ts, capture_dt
                )

            # Process deslagging zone (suppressed during tapping or active pouring cycle)
            if self.enable_deslagging and self._deslagging_mask is not None and self._deslagging_pixel_count > 0:
                if self._is_deslagging_suppressed():
                    # Reset tracker counters so partial counts don't carry over
                    self.deslagging_tracker.start_counter = 0
                    self.deslagging_tracker.end_counter = 0
                else:
                    # Check if we should use blob detection for deslagging
                    if "min_blob_area" in self._deslagging_config:
                        self._process_zone_blobs(
                            self._crop_to_bbox(gray, self._deslagging_bbox),
                            self._deslagging_mask, self._deslagging_pixel_count,
                            self.deslagging_tracker, self._deslagging_config, frame_for_ss,
                            capture_ts, capture_dt
                        )
                    else:
                        self._process_zone(
                            self._crop_to_bbox(gray, self._deslagging_bbox),
                            self._deslagging_mask, self._deslagging_pixel_count,
                            self.deslagging_tracker, frame_for_ss, capture_ts, capture_dt
                        )

            # Process spectro zone (suppressed during tapping or active pouring cycle)
            if self.enable_spectro and self._spectro_mask is not None and self._spectro_pixel_count > 0:
                if self._is_deslagging_suppressed():
                    self.spectro_tracker.start_counter = 0
                    self.spectro_tracker.end_counter = 0
                else:
                    # Check if we should use blob detection for spectro
                    if "min_blob_area" in self._spectro_config:
                        self._process_zone_blobs(
                            self._crop_to_bbox(gray, self._spectro_bbox),
                            self._spectro_mask, self._spectro_pixel_count,
                            self.spectro_tracker, self._spectro_config, frame_for_ss,
                            capture_ts, capture_dt
                        )
                    else:
                        self._process_zone(
                            self._crop_to_bbox(gray, self._spectro_bbox),
                            self._spectro_mask, self._spectro_pixel_count,
                            self.spectro_tracker, frame_for_ss, capture_ts, capture_dt
                        )

        except Exception as e:
            logger.error(f"BrightnessProcessor error: {e}", exc_info=True)

    def _process_zone_blobs(self, gray, mask, pixel_count, tracker, zone_cfg, frame_rgba,
                             capture_ts=None, capture_dt=None):
        """Process a single zone using molten blob logic (contours)."""
        threshold = tracker.brightness_threshold
        min_area = zone_cfg.get("min_blob_area", 50)
        max_ar = zone_cfg.get("max_aspect_ratio", 0.0)
        max_cov = zone_cfg.get("max_coverage", 0.0)

        # 1. Threshold within ROI
        # We need a copy of the ROI area or apply threshold on white-on-black mask
        # Optimization: use cv2.bitwise_and if gray is not already masked
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        # Apply mask
        thresh = cv2.bitwise_and(thresh, mask)

        # 2. Find Contours (Blobs)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_blobs = []
        max_blob_area = 0.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # Aspect Ratio check
            if max_ar > 0.0:
                x, y, w, h = cv2.boundingRect(cnt)
                ar = max(w, h) / max(1.0, min(w, h))
                if ar > max_ar:
                    continue

            # Coverage check (blob area / zone area)
            if max_cov > 0.0:
                coverage = area / pixel_count if pixel_count > 0 else 0.0
                if coverage > max_cov:
                    continue

            # Valid blob found
            valid_blobs.append(cnt)
            max_blob_area = max(max_blob_area, area)

        has_valid_blobs = len(valid_blobs) > 0
        
        # update white ratio for display purposes even if using blob logic
        white_pixels = np.sum(thresh > 0)
        white_ratio = white_pixels / pixel_count if pixel_count > 0 else 0.0
        self._last_white_ratios[tracker.name] = white_ratio

        # 3. Update Tracker
        event = tracker.update_blob_logic(has_valid_blobs, capture_ts, capture_dt)
        if event:
            event.setdefault("zone_name", self._zone_name_by_event_type.get(tracker.name, ""))
            if event.get("phase") == "start":
                self._handle_event_start(event, frame_rgba, white_ratio)
            else:
                self._handle_event(event, frame_rgba, white_ratio)

    def _process_zone(self, gray, mask, pixel_count, tracker, frame_rgba,
                       capture_ts=None, capture_dt=None):
        """Process a single brightness zone."""
        threshold = tracker.brightness_threshold

        # Threshold: white pixels where Y > threshold within ROI
        white_pixels = np.sum((gray > threshold) & (mask > 0))
        white_ratio = white_pixels / pixel_count if pixel_count > 0 else 0.0
        self._last_white_ratios[tracker.name] = white_ratio

        # Periodic tapping diagnostic log (~10s interval at 25fps; DEBUG — see
        # Edge_Optimization_Plan.md Phase 0.7, event start/end stay at INFO)
        if tracker.name == "tapping":
            if not hasattr(self, '_tap_log_counter'):
                self._tap_log_counter = 0
            self._tap_log_counter += 1
            if self._tap_log_counter >= 250:
                self._tap_log_counter = 0
                logger.debug(
                    "[tapping] ratio=%.3f (need>=%.2f) thresh=Y>%d on=%d/%d state=%s",
                    white_ratio,
                    tracker.start_white_ratio,
                    tracker.brightness_threshold,
                    tracker.start_counter,
                    tracker.start_frame_count,
                    "ACTIVE" if tracker.is_active else "IDLE",
                )

        # Update state machine
        event = tracker.update(white_ratio, capture_ts, capture_dt)

        if event:
            event.setdefault("zone_name", self._zone_name_by_event_type.get(tracker.name, ""))
            if event.get("phase") == "start":
                self._handle_event_start(event, frame_rgba, white_ratio)
            else:
                self._handle_event(event, frame_rgba, white_ratio)

    def add_inference_display_meta(self, batch_meta, frame_meta, force: bool = False):
        """Attach DS-native overlay for tapping/deslagging/spectro status + ROI bounds.

        force=True bypasses the enable_display_meta guard — used by the decoupled-mode
        recording display_meta writer probe which runs on the main path.
        """
        if not force and not self.enable_display_meta:
            return
        display_meta = None
        try:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                return
            display_meta.num_labels = 0
            display_meta.num_lines = 0
            display_meta.num_rects = 0
            display_meta.num_circles = 0

            # Scale overlay text when recording is downscaled
            scale_up = 1.0
            try:
                target_w = int(getattr(self.config, "INFERENCE_VIDEO_WIDTH", 0) or 0)
                if self._frame_shape and target_w and target_w < self._frame_shape[1]:
                    scale_up = min(3.0, self._frame_shape[1] / float(target_w))
            except Exception:
                scale_up = 1.0

            # Text overlays
            labels = []
            header = "MELTING EVENTS"
            labels.append((header, (1.0, 1.0, 1.0, 1.0)))

            active_events = []
            if self.tapping_tracker.is_active:
                active_events.append("TAPPING")
            if self.deslagging_tracker.is_active:
                active_events.append("DESLAG")
            if self.spectro_tracker.is_active:
                active_events.append("SPECTRO")
            active_txt = "ACTIVE: " + (", ".join(active_events) if active_events else "NONE")
            labels.append((active_txt, (0.0, 1.0, 0.0, 1.0)))

            def _status_line(name, active, ratio):
                state = "ON" if active else "OFF"
                return f"{name}: {state}  ratio={ratio:.3f}"

            labels.append((
                _status_line("TAPPING", self.tapping_tracker.is_active,
                             self._last_white_ratios.get("tapping", 0.0)),
                (1.0, 0.65, 0.0, 1.0),
            ))
            labels.append((
                _status_line("DESLAG", self.deslagging_tracker.is_active,
                             self._last_white_ratios.get("deslagging", 0.0)),
                (1.0, 0.0, 0.0, 1.0),
            ))
            labels.append((
                _status_line("SPECTRO", self.spectro_tracker.is_active,
                             self._last_white_ratios.get("spectro", 0.0)),
                (0.0, 1.0, 1.0, 1.0),
            ))

            base_x = 10
            base_y = max(45, int(round(45 * scale_up)))
            line_h = max(18, int(round(18 * scale_up)))
            display_meta.num_labels = min(len(labels), len(display_meta.text_params))
            for i in range(display_meta.num_labels):
                txt = display_meta.text_params[i]
                txt.display_text = labels[i][0]
                txt.x_offset = base_x
                txt.y_offset = base_y + i * line_h
                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = max(12, int(round(12 * scale_up)))
                r, g, b, a = labels[i][1]
                txt.font_params.font_color.set(r, g, b, a)
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.55)

            # ROI polygon lines (tilted to match screenshots)
            line_idx = 0
            max_lines = len(getattr(display_meta, "line_params", []))

            def _add_roi_poly(roi_pts, color):
                nonlocal line_idx
                if not roi_pts or max_lines == 0:
                    return
                n = len(roi_pts)
                if n < 2:
                    return
                for i in range(n):
                    if line_idx >= max_lines:
                        break
                    x1, y1 = roi_pts[i]
                    x2, y2 = roi_pts[(i + 1) % n]
                    line = display_meta.line_params[line_idx]
                    line.x1 = int(max(0, x1))
                    line.y1 = int(max(0, y1))
                    line.x2 = int(max(0, x2))
                    line.y2 = int(max(0, y2))
                    line.line_width = max(2, int(round(2 * scale_up)))
                    line.line_color.set(*color)
                    line_idx += 1

            if max_lines > 0:
                f_h, f_w = self._frame_shape if self._frame_shape else (1080, 1920)
                for poly in self._get_zone_pts_list(self._tapping_config, f_w, f_h):
                    _add_roi_poly(poly.tolist(), (1.0, 0.65, 0.0, 1.0))
                for poly in self._get_zone_pts_list(self._deslagging_config, f_w, f_h):
                    _add_roi_poly(poly.tolist(), (1.0, 0.0, 0.0, 1.0))
                for poly in self._get_zone_pts_list(self._spectro_config, f_w, f_h):
                    _add_roi_poly(poly.tolist(), (0.0, 1.0, 1.0, 1.0))
                display_meta.num_lines = line_idx
                display_meta.num_rects = 0
            else:
                # Fallback: ROI bounding rectangles if line params unavailable
                rect_idx = 0
                max_rects = len(display_meta.rect_params)
                f_h, f_w = self._frame_shape if self._frame_shape else (1080, 1920)

                def _add_roi_rect(roi_pts, color):
                    nonlocal rect_idx
                    if not roi_pts or rect_idx >= max_rects:
                        return
                    xs = [p[0] for p in roi_pts]
                    ys = [p[1] for p in roi_pts]
                    if not xs or not ys:
                        return
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    rect = display_meta.rect_params[rect_idx]
                    rect.left = int(max(0, x1))
                    rect.top = int(max(0, y1))
                    rect.width = int(max(1, x2 - x1))
                    rect.height = int(max(1, y2 - y1))
                    rect.border_width = max(2, int(round(2 * scale_up)))
                    rect.has_bg_color = 0
                    rect.border_color.set(*color)
                    rect_idx += 1

                for poly in self._get_zone_pts_list(self._tapping_config, f_w, f_h):
                    _add_roi_rect(poly.tolist(), (1.0, 0.65, 0.0, 1.0))
                for poly in self._get_zone_pts_list(self._deslagging_config, f_w, f_h):
                    _add_roi_rect(poly.tolist(), (1.0, 0.0, 0.0, 1.0))
                for poly in self._get_zone_pts_list(self._spectro_config, f_w, f_h):
                    _add_roi_rect(poly.tolist(), (0.0, 1.0, 1.0, 1.0))

                display_meta.num_rects = rect_idx
        except Exception as exc:
            logger.error(f"[osd] Failed to attach brightness display meta: {exc}", exc_info=True)
        finally:
            if display_meta is not None:
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    def draw_cpu_overlay(self, frame_bgr: np.ndarray) -> None:
        """Draw tapping/deslagging/spectro status and ROI polygons onto a BGR frame.

        Called from the MJPEG streaming probe as a CPU replacement for nvosd GPU rendering.
        Reads cached state — safe to call from any thread after analysis branch has run.
        """
        if not self.enable_display_meta:
            return
        try:
            h, w = frame_bgr.shape[:2]

            # Left-side status panel
            active_names = [
                n for n, t in [
                    ("TAPPING", self.tapping_tracker.is_active),
                    ("DESLAG", self.deslagging_tracker.is_active),
                    ("SPECTRO", self.spectro_tracker.is_active),
                ] if t
            ]
            lines = [
                ("MELTING EVENTS", (255, 255, 255)),
                ("ACTIVE: " + (", ".join(active_names) if active_names else "NONE"), (0, 255, 0)),
                (
                    f"TAPPING: {'ON' if self.tapping_tracker.is_active else 'OFF'}"
                    f"  ratio={self._last_white_ratios.get('tapping', 0.0):.3f}",
                    (0, 165, 255),
                ),
                (
                    f"DESLAG:  {'ON' if self.deslagging_tracker.is_active else 'OFF'}"
                    f"  ratio={self._last_white_ratios.get('deslagging', 0.0):.3f}",
                    (0, 0, 255),
                ),
                (
                    f"SPECTRO: {'ON' if self.spectro_tracker.is_active else 'OFF'}"
                    f"  ratio={self._last_white_ratios.get('spectro', 0.0):.3f}",
                    (255, 255, 0),
                ),
            ]
            y = 30
            for text, color in lines:
                tw = len(text) * 9
                cv2.rectangle(frame_bgr, (5, y - 16), (5 + tw, y + 4), (0, 0, 0), -1)
                cv2.putText(frame_bgr, text, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                y += 22

            # ROI polygon outlines
            for polys, color in [
                (self._get_zone_pts_list(self._tapping_config, w, h), (0, 165, 255)),
                (self._get_zone_pts_list(self._deslagging_config, w, h), (0, 0, 255)),
                (self._get_zone_pts_list(self._spectro_config, w, h), (255, 255, 0)),
            ]:
                for pts in polys:
                    cv2.polylines(frame_bgr, [pts.reshape(-1, 1, 2)], True, color, 2)
        except Exception as exc:
            logger.debug("[cpu-overlay] brightness draw error: %s", exc)

    def _handle_event_start(self, event, frame_rgba, white_ratio=0.0):
        """Handle tapping start screenshot."""
        event_type = event["type"]
        if event_type != "tapping":
            return
        logger.info(f"[{event_type}] Start detected: {event['start']}")
        self._save_annotated_screenshot(
            frame_rgba, event, white_ratio, phase="start"
        )

    def _handle_event(self, event, frame_rgba, white_ratio=0.0):
        """Handle a completed tapping/deslagging/spectro event."""
        event_type = event["type"]
        logger.info(
            f"[{event_type}] Event: {event['start']} -> {event['end']} "
            f"({event['duration_sec']}s)"
        )

        # Save annotated screenshot with ROI regions and event details
        screenshot_path = self._save_annotated_screenshot(
            frame_rgba, event, white_ratio, phase="end"
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
                zone_name=event.get("zone_name", ""),
            )
        except Exception as e:
            logger.error(f"Failed to insert {event_type} event: {e}")

        # Push to heat cycle manager for aggregation
        if self.heat_cycle_manager:
            try:
                if event_type == "tapping":
                    self.heat_cycle_manager.add_tapping_event(
                        start_wall=event["start_wall"],
                        start_dt=event["start_datetime"],
                        end_wall=event["end_wall"],
                        end_dt=event["end_datetime"],
                        duration=event["duration_sec"],
                        zone_name=event.get("zone_name"),
                    )
                elif event_type == "deslagging":
                    self.heat_cycle_manager.add_deslagging_event(
                        start_wall=event["start_wall"],
                        start_dt=event["start_datetime"],
                        end_wall=event["end_wall"],
                        end_dt=event["end_datetime"],
                        duration=event["duration_sec"],
                        zone_name=event.get("zone_name"),
                    )
                elif event_type == "spectro":
                    self.heat_cycle_manager.add_spectro_event(
                        start_wall=event["start_wall"],
                        start_dt=event["start_datetime"],
                        end_wall=event["end_wall"],
                        end_dt=event["end_datetime"],
                        duration=event["duration_sec"],
                        zone_name=event.get("zone_name"),
                    )
            except Exception as e:
                logger.error(f"Failed to push {event_type} to heat cycle manager: {e}")

    def _save_annotated_screenshot(self, frame_rgba, event, white_ratio, phase="end"):
        """Save screenshot with ROI region overlay, event details, and annotations."""
        event_type = event["type"]
        try:
            annotated = prepare_frame(frame_rgba)
            phase = phase or event.get("phase", "end")
            screenshot_dt = event.get(
                "end_datetime" if phase == "end" else "start_datetime"
            ) or datetime.now()

            # Pick ROI config and color per event type (coordinates scaled to frame resolution)
            f_h, f_w = frame_rgba.shape[:2]
            if event_type == "tapping":
                roi_pts_list = self._get_zone_pts_list(self._tapping_config, f_w, f_h)
                roi_color = (0, 165, 255)  # Orange
                threshold = self.tapping_tracker.brightness_threshold
            elif event_type == "spectro":
                roi_pts_list = self._get_zone_pts_list(self._spectro_config, f_w, f_h)
                roi_color = (255, 255, 0)  # Cyan
                threshold = self.spectro_tracker.brightness_threshold
            else:
                roi_pts_list = self._get_zone_pts_list(self._deslagging_config, f_w, f_h)
                roi_color = (0, 0, 255)  # Red
                threshold = self.deslagging_tracker.brightness_threshold

            # Draw ROI regions with semi-transparent fill + outline (one per zone)
            for roi_pts in roi_pts_list:
                draw_roi_overlay(annotated, roi_pts, roi_color,
                                 label=f"{event_type.upper()} ROI", alpha=0.2)

            # Build extra info lines
            extra_lines = []
            if phase == "end" and "duration_sec" in event:
                extra_lines.append(f"Duration: {event['duration_sec']}s")
            if phase == "end":
                extra_lines.append(f"Start: {event['start']}  End: {event['end']}")
            else:
                extra_lines.append(f"Start: {event['start']}")
            extra_lines.append(f"Threshold: Y>{threshold}  White ratio: {white_ratio:.3f}")

            # Standard header/footer
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
