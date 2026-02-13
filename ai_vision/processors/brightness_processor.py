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
from utils.utils import save_screenshot, generate_sync_id

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
                 heat_cycle_manager=None):
        """
        Args:
            zones_config: Dict with tapping/deslagging/spectro zone configs from zones.json
            db_manager: HiConDatabase instance
            config: Configuration module
            screenshot_dir: Path for event screenshots
            heat_cycle_manager: Optional shared HeatCycleManager for tapping/deslagging aggregation
        """
        self.db_manager = db_manager
        self.config = config
        self.heat_cycle_manager = heat_cycle_manager
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.customer_id = config.CUSTOMER_ID
        self.camera_id = config.CAMERA_ID_STREAM_0
        self.location = config.LOCATION

        # Build ROI masks (will be created on first frame when we know dimensions)
        self._tapping_config = zones_config.get('tapping', {})
        self._deslagging_config = zones_config.get('deslagging', {})
        self._spectro_config = zones_config.get('spectro', {})
        self._masks_built = False
        self._frame_shape = None
        self._tapping_mask = None
        self._deslagging_mask = None
        self._spectro_mask = None
        self._tapping_pixel_count = 0
        self._deslagging_pixel_count = 0
        self._spectro_pixel_count = 0

        # State machines
        self.tapping_tracker = BrightnessTracker(
            name="tapping",
            brightness_threshold=self._tapping_config.get('brightness_threshold', 180),
            start_white_ratio=self._tapping_config.get('start_white_ratio', 0.80),
            start_frame_count=self._tapping_config.get('start_frame_count', 10),
            end_white_ratio=self._tapping_config.get('end_white_ratio', 0.60),
            end_frame_count=self._tapping_config.get('end_frame_count', 20),
        )

        self.deslagging_tracker = BrightnessTracker(
            name="deslagging",
            brightness_threshold=self._deslagging_config.get('brightness_threshold', 250),
            start_white_ratio=self._deslagging_config.get('start_white_ratio', 0.01),
            start_frame_count=self._deslagging_config.get('start_frame_count', 10),
            end_white_ratio=self._deslagging_config.get('end_white_ratio', 0.01),
            end_frame_count=self._deslagging_config.get('end_frame_count', 15),
        )

        self.spectro_tracker = BrightnessTracker(
            name="spectro",
            brightness_threshold=self._spectro_config.get('brightness_threshold', 250),
            start_white_ratio=self._spectro_config.get('start_white_ratio', 0.03),
            start_frame_count=self._spectro_config.get('start_frame_count', 10),
            end_white_ratio=self._spectro_config.get('end_white_ratio', 0.03),
            end_frame_count=self._spectro_config.get('end_frame_count', 15),
            max_white_ratio=self._spectro_config.get('max_white_ratio', 0.20),
        )

        logger.info("BrightnessProcessor initialized (tapping + deslagging + spectro)")

    def _build_masks(self, frame_h, frame_w):
        """Build ROI masks once we know frame dimensions."""
        # Tapping quad ROI
        tapping_pts = self._tapping_config.get('roi_points', [])
        if tapping_pts:
            pts = np.array(tapping_pts, dtype=np.int32)
            self._tapping_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(self._tapping_mask, [pts], 255)
            self._tapping_pixel_count = int(np.sum(self._tapping_mask > 0))
            logger.info(f"Tapping ROI mask: {self._tapping_pixel_count} pixels")

        # Deslagging polygon ROI
        deslag_pts = self._deslagging_config.get('roi_points', [])
        if deslag_pts:
            pts = np.array(deslag_pts, dtype=np.int32)
            self._deslagging_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(self._deslagging_mask, [pts], 255)
            self._deslagging_pixel_count = int(np.sum(self._deslagging_mask > 0))
            logger.info(f"Deslagging ROI mask: {self._deslagging_pixel_count} pixels")

        # Spectro polygon ROI
        spectro_pts = self._spectro_config.get('roi_points', [])
        if spectro_pts:
            pts = np.array(spectro_pts, dtype=np.int32)
            self._spectro_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(self._spectro_mask, [pts], 255)
            self._spectro_pixel_count = int(np.sum(self._spectro_mask > 0))
            logger.info(f"Spectro ROI mask: {self._spectro_pixel_count} pixels")

        self._masks_built = True

    def _is_deslagging_suppressed(self):
        """
        Deslagging is suppressed when tapping or pouring cycle is active.
        Molten metal brightness during tapping/pouring causes false deslagging triggers.
        """
        # Suppress during active tapping
        if self.tapping_tracker.is_active:
            return True

        # Suppress during active pouring cycle (trolley locked = pouring in progress)
        if self.heat_cycle_manager and self.heat_cycle_manager.active_cycle:
            if self.heat_cycle_manager.active_cycle.locked_trolley_id is not None:
                return True

        return False

    def process_frame_with_array(self, frame, frame_meta):
        """
        Process a pre-extracted frame for tapping, deslagging, and spectro detection.

        Called from osd_sink_pad probe on Stream 0.
        Frame is already extracted and will be unmapped by the caller.

        Args:
            frame: RGBA numpy array (already extracted via get_nvds_buf_surface)
            frame_meta: NvDsFrameMeta
        """
        try:
            # Build masks on first frame
            if not self._masks_built:
                self._build_masks(frame.shape[0], frame.shape[1])

            # Convert RGBA to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

            # Process tapping zone
            if self._tapping_mask is not None and self._tapping_pixel_count > 0:
                self._process_zone(
                    gray, self._tapping_mask, self._tapping_pixel_count,
                    self.tapping_tracker, frame
                )

            # Process deslagging zone (suppressed during tapping or active pouring cycle)
            if self._deslagging_mask is not None and self._deslagging_pixel_count > 0:
                if self._is_deslagging_suppressed():
                    # Reset tracker counters so partial counts don't carry over
                    self.deslagging_tracker.start_counter = 0
                    self.deslagging_tracker.end_counter = 0
                else:
                    self._process_zone(
                        gray, self._deslagging_mask, self._deslagging_pixel_count,
                        self.deslagging_tracker, frame
                    )

            # Process spectro zone (suppressed during tapping or active pouring cycle)
            if self._spectro_mask is not None and self._spectro_pixel_count > 0:
                if self._is_deslagging_suppressed():
                    self.spectro_tracker.start_counter = 0
                    self.spectro_tracker.end_counter = 0
                else:
                    self._process_zone(
                        gray, self._spectro_mask, self._spectro_pixel_count,
                        self.spectro_tracker, frame
                    )

        except Exception as e:
            logger.error(f"BrightnessProcessor error: {e}", exc_info=True)

    def _process_zone(self, gray, mask, pixel_count, tracker, frame_rgba):
        """Process a single brightness zone."""
        threshold = tracker.brightness_threshold

        # Threshold: white pixels where Y > threshold within ROI
        white_pixels = np.sum((gray > threshold) & (mask > 0))
        white_ratio = white_pixels / pixel_count if pixel_count > 0 else 0.0

        # Update state machine
        event = tracker.update(white_ratio)

        if event:
            self._handle_event(event, frame_rgba, white_ratio)

    def _handle_event(self, event, frame_rgba, white_ratio=0.0):
        """Handle a completed tapping/deslagging/spectro event."""
        event_type = event["type"]
        logger.info(
            f"[{event_type}] Event: {event['start']} -> {event['end']} "
            f"({event['duration_sec']}s)"
        )

        # Save annotated screenshot with ROI regions and event details
        screenshot_path = self._save_annotated_screenshot(
            frame_rgba, event, white_ratio
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
                    )
                elif event_type == "deslagging":
                    self.heat_cycle_manager.add_deslagging_event(
                        start_wall=event["start_wall"],
                        start_dt=event["start_datetime"],
                        end_wall=event["end_wall"],
                        end_dt=event["end_datetime"],
                        duration=event["duration_sec"],
                    )
                elif event_type == "spectro":
                    self.heat_cycle_manager.add_spectro_event(
                        start_wall=event["start_wall"],
                        start_dt=event["start_datetime"],
                        end_wall=event["end_wall"],
                        end_dt=event["end_datetime"],
                        duration=event["duration_sec"],
                    )
            except Exception as e:
                logger.error(f"Failed to push {event_type} to heat cycle manager: {e}")

    def _save_annotated_screenshot(self, frame_rgba, event, white_ratio):
        """Save screenshot with ROI region overlay, event details, and annotations."""
        try:
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            annotated = frame_bgr.copy()
            h, w = annotated.shape[:2]
            event_type = event["type"]

            # Pick ROI config and color per event type
            if event_type == "tapping":
                roi_pts = self._tapping_config.get('roi_points', [])
                roi_color = (0, 165, 255)  # Orange
                threshold = self.tapping_tracker.brightness_threshold
            elif event_type == "spectro":
                roi_pts = self._spectro_config.get('roi_points', [])
                roi_color = (255, 255, 0)  # Cyan
                threshold = self.spectro_tracker.brightness_threshold
            else:
                roi_pts = self._deslagging_config.get('roi_points', [])
                roi_color = (0, 0, 255)  # Red
                threshold = self.deslagging_tracker.brightness_threshold

            # Draw ROI region with semi-transparent fill + outline
            if roi_pts:
                pts = np.array(roi_pts, dtype=np.int32)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], roi_color)
                cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
                cv2.polylines(annotated, [pts], True, roi_color, 2)
                # Label inside ROI
                cx = int(np.mean([p[0] for p in roi_pts]))
                cy = int(np.mean([p[1] for p in roi_pts]))
                cv2.putText(annotated, f"{event_type.upper()} ROI",
                           (cx - 60, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

            # Event title bar
            cv2.putText(annotated, f"{event_type.upper()} EVENT END", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Duration
            cv2.putText(annotated, f"Duration: {event['duration_sec']}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Start/end times
            cv2.putText(annotated,
                       f"Start: {event['start']}  End: {event['end']}",
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Brightness analysis info
            cv2.putText(annotated,
                       f"Threshold: Y>{threshold}  White ratio: {white_ratio:.3f}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Camera ID
            cv2.putText(annotated, f"CAM: {self.camera_id}", (w - 200, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{event_type}_end_{timestamp}.jpg"
            filepath = self.screenshot_dir / filename
            cv2.imwrite(str(filepath), annotated)
            logger.info(f"Saved {event_type} screenshot: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving {event_type} screenshot: {e}")
            return ""
