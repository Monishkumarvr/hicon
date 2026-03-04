"""
Pyrometer Processor - Rod insertion detection on Stream 1 (Pyrometer Camera).

Reads NvDsObjectMeta from nvinfer (YOLO26 custom parser).
Applies zone check + temporal frame counting for event start/end.
"""
import logging
import time
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path

import pyds

from utils.utils import generate_sync_id
from utils.screenshot import (prepare_frame, add_header, add_footer,
                               draw_roi_overlay, save as save_screenshot)

logger = logging.getLogger(__name__)


class PyrometerProcessor:
    """
    Detect pyrometer rod insertion events using nvinfer detections + zone filtering.

    Algorithm:
    1. Filter detections: confidence >= threshold
    2. Zone check: bbox top-left AND bottom-center must be inside polygon
    3. Temporal: N consecutive in-zone frames → EVENT START
                 N consecutive absent frames → EVENT END
    """

    def __init__(self, zone_config, db_manager, config, screenshot_dir, heat_cycle_manager=None):
        """
        Args:
            zone_config: Pyrometer zone config from zones.json
            db_manager: HiConDatabase instance
            config: Configuration module
            screenshot_dir: Path for event screenshots
        """
        self.db_manager = db_manager
        self.config = config
        self.heat_cycle_manager = heat_cycle_manager
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.customer_id = config.CUSTOMER_ID
        self.camera_id = config.CAMERA_ID_STREAM_1
        self.location = config.LOCATION

        # Zone polygon: list of (x, y) tuples
        self.zone_polygon = zone_config.get('zone_polygon', [])
        self.confidence_threshold = zone_config.get('confidence_threshold', 0.25)
        self.temporal_in_frames = zone_config.get('temporal_in_frames', 10)
        self.temporal_out_frames = zone_config.get('temporal_out_frames', 10)

        # State
        self.state = "IDLE"  # IDLE or ACTIVE
        self.in_zone_counter = 0
        self.out_zone_counter = 0
        self.event_start_time = None
        self.event_start_datetime = None
        self.event_sync_id = None

        # Keep latest frame + detections for screenshot on event transitions
        self._last_frame = None
        self._last_detections = []

        logger.info(
            f"PyrometerProcessor initialized: conf>={self.confidence_threshold}, "
            f"zone={len(self.zone_polygon)} pts, "
            f"temporal_in={self.temporal_in_frames}, temporal_out={self.temporal_out_frames}"
        )

    def process_frame(self, frame_meta, frame=None):
        """
        Process a single frame's detections from nvinfer.

        Called from post-nvinfer probe on Stream 1.

        Args:
            frame_meta: NvDsFrameMeta from batch meta
            frame: RGBA numpy array (optional, for screenshots)
        """
        try:
            rod_in_zone = False
            detections = []

            # Iterate detections
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Filter by confidence
                if obj_meta.confidence >= self.confidence_threshold:
                    # Get bbox
                    rect = obj_meta.rect_params
                    x1 = rect.left
                    y1 = rect.top
                    x2 = x1 + rect.width
                    y2 = y1 + rect.height
                    conf = obj_meta.confidence

                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'in_zone': False,
                    })

                    # Zone check: top-left AND bottom-center in polygon
                    top_left = (x1, y1)
                    bottom_center = ((x1 + x2) / 2, y2)

                    if (self._point_in_polygon(top_left, self.zone_polygon) and
                            self._point_in_polygon(bottom_center, self.zone_polygon)):
                        rod_in_zone = True
                        detections[-1]['in_zone'] = True

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Store latest frame + detections for screenshot capture
            if frame is not None:
                self._last_frame = frame.copy()
            self._last_detections = detections

            # Update temporal state machine
            self._update_state(rod_in_zone)

        except Exception as e:
            logger.error(f"PyrometerProcessor error: {e}", exc_info=True)

    def _update_state(self, rod_in_zone: bool):
        """Update temporal state machine."""
        if self.state == "IDLE":
            if rod_in_zone:
                self.in_zone_counter += 1
                if self.in_zone_counter >= self.temporal_in_frames:
                    # Transition to ACTIVE
                    self.state = "ACTIVE"
                    self.event_start_time = time.time()
                    self.event_start_datetime = datetime.now()
                    self.event_sync_id = generate_sync_id("pyro")
                    self.in_zone_counter = 0
                    self.out_zone_counter = 0
                    logger.info(
                        f"[pyrometer] ROD DETECTED - sustained {self.temporal_in_frames} frames"
                    )
                    # Save start screenshot
                    self._save_event_screenshot("PYROMETER ROD START")
            else:
                self.in_zone_counter = 0

        elif self.state == "ACTIVE":
            if not rod_in_zone:
                self.out_zone_counter += 1
                if self.out_zone_counter >= self.temporal_out_frames:
                    # Transition to IDLE - emit event
                    self._emit_event()
                    self.state = "IDLE"
                    self.out_zone_counter = 0
                    self.in_zone_counter = 0
            else:
                self.out_zone_counter = 0

    def _emit_event(self):
        """Emit completed pyrometer event."""
        end_time = time.time()
        end_datetime = datetime.now()
        duration = end_time - self.event_start_time

        logger.info(
            f"[pyrometer] ROD REMOVED - event duration={duration:.1f}s"
        )

        # Save end screenshot
        screenshot_path = self._save_event_screenshot(
            "PYROMETER ROD END",
            duration=duration,
        )

        try:
            self.db_manager.insert_melting_event(
                sync_id=self.event_sync_id,
                customer_id=self.customer_id,
                event_type="pyrometer",
                start_time=self.event_start_datetime.isoformat(),
                end_time=end_datetime.isoformat(),
                duration_sec=round(duration, 1),
                camera_id=self.camera_id,
                location=self.location,
                screenshot_path=screenshot_path or "",
            )
        except Exception as e:
            logger.error(f"Failed to insert pyrometer event: {e}")

        # Push to heat cycle manager for aggregation
        if self.heat_cycle_manager:
            try:
                self.heat_cycle_manager.add_pyrometer_event(
                    start_wall=self.event_start_time,
                    start_dt=self.event_start_datetime,
                    end_wall=end_time,
                    end_dt=end_datetime,
                    duration=round(duration, 1),
                )
            except Exception as e:
                logger.error(f"Failed to push pyrometer to heat cycle manager: {e}")

        self.event_start_time = None
        self.event_start_datetime = None
        self.event_sync_id = None

    def _save_event_screenshot(self, title, duration=None):
        """Save annotated screenshot with zone polygon, detections, and event details."""
        if self._last_frame is None:
            return None

        try:
            annotated = prepare_frame(self._last_frame)

            # Draw zone polygon with semi-transparent fill + outline
            if self.zone_polygon:
                draw_roi_overlay(annotated, self.zone_polygon, (255, 200, 0),
                                 label="DETECTION ZONE")

            # Draw detection bboxes
            for det in self._last_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                in_zone = det['in_zone']
                color = (0, 255, 0) if in_zone else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label = f"Rod {conf:.2f}"
                if in_zone:
                    label += " [IN ZONE]"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Standard header/footer
            extra_lines = [f"Duration: {duration:.1f}s"] if duration is not None else None
            add_header(annotated, title, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       extra_lines)
            status = (f"Conf threshold: {self.confidence_threshold}  "
                      f"Temporal: {self.temporal_in_frames}in/{self.temporal_out_frames}out frames")
            add_footer(annotated, self.camera_id, status)

            tag = "start" if "START" in title else "end"
            return save_screenshot(annotated, "pyrometer", tag, datetime.now(),
                                   self.screenshot_dir)

        except Exception as e:
            logger.error(f"Error saving pyrometer screenshot: {e}")
            return None

    def add_inference_display_meta(self, batch_meta, frame_meta):
        """Attach DS-native overlay for pyrometer zone + detection status."""
        try:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                return

            # Scale overlay for downscaled recording
            scale_up = 1.0
            try:
                target_w = int(getattr(self.config, "INFERENCE_VIDEO_WIDTH", 0) or 0)
                # Pyrometer is 1920x1080, use frame_meta dimensions
                frame_w = frame_meta.source_frame_width or 1920
                if target_w and target_w < frame_w:
                    scale_up = min(3.0, frame_w / float(target_w))
            except Exception:
                scale_up = 1.0

            # Status text
            labels = []
            header = "PYROMETER ROD"
            labels.append((header, (1.0, 1.0, 1.0, 1.0)))

            state_txt = f"STATE: {'INSERTED' if self.state == 'ACTIVE' else 'NOT DETECTED'}"
            state_color = (0.0, 1.0, 0.0, 1.0) if self.state == 'ACTIVE' else (0.8, 0.8, 0.8, 1.0)
            labels.append((state_txt, state_color))

            if self.state == 'ACTIVE' and self.event_start_time:
                duration = time.time() - self.event_start_time
                labels.append((f"Duration: {duration:.1f}s", (0.0, 1.0, 1.0, 1.0)))

            # Render text
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

            # Draw zone polygon
            line_idx = 0
            max_lines = len(getattr(display_meta, "line_params", []))
            if max_lines > 0 and self.zone_polygon:
                n = len(self.zone_polygon)
                for i in range(n):
                    if line_idx >= max_lines:
                        break
                    x1, y1 = self.zone_polygon[i]
                    x2, y2 = self.zone_polygon[(i + 1) % n]
                    line = display_meta.line_params[line_idx]
                    line.x1 = int(max(0, x1))
                    line.y1 = int(max(0, y1))
                    line.x2 = int(max(0, x2))
                    line.y2 = int(max(0, y2))
                    line.line_width = max(2, int(round(2 * scale_up)))
                    # Magenta color for pyrometer zone
                    line.line_color.set(1.0, 0.0, 1.0, 1.0)
                    line_idx += 1

            display_meta.num_lines = line_idx
            display_meta.num_rects = 0
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        except Exception as exc:
            logger.error(f"[pyrometer] Failed to attach display meta: {exc}", exc_info=True)

    @staticmethod
    def _point_in_polygon(point, polygon):
        """Ray-casting point-in-polygon test."""
        x, y = point
        n = len(polygon)
        if n < 3:
            return False

        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
