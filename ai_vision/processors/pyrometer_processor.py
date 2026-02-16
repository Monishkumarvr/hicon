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

from utils.utils import save_screenshot, generate_sync_id

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
            frame_bgr = cv2.cvtColor(self._last_frame, cv2.COLOR_RGBA2BGR)
            annotated = frame_bgr.copy()
            h, w = annotated.shape[:2]

            # Draw zone polygon with semi-transparent fill + outline
            if self.zone_polygon:
                pts = np.array(self.zone_polygon, dtype=np.int32)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], (255, 200, 0))  # Cyan-ish
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                cv2.polylines(annotated, [pts], True, (255, 200, 0), 2)
                # Label the zone
                cx = int(np.mean([p[0] for p in self.zone_polygon]))
                cy = int(np.mean([p[1] for p in self.zone_polygon]))
                cv2.putText(annotated, "DETECTION ZONE",
                           (cx - 70, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Draw detection bboxes
            for det in self._last_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                in_zone = det['in_zone']
                color = (0, 255, 0) if in_zone else (0, 0, 255)  # Green if in zone, red if not
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label = f"Rod {conf:.2f}"
                if in_zone:
                    label += " [IN ZONE]"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Event title
            cv2.putText(annotated, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Timestamp
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated, now_str, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Duration (for end events)
            if duration is not None:
                cv2.putText(annotated, f"Duration: {duration:.1f}s", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Confidence threshold info
            cv2.putText(annotated,
                       f"Conf threshold: {self.confidence_threshold}  "
                       f"Temporal: {self.temporal_in_frames}in/{self.temporal_out_frames}out frames",
                       (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Camera ID
            cv2.putText(annotated, f"CAM: {self.camera_id}", (w - 200, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "start" if "START" in title else "end"
            filename = f"pyrometer_{tag}_{timestamp}.jpg"
            filepath = self.screenshot_dir / filename
            cv2.imwrite(str(filepath), annotated)
            logger.info(f"Saved pyrometer screenshot: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving pyrometer screenshot: {e}")
            return None

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
