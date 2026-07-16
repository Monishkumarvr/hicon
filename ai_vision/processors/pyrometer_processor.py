"""
Pyrometer Processor - Rod insertion detection on Stream 1 (Pyrometer Camera).

Reads NvDsObjectMeta from nvinfer (YOLO26 custom parser).
Multi-zone support: each zone tracks rod presence independently with
temporal frame counting for event start/end.
"""
import logging
import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import pyds

from utils.utils import generate_sync_id
from utils.screenshot import (prepare_frame, add_header, add_footer,
                               draw_roi_overlay, save as save_screenshot,
                               _to_coco_bbox)

logger = logging.getLogger(__name__)

_PYROMETER_CATEGORIES = [{"id": 0, "name": "rod"}]


@dataclass
class PyroZoneState:
    """Per-zone state for pyrometer rod detection."""
    name: str
    zone: List[Tuple[float, float]]  # polygon vertices as (x, y) tuples
    exclusive_group: Optional[str] = None  # zones sharing a group suppress each other
    state: str = "IDLE"              # IDLE or ACTIVE
    in_zone_counter: int = 0
    out_zone_counter: int = 0
    last_in_zone_time: float = 0.0   # wall time of last rod-in-zone frame
    event_start_time: float = 0.0
    event_start_datetime: Optional[datetime] = None
    event_sync_id: str = ""


class PyrometerProcessor:
    """
    Detect pyrometer rod insertion events using nvinfer detections + zone filtering.

    Multi-zone: each zone tracks rod presence independently.

    Algorithm per zone:
    1. Filter detections: confidence >= threshold
    2. Zone check: bbox top-left AND bottom-left must be inside polygon
    3. Temporal: N consecutive in-zone frames → EVENT START
                 N consecutive absent frames → EVENT END
    """

    def __init__(self, zone_config, db_manager, config, screenshot_dir,
                 heat_cycle_manager=None, screenshot_writer=None):
        self.db_manager = db_manager
        self.config = config
        self.heat_cycle_manager = heat_cycle_manager
        self.screenshot_writer = screenshot_writer
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.customer_id = config.CUSTOMER_ID
        self.camera_id = config.CAMERA_ID_STREAM_1
        self.location = config.LOCATION

        # Shared thresholds
        self.confidence_threshold = zone_config.get('confidence_threshold', 0.25)
        self.min_area_px2 = zone_config.get(
            'min_area_px2',
            getattr(config, 'PYROMETER_MIN_AREA_PX2', 7000),
        )
        self.temporal_in_frames = zone_config.get('temporal_in_frames', 10)
        self.temporal_out_frames = zone_config.get('temporal_out_frames', 10)

        # Build zone states — multi-zone or legacy single-zone fallback
        self.zone_states: Dict[str, PyroZoneState] = {}
        zones_data = zone_config.get('zones', {})
        if zones_data:
            for name, zcfg in zones_data.items():
                pts = zcfg.get('roi_points', [])
                if pts:
                    polygon = [(float(p[0]), float(p[1])) for p in pts]
                    exclusive_group = zcfg.get('exclusive_group') or None
                    self.zone_states[name] = PyroZoneState(
                        name=name, zone=polygon, exclusive_group=exclusive_group
                    )
        else:
            # Legacy single-zone fallback
            legacy_pts = zone_config.get('zone_polygon', [])
            if legacy_pts:
                polygon = [(float(p[0]), float(p[1])) for p in legacy_pts]
                self.zone_states["zone-1"] = PyroZoneState(name="zone-1", zone=polygon)

        # Keep latest frame + detections for screenshot on event transitions
        self._last_frame = None
        self._last_detections = []

        zone_names = ", ".join(self.zone_states.keys())
        logger.info(
            f"PyrometerProcessor initialized: {len(self.zone_states)} zones ({zone_names}), "
            f"conf>={self.confidence_threshold}, min_area={self.min_area_px2}px², "
            f"temporal_in={self.temporal_in_frames}, temporal_out={self.temporal_out_frames}"
        )

    @property
    def is_active(self) -> bool:
        """True if any zone is currently active."""
        return any(zs.state == "ACTIVE" for zs in self.zone_states.values())

    def needs_frame(self, obj_count: int) -> bool:
        return bool(obj_count > 0 or self.is_active)

    def process_frame(self, frame_meta, frame=None, timestamp=None, datetime_obj=None):
        """
        Process a single frame's detections from nvinfer.

        Called from post-nvinfer probe on Stream 1.
        """
        try:
            now_wall = timestamp if timestamp is not None else time.time()
            now_dt = datetime_obj if datetime_obj is not None else datetime.now()
            detections = []

            # Iterate nvinfer detections (once for all zones)
            raw_confs = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                raw_confs.append(round(obj_meta.confidence, 3))

                if obj_meta.confidence >= self.confidence_threshold:
                    rect = obj_meta.rect_params
                    x1 = rect.left
                    y1 = rect.top
                    x2 = x1 + rect.width
                    y2 = y1 + rect.height
                    area = rect.width * rect.height

                    if area <= self.min_area_px2:
                        raw_confs.append(
                            f"{round(obj_meta.confidence, 3)}(area={area:.0f}<={self.min_area_px2})"
                        )
                    else:
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': obj_meta.confidence,
                            'top_left': (x1, y1),
                            'bottom_left': (x1, y2),
                        })

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Store latest frame + detections for screenshot capture
            if frame is not None:
                self._last_frame = frame.copy()
            self._last_detections = detections

            # Periodic raw detection log — shows all confidences regardless of threshold.
            # ~10s cadence, DEBUG (Edge_Optimization_Plan.md Phase 0.7; event start/end
            # below stays at INFO).
            if not hasattr(self, '_pyro_log_counter'):
                self._pyro_log_counter = 0
            self._pyro_log_counter += 1
            if self._pyro_log_counter >= 250:
                self._pyro_log_counter = 0
                if raw_confs:
                    logger.debug("[pyrometer] raw detections: %s (threshold=%.2f passed=%d)",
                                raw_confs, self.confidence_threshold, len(detections))
                else:
                    logger.debug("[pyrometer] no detections this frame (threshold=%.2f)",
                                self.confidence_threshold)

            # Check each zone independently
            for zs in self.zone_states.values():
                rod_in_zone = self._any_detection_in_zone(detections, zs.zone)
                event = self._update_zone_state(zs, rod_in_zone, now_wall=now_wall, now_dt=now_dt)
                if event:
                    if event["phase"] == "start":
                        furnace_label = None
                        if self.heat_cycle_manager:
                            ac = self.heat_cycle_manager.active_cycle
                            if ac and ac.furnace_label:
                                furnace_label = ac.furnace_label
                        start_label = furnace_label or zs.name
                        self._save_event_screenshot(
                            f"PYROMETER ROD START ({start_label})",
                            zone_state=zs,
                            capture_dt=event["start_datetime"],
                        )
                    elif event["phase"] == "end":
                        self._emit_event(event, zs)

        except Exception as e:
            logger.error(f"PyrometerProcessor error: {e}", exc_info=True)

    def _any_detection_in_zone(self, detections, zone):
        """Check if any detection has top-left AND bottom-left inside zone polygon."""
        for det in detections:
            if (self._point_in_polygon(det['top_left'], zone) and
                    self._point_in_polygon(det['bottom_left'], zone)):
                return True
        return False

    def _update_zone_state(self, zs: PyroZoneState, rod_in_zone: bool, *, now_wall: float, now_dt: datetime):
        """Update temporal state machine for a single zone. Returns event dict or None."""
        if zs.state == "IDLE":
            if rod_in_zone:
                zs.in_zone_counter += 1
                zs.last_in_zone_time = now_wall
                if zs.in_zone_counter >= self.temporal_in_frames:
                    zs.state = "ACTIVE"
                    zs.event_start_time = now_wall
                    zs.event_start_datetime = now_dt
                    zs.event_sync_id = generate_sync_id("pyro")
                    zs.in_zone_counter = 0
                    zs.out_zone_counter = 0
                    logger.info(
                        f"[pyrometer] Zone '{zs.name}': ROD DETECTED — "
                        f"sustained {self.temporal_in_frames} frames"
                    )
                    return {
                        "type": "pyrometer",
                        "phase": "start",
                        "zone_name": zs.name,
                        "start_wall": zs.event_start_time,
                        "start_datetime": zs.event_start_datetime,
                    }
            else:
                # With nvinfer interval>0, skipped frames produce no detections and would
                # instantly reset the counter. Only reset after the rod has been absent
                # long enough for multiple inference cycles to have run (~0.5s covers
                # 4 inference cycles at interval=2, 25fps).
                if zs.in_zone_counter > 0 and (now_wall - zs.last_in_zone_time) > 0.5:
                    zs.in_zone_counter = 0

        elif zs.state == "ACTIVE":
            if not rod_in_zone:
                zs.out_zone_counter += 1
                if zs.out_zone_counter >= self.temporal_out_frames:
                    end_time = now_wall
                    end_datetime = now_dt
                    duration = end_time - zs.event_start_time
                    zs.state = "IDLE"
                    zs.out_zone_counter = 0
                    zs.in_zone_counter = 0
                    logger.info(
                        f"[pyrometer] Zone '{zs.name}': ROD REMOVED — "
                        f"duration={duration:.1f}s"
                    )
                    return {
                        "type": "pyrometer",
                        "phase": "end",
                        "zone_name": zs.name,
                        "sync_id": zs.event_sync_id,
                        "start_wall": zs.event_start_time,
                        "start_datetime": zs.event_start_datetime,
                        "end_wall": end_time,
                        "end_datetime": end_datetime,
                        "duration_sec": round(duration, 1),
                    }
            else:
                zs.out_zone_counter = 0

        return None

    def _emit_event(self, event, zs: PyroZoneState):
        """Emit completed pyrometer event — DB insert + heat cycle + screenshot."""
        zone_name = event["zone_name"]
        duration = event["duration_sec"]

        # Tag event to the active furnace (set by tapping/pouring earlier in the heat).
        # Falls back to the detection zone name if no heat cycle is active yet.
        furnace_label = None
        if self.heat_cycle_manager:
            ac = self.heat_cycle_manager.active_cycle
            if ac and ac.furnace_label:
                furnace_label = ac.furnace_label  # e.g. "Furnace1" or "Furnace2"
        if not furnace_label:
            logger.warning("[pyrometer] No active furnace label — tagging event as '%s'", zone_name)
        event_zone_name = furnace_label or zone_name

        # Save end screenshot
        screenshot_path = self._save_event_screenshot(
            f"PYROMETER ROD END ({event_zone_name})",
            duration=duration,
            zone_state=zs,
            capture_dt=event["end_datetime"],
        )

        try:
            self.db_manager.insert_melting_event(
                sync_id=event["sync_id"],
                customer_id=self.customer_id,
                event_type="pyrometer",
                start_time=event["start_datetime"].isoformat(),
                end_time=event["end_datetime"].isoformat(),
                duration_sec=duration,
                camera_id=self.camera_id,
                location=self.location,
                screenshot_path=screenshot_path or "",
                zone_name=event_zone_name,
            )
        except Exception as e:
            logger.error(f"Failed to insert pyrometer event: {e}")

        if self.heat_cycle_manager:
            try:
                self.heat_cycle_manager.add_pyrometer_event(
                    start_wall=event["start_wall"],
                    start_dt=event["start_datetime"],
                    end_wall=event["end_wall"],
                    end_dt=event["end_datetime"],
                    duration=duration,
                    zone_name=event_zone_name,
                )
            except Exception as e:
                logger.error(f"Failed to push pyrometer to heat cycle manager: {e}")

        # Reset zone event fields
        zs.event_start_time = 0.0
        zs.event_start_datetime = None
        zs.event_sync_id = ""

    def _save_event_screenshot(self, title, duration=None, zone_state=None, capture_dt=None):
        """Save annotated screenshot with zone polygon, detections, and event details."""
        if self._last_frame is None:
            return None

        try:
            raw_bgr = prepare_frame(self._last_frame)
            annotated = raw_bgr.copy()

            # Draw zone polygon(s)
            if zone_state and zone_state.zone:
                draw_roi_overlay(annotated, zone_state.zone, (255, 200, 0),
                                 label=f"ZONE: {zone_state.name}")
            else:
                # Draw all zones
                for zs in self.zone_states.values():
                    draw_roi_overlay(annotated, zs.zone, (255, 200, 0),
                                     label=f"ZONE: {zs.name}")

            # Draw detection bboxes
            for det in self._last_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label = f"Rod {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            screenshot_dt = capture_dt or datetime.now()
            # Standard header/footer
            extra_lines = [f"Duration: {duration:.1f}s"] if duration is not None else None
            add_header(annotated, title, screenshot_dt.strftime("%Y-%m-%d %H:%M:%S"),
                       extra_lines)
            status = (f"Conf>={self.confidence_threshold}  "
                      f"Temporal: {self.temporal_in_frames}in/{self.temporal_out_frames}out")
            add_footer(annotated, self.camera_id, status)

            ann_list = []
            for i, det in enumerate(self._last_detections or [], start=1):
                x1, y1, x2, y2 = det['bbox']
                ann_list.append({
                    "id": i, "category_id": 0,
                    "bbox": _to_coco_bbox(x1, y1, x2, y2),
                    "area": (x2 - x1) * (y2 - y1),
                    "confidence": round(float(det['confidence']), 4),
                    "iscrowd": 0,
                })

            tag = "start" if "START" in title else "end"
            zone_tag = f"_{zone_state.name}" if zone_state else ""
            return save_screenshot(
                annotated, f"pyrometer{zone_tag}", tag, screenshot_dt,
                self.screenshot_dir,
                annotations=ann_list or None,
                raw_frame=raw_bgr,
                event_type=f"rod_{tag}",
                camera_id=str(self.camera_id),
                categories=_PYROMETER_CATEGORIES,
                writer=self.screenshot_writer,
            )

        except Exception as e:
            logger.error(f"Error saving pyrometer screenshot: {e}")
            return None

    def add_inference_display_meta(self, batch_meta, frame_meta):
        """Attach DS-native overlay for pyrometer zones + detection status."""
        try:
            # Scale overlay for downscaled recording
            scale_up = 1.0
            try:
                target_w = int(getattr(self.config, "INFERENCE_VIDEO_WIDTH", 0) or 0)
                frame_w = frame_meta.source_frame_width or 1280
                if target_w and target_w < frame_w:
                    scale_up = min(3.0, frame_w / float(target_w))
            except Exception:
                scale_up = 1.0

            # Status text (first display_meta)
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                return
            display_meta.num_labels = 0
            display_meta.num_lines = 0
            display_meta.num_rects = 0
            display_meta.num_circles = 0

            labels = []
            labels.append(("PYROMETER ROD", (1.0, 1.0, 1.0, 1.0)))

            # Per-zone status
            for zs in self.zone_states.values():
                if zs.state == "ACTIVE":
                    dur = time.time() - zs.event_start_time if zs.event_start_time else 0
                    labels.append(
                        (f"{zs.name}: INSERTED ({dur:.1f}s)", (0.0, 1.0, 0.0, 1.0))
                    )
                else:
                    labels.append(
                        (f"{zs.name}: idle", (0.8, 0.8, 0.8, 1.0))
                    )

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

            display_meta.num_lines = 0
            display_meta.num_rects = 0
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            display_meta = None  # ownership transferred to frame_meta

            # Draw zone polygons (separate display_meta per zone for 16-line limit)
            for zs in self.zone_states.values():
                color = (1.0, 0.5, 0.0, 1.0) if zs.state == "ACTIVE" else (0.0, 1.0, 1.0, 1.0)
                self._draw_zone_polygon(batch_meta, frame_meta, zs.zone, color, scale_up)

        except Exception as exc:
            logger.error(f"[pyrometer] Failed to attach display meta: {exc}", exc_info=True)
        finally:
            if display_meta is not None:
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    def _draw_zone_polygon(self, batch_meta, frame_meta, polygon, color, scale_up=1.0):
        """Draw a polygon outline using NvOSD line segments."""
        n = len(polygon)
        if n < 3:
            return

        dm = None
        try:
            dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not dm:
                return

            line_idx = 0
            max_lines = 16  # NvOSD limit per display_meta
            for i in range(n):
                if line_idx >= max_lines:
                    dm.num_lines = line_idx
                    dm.num_labels = 0
                    dm.num_rects = 0
                    pyds.nvds_add_display_meta_to_frame(frame_meta, dm)
                    dm = None  # ownership transferred
                    dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    if not dm:
                        return
                    line_idx = 0

                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]
                line = dm.line_params[line_idx]
                line.x1 = int(max(0, x1))
                line.y1 = int(max(0, y1))
                line.x2 = int(max(0, x2))
                line.y2 = int(max(0, y2))
                line.line_width = max(2, int(round(2 * scale_up)))
                line.line_color.set(*color)
                line_idx += 1

            dm.num_lines = line_idx
            dm.num_labels = 0
            dm.num_rects = 0
            pyds.nvds_add_display_meta_to_frame(frame_meta, dm)
            dm = None  # ownership transferred
        finally:
            if dm is not None:
                pyds.nvds_add_display_meta_to_frame(frame_meta, dm)

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
