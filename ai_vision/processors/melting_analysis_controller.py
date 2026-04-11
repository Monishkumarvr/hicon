"""
Python-side orchestration for native CUDA melting detections.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import cv2
import numpy as np

from processors.brightness_processor import BrightnessProcessor
from utils.screenshot import (
    prepare_frame,
    draw_roi_overlay,
    add_header,
    add_footer,
    save as save_screenshot,
)
from utils.utils import generate_sync_id

logger = logging.getLogger(__name__)


class MeltingAnalysisController(BrightnessProcessor):
    """
    Consume native melting meta and reuse the existing Python event chain.
    """

    def __init__(self, zones_config, db_manager, config, screenshot_dir,
                 heat_cycle_manager=None, enable_display_meta=True,
                 screenshot_writer=None):
        super().__init__(
            zones_config=zones_config,
            db_manager=db_manager,
            config=config,
            screenshot_dir=screenshot_dir,
            heat_cycle_manager=heat_cycle_manager,
            enable_display_meta=enable_display_meta,
            screenshot_writer=screenshot_writer,
        )
        self._zone_orders = {
            "tapping": self._ordered_zone_names(self._tapping_config),
            "deslagging": self._ordered_zone_names(self._deslagging_config),
            "spectro": self._ordered_zone_names(self._spectro_config),
        }
        self._native_active = {
            "tapping": {name: False for name in self._zone_orders["tapping"]},
            "deslagging": {name: False for name in self._zone_orders["deslagging"]},
            "spectro": {name: False for name in self._zone_orders["spectro"]},
        }
        self._event_starts = {
            "tapping": {},
            "deslagging": {},
            "spectro": {},
        }
        logger.info("MeltingAnalysisController initialized (native CUDA meta -> Python events)")

    @staticmethod
    def _ordered_zone_names(cfg):
        zones = cfg.get("zones", {})
        if zones:
            return list(zones.keys())
        if cfg.get("roi_points"):
            return ["zone-1"]
        return []

    def _zone_cfg(self, event_type, zone_name):
        cfg = {
            "tapping": self._tapping_config,
            "deslagging": self._deslagging_config,
            "spectro": self._spectro_config,
        }[event_type]
        zones = cfg.get("zones", {})
        return zones.get(zone_name, cfg)

    def _zone_points(self, event_type, zone_name, frame_shape=None):
        zone_cfg = self._zone_cfg(event_type, zone_name)
        pts = zone_cfg.get("roi_points", [])
        if not pts:
            return []

        if frame_shape is None:
            return pts

        frame_h, frame_w = frame_shape[:2]
        ref_size = zone_cfg.get("annotation_size")
        if ref_size and len(ref_size) == 2:
            ref_w, ref_h = ref_size
        else:
            ref_w, ref_h = self._ref_w, self._ref_h
        return self._scale_pts(pts, frame_w, frame_h, ref_w, ref_h)

    def needs_frame(self, native_state):
        if native_state is None:
            return False
        return bool(self._peek_transitions(native_state))

    def _iter_zone_states(self, native_state):
        for event_type in ("tapping", "deslagging", "spectro"):
            values = getattr(native_state, event_type)
            zone_names = self._zone_orders[event_type]
            for idx, zone_name in enumerate(zone_names):
                zone_state = values[idx] if idx < len(values) else None
                yield event_type, zone_name, zone_state

    def _peek_transitions(self, native_state):
        transitions = []
        for event_type, zone_name, zone_state in self._iter_zone_states(native_state):
            current_active = bool(zone_state and zone_state.valid and zone_state.active)
            previous_active = self._native_active[event_type].get(zone_name, False)
            if current_active != previous_active:
                transitions.append((event_type, zone_name, zone_state, current_active))
        return transitions

    def process_native_state(self, native_state, frame_meta, frame=None,
                             batch_meta=None, timestamp=None, datetime_obj=None):
        del batch_meta
        if native_state is None:
            return

        now_wall = timestamp if timestamp is not None else time.time()
        now_dt = datetime_obj if datetime_obj is not None else datetime.now()
        transitions = self._peek_transitions(native_state)
        if not transitions:
            return

        for event_type, zone_name, zone_state, current_active in transitions:
            state_map = self._native_active[event_type]
            state_map[zone_name] = current_active
            if current_active:
                self._event_starts[event_type][zone_name] = (now_wall, now_dt)
                event = {
                    "type": event_type,
                    "phase": "start",
                    "zone_name": zone_name,
                    "start": now_dt.isoformat(),
                    "start_wall": now_wall,
                    "start_datetime": now_dt,
                    "white_ratio": getattr(zone_state, "white_ratio", 0.0) if zone_state else 0.0,
                    "raw_count": getattr(zone_state, "raw_count", 0) if zone_state else 0,
                    "filtered_count": getattr(zone_state, "filtered_count", 0) if zone_state else 0,
                }
                self._handle_native_event_start(event, frame)
            else:
                start_wall, start_dt = self._event_starts[event_type].pop(zone_name, (now_wall, now_dt))
                duration = max(0.0, now_wall - start_wall)
                event = {
                    "type": event_type,
                    "phase": "end",
                    "zone_name": zone_name,
                    "start": start_dt.isoformat(),
                    "end": now_dt.isoformat(),
                    "duration_sec": round(duration, 1),
                    "start_wall": start_wall,
                    "end_wall": now_wall,
                    "start_datetime": start_dt,
                    "end_datetime": now_dt,
                    "white_ratio": getattr(zone_state, "white_ratio", 0.0) if zone_state else 0.0,
                    "raw_count": getattr(zone_state, "raw_count", 0) if zone_state else 0,
                    "filtered_count": getattr(zone_state, "filtered_count", 0) if zone_state else 0,
                    "max_blob_area": getattr(zone_state, "max_blob_area", 0.0) if zone_state else 0.0,
                    "max_blob_brightness": getattr(zone_state, "max_blob_brightness", 0.0) if zone_state else 0.0,
                }
                self._handle_native_event(event, frame)

    def _handle_native_event_start(self, event, frame_rgba):
        if event["type"] != "tapping":
            return
        logger.info(
            "[%s][%s] Start detected: %s",
            event["type"],
            event["zone_name"],
            event["start"],
        )
        if frame_rgba is not None:
            self._save_native_screenshot(frame_rgba, event, phase="start")

    def _handle_native_event(self, event, frame_rgba):
        event_type = event["type"]
        logger.info(
            "[%s][%s] Event: %s -> %s (%ss)",
            event_type,
            event["zone_name"],
            event["start"],
            event["end"],
            event["duration_sec"],
        )

        screenshot_path = ""
        if frame_rgba is not None:
            screenshot_path = self._save_native_screenshot(frame_rgba, event, phase="end")

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
            logger.error("Failed to insert %s event: %s", event_type, e)

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
                logger.error("Failed to push %s to heat cycle manager: %s", event_type, e)

    def _save_native_screenshot(self, frame_rgba, event, phase="end"):
        try:
            annotated = prepare_frame(frame_rgba)
            zone_points = self._zone_points(
                event["type"],
                event["zone_name"],
                annotated.shape,
            )
            roi_color = {
                "tapping": (0, 165, 255),
                "deslagging": (0, 0, 255),
                "spectro": (255, 255, 0),
            }[event["type"]]
            if zone_points:
                draw_roi_overlay(
                    annotated,
                    zone_points,
                    roi_color,
                    label=f"{event['type'].upper()} {event['zone_name']}",
                    alpha=0.2,
                )

            extra_lines = [
                f"Zone: {event['zone_name']}",
            ]
            if event["type"] == "tapping":
                extra_lines.append(
                    f"White Ratio: {event.get('white_ratio', 0.0):.3f}"
                )
            else:
                extra_lines.append(
                    f"Blobs: raw={event.get('raw_count', 0)} filtered={event.get('filtered_count', 0)}"
                )
                if event.get("max_blob_area", 0.0) > 0.0:
                    extra_lines.append(
                        f"Max Blob: area={event.get('max_blob_area', 0.0):.1f} "
                        f"brightness={event.get('max_blob_brightness', 0.0):.1f}"
                    )

            screenshot_dt = event.get(
                "end_datetime" if phase == "end" else "start_datetime"
            ) or datetime.now()
            title = f"{event['type'].upper()} EVENT {phase.upper()}"
            add_header(
                annotated,
                title,
                screenshot_dt.strftime("%Y-%m-%d %H:%M:%S"),
                extra_lines,
            )
            add_footer(annotated, self.camera_id, f"Phase: {phase.upper()}")

            safe_zone = event["zone_name"].replace("/", "_")
            return save_screenshot(
                annotated,
                event["type"],
                f"{safe_zone}_{phase}",
                screenshot_dt,
                self.screenshot_dir,
                event_type=event["type"],
                camera_id=str(self.camera_id),
                writer=self.screenshot_writer,
            )
        except Exception as e:
            logger.error("Failed to save native %s screenshot: %s", event["type"], e, exc_info=True)
            return ""
