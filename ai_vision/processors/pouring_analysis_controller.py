"""
Hybrid native-plus-Python pouring controller.

The C++ plugin exports one authoritative trolley/mouth/session snapshot per
frame. Python consumes that state, samples brightness from the shared analysis
frame, and reuses the mature PouringProcessor business flow for pour events,
mould handling, screenshots, DB writes, heat-cycle updates, and sync.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from processors.pouring_meta_reader import (
    DecodedPouringState,
    EVENT_NONE,
    EVENT_SESSION_END,
    EVENT_SESSION_START,
)
from processors.pouring_processor import PouringProcessor

logger = logging.getLogger(__name__)


class HybridPouringController(PouringProcessor):
    """Drive pouring business logic from native session/probe state."""

    def needs_frame(self, native_state: Optional[DecodedPouringState]) -> bool:
        if native_state is None:
            return bool(self.session_active or self.pour_active)
        return bool(
            native_state.session_active
            or native_state.event != EVENT_NONE
            or self.session_active
            or self.pour_active
        )

    def process_native_state(
        self,
        frame_meta,
        native_state: Optional[DecodedPouringState],
        frame,
        timestamp,
        datetime_obj,
        batch_meta=None,
    ):
        self._frame_count += 1

        frame_w = int(getattr(frame_meta, 'source_frame_width', 0) or 0) or self._frame_w or 1280
        frame_h = int(getattr(frame_meta, 'source_frame_height', 0) or 0) or self._frame_h or 720
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._update_runtime_geometry(frame_w, frame_h)

        if native_state is None:
            self._check_cycle_timeout(timestamp, datetime_obj, [], [], frame)
            self._finalize_heat_cycles_if_due(timestamp, datetime_obj)
            return

        mouths, actual_trolleys = self._extract_detections(frame_meta)
        target_trolley = self._resolve_native_target_trolley(native_state, actual_trolleys)
        trolleys = list(actual_trolleys)
        if target_trolley is not None and not any(
            trolley.get("track_id") == target_trolley.get("track_id") for trolley in trolleys
        ):
            trolleys.append(target_trolley)
        best_mouth = (
            self._select_best_mouth_for_trolley(mouths, target_trolley)
            if mouths and target_trolley
            else None
        )
        mouth_in_trolley = bool(native_state.mouth_present_in_trolley and target_trolley)

        if target_trolley and self.trolley_locked:
            if self.locked_trolley_id == target_trolley["track_id"]:
                self.locked_trolley_bbox = target_trolley['bbox']
            elif native_state.session_active:
                self._relock_trolley(target_trolley, timestamp, reason="native_active_trolley")

        self._sync_session_from_native(
            native_state=native_state,
            target_trolley=target_trolley,
            mouths=mouths,
            trolleys=trolleys,
            frame=frame,
            timestamp=timestamp,
            datetime_obj=datetime_obj,
        )

        if native_state.probe_valid:
            self._last_probe_base = (
                int(round(native_state.probe_x_px)),
                int(round(native_state.probe_y_px)),
            )

        if (
            self.heat_cycle_manager
            and self.session_active
            and mouth_in_trolley
        ):
            presence_track_id = self._presence_track_id(native_state, best_mouth, target_trolley)
            if presence_track_id > 0:
                self.heat_cycle_manager.update_pouring_session_presence(
                    presence_track_id, timestamp, datetime_obj
                )

        pour_probe_mouth = None
        if self.session_active and frame is not None:
            pour_probe_mouth = self._update_pour(
                mouths,
                frame,
                timestamp,
                datetime_obj,
                trolleys,
                target_trolley,
            )

        mould_mouth = pour_probe_mouth or best_mouth
        if self.session_active and self.pour_active and mould_mouth and target_trolley:
            self._update_mould_counter(mould_mouth, target_trolley, timestamp, datetime_obj)

        self._check_cycle_timeout(timestamp, datetime_obj, mouths, trolleys, frame)
        self._finalize_heat_cycles_if_due(timestamp, datetime_obj)

        if self.enable_inference_video and self.enable_display_meta and batch_meta is not None:
            self._add_inference_display_meta(
                batch_meta=batch_meta,
                frame_meta=frame_meta,
                mouths=mouths,
                trolleys=trolleys,
                target_trolley=target_trolley,
                timestamp=timestamp,
                datetime_obj=datetime_obj,
            )

    def _sync_session_from_native(
        self,
        native_state: DecodedPouringState,
        target_trolley,
        mouths,
        trolleys,
        frame,
        timestamp,
        datetime_obj: datetime,
    ):
        if native_state.mouth_present_in_trolley:
            self.mouth_last_seen_in_trolley = timestamp
            self.mouth_absent_since = None

        if native_state.event == EVENT_SESSION_END:
            if self.session_active:
                self._end_session(timestamp, datetime_obj, mouths, trolleys, frame)
            else:
                self._clear_session_state()
            return

        if native_state.event == EVENT_SESSION_START:
            if not self.session_active:
                self._start_session(target_trolley, timestamp, datetime_obj, mouths, trolleys, frame)
            self.mouth_inside_since = None
            self.mouth_absent_since = None
            return

        if native_state.session_active:
            if not self.session_active:
                logger.info(
                    "[hybrid-pouring] Bootstrapping Python session state from native session_active"
                )
                self._start_session(target_trolley, timestamp, datetime_obj, mouths, trolleys, frame)
            self.mouth_inside_since = None
        elif self.session_active:
            self._end_session(timestamp, datetime_obj, mouths, trolleys, frame)

    def _clear_session_state(self):
        self.session_active = False
        self.session_start_time = None
        self.session_start_datetime = None
        self.mouth_inside_since = None
        self.mouth_absent_since = None
        self.pour_active = False
        self.pour_start_time = None
        self.pour_start_datetime = None
        self.pour_sync_id = None
        self.active_mould_id = None
        self.active_mould_start_time = None
        self.active_mould_start_datetime = None
        self.active_mould_start_norm = None
        self.brightness_above_since = None
        self.brightness_below_since = None
        self.pour_on_count = 0
        self.pour_off_count = 0
        self._last_probe_tail_brightness = None
        self._last_probe_is_pouring = None
        self._clear_frozen_probe_state()
        self._clear_active_probe_state()

    def _resolve_native_target_trolley(self, native_state: DecodedPouringState, actual_trolleys):
        if native_state.trolley_track_id > 0:
            for trolley in actual_trolleys:
                if trolley.get("track_id") == native_state.trolley_track_id:
                    return trolley

        proxy = self._build_proxy_trolley(native_state)
        if proxy is None:
            return None

        if not actual_trolleys:
            return proxy

        best = None
        best_iou = -1.0
        for trolley in actual_trolleys:
            iou = self._bbox_iou(proxy["bbox"], trolley["bbox"])
            if iou > best_iou:
                best = trolley
                best_iou = iou
        return best if best_iou > 0.0 else proxy

    def _build_proxy_trolley(self, native_state: DecodedPouringState):
        if native_state.trolley_track_id <= 0:
            return None
        bbox = self._bbox_to_ints(native_state.trolley_bbox)
        return {
            "track_id": native_state.trolley_track_id,
            "confidence": 1.0,
            "bbox": bbox,
            "center": ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
            "bottom_center": ((bbox[0] + bbox[2]) // 2, bbox[3]),
        }

    def _build_proxy_mouth(self, native_state: DecodedPouringState):
        if native_state.mouth_track_id <= 0 and not native_state.probe_valid:
            return None

        bbox = self._proxy_mouth_bbox(native_state)
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        bottom_center = (
            int(round(native_state.probe_x_px)) if native_state.probe_valid else center[0],
            int(round(native_state.probe_y_px - self.probe_below_px))
            if native_state.probe_valid else bbox[3],
        )

        native_norm = None
        if native_state.has_valid_mouth_norm:
            native_norm = (native_state.mouth_norm_x, native_state.mouth_norm_y)

        return {
            "track_id": native_state.mouth_track_id or native_state.trolley_track_id,
            "confidence": 1.0,
            "bbox": bbox,
            "center": center,
            "bottom_center": bottom_center,
            "native_norm": native_norm,
        }

    def _proxy_mouth_bbox(self, native_state: DecodedPouringState):
        bbox = self._bbox_to_ints(native_state.mouth_bbox)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            return bbox

        if native_state.probe_valid:
            base_x = int(round(native_state.probe_x_px))
            base_y = int(round(native_state.probe_y_px - self.probe_below_px))
            return (base_x - 1, base_y - 1, base_x + 1, base_y + 1)

        return (0, 0, 1, 1)

    @staticmethod
    def _bbox_to_ints(bbox):
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        return (
            x1,
            y1,
            max(x2, x1 + 1),
            max(y2, y1 + 1),
        )

    @staticmethod
    def _presence_track_id(native_state, best_mouth, target_trolley):
        if best_mouth and best_mouth.get("track_id"):
            return int(best_mouth["track_id"])
        if native_state.mouth_track_id > 0:
            return int(native_state.mouth_track_id)
        if target_trolley and target_trolley.get("track_id"):
            return int(target_trolley["track_id"])
        return 0
