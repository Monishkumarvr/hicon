"""
Pouring Processor - HiCon pouring detection per standalone pouring system documentation.

Three sub-systems running in the OSD sink pad probe on Stream 0:
1. Session Manager:  ladle_mouth center inside expanded trolley bbox → session lifecycle
2. Pour Detector:    multi-probe brightness below ladle_mouth → pour start/end
3. Mould Counter:    reference-style segment split + spatial clustering → mould count

Key behaviors:
- Trolley locking: lock onto trolley where first pour starts, ignore others
- EDGE_EXPAND: expand trolley bbox by 180px on all sides (reference parity)
- Session persistence: mould data preserved across session boundaries for same locked trolley
- Pouring cycle timeout: 5 min mouth absence from locked trolley → reset everything

Model classes: ladle_mouth (class 0), trolley (class 1).
"""

import cv2
import math
import numpy as np
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pyds

from utils.utils import generate_sync_id
from utils.screenshot import (prepare_frame, add_header, add_footer,
                               save as save_screenshot, _to_coco_bbox)

logger = logging.getLogger(__name__)

# Class IDs from labels_pouring.txt
CLASS_MOUTH = 0
CLASS_TROLLEY = 1

_POURING_CATEGORIES = [
    {"id": CLASS_MOUTH,   "name": "ladle_mouth"},
    {"id": CLASS_TROLLEY, "name": "trolley"},
]


class PouringProcessor:
    """
    Pouring detection with trolley locking, multi-probe brightness, and mould counting.

    State Machine:
    IDLE → mouth in expanded trolley >=1.0s → SESSION ACTIVE
    SESSION ACTIVE → head>=205 & tail>=160 for 0.20s → POUR START → LOCK TROLLEY (first time)
    POUR ACTIVE → no valid pouring probe for 0.80s → POUR END (min 2.0s) → mould counted
    SESSION ACTIVE → mouth absent >0.6s + 1.5s → SESSION END (keep mould data)
    SESSION END → mouth returns to locked trolley >=1.0s → SESSION RESTART (resume counting)
    ANY → mouth absent from locked trolley 5 min → CYCLE END (reset everything)
    """

    def __init__(self, db_manager, config, screenshot_dir: str, heat_cycle_manager=None,
                 camera_id_override: str = None, enable_display_meta: bool = True,
                 screenshot_writer=None):
        self.db_manager = db_manager
        self.config = config
        self.enable_display_meta = enable_display_meta
        self.screenshot_writer = screenshot_writer
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Customer/location
        self.customer_id = config.CUSTOMER_ID
        self.location = config.LOCATION
        self.camera_id = camera_id_override if camera_id_override else config.CAMERA_ID_STREAM_0

        # Detection thresholds
        self.mouth_conf = config.MOUTH_CONFIDENCE
        self.trolley_conf = config.TROLLEY_CONFIDENCE

        # Session timing
        self.session_start_dur = config.SESSION_START_DURATION
        self.session_end_dur = config.SESSION_END_DURATION

        # Reference resolution for pixel-space pouring geometry.
        self.pour_ref_width = int(getattr(config, 'POUR_REF_WIDTH', 1920))
        self.pour_ref_height = int(getattr(config, 'POUR_REF_HEIGHT', 1080))

        # Pour probe (multi-probe offsets)
        self.ref_probe_below_px = int(config.POUR_PROBE_BELOW_PX)
        self.ref_probe_offsets = [tuple(offset) for offset in config.POUR_PROBE_OFFSETS]
        self.ref_probe_radius = int(config.POUR_PROBE_RADIUS_PX)
        self.brightness_start = config.POUR_BRIGHTNESS_START
        self.brightness_end = config.POUR_BRIGHTNESS_END
        self.pour_start_dur = config.POUR_START_DURATION
        self.pour_end_dur = config.POUR_END_DURATION
        self.pour_min_dur = config.POUR_MIN_DURATION
        self.ref_probe_tail_dy = int(getattr(config, 'PROBE_TAIL_DY', 20))

        # Mould counting
        self.displacement_thresh = config.MOULD_DISPLACEMENT_THRESHOLD
        self.sustained_dur = config.MOULD_SUSTAINED_DURATION
        self.r_cluster = config.CLUSTER_R_CLUSTER
        self.r_merge_x = float(getattr(config, 'CLUSTER_R_MERGE_X', 0.08))
        self.r_merge_y = float(getattr(config, 'CLUSTER_R_MERGE_Y', 0.08))
        self.r_merge = config.CLUSTER_R_MERGE
        self.cluster_backtrack_guard = int(getattr(config, 'CLUSTER_BACKTRACK_CID_GUARD', 1))
        self.mould_switch_min_pour = config.MOULD_SWITCH_MIN_POUR_S
        self.min_cluster_pour_s = config.MIN_CLUSTER_POUR_S
        self.split_cooldown_s = float(getattr(config, 'MOULD_SPLIT_COOLDOWN_S', 1.5))
        self.split_rearm_baseline_s = float(getattr(config, 'MOULD_SPLIT_REARM_BASELINE_S', 0.5))
        self.split_dom_ratio = float(getattr(config, 'MOULD_SPLIT_DOM_RATIO', 1.35))
        self.axis_only_min_mag = float(getattr(config, 'MOULD_AXIS_ONLY_MIN_MAG', 0.05))
        self.ref_split_rearm_dx_px = float(getattr(config, 'MOULD_SPLIT_REARM_DX_PX', 10.0))
        self.ref_split_rearm_dy_px = float(getattr(config, 'MOULD_SPLIT_REARM_DY_PX', 14.0))
        # Pixel-based axis thresholds for split trigger (OR gate with magnitude)
        self.ref_split_min_dx_px = float(getattr(config, 'MOULD_SPLIT_MIN_DX_PX', 12.0))
        self.ref_split_min_dy_px = float(getattr(config, 'MOULD_SPLIT_MIN_DY_PX', 12.0))
        self.log_mould_displacement = bool(getattr(config, 'LOG_MOULD_DISPLACEMENT', False))
        self.mould_disp_log_interval_s = float(getattr(config, 'MOULD_DISP_LOG_INTERVAL_S', 0.25))
        self.fps = getattr(config, 'RTSP_FPS', 25.0)
        self.pour_start_frames = max(1, int(round(self.pour_start_dur * self.fps)))
        self.pour_end_frames = max(1, int(round(self.pour_end_dur * self.fps)))

        # Edge expand and tolerances
        self.ref_edge_expand_px = float(config.EDGE_EXPAND_PX)
        self.mouth_missing_tol = config.MOUTH_MISSING_TOL_S
        self.mouth_hold_s = float(getattr(config, 'MOUTH_HOLD_S', 0.4))
        self.cycle_timeout = config.POURING_CYCLE_TIMEOUT_S

        # Runtime-scaled pixel geometry (derived from reference-space constants).
        self._geometry_frame_size: Optional[Tuple[int, int]] = None
        self.geometry_scale_x = 1.0
        self.geometry_scale_y = 1.0
        self.probe_below_px = self.ref_probe_below_px
        self.probe_offsets = list(self.ref_probe_offsets)
        self.probe_radius = self.ref_probe_radius
        self.probe_tail_dy = self.ref_probe_tail_dy
        self.edge_expand_x_px = self.ref_edge_expand_px
        self.edge_expand_y_px = self.ref_edge_expand_px
        self.split_rearm_dx_px = self.ref_split_rearm_dx_px
        self.split_rearm_dy_px = self.ref_split_rearm_dy_px
        self.split_min_dx_px = self.ref_split_min_dx_px
        self.split_min_dy_px = self.ref_split_min_dy_px

        # Heat cycle manager (shared, injected from pipeline)
        self.heat_cycle_manager = heat_cycle_manager

        # --- Trolley locking state ---
        self.locked_trolley_id: Optional[int] = None
        self.locked_trolley_bbox: Optional[Tuple[int, int, int, int]] = None
        self.trolley_locked = False
        self.relock_hold_s = 0.8
        self.phantom_trolley_timeout = config.PHANTOM_TROLLEY_TIMEOUT_S
        self.trolley_last_detected_time: Optional[float] = None  # last time trolley was physically seen (not phantom)

        # --- Session state ---
        self.session_active = False
        self.mouth_inside_since: Optional[float] = None   # timestamp when mouth entered expanded trolley
        self.mouth_absent_since: Optional[float] = None    # timestamp when mouth left expanded trolley
        self.session_start_time: Optional[float] = None
        self.session_start_datetime: Optional[datetime] = None

        # --- Pour state ---
        self.pour_active = False
        self.brightness_above_since: Optional[float] = None
        self.brightness_below_since: Optional[float] = None
        self.pour_on_count = 0
        self.pour_off_count = 0
        self.pour_start_time: Optional[float] = None
        self.pour_start_datetime: Optional[datetime] = None
        self.pour_sync_id: Optional[str] = None
        self.pour_slno: Optional[int] = None
        self.active_mould_id: Optional[int] = None  # mould id tied to current active pour record
        self.active_mould_start_time: Optional[float] = None
        self.active_mould_start_datetime: Optional[datetime] = None
        self.active_mould_start_norm: Optional[Tuple[float, float]] = None
        self.last_pour_duration: float = 0.0  # duration of last completed pour
        self.active_probe_track_id: Optional[int] = None
        self.active_probe_last_seen_frame: int = -999999
        self.active_probe_bbox: Optional[Tuple[int, int, int, int]] = None
        self.active_probe_bbox_valid = False
        self.active_probe_pt_px = (0.0, 0.0)
        self.active_probe_pt_valid = False
        self.active_probe_from_hold = False
        self.frozen_probe_active = False
        self.frozen_probe_x = 0.0
        self.frozen_probe_y = 0.0
        self.frozen_probe_bbox: Optional[Tuple[int, int, int, int]] = None
        self.frozen_probe_bbox_valid = False

        # --- Mould counter state ---
        self.anchor_position: Optional[Tuple[float, float]] = None  # normalized mouth pos (trolley-relative)
        self.anchor_set = False  # anchor set on pour start
        self.displacement_hold_frames: Optional[int] = None  # frame counter for sustained displacement
        self.sustained_hold_frames = int(round(self.sustained_dur * self.fps))
        self.mouth_hold_frames = int(round(self.mouth_hold_s * self.fps))
        # Direction consistency guard for split hold
        self.split_hold_quadrant: Optional[int] = None  # 1, 2, 3, or 4
        self.split_rearm_required: bool = False
        self.split_rearm_below_since: Optional[float] = None
        self.split_rearm_axis: Optional[str] = None  # 'x' or 'y'
        self.moved_positions: List[Tuple[float, float, float]] = []  # normalized (x, y) + pour duration
        self.mould_count = 0  # monotonic accepted mould count
        self.clustered_mould_count = 0  # spatial validation metric only
        self.next_mould_id = 1
        self.last_split_time: Optional[float] = None
        self._last_mould_disp_log_ts: Optional[float] = None

        # --- Last known probe state (for screenshot annotations) ---
        self._last_probe_base: Optional[Tuple[int, int]] = None  # (base_x, base_y) before offsets
        self._last_probe_brightness: Optional[float] = None
        self._last_probe_tail_brightness: Optional[float] = None
        self._last_probe_is_pouring: Optional[bool] = None

        # --- Cycle timeout tracking ---
        self.mouth_last_seen_in_trolley: Optional[float] = None  # last time mouth was inside locked trolley region
        self.cycle_start_time: Optional[float] = None
        self.cycle_start_datetime: Optional[datetime] = None

        # Frame dimensions (set on first frame)
        self._frame_w = 0
        self._frame_h = 0

        # Frame counter
        self._frame_count = 0
        self._relock_candidate_id: Optional[int] = None
        self._relock_candidate_since: Optional[float] = None

        # --- DS-native inference overlay toggle (recorded post-OSD via tee branch) ---
        self.enable_inference_video = bool(getattr(config, 'ENABLE_INFERENCE_VIDEO', False))

        # --- Cached display state for decoupled-mode recording overlay probe ---
        # Updated by process_frame() on the analysis branch; read by the main-path
        # display_meta writer probe (osd_sink_pad_probe_stream0_display_meta).
        self._cached_display_data: Optional[Dict] = None

        # --- Per-mould timing tracker (for live overlay) ---
        self.mould_completed_times = {}  # mould_id → total frames accumulated
        self.mould_records: List[Dict[str, Any]] = []  # accepted moulds with timing/axis/cluster metadata
        self.completed_segments: List[Dict[str, Any]] = []
        self._active_segment: Optional[Dict[str, Any]] = None
        self._synced_mould_slot_ids = set()
        self._pour_start_time_wall = None  # Wall-clock timestamp for live overlay display

        logger.info("PouringProcessor initialized (aligned with standalone doc)")
        logger.info(f"  Mouth conf: {self.mouth_conf}, Trolley conf: {self.trolley_conf}")
        logger.info(f"  Session: start={self.session_start_dur}s, end={self.session_end_dur}s")
        logger.info(
            "  Pour geometry ref-space: %dx%d",
            self.pour_ref_width,
            self.pour_ref_height,
        )
        logger.info(f"  Pour: multi-probe offsets={self.probe_offsets}, below={self.probe_below_px}px")
        logger.info(f"  Pour brightness: start>{self.brightness_start}, end<{self.brightness_end}")
        logger.info(
            f"  Pour timing: start={self.pour_start_dur}s ({self.pour_start_frames}f), "
            f"end={self.pour_end_dur}s ({self.pour_end_frames}f), min={self.pour_min_dur}s"
        )
        logger.info(
            f"  Mould: disp>{self.displacement_thresh}, sustained={self.sustained_dur}s, "
            f"min_pour={self.mould_switch_min_pour}s, backtrack_guard={self.cluster_backtrack_guard}"
        )
        logger.info(
            f"  Clustering: r_cluster={self.r_cluster:.3f}, r_merge_x={self.r_merge_x:.3f}, "
            f"r_merge_y={self.r_merge_y:.3f}, min_cluster_pour={self.min_cluster_pour_s}s"
        )
        logger.info(
            f"  Split gates: min_dx_px={self.split_min_dx_px:.1f}, min_dy_px={self.split_min_dy_px:.1f}, "
            f"dom_ratio={self.split_dom_ratio:.2f}, axis_only_min_mag={self.axis_only_min_mag:.3f}, "
            f"rearm_dx_px={self.split_rearm_dx_px:.1f}, rearm_dy_px={self.split_rearm_dy_px:.1f}, "
            f"rearm_baseline={self.split_rearm_baseline_s:.2f}s, cooldown={self.split_cooldown_s:.2f}s"
        )
        logger.info(
            f"  Mould displacement logging: enabled={self.log_mould_displacement}, "
            f"interval={self.mould_disp_log_interval_s}s"
        )
        logger.info(
            f"  Edge expand: x={self.edge_expand_x_px:.1f}px y={self.edge_expand_y_px:.1f}px, Mouth missing tol: {self.mouth_missing_tol}s, "
            f"mouth hold: {self.mouth_hold_s}s"
        )
        logger.info(f"  Cycle timeout: {self.cycle_timeout}s, Phantom trolley timeout: {self.phantom_trolley_timeout:.0f}s")
        logger.info(
            "  DS-native inference overlay enabled=%s (display_meta=%s)",
            self.enable_inference_video,
            self.enable_display_meta,
        )

    def needs_frame(self, obj_count: int) -> bool:
        return bool(obj_count > 0 or self.pour_active)

    @staticmethod
    def _scale_pixel_value(value, scale, minimum=0, as_float=False):
        scaled = float(value) * float(scale)
        if as_float:
            return max(float(minimum), scaled)
        return max(int(minimum), int(round(scaled)))

    @staticmethod
    def _scale_offset_value(value, scale):
        return int(round(float(value) * float(scale)))

    def _scaled_probe_offsets(self, scale_x, scale_y):
        seen = set()
        scaled = []
        for dx, dy in self.ref_probe_offsets:
            item = (
                self._scale_offset_value(dx, scale_x),
                self._scale_offset_value(dy, scale_y),
            )
            if item in seen:
                continue
            seen.add(item)
            scaled.append(item)
        return scaled or [(0, 0)]

    def _update_runtime_geometry(self, frame_w, frame_h):
        if frame_w <= 0 or frame_h <= 0:
            return

        frame_size = (int(frame_w), int(frame_h))
        if self._geometry_frame_size == frame_size:
            return

        ref_w = max(1, int(self.pour_ref_width))
        ref_h = max(1, int(self.pour_ref_height))
        scale_x = float(frame_w) / float(ref_w)
        scale_y = float(frame_h) / float(ref_h)
        scale_min = min(scale_x, scale_y)

        self.geometry_scale_x = scale_x
        self.geometry_scale_y = scale_y
        self.probe_below_px = self._scale_pixel_value(self.ref_probe_below_px, scale_y, minimum=1)
        self.probe_radius = self._scale_pixel_value(self.ref_probe_radius, scale_min, minimum=1)
        self.probe_tail_dy = self._scale_pixel_value(self.ref_probe_tail_dy, scale_y, minimum=1)
        self.probe_offsets = self._scaled_probe_offsets(scale_x, scale_y)
        self.edge_expand_x_px = self._scale_pixel_value(
            self.ref_edge_expand_px, scale_x, minimum=0.0, as_float=True
        )
        self.edge_expand_y_px = self._scale_pixel_value(
            self.ref_edge_expand_px, scale_y, minimum=0.0, as_float=True
        )
        self.split_min_dx_px = float(self._scale_pixel_value(self.ref_split_min_dx_px, scale_x, minimum=0))
        self.split_min_dy_px = float(self._scale_pixel_value(self.ref_split_min_dy_px, scale_y, minimum=0))
        self.split_rearm_dx_px = float(self._scale_pixel_value(self.ref_split_rearm_dx_px, scale_x, minimum=0))
        self.split_rearm_dy_px = float(self._scale_pixel_value(self.ref_split_rearm_dy_px, scale_y, minimum=0))
        self._geometry_frame_size = frame_size

        logger.info(
            "[pour-geometry] ref=%dx%d actual=%dx%d scale=(%.3f, %.3f) "
            "edge_expand=(%.1f, %.1f) probe_below=%d probe_radius=%d probe_tail_dy=%d "
            "probe_offsets=%s split_min=(%.1f, %.1f) split_rearm=(%.1f, %.1f)",
            ref_w,
            ref_h,
            frame_w,
            frame_h,
            scale_x,
            scale_y,
            self.edge_expand_x_px,
            self.edge_expand_y_px,
            self.probe_below_px,
            self.probe_radius,
            self.probe_tail_dy,
            self.probe_offsets,
            self.split_min_dx_px,
            self.split_min_dy_px,
            self.split_rearm_dx_px,
            self.split_rearm_dy_px,
        )

    # =========================================================================
    # Main entry point (called from OSD sink pad probe)
    # =========================================================================

    def process_frame(self, frame_meta, frame, timestamp, datetime_obj,
                      frame_num=None, batch_meta=None):
        """
        Process a single frame from Stream 0.

        Args:
            frame_meta: NvDsFrameMeta from DeepStream batch meta
            frame: RGBA numpy array (may be None if extraction failed)
            timestamp: time.time() wall clock
            datetime_obj: datetime.now()
            frame_num: optional frame counter for logging
            batch_meta: NvDsBatchMeta (optional, for custom OSD overlays)
        """
        self._frame_count += 1

        if frame is not None:
            if hasattr(frame, "ndim") and frame.ndim == 2:
                frame_h = frame.shape[0] * 2 // 3
                frame_w = frame.shape[1]
            else:
                frame_h, frame_w = frame.shape[:2]
        else:
            frame_w = getattr(self.config, 'STREAM_0_MUX_WIDTH', 1600)
            frame_h = getattr(self.config, 'STREAM_0_MUX_HEIGHT', 900)

        self._frame_w = frame_w
        self._frame_h = frame_h
        self._update_runtime_geometry(frame_w, frame_h)

        # 1. Extract detections
        mouths, trolleys = self._extract_detections(frame_meta)

        if frame_num and frame_num % 1000 == 0:
            logger.info(
                f"Stream 0 Frame {frame_num}: {len(mouths)} mouths, {len(trolleys)} trolleys"
            )

        # 2. Find the relevant trolley (locked or best candidate)
        target_trolley = self._get_target_trolley(trolleys, timestamp, mouths)

        # 3. Check mouth-in-trolley (with EDGE_EXPAND)
        mouth_in_trolley = False
        best_mouth = None
        if target_trolley and mouths:
            # IMPORTANT: choose mouth only from the target trolley expanded region.
            # This prevents global max-confidence mouth in other areas from driving pour logic.
            best_mouth = self._select_best_mouth_for_trolley(mouths, target_trolley)
            mouth_in_trolley = best_mouth is not None
            if best_mouth:
                # Keep latest probe base available for session/pour screenshots and OSD overlay.
                mx, my_bottom = best_mouth['bottom_center']
                self._last_probe_base = (mx, my_bottom + self.probe_below_px)
                if not self.session_active:
                    self._last_probe_brightness = None
                    self._last_probe_tail_brightness = None
                    self._last_probe_is_pouring = None

        # 4. Update trolley bbox (keep latest position for locked trolley; skip phantom)
        if target_trolley and self.trolley_locked and not target_trolley.get('is_phantom', False):
            self.locked_trolley_bbox = target_trolley['bbox']

        # 5. Session manager
        self._update_session(mouth_in_trolley, best_mouth, target_trolley,
                             mouths, trolleys, timestamp, datetime_obj, frame)

        # 6. Refresh heat cycle presence before any pour-start business logic so
        # fresh pours inherit the current heat_no on their very first DB row.
        if (
            self.heat_cycle_manager
            and self.session_active
            and mouth_in_trolley
            and best_mouth is not None
        ):
            self.heat_cycle_manager.update_pouring_session_presence(
                int(best_mouth['track_id']), timestamp, datetime_obj
            )

        # 7. Pour detector (during active session, with frame data)
        pour_probe_mouth = None
        if self.session_active and frame is not None:
            pour_probe_mouth = self._update_pour(
                mouths, frame, timestamp, datetime_obj, trolleys, target_trolley
            )

        # 8. Mould counter (during active session)
        mould_mouth = pour_probe_mouth or best_mouth
        if self.session_active and self.pour_active and mould_mouth and target_trolley:
            self._update_mould_counter(mould_mouth, target_trolley, timestamp, datetime_obj)

        # 9. Pouring cycle timeout check (5 min)
        self._check_cycle_timeout(timestamp, datetime_obj, mouths, trolleys, frame)

        # 10. Finalize heat cycles periodically
        self._finalize_heat_cycles_if_due(timestamp, datetime_obj)

        # 11. Cache latest detection state for the decoupled-mode recording display_meta probe.
        # The main-path probe (osd_sink_pad_probe_stream0_display_meta) reads this to write
        # NvDsDisplayMeta on the main GStreamer path without re-running analysis.
        self._cached_display_data = {
            'mouths': mouths,
            'trolleys': trolleys,
            'target_trolley': target_trolley,
            'timestamp': timestamp,
            'datetime_obj': datetime_obj,
        }

        # 12. DS-native overlay annotations (rendered by nvosd and captured via tee branch)
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

    # =========================================================================
    # Detection extraction
    # =========================================================================

    def _extract_detections(self, frame_meta):
        """Extract ladle_mouth and trolley detections from NvDsObjectMeta."""
        mouths = []
        trolleys = []

        if frame_meta is None:
            return mouths, trolleys

        l_obj = getattr(frame_meta, 'obj_meta_list', None)
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            class_id = int(obj_meta.class_id)
            conf = float(obj_meta.confidence)
            rect = obj_meta.rect_params
            x1 = int(rect.left)
            y1 = int(rect.top)
            x2 = int(rect.left + rect.width)
            y2 = int(rect.top + rect.height)
            track_id = int(obj_meta.object_id)

            det = {
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'track_id': track_id,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'bottom_center': ((x1 + x2) // 2, y2),
            }

            if class_id == CLASS_MOUTH and conf >= self.mouth_conf:
                mouths.append(det)
            elif class_id == CLASS_TROLLEY and conf >= self.trolley_conf:
                trolleys.append(det)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        return mouths, trolleys

    # =========================================================================
    # Trolley targeting (locking)
    # =========================================================================

    def _get_target_trolley(self, trolleys, timestamp=None, mouths=None):
        """Get the target trolley: locked one if exists, else best candidate.

        When the locked trolley disappears (e.g., occluded by ladle during pour),
        returns a phantom trolley from the last known bbox for up to
        phantom_trolley_timeout seconds, allowing sessions to continue uninterrupted.
        """
        ts = timestamp or time.time()

        if self.trolley_locked and self.locked_trolley_id is not None:
            # 1. Find locked trolley by track_id
            for t in trolleys:
                if t['track_id'] == self.locked_trolley_id:
                    self.trolley_last_detected_time = ts  # real detection — update clock
                    self._relock_candidate_id = None
                    self._relock_candidate_since = None
                    return t

            # 2. Locked trolley missing — try relock if any trolleys are visible
            if trolleys:
                relock = self._select_relock_trolley(trolleys, mouths=mouths)
                if relock and self._should_relock_trolley(relock, trolleys, ts):
                    self._relock_trolley(relock, ts, reason="missing_locked_id")
                    self.trolley_last_detected_time = ts
                    self._relock_candidate_id = None
                    self._relock_candidate_since = None
                    return relock

            # 3. No trolleys visible (or relock failed) — use phantom frozen bbox
            if (self.locked_trolley_bbox is not None and
                    self.trolley_last_detected_time is not None):
                phantom_age = ts - self.trolley_last_detected_time
                if phantom_age <= self.phantom_trolley_timeout:
                    x1, y1, x2, y2 = self.locked_trolley_bbox
                    logger.debug(
                        "[trolley] phantom bbox (age=%.1fs, id=%s, bbox=%s)",
                        phantom_age, self.locked_trolley_id, self.locked_trolley_bbox,
                    )
                    return {
                        'track_id': self.locked_trolley_id,
                        'bbox': self.locked_trolley_bbox,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'confidence': 0.0,
                        'is_phantom': True,
                    }
            return None

        # No lock yet — return highest-confidence trolley
        if not trolleys:
            return None
        return max(trolleys, key=lambda t: t['confidence'])

    @staticmethod
    def _bbox_iou(a, b):
        """Compute IoU between two bboxes (x1,y1,x2,y2)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def _select_relock_trolley(self, trolleys, mouths=None):
        """Pick best trolley to relock when locked ID is missing."""
        def _has_mouth_inside(trolley):
            if not mouths:
                return False
            return any(self._is_mouth_in_expanded_trolley(m, trolley) for m in mouths)

        # Prefer candidates that currently contain a mouth.
        candidates = [t for t in trolleys if _has_mouth_inside(t)]
        if not candidates:
            # No strong relock signal in this frame.
            return None

        if self.locked_trolley_bbox:
            best = None
            best_iou = -1.0
            for t in candidates:
                iou = self._bbox_iou(self.locked_trolley_bbox, t['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best = t
            if best is not None:
                return best
        return max(candidates, key=lambda t: t['confidence'])

    @staticmethod
    def _bbox_center_distance_norm(a, b):
        """Center distance normalized by locked bbox max dimension."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        base = max(ax2 - ax1, ay2 - ay1, 1)
        return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / float(base)

    def _should_relock_trolley(self, candidate, trolleys, timestamp):
        """Gate relock to avoid jitter while allowing move/new trolley transitions."""
        cid = candidate['track_id']
        if self._relock_candidate_id != cid:
            self._relock_candidate_id = cid
            self._relock_candidate_since = timestamp
            return False

        if self._relock_candidate_since is None:
            self._relock_candidate_since = timestamp
            return False

        if timestamp - self._relock_candidate_since < self.relock_hold_s:
            return False

        if not self.locked_trolley_bbox:
            return True

        iou = self._bbox_iou(self.locked_trolley_bbox, candidate['bbox'])
        moved_norm = self._bbox_center_distance_norm(self.locked_trolley_bbox, candidate['bbox'])
        moved = moved_norm >= self.displacement_thresh
        same_physical = iou >= 0.25
        new_trolley_present = len(trolleys) >= 2
        return same_physical or moved or new_trolley_present

    def _relock_trolley(self, trolley, timestamp, reason=""):
        """Re-lock to a new trolley (e.g., tracker ID switched)."""
        prev_id = self.locked_trolley_id
        self.locked_trolley_id = trolley['track_id']
        self.locked_trolley_bbox = trolley['bbox']
        self.trolley_locked = True
        self.mouth_last_seen_in_trolley = timestamp
        if self.heat_cycle_manager:
            self.heat_cycle_manager.lock_trolley(trolley['track_id'])
        logger.info(
            f"[trolley] RELOCK T{prev_id} -> T{trolley['track_id']} "
            f"at bbox {trolley['bbox']} ({reason})"
        )

    def _is_mouth_in_expanded_trolley(self, mouth, trolley):
        """Check if mouth center is inside trolley bbox with only the top edge expanded."""
        mx, my = mouth['center']
        tx1, ty1, tx2, ty2 = trolley['bbox']
        return tx1 <= mx <= tx2 and (ty1 - self.edge_expand_y_px) <= my <= ty2

    def _lock_trolley(self, trolley, timestamp):
        """Lock onto a trolley (happens on first pour start)."""
        self.locked_trolley_id = trolley['track_id']
        self.locked_trolley_bbox = trolley['bbox']
        self.trolley_locked = True
        self.mouth_last_seen_in_trolley = timestamp
        if self.cycle_start_time is None:
            self.cycle_start_time = timestamp
            self.cycle_start_datetime = datetime.now()
        logger.info(f"[trolley] LOCKED to T{trolley['track_id']} at bbox {trolley['bbox']}")

        # Also lock in heat cycle manager
        if self.heat_cycle_manager:
            self.heat_cycle_manager.lock_trolley(trolley['track_id'])

    def _select_best_mouth_for_trolley(self, mouths, trolley):
        """Select highest-confidence mouth whose probe point falls inside the trolley bbox (top edge expanded)."""
        if not mouths or not trolley:
            return None
        tx1, ty1, tx2, ty2 = trolley['bbox']
        top_expanded = (tx1, ty1 - self.edge_expand_y_px, tx2, ty2)
        candidates = [m for m in mouths
                      if self._point_in_bbox(*self._mouth_probe_point(m), top_expanded)]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m['confidence'])

    def _get_locked_trolley(self, trolleys):
        """Return locked trolley from current detections if available."""
        if not self.trolley_locked or self.locked_trolley_id is None or not trolleys:
            return None
        for trolley in trolleys:
            if trolley['track_id'] == self.locked_trolley_id:
                return trolley
        return None

    # =========================================================================
    # Sub-system 1: Session Manager (with mouth missing tolerance)
    # =========================================================================

    def _update_session(self, mouth_in_trolley, best_mouth, target_trolley,
                        mouths, trolleys, timestamp, datetime_obj, frame):
        """Session state machine with MOUTH_MISSING_TOL_S tolerance."""

        if mouth_in_trolley:
            # Mouth is inside (expanded) trolley
            self.mouth_absent_since = None
            self.mouth_last_seen_in_trolley = timestamp

            if not self.session_active:
                # Track time inside for session start
                if self.mouth_inside_since is None:
                    self.mouth_inside_since = timestamp
                elif timestamp - self.mouth_inside_since >= self.session_start_dur:
                    self._start_session(target_trolley, timestamp, datetime_obj,
                                        mouths, trolleys, frame)
            else:
                # Session active, mouth still inside — reset inside tracker
                self.mouth_inside_since = timestamp

        else:
            # Mouth NOT inside (expanded) trolley
            self.mouth_inside_since = None

            if self.session_active:
                # Reference: end session if out_count >= N_EXIT (session_end_dur of consecutive
                # absence) OR probe_missing > MOUTH_MISSING_TOL — whichever fires first.
                if self.mouth_absent_since is None:
                    self.mouth_absent_since = timestamp
                else:
                    absence_duration = timestamp - self.mouth_absent_since
                    # Condition 1: sustained absence >= session_end_dur (N_EXIT path)
                    # Condition 2: probe not seen for > mouth_missing_tol (probe_missing path)
                    probe_missing = (
                        timestamp - self.mouth_last_seen_in_trolley
                        if self.mouth_last_seen_in_trolley is not None
                        else float('inf')
                    )
                    if absence_duration >= self.session_end_dur or probe_missing > self.mouth_missing_tol:
                        self._end_session(timestamp, datetime_obj,
                                          mouths, trolleys, frame)

    def _start_session(self, trolley, timestamp, datetime_obj, mouths, trolleys, frame):
        """Start a new pouring session."""
        self.session_active = True
        self.session_start_time = timestamp
        self.session_start_datetime = datetime_obj
        self.mouth_inside_since = None

        tid = trolley['track_id'] if trolley else '?'
        logger.info(
            f"[session] START - mouth inside trolley T{tid} "
            f"for {self.session_start_dur}s"
        )
        probe_base = self._last_probe_base or self._get_probe_base_from_mouths(mouths)
        self._save_event_screenshot(
            "SESSION START", mouths, trolleys, frame, datetime_obj,
            probe_point=probe_base,
            probe_brightness=self._last_probe_brightness,
        )

    def _end_session(self, timestamp, datetime_obj, mouths, trolleys, frame):
        """End the current pouring session. Preserves mould data for locked trolley."""
        # End any active pour first
        if self.pour_active:
            self._end_pour(timestamp, datetime_obj, mouths, trolleys, frame)

        duration = timestamp - self.session_start_time if self.session_start_time else 0
        logger.info(
            f"[session] END - duration={duration:.1f}s, moulds={self.mould_count}"
            f" (mould data preserved for locked trolley)"
        )
        probe_base = self._last_probe_base or self._get_probe_base_from_mouths(mouths)
        self._save_event_screenshot(
            "SESSION END", mouths, trolleys, frame, datetime_obj,
            probe_point=probe_base,
            probe_brightness=self._last_probe_brightness,
            extra_info=f"Duration: {duration:.1f}s  Moulds: {self.mould_count}"
        )

        # Reset session state but KEEP mould data (persistence across sessions)
        self.session_active = False
        self.session_start_time = None
        self.session_start_datetime = None
        self.mouth_inside_since = None
        self.mouth_absent_since = None

        # Pour state reset
        self.brightness_above_since = None
        self.brightness_below_since = None
        self.pour_on_count = 0
        self.pour_off_count = 0
        self._clear_frozen_probe_state()
        self._clear_active_probe_state()

        # Do NOT reset: mould_count, moved_positions, anchor_position, locked trolley
        # These persist until pouring cycle ends (5 min timeout)

        if self.heat_cycle_manager:
            self.heat_cycle_manager.notify_session_ended()

    # =========================================================================
    # Sub-system 2: Pour Detector (multi-probe)
    # =========================================================================

    @staticmethod
    def _point_in_bbox(px, py, bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= px <= x2 and y1 <= py <= y2

    def _mouth_probe_point(self, mouth):
        mx, my_bottom = mouth['bottom_center']
        return float(mx), float(my_bottom + self.probe_below_px)

    def _measure_head_tail_scores(self, frame, probe_x, probe_y):
        head_score = self._measure_multi_probe_brightness(frame, probe_x, probe_y)
        tail_score = self._measure_multi_probe_brightness(
            frame, probe_x, probe_y + self.probe_tail_dy
        )
        return head_score, tail_score

    def _probe_is_pouring(self, head_score, tail_score):
        return head_score >= self.brightness_start and tail_score >= self.brightness_end

    def _build_probe_hits(self, mouths, target_trolley, frame):
        if not target_trolley or frame is None:
            return []

        hits = []
        trolley_bbox = target_trolley['bbox']
        for mouth in mouths or []:
            probe_x, probe_y = self._mouth_probe_point(mouth)
            if not self._point_in_bbox(probe_x, probe_y, trolley_bbox):
                continue

            head_score, tail_score = self._measure_head_tail_scores(frame, probe_x, probe_y)
            hits.append(
                {
                    "mouth": mouth,
                    "probe_x": probe_x,
                    "probe_y": probe_y,
                    "head_score": head_score,
                    "tail_score": tail_score,
                    "is_pouring": self._probe_is_pouring(head_score, tail_score),
                }
            )
        return hits

    def _select_probe_hit(self, probe_hits):
        if not probe_hits:
            return None

        for hit in probe_hits:
            if hit["mouth"].get("track_id") == self.active_probe_track_id:
                return hit

        pouring_hits = [hit for hit in probe_hits if hit["is_pouring"]]
        if pouring_hits:
            return max(
                pouring_hits,
                key=lambda hit: (
                    hit["tail_score"],
                    hit["head_score"],
                    hit["mouth"].get("confidence", 0.0),
                ),
            )

        return max(
            probe_hits,
            key=lambda hit: (
                hit["head_score"],
                hit["mouth"].get("confidence", 0.0),
            ),
        )

    def _update_active_probe_state(self, selected_hit):
        mouth = selected_hit["mouth"]
        self.active_probe_track_id = mouth.get("track_id")
        self.active_probe_bbox = tuple(mouth.get("bbox", (0, 0, 0, 0)))
        self.active_probe_bbox_valid = True
        self.active_probe_pt_px = (
            float(selected_hit["probe_x"]),
            float(selected_hit["probe_y"]),
        )
        self.active_probe_pt_valid = True
        self.active_probe_last_seen_frame = self._frame_count
        self.active_probe_from_hold = False

    def _clear_active_probe_state(self):
        self.active_probe_track_id = None
        self.active_probe_last_seen_frame = -999999
        self.active_probe_bbox = None
        self.active_probe_bbox_valid = False
        self.active_probe_pt_px = (0.0, 0.0)
        self.active_probe_pt_valid = False
        self.active_probe_from_hold = False

    def _clear_frozen_probe_state(self):
        self.frozen_probe_active = False
        self.frozen_probe_x = 0.0
        self.frozen_probe_y = 0.0
        self.frozen_probe_bbox = None
        self.frozen_probe_bbox_valid = False

    def _update_pour(self, mouths, frame, timestamp, datetime_obj, trolleys, target_trolley):
        """Reference-style head/tail probe gating on the supported Python path."""
        probe_hits = self._build_probe_hits(mouths, target_trolley, frame)
        any_probe_pouring = any(hit["is_pouring"] for hit in probe_hits)
        selected = self._select_probe_hit(probe_hits)

        if selected is not None:
            self._update_active_probe_state(selected)
        else:
            self.active_probe_from_hold = False

        use_probe = False
        using_hold = False
        selected_mouth = selected["mouth"] if selected is not None else None
        probe_x_use = 0.0
        probe_y_use = 0.0
        head_score = 0.0
        tail_score = 0.0
        active_probe_pouring = False

        if self.pour_active and self.frozen_probe_active:
            probe_x_use = self.frozen_probe_x
            probe_y_use = self.frozen_probe_y
            head_score, tail_score = self._measure_head_tail_scores(frame, probe_x_use, probe_y_use)
            active_probe_pouring = self._probe_is_pouring(head_score, tail_score)
            use_probe = True
        elif selected is not None:
            probe_x_use = float(selected["probe_x"])
            probe_y_use = float(selected["probe_y"])
            head_score = float(selected["head_score"])
            tail_score = float(selected["tail_score"])
            active_probe_pouring = bool(selected["is_pouring"])
            use_probe = True
        elif (
            self.pour_active
            and self.active_probe_pt_valid
            and (self._frame_count - self.active_probe_last_seen_frame) <= self.mouth_hold_frames
        ):
            probe_x_use, probe_y_use = self.active_probe_pt_px
            head_score, tail_score = self._measure_head_tail_scores(frame, probe_x_use, probe_y_use)
            active_probe_pouring = self._probe_is_pouring(head_score, tail_score)
            use_probe = True
            using_hold = True
            self.active_probe_from_hold = True

        if not use_probe:
            self.pour_on_count = 0 if not self.pour_active else self.pour_on_count
            self._last_probe_brightness = None
            self._last_probe_tail_brightness = None
            self._last_probe_is_pouring = None
            return None

        self._last_probe_base = (int(round(probe_x_use)), int(round(probe_y_use)))
        self._last_probe_brightness = head_score
        self._last_probe_tail_brightness = tail_score
        self._last_probe_is_pouring = self._probe_is_pouring(head_score, tail_score)

        if not self.pour_active:
            self.pour_on_count = (self.pour_on_count + 1) if any_probe_pouring else 0
            self.pour_off_count = 0
            self.brightness_above_since = None
            self.brightness_below_since = None

            if self.pour_on_count >= self.pour_start_frames and selected_mouth is not None:
                # If the tracked (selected) probe isn't the one triggering pouring,
                # anchor the frozen probe to the actually-pouring hit instead so that
                # the active pour monitors the correct location.
                start_brightness = head_score
                start_probe_x = int(round(probe_x_use))
                start_probe_y = int(round(probe_y_use))
                start_mouth = selected_mouth
                if selected is None or not selected["is_pouring"]:
                    pouring_hits = [h for h in probe_hits if h["is_pouring"]]
                    if pouring_hits:
                        best_pour_hit = max(pouring_hits, key=lambda h: h["head_score"])
                        start_brightness = best_pour_hit["head_score"]
                        start_probe_x = int(round(best_pour_hit["probe_x"]))
                        start_probe_y = int(round(best_pour_hit["probe_y"]))
                        start_mouth = best_pour_hit["mouth"]
                self._start_pour(
                    timestamp,
                    datetime_obj,
                    mouths,
                    trolleys,
                    frame,
                    start_brightness,
                    start_probe_x,
                    start_probe_y,
                    target_trolley,
                    start_mouth,
                )
                self.pour_on_count = 0
                self.pour_off_count = 0
                self.frozen_probe_active = True
                self.frozen_probe_x = float(start_probe_x)
                self.frozen_probe_y = float(start_probe_y)
                if start_mouth is not None:
                    self.frozen_probe_bbox = tuple(start_mouth.get("bbox", (0, 0, 0, 0)))
                    self.frozen_probe_bbox_valid = True
                elif self.active_probe_bbox_valid:
                    self.frozen_probe_bbox = self.active_probe_bbox
                    self.frozen_probe_bbox_valid = True
                else:
                    self.frozen_probe_bbox = None
                    self.frozen_probe_bbox_valid = False
            return selected_mouth

        trolley_still_pouring = any_probe_pouring or (using_hold and active_probe_pouring)
        self.pour_on_count = 0
        self.pour_off_count = 0 if trolley_still_pouring else (self.pour_off_count + 1)
        self.brightness_above_since = None
        self.brightness_below_since = None

        if self.pour_off_count >= self.pour_end_frames:
            self._end_pour(
                timestamp,
                datetime_obj,
                mouths,
                trolleys,
                frame,
                best_mouth=selected_mouth,
            )
            self.pour_on_count = 0
            self.pour_off_count = 0
            self._clear_frozen_probe_state()

        return selected_mouth

    def _measure_multi_probe_brightness(self, frame, base_x, base_y):
        """Measure average brightness across multiple probe patches.

        Reference parity:
        - 5 probes around the mouth probe point
        - patch score = HSV Value = max(R, G, B)
        - frame score = mean of valid patch scores
        """
        if frame is None:
            return 0.0
        if not hasattr(frame, "ndim"):
            return 0.0

        if frame.ndim == 2:
            return self._measure_multi_probe_brightness_nv12(frame, base_x, base_y)

        h, w = frame.shape[:2]
        r = self.probe_radius
        base_x_i = int(round(float(base_x)))
        base_y_i = int(round(float(base_y)))
        values = []

        for dx, dy in self.probe_offsets:
            px = int(base_x_i + dx)
            py = int(base_y_i + dy)
            x1 = max(0, int(px - r))
            y1 = max(0, int(py - r))
            x2 = min(w, int(px + r))
            y2 = min(h, int(py + r))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.ndim != 3 or roi.shape[2] < 3:
                continue
            rgb = roi[:, :, :3]
            values.append(float(np.mean(np.max(rgb, axis=2))))

        return sum(values) / len(values) if values else 0.0

    def _measure_multi_probe_brightness_nv12(self, frame, base_x, base_y):
        """Measure reference-style V-channel brightness from a tiny NV12 ROI."""
        visible_h = frame.shape[0] * 2 // 3
        w = frame.shape[1]
        r = self.probe_radius
        base_x_i = int(round(float(base_x)))
        base_y_i = int(round(float(base_y)))
        boxes = []
        for dx, dy in self.probe_offsets:
            px = int(base_x_i + dx)
            py = int(base_y_i + dy)
            x1 = max(0, int(px - r))
            y1 = max(0, int(py - r))
            x2 = min(w, int(px + r))
            y2 = min(visible_h, int(py + r))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        if not boxes:
            return 0.0

        roi_x1 = min(b[0] for b in boxes)
        roi_y1 = min(b[1] for b in boxes)
        roi_x2 = max(b[2] for b in boxes)
        roi_y2 = max(b[3] for b in boxes)

        roi_x1 = max(0, roi_x1 & ~1)
        roi_y1 = max(0, roi_y1 & ~1)
        roi_x2 = min(w, (roi_x2 + 1) & ~1)
        roi_y2 = min(visible_h, (roi_y2 + 1) & ~1)
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return 0.0

        y_plane = frame[:visible_h, :]
        uv_plane = frame[visible_h:, :]
        roi_y = y_plane[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_uv = uv_plane[roi_y1 // 2:roi_y2 // 2, roi_x1:roi_x2]
        if roi_y.size == 0 or roi_uv.size == 0:
            return 0.0

        nv12_roi = np.vstack([roi_y, roi_uv])
        rgb_roi = cv2.cvtColor(nv12_roi, cv2.COLOR_YUV2RGB_NV12)
        values = []
        for x1, y1, x2, y2 in boxes:
            rx1 = x1 - roi_x1
            ry1 = y1 - roi_y1
            rx2 = x2 - roi_x1
            ry2 = y2 - roi_y1
            roi = rgb_roi[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue
            values.append(float(np.mean(np.max(roi, axis=2))))
        return sum(values) / len(values) if values else 0.0

    def _prepare_frame_for_annotation(self, frame):
        """Convert NV12 or RGBA frames into a screenshot-safe RGBA image."""
        if frame is None:
            return None
        if frame.ndim == 2:
            frame_h = frame.shape[0] * 2 // 3
            gray = frame[:frame_h, :]
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
        return frame

    @staticmethod
    def _dominant_axis(dx_px: float, dy_px: float) -> str:
        """Return dominant displacement axis label for split diagnostics."""
        adx = abs(dx_px)
        ady = abs(dy_px)
        if adx > ady:
            return "x"
        if ady > adx:
            return "y"
        return "xy"

    @staticmethod
    def _segment_duration(segment):
        return max(0.0, float(segment["end_time"] - segment["start_time"]))

    @staticmethod
    def _segment_representative_point(segment):
        points = [sample["norm"] for sample in segment.get("samples", []) if sample.get("norm") is not None]
        if not points:
            return None
        xs = sorted(point[0] for point in points)
        ys = sorted(point[1] for point in points)
        mid = len(xs) // 2
        if len(xs) % 2 == 0:
            return ((xs[mid - 1] + xs[mid]) / 2.0, (ys[mid - 1] + ys[mid]) / 2.0)
        return (xs[mid], ys[mid])

    def _append_active_mould_sample(self, timestamp, datetime_obj, mouth, trolley):
        if self._active_segment is None or mouth is None or trolley is None:
            return
        try:
            norm_x, norm_y, _, _ = self._normalize_mouth_position(mouth, trolley)
        except Exception:
            return

        samples = self._active_segment["samples"]
        if samples and abs(samples[-1]["time"] - timestamp) < 1e-6:
            samples[-1]["norm"] = (norm_x, norm_y)
            samples[-1]["datetime"] = datetime_obj
        else:
            samples.append(
                {
                    "time": float(timestamp),
                    "datetime": datetime_obj,
                    "norm": (norm_x, norm_y),
                }
            )
        self._active_segment["end_time"] = float(timestamp)
        self._active_segment["end_datetime"] = datetime_obj

    def _split_segment_by_motion(self, segment, out):
        samples = [sample for sample in segment.get("samples", []) if sample.get("norm") is not None]
        if len(samples) < max(3, self.sustained_hold_frames + 1):
            out.append(segment)
            return

        anchor_x, anchor_y = samples[0]["norm"]
        hold = 0
        for idx in range(1, len(samples)):
            curr_x, curr_y = samples[idx]["norm"]
            if math.sqrt((curr_x - anchor_x) ** 2 + (curr_y - anchor_y) ** 2) > self.displacement_thresh:
                hold += 1
            else:
                hold = 0

            if hold >= self.sustained_hold_frames:
                split_idx = idx - self.sustained_hold_frames
                left_samples = samples[:split_idx + 1]
                right_samples = samples[split_idx + 1:]
                if not left_samples or not right_samples:
                    break

                left = {
                    "start_time": left_samples[0]["time"],
                    "start_datetime": left_samples[0]["datetime"],
                    "end_time": left_samples[-1]["time"],
                    "end_datetime": left_samples[-1]["datetime"],
                    "samples": left_samples,
                    "ladle_track_id": segment.get("ladle_track_id", 0),
                }
                right = {
                    "start_time": right_samples[0]["time"],
                    "start_datetime": right_samples[0]["datetime"],
                    "end_time": right_samples[-1]["time"],
                    "end_datetime": right_samples[-1]["datetime"],
                    "samples": right_samples,
                    "ladle_track_id": segment.get("ladle_track_id", 0),
                }
                out.append(left)
                self._split_segment_by_motion(right, out)
                return

        out.append(segment)

    def _assign_to_cluster(self, point, clusters, r_override=None):
        latest_cid = max((cluster["cid"] for cluster in clusters), default=0)
        min_allowed_cid = max(1, latest_cid - self.cluster_backtrack_guard)
        effective_r = r_override if r_override is not None else self.r_cluster

        best_idx = -1
        best_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            if cluster["cid"] < min_allowed_cid:
                continue
            cx, cy = cluster["centroid"]
            dist = math.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx if best_dist <= effective_r else -1

    def _merge_clusters(self, clusters, r_override=None):
        effective_r = r_override if r_override is not None else self.r_merge
        out = []
        used = [False] * len(clusters)
        next_cid = 1
        for i, cluster in enumerate(clusters):
            if used[i]:
                continue
            used[i] = True
            group = [cluster]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                dist = math.sqrt(
                    (cluster["centroid"][0] - clusters[j]["centroid"][0]) ** 2 +
                    (cluster["centroid"][1] - clusters[j]["centroid"][1]) ** 2
                )
                if dist <= effective_r:
                    used[j] = True
                    group.append(clusters[j])

            merged_segments = []
            sx = 0.0
            sy = 0.0
            for item in group:
                sx += item["centroid"][0]
                sy += item["centroid"][1]
                merged_segments.extend(item["segments"])
            out.append(
                {
                    "cid": next_cid,
                    "centroid": (sx / len(group), sy / len(group)),
                    "segments": merged_segments,
                }
            )
            next_cid += 1
        return out

    def _rescue_refine_clusters(self, baseline_clusters):
        """
        Re-cluster segments within suspicious baseline clusters using a
        locally-computed tighter radius. Only splits clusters that contain
        clearly distinct mould positions (significant internal x-gap > 0.008).
        Returns a flat list of clusters with renumbered cids.
        """
        refined = []
        for cluster in baseline_clusters:
            segs = cluster["segments"]
            if len(segs) < 2:
                refined.append(cluster)
                continue

            # Collect valid x-values for this cluster's segments
            x_vals = []
            for seg in segs:
                rep = self._segment_representative_point(seg)
                if rep is not None:
                    x_vals.append(rep[0])

            if len(x_vals) < 2:
                refined.append(cluster)
                continue

            x_sorted = sorted(x_vals)
            gaps = [x_sorted[i + 1] - x_sorted[i] for i in range(len(x_sorted) - 1)]
            significant = sorted(g for g in gaps if g > 0.008)

            if not significant:
                # All pours at essentially the same position — keep merged
                refined.append(cluster)
                continue

            # Compute local radii from internal gap distribution (25th percentile)
            typical_gap = significant[max(0, len(significant) // 4)]
            local_r = min(self.r_cluster, max(0.005, typical_gap * 0.40))
            local_r_merge = min(self.r_merge, max(0.003, local_r * 0.35))

            # Re-cluster this cluster's segments with local radii
            sub_clusters = []
            sub_cid = 1
            for seg in sorted(segs, key=lambda s: s["start_time"]):
                rep = self._segment_representative_point(seg)
                if rep is None:
                    continue
                idx = self._assign_to_cluster(rep, sub_clusters, r_override=local_r)
                if idx == -1:
                    sub_clusters.append({"cid": sub_cid, "centroid": rep, "segments": [seg]})
                    sub_cid += 1
                else:
                    sub_clusters[idx]["segments"].append(seg)
                    reps = [self._segment_representative_point(s) for s in sub_clusters[idx]["segments"]]
                    reps = [rp for rp in reps if rp is not None]
                    if reps:
                        sub_clusters[idx]["centroid"] = (
                            sum(p[0] for p in reps) / len(reps),
                            sum(p[1] for p in reps) / len(reps),
                        )

            sub_merged = self._merge_clusters(sub_clusters, r_override=local_r_merge)
            refined.extend(sub_merged)

        # Renumber cids sequentially
        for i, c in enumerate(refined, start=1):
            c["cid"] = i
        return refined

    def _build_clusters(self, segments):
        split_segments = []
        for segment in segments:
            self._split_segment_by_motion(segment, split_segments)

        valid_segment_count = sum(
            1 for seg in split_segments
            if self._segment_duration(seg) >= self.pour_min_dur
            and self._segment_representative_point(seg) is not None
        )

        clusters = []
        next_cid = 1
        for segment in split_segments:
            if self._segment_duration(segment) < self.pour_min_dur:
                continue
            rep = self._segment_representative_point(segment)
            if rep is None:
                continue

            idx = self._assign_to_cluster(rep, clusters)
            if idx == -1:
                clusters.append(
                    {
                        "cid": next_cid,
                        "centroid": rep,
                        "segments": [segment],
                    }
                )
                next_cid += 1
            else:
                clusters[idx]["segments"].append(segment)
                reps = [self._segment_representative_point(seg) for seg in clusters[idx]["segments"]]
                reps = [rep_point for rep_point in reps if rep_point is not None]
                if reps:
                    clusters[idx]["centroid"] = (
                        sum(point[0] for point in reps) / len(reps),
                        sum(point[1] for point in reps) / len(reps),
                    )

        merged = self._merge_clusters(clusters)

        # Rescue refinement: only for heavily over-collapsed heats
        baseline_count = len(merged)
        if (valid_segment_count >= 18
                and baseline_count > 0
                and baseline_count / valid_segment_count <= 0.65
                and valid_segment_count - baseline_count >= 8):
            merged = self._rescue_refine_clusters(merged)
            logger.info(
                "Cluster rescue: segs=%d baseline=%d refined=%d ratio=%.2f gap=%d",
                valid_segment_count, baseline_count, len(merged),
                baseline_count / valid_segment_count,
                valid_segment_count - baseline_count,
            )

        valid = []
        for cluster in merged:
            total_duration = sum(self._segment_duration(seg) for seg in cluster["segments"])
            if total_duration >= self.min_cluster_pour_s:
                valid.append(cluster)
        return valid

    def _sync_mould_records_to_heat_cycle(self):
        if not self.heat_cycle_manager:
            return

        # Group segments by spatial cluster — one MouldPouringRecord per cluster.
        by_cluster = defaultdict(list)
        for record in self.mould_records:
            by_cluster[record["cluster_id"]].append(record)

        for cid in sorted(by_cluster):
            slot_id = f"C{cid}"
            if slot_id in self._synced_mould_slot_ids:
                continue

            recs = by_cluster[cid]
            ladle_track_id = int(
                recs[0].get("ladle_track_id") or self.locked_trolley_id or 0
            )
            if ladle_track_id <= 0:
                continue

            total_dur = sum(float(r["duration_s"]) for r in recs)
            start_rec = min(recs, key=lambda r: r["start_time_wall"])
            end_rec = max(recs, key=lambda r: r["end_time_wall"])

            self.heat_cycle_manager.add_pouring_to_cycle(
                ladle_track_id=ladle_track_id,
                mould_id=f"MOULD_C{cid}",
                mould_track_id=ladle_track_id,
                start_time=start_rec["start_time_wall"],
                start_datetime=start_rec["start_datetime_obj"],
                sync_id=self.pour_sync_id or "",
                slno=self.pour_slno or 0,
            )
            self.heat_cycle_manager.update_pouring_end(
                ladle_track_id=ladle_track_id,
                mould_id=f"MOULD_C{cid}",
                end_time=end_rec["end_time_wall"],
                end_datetime=end_rec["end_datetime_obj"],
                duration_seconds=total_dur,
            )
            self._synced_mould_slot_ids.add(slot_id)

    def _materialize_mould_records(self, include_active=False):
        segments = list(self.completed_segments)
        if include_active and self._active_segment is not None:
            active_copy = {
                "start_time": self._active_segment["start_time"],
                "start_datetime": self._active_segment["start_datetime"],
                "end_time": self._active_segment["end_time"],
                "end_datetime": self._active_segment["end_datetime"],
                "samples": list(self._active_segment["samples"]),
                "ladle_track_id": self._active_segment.get("ladle_track_id", 0),
            }
            if self._segment_duration(active_copy) >= self.pour_min_dur:
                segments.append(active_copy)

        clusters = self._build_clusters(segments)
        records = []
        for cluster in sorted(clusters, key=lambda item: item["cid"]):
            cluster_segments = sorted(cluster["segments"], key=lambda seg: seg["start_time"])
            for segment in cluster_segments:
                rep = self._segment_representative_point(segment)
                records.append(
                    {
                        "cluster_id": cluster["cid"],
                        "start_time_wall": segment["start_time"],
                        "start_datetime_obj": segment["start_datetime"],
                        "end_time_wall": segment["end_time"],
                        "end_datetime_obj": segment["end_datetime"],
                        "start_iso": segment["start_datetime"].isoformat() if segment["start_datetime"] else "",
                        "end_iso": segment["end_datetime"].isoformat() if segment["end_datetime"] else "",
                        "duration_s": round(self._segment_duration(segment), 2),
                        "split_axis": "",
                        "split_dx_px": 0.0,
                        "split_dy_px": 0.0,
                        "start_norm": segment["samples"][0]["norm"] if segment["samples"] else None,
                        "end_norm": segment["samples"][-1]["norm"] if segment["samples"] else None,
                        "rep_norm": rep,
                        "ladle_track_id": segment.get("ladle_track_id", 0),
                    }
                )

        records.sort(key=lambda record: (record["start_time_wall"], record["end_time_wall"]))
        self.mould_records = []
        self.mould_completed_times.clear()
        for mould_no, record in enumerate(records, start=1):
            record["mould_no"] = mould_no
            record["slot_id"] = mould_no
            self.mould_records.append(record)
            self.mould_completed_times[mould_no] = int(max(1, round(record["duration_s"] * self.fps)))

        self.mould_count = len(self.mould_records)
        self.clustered_mould_count = len(clusters)

    def _open_active_mould(self, timestamp, datetime_obj, mouth, trolley=None):
        """Start a new raw pouring segment for later reference-style splitting."""
        self.active_mould_id = self.next_mould_id
        self.next_mould_id += 1
        self.active_mould_start_time = timestamp
        self.active_mould_start_datetime = datetime_obj
        self.active_mould_start_norm = None
        self._active_segment = {
            "start_time": float(timestamp),
            "start_datetime": datetime_obj,
            "end_time": float(timestamp),
            "end_datetime": datetime_obj,
            "samples": [],
            "ladle_track_id": int(mouth["track_id"]) if mouth is not None else 0,
        }
        if mouth is not None and trolley is not None:
            try:
                norm_x, norm_y, _, _ = self._normalize_mouth_position(mouth, trolley)
                self.active_mould_start_norm = (norm_x, norm_y)
            except Exception:
                self.active_mould_start_norm = None
        self._append_active_mould_sample(timestamp, datetime_obj, mouth, trolley)

    def _close_active_mould(
        self,
        timestamp,
        datetime_obj,
        mouth,
        min_duration_s,
        close_axis: str = "end",
        split_dx_px: float = 0.0,
        split_dy_px: float = 0.0,
        trolley=None,
    ):
        """Finalize the current raw pouring segment and rebuild mould clusters."""
        if self.active_mould_id is None or self.active_mould_start_time is None:
            return None

        self._append_active_mould_sample(timestamp, datetime_obj, mouth, trolley)

        duration = max(0.0, timestamp - self.active_mould_start_time)
        if duration < float(min_duration_s):
            return None

        if self._active_segment is not None:
            self._active_segment["end_time"] = float(timestamp)
            self._active_segment["end_datetime"] = datetime_obj
            self.completed_segments.append(self._active_segment)

        self._active_segment = None
        self._materialize_mould_records(include_active=False)
        self._sync_mould_records_to_heat_cycle()
        return self.mould_records[-1] if self.mould_records else None

    def _start_pour(self, timestamp, datetime_obj, mouths, trolleys, frame,
                    brightness, probe_x, probe_y, target_trolley, best_mouth):
        """Start a pouring event. Lock trolley on first pour."""
        self.pour_active = True
        self.pour_start_time = timestamp
        self.pour_start_datetime = datetime_obj
        self._pour_start_time_wall = timestamp  # For live overlay display
        self.pour_sync_id = generate_sync_id('pour')
        self.active_mould_id = None
        self.active_mould_start_time = None
        self.active_mould_start_datetime = None
        self.active_mould_start_norm = None
        self._active_segment = None
        self.brightness_above_since = None
        self.brightness_below_since = None
        self.pour_on_count = 0
        self.pour_off_count = 0
        self.active_probe_from_hold = False

        # LOCK TROLLEY on first pour start
        if not self.trolley_locked and target_trolley:
            self._lock_trolley(target_trolley, timestamp)

        # Set mould anchor on pour start (mouth position relative to trolley)
        if target_trolley and best_mouth:
            self._set_anchor_on_pour_start(best_mouth, target_trolley)

        logger.info(
            f"[pour] START - brightness={brightness:.0f}>{self.brightness_start} "
            f"at ({probe_x},{probe_y}), trolley_locked={self.trolley_locked}"
        )

        # DB insert
        try:
            date_str = datetime_obj.strftime("%Y-%m-%d")
            hour = datetime_obj.hour
            shift = "DAY" if 6 <= hour < 18 else "NIGHT"
            heat_no = ""
            pour_location = self.location
            if self.heat_cycle_manager and self.heat_cycle_manager.active_cycle:
                active_cycle = self.heat_cycle_manager.active_cycle
                heat_no = active_cycle.heat_no
                pour_location = self.heat_cycle_manager.location_with_furnace(
                    self.location,
                    getattr(active_cycle, "furnace_label", ""),
                )
            self.pour_slno = self.db_manager.insert_pouring_event(
                sync_id=self.pour_sync_id,
                customer_id=self.customer_id,
                date=date_str,
                shift=shift,
                heat_no=heat_no,
                ladle_number="",
                location=pour_location,
                camera_id=self.camera_id,
                pouring_start_time=datetime_obj.isoformat(),
            )
        except Exception as e:
            logger.error(f"Failed to insert pour start: {e}")

        # Start active mould slot for this pour.
        bm = best_mouth
        if bm is None and mouths:
            locked_trolley = self._get_locked_trolley(trolleys)
            bm = self._select_best_mouth_for_trolley(mouths, locked_trolley) if locked_trolley else None
        if bm is None and mouths:
            bm = max(mouths, key=lambda m: m['confidence'])
        self._open_active_mould(timestamp, datetime_obj, bm, target_trolley)

        # Screenshot
        self._save_event_screenshot(
            "POUR START", mouths, trolleys, frame, datetime_obj,
            probe_point=(probe_x, probe_y), probe_brightness=brightness
        )

    def _end_pour(self, timestamp, datetime_obj, mouths, trolleys, frame, best_mouth=None):
        """End the current pouring event."""
        if not self.pour_active or self.pour_start_time is None:
            return

        # Back-date pour end to when brightness dropped, not when the hold-count completed
        effective_end_ts = timestamp - self.pour_end_dur
        effective_end_dt = datetime_obj - timedelta(seconds=self.pour_end_dur)
        duration = effective_end_ts - self.pour_start_time
        self.pour_active = False
        self.brightness_above_since = None
        self.brightness_below_since = None
        self.pour_on_count = 0
        self.pour_off_count = 0
        self._clear_frozen_probe_state()

        if duration < self.pour_min_dur:
            logger.info(f"[pour] DISCARDED - duration={duration:.1f}s < {self.pour_min_dur}s minimum")
            if self.pour_sync_id:
                try:
                    self.db_manager.delete_pouring_event(self.pour_sync_id)
                except Exception as e:
                    logger.warning(f"[pour] Failed to clean up discarded pour row: {e}")
                self.pour_sync_id = None
                self.pour_slno = None
            self.pour_start_time = None
            self.pour_start_datetime = None
            self._active_segment = None
            self.active_mould_id = None
            self.active_mould_start_time = None
            self.active_mould_start_datetime = None
            self.active_mould_start_norm = None
            self._materialize_mould_records(include_active=False)
            self.displacement_hold_frames = None
            self.split_hold_quadrant = None
            self.split_rearm_required = False
            self.split_rearm_below_since = None
            self.split_rearm_axis = None
            self._last_probe_is_pouring = None
            return

        self.last_pour_duration = duration

        bm = best_mouth
        if bm is None and mouths:
            locked_trolley = self._get_locked_trolley(trolleys)
            bm = self._select_best_mouth_for_trolley(mouths, locked_trolley) if locked_trolley else None
        if bm is None and mouths:
            bm = max(mouths, key=lambda m: m['confidence'])

        locked_trolley = self._get_locked_trolley(trolleys)

        # Finalize current active mould on pour end.
        closed = self._close_active_mould(
            timestamp=timestamp,
            datetime_obj=datetime_obj,
            mouth=bm,
            min_duration_s=self.pour_min_dur,
            close_axis="end",
            split_dx_px=0.0,
            split_dy_px=0.0,
            trolley=locked_trolley,
        )
        if not closed:
            if self.active_mould_start_time is not None:
                age = timestamp - self.active_mould_start_time
                logger.info(
                    f"[mould] Skip finalization at pour end: active_mould={self.active_mould_id}, "
                    f"duration={age:.1f}s"
                )
            else:
                logger.info(
                    f"[mould] Skip finalization at pour end: active_mould={self.active_mould_id}, no start"
                )
        else:
            cluster_txt = f"C{closed['cluster_id']}" if closed.get('cluster_id') else "C-"
            logger.info(
                "[mould] Mould #%s completed: %.1fs | %s | axis=%s",
                closed.get("mould_no"),
                float(closed.get("duration_s", 0.0)),
                cluster_txt,
                str(closed.get("split_axis", "-")).upper(),
            )

        logger.info(f"[pour] END - duration={duration:.1f}s, moulds={self.mould_count}")
        for rec in self._get_mould_breakdown():
            cluster_txt = f"C{rec['cluster_id']}" if rec.get("cluster_id") is not None else "C-"
            logger.info(
                "[mould-summary] M#%s %s axis=%s duration=%.1fs",
                rec.get("mould_no"),
                cluster_txt,
                str(rec.get("axis", "-")).upper(),
                float(rec.get("duration_s", 0.0)),
            )
        self._pour_start_time_wall = None  # Clear for next pour

        # DB update
        try:
            mould_wise = {
                "mould_count": self.mould_count,
                "clustered_mould_count": self.clustered_mould_count,
                "last_pour_duration": round(duration, 1),
                "moulds": self._get_mould_breakdown(),
            }
            self.db_manager.update_pouring_end(
                sync_id=self.pour_sync_id,
                pouring_end_time=effective_end_dt.isoformat(),
                total_pouring_time=str(int(duration)),
                mould_wise_pouring_time=mould_wise,
            )
        except Exception as e:
            logger.error(f"Failed to update pour end: {e}")

        # Screenshot (include last known probe points)
        probe_base = self._last_probe_base or self._get_probe_base_from_mouths(mouths)
        self._save_event_screenshot(
            "POUR END", mouths, trolleys, frame, effective_end_dt,
            probe_point=probe_base,
            probe_brightness=self._last_probe_brightness,
            extra_info=f"Duration: {duration:.1f}s  Moulds: {self.mould_count}"
        )

        self.pour_start_time = None
        self.pour_start_datetime = None
        self.pour_sync_id = None
        self.active_mould_id = None
        self.active_mould_start_time = None
        self.active_mould_start_datetime = None
        self.active_mould_start_norm = None
        self.displacement_hold_frames = None
        self.split_hold_quadrant = None
        self.split_rearm_required = False
        self.split_rearm_below_since = None
        self.split_rearm_axis = None

    # =========================================================================
    # Sub-system 3: Mould Counter (trolley-relative anchor)
    # =========================================================================

    def _get_mould_norm_bbox(self, trolley):
        """Normalization bbox aligned with full expanded trolley geometry."""
        tx1, ty1, tx2, ty2 = trolley['bbox']
        return (
            tx1 - self.edge_expand_x_px,
            ty1 - self.edge_expand_y_px,
            tx2 + self.edge_expand_x_px,
            ty2 + self.edge_expand_y_px,
        )

    def _normalize_mouth_position(self, mouth, trolley):
        """Normalize probe point to [0,1] in the raw trolley coordinate space."""
        px, py = self._mouth_probe_point(mouth)
        tx1, ty1, tx2, ty2 = trolley['bbox']
        tw = max(tx2 - tx1, 1)
        th = max(ty2 - ty1, 1)
        return (
            max(0.0, min(1.0, (px - tx1) / tw)),
            max(0.0, min(1.0, (py - ty1) / th)),
            tw,
            th,
        )

    def _set_anchor_on_pour_start(self, mouth, trolley):
        """Set mould anchor = mouth position normalized to expanded trolley dimensions."""
        norm_x, norm_y, _, _ = self._normalize_mouth_position(mouth, trolley)
        self.anchor_position = (norm_x, norm_y)
        self.anchor_set = True
        self.displacement_hold_frames = None
        self.split_hold_quadrant = None
        self.split_rearm_required = False
        self.split_rearm_below_since = None
        self.split_rearm_axis = None

    def _update_mould_counter(self, mouth, trolley, timestamp, datetime_obj):
        """Collect active pour samples and update reference-style mould preview."""
        if self._active_segment is None:
            return

        self._append_active_mould_sample(timestamp, datetime_obj, mouth, trolley)
        if self.log_mould_displacement:
            should_log = (
                self._last_mould_disp_log_ts is None or
                (timestamp - self._last_mould_disp_log_ts) >= self.mould_disp_log_interval_s
            )
            if should_log and self._active_segment is not None:
                rep = self._segment_representative_point(self._active_segment)
                logger.info(
                    "[mould] Active segment samples=%d rep_norm=%s duration=%.2fs",
                    len(self._active_segment["samples"]),
                    rep,
                    self._segment_duration(self._active_segment),
                )
                self._last_mould_disp_log_ts = timestamp

        self._materialize_mould_records(include_active=True)

    def _recompute_clusters(self):
        """Recompute clustered mould records from completed raw segments."""
        self._materialize_mould_records(include_active=False)

    # =========================================================================
    # Canonical Merge Helper Methods
    # =========================================================================

    def _get_mould_breakdown(self):
        """Return per-mould timings with cluster and split-axis attribution."""
        breakdown = []
        for rec in self.mould_records:
            cluster = rec.get("cluster_id")
            breakdown.append(
                {
                    "mould_no": int(rec.get("mould_no", 0)),
                    "mould_slot_id": int(rec.get("slot_id", 0)),
                    "cluster_id": int(cluster) if cluster is not None else None,
                    "axis": str(rec.get("split_axis", "")),
                    "duration_s": round(float(rec.get("duration_s", 0.0)), 2),
                    "start": rec.get("start_iso", ""),
                    "end": rec.get("end_iso", ""),
                    "split_dx_px": round(float(rec.get("split_dx_px", 0.0)), 1),
                    "split_dy_px": round(float(rec.get("split_dy_px", 0.0)), 1),
                }
            )
        return breakdown

    # =========================================================================
    # Pouring Cycle Timeout (5 min)
    # =========================================================================

    def _check_cycle_timeout(self, timestamp, datetime_obj, mouths, trolleys, frame):
        """Check if mouth has been absent from locked trolley for 5 minutes."""
        if not self.trolley_locked:
            return

        if self.mouth_last_seen_in_trolley is None:
            return

        absence = timestamp - self.mouth_last_seen_in_trolley
        if absence >= self.cycle_timeout:
            logger.info(
                f"[cycle] TIMEOUT - mouth absent from locked trolley T{self.locked_trolley_id} "
                f"for {absence:.0f}s >= {self.cycle_timeout}s — ending pouring cycle"
            )

            # End any active pour/session
            if self.pour_active:
                self._end_pour(timestamp, datetime_obj, mouths, trolleys, frame)
            if self.session_active:
                self._end_session(timestamp, datetime_obj, mouths, trolleys, frame)

            # Reset ALL state — definitive end of pouring cycle
            self._reset_all_state()

    def _reset_all_state(self):
        """Reset all pouring cycle state (on cycle end/timeout)."""
        self.locked_trolley_id = None
        self.locked_trolley_bbox = None
        self.trolley_locked = False
        self.trolley_last_detected_time = None
        self.session_active = False
        self.mouth_inside_since = None
        self.mouth_absent_since = None
        self.session_start_time = None
        self.session_start_datetime = None
        self.pour_active = False
        self.brightness_above_since = None
        self.brightness_below_since = None
        self.pour_on_count = 0
        self.pour_off_count = 0
        self.pour_start_time = None
        self.pour_start_datetime = None
        self.pour_sync_id = None
        self.pour_slno = None
        self.active_mould_id = None
        self.active_mould_start_time = None
        self.active_mould_start_datetime = None
        self.active_mould_start_norm = None
        self.last_pour_duration = 0.0
        self.anchor_position = None
        self.anchor_set = False
        self.displacement_hold_frames = None
        self.split_hold_quadrant = None
        self.split_rearm_required = False
        self.split_rearm_below_since = None
        self.split_rearm_axis = None
        self.moved_positions.clear()
        self.mould_records.clear()
        self.completed_segments.clear()
        self._active_segment = None
        self._synced_mould_slot_ids.clear()
        self.mould_completed_times.clear()
        self.mould_count = 0
        self.clustered_mould_count = 0
        self.next_mould_id = 1
        self.last_split_time = None
        self._last_mould_disp_log_ts = None
        self.mouth_last_seen_in_trolley = None
        self.cycle_start_time = None
        self.cycle_start_datetime = None
        self._last_probe_base = None
        self._last_probe_brightness = None
        self._last_probe_tail_brightness = None
        self._last_probe_is_pouring = None
        self._clear_frozen_probe_state()
        self._clear_active_probe_state()
        self._relock_candidate_id = None
        self._relock_candidate_since = None
        logger.info("[cycle] All state reset — ready for new pouring cycle")

    # =========================================================================
    # Heat cycle DB insertion
    # =========================================================================

    def _insert_heat_cycle_to_db(self, cycle):
        """Insert finalized heat cycle to database."""
        if not cycle.mould_pourings and not cycle.tapping_events:
            logger.warning(
                f"Heat cycle {cycle.heat_no} has no pouring or tapping data — skipping DB insert"
            )
            return

        date_str = cycle.cycle_start_datetime.strftime("%Y-%m-%d")
        hour = cycle.cycle_start_datetime.hour
        shift = "DAY" if 6 <= hour < 18 else "NIGHT"
        cycle_location = self.location
        if self.heat_cycle_manager:
            cycle_location = self.heat_cycle_manager.location_with_furnace(
                self.location,
                getattr(cycle, "furnace_label", ""),
            )

        try:
            self.db_manager.insert_heat_cycle(
                sync_id=generate_sync_id('heat_cycle'),
                customer_id=self.customer_id,
                date=date_str,
                shift=shift,
                heat_no=cycle.heat_no,
                ladle_number="",
                location=cycle_location,
                camera_id=self.camera_id,
                cycle_start_time=cycle.cycle_start_datetime.isoformat(),
                cycle_end_time=cycle.cycle_end_datetime.isoformat() if cycle.cycle_end_datetime else datetime.now().isoformat(),
                pouring_start_time=cycle.pouring_start_time.isoformat() if cycle.pouring_start_time else "",
                pouring_end_time=cycle.pouring_end_time.isoformat() if cycle.pouring_end_time else "",
                total_pouring_time=str(cycle.total_pouring_time or 0),
                mould_wise_pouring_time=cycle.mould_wise_pouring_time or [],
                tapping_start_time=cycle.tapping_start_datetime.isoformat() if cycle.tapping_start_datetime else None,
                tapping_end_time=cycle.tapping_end_datetime.isoformat() if cycle.tapping_end_datetime else None,
                tapping_events=cycle.tapping_events if cycle.tapping_events else None,
                deslagging_events=cycle.deslagging_events if cycle.deslagging_events else None,
                spectro_events=cycle.spectro_events if cycle.spectro_events else None,
                pyrometer_events=cycle.pyrometer_events if cycle.pyrometer_events else None,
            )
            logger.info(
                f"HEAT CYCLE FINALIZED: {cycle.heat_no} - "
                f"{len(cycle.mould_pourings)} pours, {cycle.total_pouring_time}s total"
            )
        except Exception as e:
            logger.error(f"Failed to insert heat cycle {cycle.heat_no}: {e}")

    def _finalize_heat_cycles_if_due(self, timestamp, datetime_obj):
        """Finalize and persist any completed heat cycles on the shared cadence."""
        if not self.heat_cycle_manager or self._frame_count % 10 != 0:
            return

        finalized = self.heat_cycle_manager.check_and_finalize_cycles(
            timestamp, datetime_obj
        )
        for cycle in finalized:
            self._insert_heat_cycle_to_db(cycle)

    # =========================================================================
    # DS-native OSD overlay metadata (recorded via tee branch after nvosd)
    # =========================================================================

    def _get_probe_base_from_mouths(self, mouths):
        """Derive probe base from best mouth in-frame when no measured probe is cached."""
        if not mouths:
            return None
        best_mouth = None
        if self.trolley_locked and self.locked_trolley_bbox:
            pseudo_locked = {'bbox': self.locked_trolley_bbox}
            best_mouth = self._select_best_mouth_for_trolley(mouths, pseudo_locked)
        if best_mouth is None:
            best_mouth = max(mouths, key=lambda m: m['confidence'])
        mx, my_bottom = best_mouth['bottom_center']
        return (mx, my_bottom + self.probe_below_px)

    def _draw_probe_points(self, annotated, probe_point, probe_brightness):
        """Draw configured probe circles and current brightness value."""
        if not probe_point:
            return
        for dx, dy in self.probe_offsets:
            px = probe_point[0] + dx
            py = probe_point[1] + dy
            probe_on = self._last_probe_is_pouring
            if probe_on is None:
                probe_on = (probe_brightness or 0) >= self.brightness_start
            color = (0, 255, 0) if probe_on else (0, 0, 255)
            cv2.circle(annotated, (px, py), self.probe_radius + 2, color, -1)
        if probe_brightness is not None:
            cv2.putText(
                annotated,
                f"B:{probe_brightness:.0f}",
                (probe_point[0] + 50, probe_point[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    def _add_inference_display_meta(self, batch_meta, frame_meta, mouths, trolleys,
                                    target_trolley, timestamp, datetime_obj):
        """Attach 2-line technical status + clean per-mould panel overlay (nvosd)."""
        display_meta = None
        try:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                return

            # Scale overlay text when recording is downscaled
            scale_up = 1.0
            try:
                target_w = int(getattr(self.config, "INFERENCE_VIDEO_WIDTH", 0) or 0)
                if self._frame_w and target_w and target_w < self._frame_w:
                    scale_up = min(2.5, self._frame_w / float(target_w))
            except Exception:
                scale_up = 1.0

            frame_w = int(getattr(frame_meta, "source_frame_width", 0) or self._frame_w or 1280)
            frame_h = int(getattr(frame_meta, "source_frame_height", 0) or self._frame_h or 720)

            cycle_age = (timestamp - self.cycle_start_time) if self.cycle_start_time else 0.0
            absence = (timestamp - self.mouth_last_seen_in_trolley) if self.mouth_last_seen_in_trolley else 0.0
            target_tid = target_trolley['track_id'] if target_trolley else "-"
            lock_tid = self.locked_trolley_id if self.locked_trolley_id is not None else "-"
            brightness_txt = f"{self._last_probe_brightness:.0f}" if self._last_probe_brightness is not None else "-"

            line_height = max(16, int(round(18 * scale_up)))
            font_size = max(11, int(round(12 * scale_up)))
            max_labels = len(display_meta.text_params)
            max_lines_by_height = max(6, (frame_h - 20) // line_height)
            line_budget = max(4, min(max_labels, max_lines_by_height))

            # Build right-panel text rows: (text, role)
            rows = [
                ("POURING INFERENCE", "title"),
                (
                    f"{datetime_obj.strftime('%H:%M:%S')} S:{'ON' if self.session_active else 'OFF'} "
                    f"P:{'ON' if self.pour_active else 'OFF'} M:{self.mould_count} C:{self.clustered_mould_count} "
                    f"B:{brightness_txt} T:{target_tid}/{lock_tid}",
                    "metrics",
                ),
            ]

            if self.trolley_locked and self.locked_trolley_id is not None:
                rows.append((f"Trolley #{self.locked_trolley_id} [LOCKED]", "locked"))
                rows.append((f"Total Moulds: {self.mould_count}", "totals"))

                records_by_no = {int(r.get("mould_no", -1)): r for r in self.mould_records}
                mould_rows = []
                for mid in sorted(self.mould_completed_times.keys()):
                    frames = self.mould_completed_times[mid]
                    time_s = frames / self.fps
                    rec = records_by_no.get(int(mid))
                    if rec:
                        cluster = rec.get("cluster_id")
                        cluster_txt = f"C{cluster}" if cluster is not None else "C-"
                        axis_txt = str(rec.get("split_axis", "-")).upper()
                    else:
                        cluster_txt = "C-"
                        axis_txt = "-"
                    mould_rows.append((f"M#{mid}: {time_s:.1f}s {cluster_txt} {axis_txt}", "mould_done"))

                if self.pour_active and self._pour_start_time_wall is not None:
                    active_s = timestamp - self._pour_start_time_wall
                    next_mid = self.mould_count + 1
                    mould_rows.append((f"M#{next_mid}: {active_s:.1f}s ACTIVE", "mould_active"))

                # Keep the latest mould rows when vertical space is limited.
                remaining = max(0, line_budget - len(rows) - 1)  # reserve 1 line for footer
                if len(mould_rows) > remaining:
                    keep = max(0, remaining - 1)
                    hidden = len(mould_rows) - keep
                    if keep > 0:
                        mould_rows = [("... +%d earlier moulds" % hidden, "trimmed")] + mould_rows[-keep:]
                    else:
                        mould_rows = [("... +%d mould rows hidden" % hidden, "trimmed")]
                rows.extend(mould_rows)
            else:
                rows.append(("No active trolley", "metrics"))

            session_age = (timestamp - self.session_start_time) if self.session_start_time else 0.0
            cycle_txt = f"{int(cycle_age // 60)}m" if cycle_age >= 60 else f"{int(cycle_age)}s"
            rows.append((f"Session:{int(session_age)}s  Cycle:{cycle_txt}  Absence:{absence:.1f}s", "footer"))

            if len(rows) > line_budget:
                rows = rows[:line_budget]

            # Place pouring panel on the right to avoid overlap with brightness panel on the left.
            panel_margin = 10
            panel_w = max(360, int(frame_w * 0.42))
            panel_w = min(panel_w, max(260, frame_w - panel_margin * 2))
            panel_x = max(panel_margin, frame_w - panel_w - panel_margin)
            panel_y = panel_margin
            panel_h = min(frame_h - panel_margin * 2, max(line_height + 12, len(rows) * line_height + 14))

            rect_idx = 0
            if len(display_meta.rect_params) > 0:
                panel_rect = display_meta.rect_params[rect_idx]
                panel_rect.left = int(panel_x)
                panel_rect.top = int(panel_y)
                panel_rect.width = int(max(1, panel_w))
                panel_rect.height = int(max(1, panel_h))
                panel_rect.border_width = max(1, int(round(2 * scale_up)))
                panel_rect.border_color.set(0.0, 1.0, 1.0, 0.65)
                panel_rect.has_bg_color = 1
                panel_rect.bg_color.set(0.0, 0.0, 0.0, 0.45)
                rect_idx += 1

            display_meta.num_labels = min(len(rows), max_labels)
            color_map = {
                "title": (1.0, 1.0, 0.0, 1.0),
                "metrics": (1.0, 1.0, 1.0, 1.0),
                "locked": (0.0, 1.0, 1.0, 1.0),
                "totals": (0.85, 0.85, 0.85, 1.0),
                "mould_done": (0.85, 0.85, 0.85, 1.0),
                "mould_active": (0.0, 1.0, 0.0, 1.0),
                "trimmed": (1.0, 0.8, 0.2, 1.0),
                "footer": (1.0, 1.0, 1.0, 1.0),
            }
            for i in range(display_meta.num_labels):
                text, role = rows[i]
                txt = display_meta.text_params[i]
                txt.display_text = text
                txt.x_offset = int(panel_x + 12)
                txt.y_offset = int(panel_y + 4 + (i + 1) * line_height)
                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = font_size
                r, g, b, a = color_map.get(role, color_map["metrics"])
                txt.font_params.font_color.set(r, g, b, a)
                txt.set_bg_clr = 0

            # Draw large probe dot (circle). Brightness already appears in panel metrics.
            if self._last_probe_base is not None and display_meta.num_circles < 16:
                base_x, base_y = self._last_probe_base
                probe_on = self._last_probe_is_pouring
                if probe_on is None:
                    probe_on = (self._last_probe_brightness or 0.0) >= self.brightness_start

                circle = display_meta.circle_params[0]
                circle.xc = int(base_x)
                circle.yc = int(base_y)
                circle.radius = max(5, int(round(5 * scale_up)))  # Visible but not huge

                if probe_on:
                    circle.circle_color.set(0.0, 1.0, 0.0, 1.0)
                    circle.has_bg_color = 1
                    circle.bg_color.set(0.0, 1.0, 0.0, 0.85)
                else:
                    circle.circle_color.set(1.0, 0.0, 0.0, 1.0)
                    circle.has_bg_color = 1
                    circle.bg_color.set(1.0, 0.0, 0.0, 0.85)

                display_meta.num_circles = 1

            # Draw expanded trolley bbox (optional dashed outline)
            if self.trolley_locked and self.locked_trolley_bbox and rect_idx < 12:
                x1, y1, x2, y2 = self.locked_trolley_bbox
                ex1 = max(0, int(round(x1 - self.edge_expand_x_px)))
                ey1 = max(0, int(round(y1 - self.edge_expand_y_px)))
                ex2 = int(round(x2 + self.edge_expand_x_px))
                ey2 = int(round(y2 + self.edge_expand_y_px))
                rect = display_meta.rect_params[rect_idx]
                rect.left = ex1
                rect.top = int(max(0, ey1))
                rect.width = int(max(1, ex2 - ex1))
                rect.height = int(max(1, ey2 - ey1))
                rect.border_width = max(1, int(round(1 * scale_up)))
                rect.border_color.set(0.0, 1.0, 0.0, 0.5)  # Semi-transparent green
                rect.has_bg_color = 0  # No fill, just outline
                rect_idx += 1

            display_meta.num_rects = rect_idx
        except Exception as exc:
            logger.error(f"[osd] Failed to attach inference display meta: {exc}", exc_info=True)
        finally:
            if display_meta is not None:
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    def write_recording_overlay(self, batch_meta, frame_meta):
        """Write display overlay from latest cached detection state.

        Called from the main-path display_meta writer probe in decoupled analysis mode.
        Uses state already computed by the analysis branch — zero additional analysis.
        Safe to call from a different GStreamer thread; reads a snapshot dict reference.
        """
        data = self._cached_display_data
        if not data or not self.enable_inference_video:
            return
        self._add_inference_display_meta(
            batch_meta=batch_meta,
            frame_meta=frame_meta,
            mouths=data['mouths'],
            trolleys=data['trolleys'],
            target_trolley=data['target_trolley'],
            timestamp=data['timestamp'],
            datetime_obj=data['datetime_obj'],
        )

    def close(self):
        """Compatibility no-op. DS-native recording is managed by pipeline/RecordingManager."""
        return None

    def __del__(self):
        """Keep backward-compatible lifecycle hook."""
        self.close()

    # =========================================================================
    # Annotated screenshot saving
    # =========================================================================

    def _save_event_screenshot(self, title, mouths, trolleys, frame, datetime_obj,
                               probe_point=None, probe_brightness=None,
                               extra_info=None):
        """Save annotated screenshot with detections, probe, and event details."""
        if frame is None:
            return None

        try:
            raw_bgr = prepare_frame(self._prepare_frame_for_annotation(frame))
            annotated = raw_bgr.copy()

            # Draw trolley bboxes (green) + expanded region
            for t in trolleys:
                x1, y1, x2, y2 = t['bbox']
                is_locked = (self.trolley_locked and t['track_id'] == self.locked_trolley_id)
                color = (0, 255, 0) if is_locked else (0, 180, 0)
                thickness = 3 if is_locked else 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                label = f"Trolley T{t['track_id']} {t['confidence']:.2f}"
                if is_locked:
                    label += " [LOCKED]"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if is_locked:
                    ex1 = max(0, int(round(x1 - self.edge_expand_x_px)))
                    ey1 = max(0, int(round(y1 - self.edge_expand_y_px)))
                    ex2 = int(round(x2 + self.edge_expand_x_px))
                    ey2 = int(round(y2 + self.edge_expand_y_px))
                    cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (0, 255, 0), 1)

            # Draw mouth bboxes (cyan)
            for m in mouths:
                x1, y1, x2, y2 = m['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"Mouth T{m['track_id']} {m['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw probe points
            self._draw_probe_points(annotated, probe_point, probe_brightness)

            # Standard header/footer
            extra_lines = [extra_info] if extra_info else None
            add_header(annotated, title, datetime_obj.strftime("%Y-%m-%d %H:%M:%S"),
                       extra_lines)
            status = f"Moulds: {self.mould_count}  Clustered: {self.clustered_mould_count}"
            if self.trolley_locked:
                status += f"  Trolley: T{self.locked_trolley_id} [LOCKED]"
            add_footer(annotated, self.camera_id, status)

            ann_list = []
            ann_id = 1
            for m in (mouths or []):
                x1, y1, x2, y2 = m['bbox']
                ann_list.append({
                    "id": ann_id, "category_id": CLASS_MOUTH,
                    "bbox": _to_coco_bbox(x1, y1, x2, y2),
                    "area": (x2 - x1) * (y2 - y1),
                    "confidence": round(float(m['confidence']), 4),
                    "track_id": int(m['track_id']),
                    "iscrowd": 0,
                })
                ann_id += 1
            for t in (trolleys or []):
                x1, y1, x2, y2 = t['bbox']
                ann_list.append({
                    "id": ann_id, "category_id": CLASS_TROLLEY,
                    "bbox": _to_coco_bbox(x1, y1, x2, y2),
                    "area": (x2 - x1) * (y2 - y1),
                    "confidence": round(float(t['confidence']), 4),
                    "track_id": int(t['track_id']),
                    "iscrowd": 0,
                })
                ann_id += 1

            tag = title.lower().replace(" ", "_")
            return save_screenshot(
                annotated, "pouring", tag, datetime_obj, self.screenshot_dir,
                annotations=ann_list or None,
                raw_frame=raw_bgr,
                event_type=tag,
                camera_id=str(self.camera_id),
                categories=_POURING_CATEGORIES,
                writer=self.screenshot_writer,
            )

        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return None
