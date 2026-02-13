"""
Pouring Processor - HiCon pouring detection per standalone pouring system documentation.

Three sub-systems running in the OSD sink pad probe on Stream 0:
1. Session Manager:  ladle_mouth center inside expanded trolley bbox → session lifecycle
2. Pour Detector:    multi-probe brightness below ladle_mouth → pour start/end
3. Mould Counter:    mouth-position anchor displacement (trolley-relative) → mould count

Key behaviors:
- Trolley locking: lock onto trolley where first pour starts, ignore others
- EDGE_EXPAND: expand trolley bbox by 200px on top edge only (ladle sits above trolley)
- Session persistence: mould data preserved across session boundaries for same locked trolley
- Pouring cycle timeout: 5 min mouth absence from locked trolley → reset everything

Model classes: ladle_mouth (class 0), trolley (class 1).
"""

import cv2
import math
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pyds

from utils.utils import generate_sync_id

logger = logging.getLogger(__name__)

# Class IDs from labels_pouring.txt
CLASS_MOUTH = 0
CLASS_TROLLEY = 1


class PouringProcessor:
    """
    Pouring detection with trolley locking, multi-probe brightness, and mould counting.

    State Machine:
    IDLE → mouth in expanded trolley >=1.0s → SESSION ACTIVE
    SESSION ACTIVE → brightness >230 for 0.25s → POUR START → LOCK TROLLEY (first time)
    POUR ACTIVE → brightness <180 for 1.0s → POUR END (min 2.0s) → mould counted
    SESSION ACTIVE → mouth absent >0.8s + 1.5s → SESSION END (keep mould data)
    SESSION END → mouth returns to locked trolley >=1.0s → SESSION RESTART (resume counting)
    ANY → mouth absent from locked trolley 5 min → CYCLE END (reset everything)
    """

    def __init__(self, db_manager, config, screenshot_dir: str, heat_cycle_manager=None):
        self.db_manager = db_manager
        self.config = config
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Customer/location
        self.customer_id = config.CUSTOMER_ID
        self.location = config.LOCATION
        self.camera_id = config.CAMERA_ID_STREAM_0

        # Detection thresholds
        self.mouth_conf = config.MOUTH_CONFIDENCE
        self.trolley_conf = config.TROLLEY_CONFIDENCE

        # Session timing
        self.session_start_dur = config.SESSION_START_DURATION
        self.session_end_dur = config.SESSION_END_DURATION

        # Pour probe (multi-probe offsets)
        self.probe_below_px = config.POUR_PROBE_BELOW_PX
        self.probe_offsets = config.POUR_PROBE_OFFSETS  # [(dx, dy), ...]
        self.probe_radius = config.POUR_PROBE_RADIUS_PX
        self.brightness_start = config.POUR_BRIGHTNESS_START
        self.brightness_end = config.POUR_BRIGHTNESS_END
        self.pour_start_dur = config.POUR_START_DURATION
        self.pour_end_dur = config.POUR_END_DURATION
        self.pour_min_dur = config.POUR_MIN_DURATION

        # Mould counting
        self.displacement_thresh = config.MOULD_DISPLACEMENT_THRESHOLD
        self.sustained_dur = config.MOULD_SUSTAINED_DURATION
        self.r_cluster = config.CLUSTER_R_CLUSTER
        self.r_merge = config.CLUSTER_R_MERGE
        self.mould_switch_min_pour = config.MOULD_SWITCH_MIN_POUR_S
        self.min_cluster_pour_s = config.MIN_CLUSTER_POUR_S

        # Edge expand and tolerances
        self.edge_expand = config.EDGE_EXPAND_PX
        self.mouth_missing_tol = config.MOUTH_MISSING_TOL_S
        self.cycle_timeout = config.POURING_CYCLE_TIMEOUT_S

        # Heat cycle manager (shared, injected from pipeline)
        self.heat_cycle_manager = heat_cycle_manager

        # --- Trolley locking state ---
        self.locked_trolley_id: Optional[int] = None
        self.locked_trolley_bbox: Optional[Tuple[int, int, int, int]] = None
        self.trolley_locked = False

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
        self.pour_start_time: Optional[float] = None
        self.pour_start_datetime: Optional[datetime] = None
        self.pour_sync_id: Optional[str] = None
        self.pour_slno: Optional[int] = None
        self.last_pour_duration: float = 0.0  # duration of last completed pour

        # --- Mould counter state ---
        self.anchor_position: Optional[Tuple[float, float]] = None  # normalized mouth pos (trolley-relative)
        self.anchor_set = False  # anchor set on pour start
        self.displacement_since: Optional[float] = None  # timestamp when displacement exceeded threshold
        self.moved_positions: List[Tuple[float, float, float]] = []  # normalized (x, y) + pour duration
        self.mould_count = 0

        # --- Last known probe state (for screenshot annotations) ---
        self._last_probe_base: Optional[Tuple[int, int]] = None  # (base_x, base_y) before offsets
        self._last_probe_brightness: Optional[float] = None

        # --- Cycle timeout tracking ---
        self.mouth_last_seen_in_trolley: Optional[float] = None  # last time mouth was inside locked trolley region
        self.cycle_start_time: Optional[float] = None
        self.cycle_start_datetime: Optional[datetime] = None

        # Frame dimensions (set on first frame)
        self._frame_w = 0
        self._frame_h = 0

        # Frame counter
        self._frame_count = 0

        # --- DS-native inference overlay toggle (recorded post-OSD via tee branch) ---
        self.enable_inference_video = bool(getattr(config, 'ENABLE_INFERENCE_VIDEO', False))

        logger.info("PouringProcessor initialized (aligned with standalone doc)")
        logger.info(f"  Mouth conf: {self.mouth_conf}, Trolley conf: {self.trolley_conf}")
        logger.info(f"  Session: start={self.session_start_dur}s, end={self.session_end_dur}s")
        logger.info(f"  Pour: multi-probe offsets={self.probe_offsets}, below={self.probe_below_px}px")
        logger.info(f"  Pour brightness: start>{self.brightness_start}, end<{self.brightness_end}")
        logger.info(f"  Pour timing: start={self.pour_start_dur}s, end={self.pour_end_dur}s, min={self.pour_min_dur}s")
        logger.info(f"  Mould: disp>{self.displacement_thresh}, sustained={self.sustained_dur}s, min_pour={self.mould_switch_min_pour}s")
        logger.info(f"  Mould cluster filter: min_cluster_pour={self.min_cluster_pour_s}s")
        logger.info(f"  Edge expand: {self.edge_expand}px, Mouth missing tol: {self.mouth_missing_tol}s")
        logger.info(f"  Cycle timeout: {self.cycle_timeout}s")
        logger.info(f"  DS-native inference overlay enabled={self.enable_inference_video}")

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

        # Set frame dimensions once
        if self._frame_w == 0:
            self._frame_w = int(getattr(frame_meta, 'source_frame_width', 0) or 0) or 1920
            self._frame_h = int(getattr(frame_meta, 'source_frame_height', 0) or 0) or 1080

        # 1. Extract detections
        mouths, trolleys = self._extract_detections(frame_meta)

        if frame_num and frame_num % 1000 == 0:
            logger.info(
                f"Stream 0 Frame {frame_num}: {len(mouths)} mouths, {len(trolleys)} trolleys"
            )

        # 2. Find the relevant trolley (locked or best candidate)
        target_trolley = self._get_target_trolley(trolleys)

        # 3. Check mouth-in-trolley (with EDGE_EXPAND)
        mouth_in_trolley = False
        best_mouth = None
        if target_trolley and mouths:
            best_mouth = max(mouths, key=lambda m: m['confidence'])
            mouth_in_trolley = self._is_mouth_in_expanded_trolley(best_mouth, target_trolley)
            # Keep latest probe base available for session/pour screenshots and OSD overlay.
            mx, my_bottom = best_mouth['bottom_center']
            self._last_probe_base = (mx, my_bottom + self.probe_below_px)
            if not self.session_active:
                self._last_probe_brightness = None

        # 4. Update trolley bbox (keep latest position for locked trolley)
        if target_trolley and self.trolley_locked:
            self.locked_trolley_bbox = target_trolley['bbox']

        # 5. Session manager
        self._update_session(mouth_in_trolley, best_mouth, target_trolley,
                             mouths, trolleys, timestamp, datetime_obj, frame)

        # 6. Pour detector (during active session, with frame data)
        if self.session_active and best_mouth and frame is not None:
            self._update_pour(best_mouth, frame, timestamp, datetime_obj,
                              mouths, trolleys, target_trolley)

        # 7. Mould counter (during active session)
        if self.session_active and best_mouth and target_trolley:
            self._update_mould_counter(best_mouth, target_trolley, timestamp)

        # 8. Update heat cycle ladle presence
        if self.heat_cycle_manager and mouths:
            bm = max(mouths, key=lambda m: m['confidence'])
            self.heat_cycle_manager.update_ladle_presence(
                bm['track_id'], timestamp, datetime_obj
            )

        # 9. Pouring cycle timeout check (5 min)
        self._check_cycle_timeout(timestamp, datetime_obj, mouths, trolleys, frame)

        # 10. Finalize heat cycles periodically
        if self.heat_cycle_manager and self._frame_count % 10 == 0:
            finalized = self.heat_cycle_manager.check_and_finalize_cycles(
                timestamp, datetime_obj
            )
            for cycle in finalized:
                self._insert_heat_cycle_to_db(cycle)

        # 11. DS-native overlay annotations (rendered by nvosd and captured via tee branch)
        if self.enable_inference_video and batch_meta is not None:
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

        l_obj = frame_meta.obj_meta_list
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

    def _get_target_trolley(self, trolleys):
        """Get the target trolley: locked one if exists, else best candidate."""
        if not trolleys:
            return None

        if self.trolley_locked and self.locked_trolley_id is not None:
            # Find locked trolley by track_id
            for t in trolleys:
                if t['track_id'] == self.locked_trolley_id:
                    return t
            # Locked trolley not detected this frame — return None
            return None

        # No lock yet — return highest-confidence trolley
        return max(trolleys, key=lambda t: t['confidence'])

    def _is_mouth_in_expanded_trolley(self, mouth, trolley):
        """Check if mouth center is inside trolley bbox expanded on top edge only.

        The ladle mouth sits above the trolley during pouring, so only the
        top edge (ey1) is expanded upward by EDGE_EXPAND_PX. Left/right/bottom
        remain at the original trolley bbox.
        """
        mx, my = mouth['center']
        tx1, ty1, tx2, ty2 = trolley['bbox']

        # Expand only the top edge (ladle is above trolley during pouring)
        ey1 = max(0, ty1 - self.edge_expand)

        return tx1 <= mx <= tx2 and ey1 <= my <= ty2

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
                # Mouth missing tolerance: brief absence < 0.8s is tolerated
                if self.mouth_absent_since is None:
                    self.mouth_absent_since = timestamp
                else:
                    absence_duration = timestamp - self.mouth_absent_since
                    if absence_duration > self.mouth_missing_tol:
                        # Past tolerance — check session end duration
                        effective_absence = absence_duration - self.mouth_missing_tol
                        if effective_absence >= self.session_end_dur:
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

        # Do NOT reset: mould_count, moved_positions, anchor_position, locked trolley
        # These persist until pouring cycle ends (5 min timeout)

    # =========================================================================
    # Sub-system 2: Pour Detector (multi-probe)
    # =========================================================================

    def _update_pour(self, best_mouth, frame, timestamp, datetime_obj,
                     mouths, trolleys, target_trolley):
        """Multi-probe brightness measurement below mouth."""
        # Calculate probe points: below mouth bottom-center by probe_below_px
        mx, my_bottom = best_mouth['bottom_center']
        probe_y = my_bottom + self.probe_below_px

        # Average brightness across all probe offsets
        brightness = self._measure_multi_probe_brightness(frame, mx, probe_y)

        # Store last known probe state for screenshot annotations
        self._last_probe_base = (mx, probe_y)
        self._last_probe_brightness = brightness

        if not self.pour_active:
            # Check for pour START
            if brightness > self.brightness_start:
                if self.brightness_above_since is None:
                    self.brightness_above_since = timestamp
                elif timestamp - self.brightness_above_since >= self.pour_start_dur:
                    self._start_pour(timestamp, datetime_obj, mouths, trolleys,
                                     frame, brightness, mx, probe_y, target_trolley)
            else:
                self.brightness_above_since = None
        else:
            # Check for pour END
            if brightness < self.brightness_end:
                if self.brightness_below_since is None:
                    self.brightness_below_since = timestamp
                elif timestamp - self.brightness_below_since >= self.pour_end_dur:
                    self._end_pour(timestamp, datetime_obj, mouths, trolleys, frame)
            else:
                self.brightness_below_since = None

    def _measure_multi_probe_brightness(self, frame, base_x, base_y):
        """Measure average brightness across multiple probe patches.

        Uses HSV Value channel (not grayscale) to isolate brightness
        independent of hue/saturation — more robust for detecting bright
        molten metal glow at varying color temperatures.

        Probe layout: 3 horizontal sample points at 50px below mouth bottom.
        Offsets [(20,0), (30,0), (40,0)] spread right from mouth center.
        Each patch is a (2r x 2r) square, mean V averaged across all patches.
        """
        if frame is None:
            return 0.0

        h, w = frame.shape[:2]
        r = self.probe_radius
        values = []

        for dx, dy in self.probe_offsets:
            px = base_x + dx
            py = base_y + dy
            x1 = max(0, px - r)
            y1 = max(0, py - r)
            x2 = min(w, px + r)
            y2 = min(h, py + r)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            # HSV Value channel — brightness independent of color
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)
            values.append(float(np.mean(hsv[:, :, 2])))

        return sum(values) / len(values) if values else 0.0

    def _start_pour(self, timestamp, datetime_obj, mouths, trolleys, frame,
                    brightness, probe_x, probe_y, target_trolley):
        """Start a pouring event. Lock trolley on first pour."""
        self.pour_active = True
        self.pour_start_time = timestamp
        self.pour_start_datetime = datetime_obj
        self.pour_sync_id = generate_sync_id('pour')
        self.brightness_above_since = None
        self.brightness_below_since = None

        # LOCK TROLLEY on first pour start
        if not self.trolley_locked and target_trolley:
            self._lock_trolley(target_trolley, timestamp)

        # Set mould anchor on pour start (mouth position relative to trolley)
        if target_trolley:
            best_mouth = max(mouths, key=lambda m: m['confidence'])
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
            ladle_number = ""
            if self.heat_cycle_manager and self.heat_cycle_manager.active_cycle:
                heat_no = self.heat_cycle_manager.active_cycle.heat_no
                ladle_number = self.heat_cycle_manager.active_cycle.ladle_number
            self.pour_slno = self.db_manager.insert_pouring_event(
                sync_id=self.pour_sync_id,
                customer_id=self.customer_id,
                date=date_str,
                shift=shift,
                heat_no=heat_no,
                ladle_number=ladle_number,
                location=self.location,
                camera_id=self.camera_id,
                pouring_start_time=datetime_obj.isoformat(),
            )
        except Exception as e:
            logger.error(f"Failed to insert pour start: {e}")

        # Add to heat cycle
        if self.heat_cycle_manager:
            best_mouth = max(mouths, key=lambda m: m['confidence'])
            self.heat_cycle_manager.add_pouring_to_cycle(
                ladle_track_id=best_mouth['track_id'],
                mould_id=f"MOULD_{self.mould_count + 1}",
                mould_track_id=best_mouth['track_id'],
                start_time=timestamp,
                start_datetime=datetime_obj,
                sync_id=self.pour_sync_id,
                slno=self.pour_slno or 0,
            )

        # Screenshot
        self._save_event_screenshot(
            "POUR START", mouths, trolleys, frame, datetime_obj,
            probe_point=(probe_x, probe_y), probe_brightness=brightness
        )

    def _end_pour(self, timestamp, datetime_obj, mouths, trolleys, frame):
        """End the current pouring event."""
        if not self.pour_active or self.pour_start_time is None:
            return

        duration = timestamp - self.pour_start_time
        self.pour_active = False
        self.brightness_above_since = None
        self.brightness_below_since = None

        if duration < self.pour_min_dur:
            logger.info(f"[pour] DISCARDED - duration={duration:.1f}s < {self.pour_min_dur}s minimum")
            self.pour_start_time = None
            self.pour_start_datetime = None
            return

        self.last_pour_duration = duration
        # Count this as a mould (first mould doesn't need move validation)
        if self.mould_count == 0:
            self.mould_count = 1
            logger.info(f"[mould] First mould counted (pour duration={duration:.1f}s)")

        logger.info(f"[pour] END - duration={duration:.1f}s, moulds={self.mould_count}")

        # DB update
        try:
            mould_wise = {
                "mould_count": self.mould_count,
                "last_pour_duration": round(duration, 1),
            }
            self.db_manager.update_pouring_end(
                sync_id=self.pour_sync_id,
                pouring_end_time=datetime_obj.isoformat(),
                total_pouring_time=str(int(duration)),
                mould_wise_pouring_time=mould_wise,
            )
        except Exception as e:
            logger.error(f"Failed to update pour end: {e}")

        # Update heat cycle
        if self.heat_cycle_manager:
            best_mouth = max(mouths, key=lambda m: m['confidence']) if mouths else None
            if best_mouth:
                self.heat_cycle_manager.update_pouring_end(
                    ladle_track_id=best_mouth['track_id'],
                    mould_id=f"MOULD_{self.mould_count}",
                    end_time=timestamp,
                    end_datetime=datetime_obj,
                    duration_seconds=duration,
                )

        # Screenshot (include last known probe points)
        probe_base = self._last_probe_base or self._get_probe_base_from_mouths(mouths)
        self._save_event_screenshot(
            "POUR END", mouths, trolleys, frame, datetime_obj,
            probe_point=probe_base,
            probe_brightness=self._last_probe_brightness,
            extra_info=f"Duration: {duration:.1f}s  Moulds: {self.mould_count}"
        )

        self.pour_start_time = None
        self.pour_start_datetime = None
        self.pour_sync_id = None

    # =========================================================================
    # Sub-system 3: Mould Counter (trolley-relative anchor)
    # =========================================================================

    def _set_anchor_on_pour_start(self, mouth, trolley):
        """Set mould anchor = mouth position normalized to trolley dimensions."""
        mx, my = mouth['center']
        tx1, ty1, tx2, ty2 = trolley['bbox']
        tw = max(tx2 - tx1, 1)
        th = max(ty2 - ty1, 1)

        # Normalize to trolley-relative coordinates
        norm_x = (mx - tx1) / tw
        norm_y = (my - ty1) / th

        self.anchor_position = (norm_x, norm_y)
        self.anchor_set = True
        self.displacement_since = None

    def _update_mould_counter(self, mouth, trolley, timestamp):
        """Track mouth-position displacement relative to trolley for mould counting."""
        if not self.anchor_set or self.anchor_position is None:
            return

        # Current mouth position normalized to trolley
        mx, my = mouth['center']
        tx1, ty1, tx2, ty2 = trolley['bbox']
        tw = max(tx2 - tx1, 1)
        th = max(ty2 - ty1, 1)
        norm_x = (mx - tx1) / tw
        norm_y = (my - ty1) / th

        # Displacement from anchor
        ax, ay = self.anchor_position
        displacement = math.sqrt((norm_x - ax) ** 2 + (norm_y - ay) ** 2)

        if displacement > self.displacement_thresh:
            if self.displacement_since is None:
                self.displacement_since = timestamp
            elif timestamp - self.displacement_since >= self.sustained_dur:
                # Mould switch validation: requires last pour >= min duration
                if self.mould_count == 0 or self.last_pour_duration >= self.mould_switch_min_pour:
                    # Record move, update anchor, increment mould count
                    self.moved_positions.append((norm_x, norm_y, self.last_pour_duration))
                    self.anchor_position = (norm_x, norm_y)
                    self.displacement_since = None
                    self._recompute_clusters()
                    logger.info(
                        f"[mould] Displacement={displacement:.3f} sustained {self.sustained_dur}s, "
                        f"mould_count={self.mould_count}"
                    )
                else:
                    logger.debug(
                        f"[mould] Displacement={displacement:.3f} but last_pour={self.last_pour_duration:.1f}s "
                        f"< min {self.mould_switch_min_pour}s — ignoring"
                    )
                    self.displacement_since = None
        else:
            self.displacement_since = None

    def _recompute_clusters(self):
        """Agglomerative clustering of moved positions to count moulds."""
        if not self.moved_positions:
            # mould_count stays as-is (at least 1 from first pour)
            return

        # Simple agglomerative clustering
        clusters = [[pos] for pos in self.moved_positions]

        # Assign to clusters
        final_clusters = []
        for points in clusters:
            pos = points[0]
            assigned = False
            for cluster in final_clusters:
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                dist = math.sqrt((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2)
                if dist < self.r_cluster:
                    cluster.append(pos)
                    assigned = True
                    break
            if not assigned:
                final_clusters.append([pos])

        # Merge nearby clusters
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(final_clusters):
                j = i + 1
                while j < len(final_clusters):
                    ci = final_clusters[i]
                    cj = final_clusters[j]
                    cx_i = sum(p[0] for p in ci) / len(ci)
                    cy_i = sum(p[1] for p in ci) / len(ci)
                    cx_j = sum(p[0] for p in cj) / len(cj)
                    cy_j = sum(p[1] for p in cj) / len(cj)
                    dist = math.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
                    if dist < self.r_merge:
                        final_clusters[i].extend(final_clusters[j])
                        final_clusters.pop(j)
                        merged = True
                    else:
                        j += 1
                i += 1

        # Reference parity: discard clusters with too little cumulative pour time
        kept_clusters = []
        for cluster in final_clusters:
            total_cluster_pour = sum(float(p[2] or 0.0) for p in cluster)
            if total_cluster_pour >= self.min_cluster_pour_s:
                kept_clusters.append(cluster)

        # +1 because first mould is counted on first pour (before any displacement)
        self.mould_count = len(kept_clusters) + 1

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
        self.session_active = False
        self.mouth_inside_since = None
        self.mouth_absent_since = None
        self.session_start_time = None
        self.session_start_datetime = None
        self.pour_active = False
        self.brightness_above_since = None
        self.brightness_below_since = None
        self.pour_start_time = None
        self.pour_start_datetime = None
        self.pour_sync_id = None
        self.pour_slno = None
        self.last_pour_duration = 0.0
        self.anchor_position = None
        self.anchor_set = False
        self.displacement_since = None
        self.moved_positions.clear()
        self.mould_count = 0
        self.mouth_last_seen_in_trolley = None
        self.cycle_start_time = None
        self.cycle_start_datetime = None
        self._last_probe_base = None
        self._last_probe_brightness = None
        logger.info("[cycle] All state reset — ready for new pouring cycle")

    # =========================================================================
    # Heat cycle DB insertion
    # =========================================================================

    def _insert_heat_cycle_to_db(self, cycle):
        """Insert finalized heat cycle to database."""
        if not cycle.mould_pourings:
            logger.warning(f"Heat cycle {cycle.heat_no} has no pourings — skipping DB insert")
            return

        date_str = cycle.cycle_start_datetime.strftime("%Y-%m-%d")
        hour = cycle.cycle_start_datetime.hour
        shift = "DAY" if 6 <= hour < 18 else "NIGHT"

        try:
            self.db_manager.insert_heat_cycle(
                sync_id=generate_sync_id('heat_cycle'),
                customer_id=self.customer_id,
                date=date_str,
                shift=shift,
                heat_no=cycle.heat_no,
                ladle_number=cycle.ladle_number,
                location=self.location,
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
            )
            logger.info(
                f"HEAT CYCLE FINALIZED: {cycle.heat_no} - "
                f"{len(cycle.mould_pourings)} pours, {cycle.total_pouring_time}s total"
            )
        except Exception as e:
            logger.error(f"Failed to insert heat cycle {cycle.heat_no}: {e}")

    # =========================================================================
    # DS-native OSD overlay metadata (recorded via tee branch after nvosd)
    # =========================================================================

    def _get_probe_base_from_mouths(self, mouths):
        """Derive probe base from best mouth in-frame when no measured probe is cached."""
        if not mouths:
            return None
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
            color = (0, 255, 0) if (probe_brightness or 0) > self.brightness_start else (0, 0, 255)
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
        """Attach custom probe/state overlays that nvosd renders on Stream 0."""
        try:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if not display_meta:
                return

            # 1) Status text block
            cycle_age = (timestamp - self.cycle_start_time) if self.cycle_start_time else 0.0
            absence = (timestamp - self.mouth_last_seen_in_trolley) if self.mouth_last_seen_in_trolley else 0.0
            target_tid = target_trolley['track_id'] if target_trolley else "-"
            lock_tid = self.locked_trolley_id if self.locked_trolley_id is not None else "-"
            brightness_txt = (
                f"{self._last_probe_brightness:.0f}"
                if self._last_probe_brightness is not None else "-"
            )
            text = (
                f"POURING INFERENCE | {datetime_obj.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"SESSION:{'ON' if self.session_active else 'OFF'} "
                f"POUR:{'ON' if self.pour_active else 'OFF'} "
                f"MOULDS:{self.mould_count} B:{brightness_txt} "
                f"TARGET_T:{target_tid} LOCK_T:{lock_tid} "
                f"CYCLE_AGE:{cycle_age:.1f}s ABSENCE:{absence:.1f}s"
            )
            display_meta.num_labels = 1
            txt = display_meta.text_params[0]
            txt.display_text = text
            txt.x_offset = 10
            txt.y_offset = 20
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 12
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)

            # 2) Probe points + expanded lock region as rectangle overlays
            rect_idx = 0
            max_rects = 12

            if self._last_probe_base is not None:
                base_x, base_y = self._last_probe_base
                probe_on = (self._last_probe_brightness or 0.0) > self.brightness_start
                for dx, dy in self.probe_offsets:
                    if rect_idx >= max_rects:
                        break
                    px = int(base_x + dx)
                    py = int(base_y + dy)
                    size = max(3, self.probe_radius)
                    rect = display_meta.rect_params[rect_idx]
                    rect.left = max(0, px - size // 2)
                    rect.top = max(0, py - size // 2)
                    rect.width = size
                    rect.height = size
                    rect.border_width = 1
                    rect.has_bg_color = 1
                    if probe_on:
                        rect.border_color.set(0.0, 1.0, 0.0, 1.0)
                        rect.bg_color.set(0.0, 1.0, 0.0, 0.9)
                    else:
                        rect.border_color.set(1.0, 0.0, 0.0, 1.0)
                        rect.bg_color.set(1.0, 0.0, 0.0, 0.9)
                    rect_idx += 1

            if self.trolley_locked and self.locked_trolley_bbox and rect_idx < max_rects:
                x1, y1, x2, y2 = self.locked_trolley_bbox
                ey1 = max(0, y1 - self.edge_expand)
                rect = display_meta.rect_params[rect_idx]
                rect.left = int(max(0, x1))
                rect.top = int(max(0, ey1))
                rect.width = int(max(1, x2 - x1))
                rect.height = int(max(1, y2 - ey1))
                rect.border_width = 1
                rect.has_bg_color = 0
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)
                rect_idx += 1

            display_meta.num_rects = rect_idx
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        except Exception as exc:
            logger.error(f"[osd] Failed to attach inference display meta: {exc}", exc_info=True)

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
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            annotated = frame_bgr.copy()
            h, w = annotated.shape[:2]

            # Draw trolley bboxes (green) + expanded region (dashed green)
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

                # Draw expanded region for locked trolley (top edge only)
                if is_locked:
                    ey1 = max(0, y1 - self.edge_expand)
                    cv2.rectangle(annotated, (x1, ey1), (x2, y2), (0, 255, 0), 1)

            # Draw mouth bboxes (cyan)
            for m in mouths:
                x1, y1, x2, y2 = m['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"Mouth T{m['track_id']} {m['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw probe points
            self._draw_probe_points(annotated, probe_point, probe_brightness)

            # Title
            cv2.putText(annotated, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Timestamp
            ts_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated, ts_str, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Extra info
            if extra_info:
                cv2.putText(annotated, extra_info, (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Mould count + trolley lock status
            status = f"Moulds: {self.mould_count}"
            if self.trolley_locked:
                status += f"  Trolley: T{self.locked_trolley_id} [LOCKED]"
            cv2.putText(annotated, status, (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Camera ID
            cv2.putText(annotated, f"CAM: {self.camera_id}", (w - 200, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            # Save
            tag = title.lower().replace(" ", "_")
            ts = datetime_obj.strftime("%Y%m%d_%H%M%S")
            filename = f"pouring_{tag}_{ts}.jpg"
            filepath = self.screenshot_dir / filename
            cv2.imwrite(str(filepath), annotated)
            logger.info(f"Saved screenshot: {filename}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return None
