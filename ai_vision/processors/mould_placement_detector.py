"""
MouldPlacementDetector: single-frame spatial edge detection for mould counting.

Algorithm (v3 — spatial):
  At each pour start, detect the brightness boundary between mould surfaces and the
  trolley bed within a single frame. No temporal comparison or baseline required.

  Sand mould against metal trolley → strong brightness boundary (different reflectance).
  Glowing poured mould → even stronger boundary (high contrast against trolley).
  Fresh oven-baked mould → moderate but detectable boundary.

  Steps per pour:
    1. Crop trolley bbox → 128×64 grayscale
    2. equalizeHist → enhance mould/trolley contrast
    3. Canny → extract brightness boundaries
    4. Dilate + morphological close → fill mould outlines into solid regions
    5. Connected components + area/aspect filter → count distinct mould objects
    6. predictive_mould_count = max count seen across all pours in the cycle

  Why single-frame beats temporal diff:
  - Pour glow is a positive feature here (stronger edge against trolley), not a problem
  - Detects mould 1 directly (no "first pour is special" offset)
  - No baseline state to maintain; robust to lighting shifts
  - Simpler: no _cycle_baseline, _S_prev, _locked_blobs

Feature flag: HICON_MOULD_TRACKING_MODE controls which count reaches the DB.
  reactive (default) — existing cluster-based count unchanged; runs for logs only.
  predictive         — predictive_mould_count replaces mould_count.
  hybrid             — predictive if > 0; reactive fallback.
"""

import cv2
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_KERNEL_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_KERNEL_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


class MouldPlacementDetector:
    """
    Counts moulds on trolley by spatial edge detection at each pour start.
    State is cycle-scoped: persists across sessions, resets on cycle timeout.
    """

    def __init__(self, config):
        self.canny_low      = int(getattr(config, 'PLACEMENT_CANNY_LOW', 40))
        self.canny_high     = int(getattr(config, 'PLACEMENT_CANNY_HIGH', 120))
        self.min_blob_area  = int(getattr(config, 'PLACEMENT_MIN_BLOB_AREA_PX', 50))
        self.max_blob_area  = int(getattr(config, 'PLACEMENT_MAX_BLOB_AREA_PX', 2000))
        self.min_aspect     = float(getattr(config, 'PLACEMENT_MIN_ASPECT', 0.15))
        self.max_aspect     = float(getattr(config, 'PLACEMENT_MAX_ASPECT', 6.67))
        self.canonical_w    = int(getattr(config, 'PLACEMENT_CANONICAL_W', 128))
        self.canonical_h    = int(getattr(config, 'PLACEMENT_CANONICAL_H', 64))
        self.edge_expand_px = int(getattr(config, 'EDGE_EXPAND_PX', 180))
        self.save_debug     = bool(getattr(config, 'PLACEMENT_SAVE_DEBUG_IMAGES', False))
        self.screenshot_dir = Path(getattr(config, 'SCREENSHOT_DIR', 'output/screenshots'))

        # Cycle-scoped state (cleared only by on_cycle_reset)
        self._max_count: int = 0      # highest mould count seen this cycle
        self._pour_index: int = 0     # cumulative across sessions within cycle
        self._session_active: bool = False

    # ─────────────────────────── lifecycle ────────────────────────────────────

    def on_session_start(self, frame_rgba, trolley_bbox, timestamp: float) -> None:
        """Mark session active. Cycle-level count is preserved across sessions."""
        self._session_active = True

    def on_session_end(self) -> None:
        self._session_active = False

    def on_cycle_reset(self) -> None:
        """Full reset on pouring cycle end / 5-min timeout."""
        self._max_count = 0
        self._pour_index = 0
        self._session_active = False

    # ──────────────────────────── pour events ─────────────────────────────────

    def on_pour_start(self, frame_rgba: Optional[np.ndarray],
                      trolley_bbox: Optional[Tuple[int, int, int, int]],
                      timestamp: float) -> None:
        """
        Detect moulds on trolley via single-frame spatial edge detection.
        Updates _max_count with the highest count seen in this cycle.
        Returns None (no per-blob assignment in this approach).
        """
        if not self._session_active:
            return None

        self._pour_index += 1

        if frame_rgba is None or trolley_bbox is None:
            logger.debug("[placement] pour_%d: no frame/trolley, skipped",
                         self._pour_index)
            return None

        roi = self._extract_roi(frame_rgba, trolley_bbox)
        if roi is None:
            return None

        enhanced = self._preprocess(roi)
        count = self._detect_moulds_single_frame(enhanced)
        self._max_count = max(self._max_count, count)

        logger.info(
            "[placement] pour_%d: %d mould(s) on trolley (cycle_max=%d)",
            self._pour_index, count, self._max_count,
        )

        if self.save_debug:
            self._save_debug_image(enhanced, self._pour_index, count, timestamp)

        return None

    def on_pour_end(self, blob_id, duration_s: float) -> None:
        """No-op — spatial detection has no per-blob state to close."""

    # ────────────────────────── query ────────────────────────────────────────

    def get_poured_blob_count(self) -> int:
        """Cycle-level predictive mould count (max seen across all pours)."""
        return self._max_count

    def get_result_dict(self):
        return {
            "predictive_mould_count": self._max_count,
            "cycle_pour_index": self._pour_index,
        }

    # ───────────────────────── internal ──────────────────────────────────────

    @staticmethod
    def _to_grayscale(frame_rgba: np.ndarray) -> np.ndarray:
        if frame_rgba.ndim == 3 and frame_rgba.shape[2] >= 3:
            return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2GRAY)
        return frame_rgba

    def _extract_roi(self, frame_rgba: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop trolley bbox from grayscale frame, resize to canonical grid."""
        gray = self._to_grayscale(frame_rgba)
        fh, fw = gray.shape[0], gray.shape[1]
        x1, y1, x2, y2 = bbox
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(fw, x2)
        y2c = min(fh, y2)
        if x2c - x1c < 8 or y2c - y1c < 4:
            return None
        crop = gray[y1c:y2c, x1c:x2c]
        return cv2.resize(crop, (self.canonical_w, self.canonical_h),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _preprocess(roi: np.ndarray) -> np.ndarray:
        """Enhance local contrast to make mould/trolley boundary more visible."""
        return cv2.equalizeHist(cv2.GaussianBlur(roi, (3, 3), 0))

    def _detect_moulds_single_frame(self, enhanced: np.ndarray) -> int:
        """
        Count distinct compact objects on trolley bed from a single equalized frame.

        Moulds create brightness boundaries against the trolley surface.
        Glowing poured moulds produce even stronger boundaries — glow is a positive
        feature here, not a problem.
        """
        # Extract all brightness boundaries (mould/trolley interface)
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        # Small dilation to thicken edges without merging adjacent moulds, then
        # close with a larger kernel to fill mould interiors into solid blobs.
        dilated = cv2.dilate(edges, _KERNEL_3x3, iterations=1)
        _, mask = cv2.threshold(dilated, 10, 255, cv2.THRESH_BINARY)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_5x5)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _KERNEL_3x3)

        n, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        count = 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_blob_area or area > self.max_blob_area:
                continue
            w = max(int(stats[i, cv2.CC_STAT_WIDTH]), 1)
            h = max(int(stats[i, cv2.CC_STAT_HEIGHT]), 1)
            aspect = w / h
            if self.min_aspect <= aspect <= self.max_aspect:
                count += 1
        return count

    def _save_debug_image(self, enhanced: np.ndarray, pour_idx: int,
                          count: int, timestamp: float) -> None:
        """Save [equalized_roi | edges | cleaned_mask] strip to screenshots dir."""
        try:
            ts = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)
            dilated = cv2.dilate(edges, _KERNEL_3x3, iterations=1)
            _, mask = cv2.threshold(dilated, 10, 255, cv2.THRESH_BINARY)
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_5x5)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _KERNEL_3x3)
            strip = np.hstack([enhanced, edges, cleaned])
            path = self.screenshot_dir / f"placement_pour{pour_idx}_n{count}_{ts}.jpg"
            cv2.imwrite(str(path), strip)
        except Exception as exc:
            logger.debug("[placement] debug image save failed: %s", exc)
