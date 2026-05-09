"""
MouldPlacementDetector: pre-pour snapshot comparison for mould instance detection.

Algorithm:
  At session start → capture S_base (trolley ROI baseline)
  At each pour start → capture S_N, compare to S_{N-1} using signed diff + edge diff
  Detect new physical objects (sand moulds placed on trolley) as compact blobs
  Lock detected blob as mould instance; assign to current pour
  Count poured instances for predictive_mould_count

Coordinate note:
  Blob centroids detected in raw trolley bbox space (crop = raw bbox, 128×64)
  are converted to expanded-bbox normalized [0,1] to match _normalize_mouth_position()
  which normalises relative to trolley_bbox ± EDGE_EXPAND_PX.

Feature flag: HICON_MOULD_TRACKING_MODE controls which count reaches the DB.
  reactive (default) — existing cluster-based count unchanged; this runs silently for logs only.
  predictive — predictive_mould_count replaces mould_count.
  hybrid — predictive if available; reactive fallback.
"""

import cv2
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_KERNEL_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_KERNEL_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


class _BlobState:
    __slots__ = (
        'blob_id', 'cx_raw', 'cy_raw', 'cx_exp', 'cy_exp',
        'area_norm', 'detected_at', 'pour_count', 'total_pour_s',
    )

    def __init__(self, blob_id: int, cx_raw: float, cy_raw: float,
                 cx_exp: float, cy_exp: float, area_norm: float, timestamp: float):
        self.blob_id = blob_id
        self.cx_raw = cx_raw
        self.cy_raw = cy_raw
        self.cx_exp = cx_exp   # expanded-bbox normalized (matches mouth norm space)
        self.cy_exp = cy_exp
        self.area_norm = area_norm
        self.detected_at = timestamp
        self.pour_count = 0
        self.total_pour_s = 0.0


class MouldPlacementDetector:
    """
    Detects mould placement events by comparing consecutive pre-pour trolley snapshots.
    Assigns each pour event to the nearest detected mould blob.
    """

    def __init__(self, config):
        self.diff_thresh     = int(getattr(config, 'PLACEMENT_DIFF_THRESH', 30))
        self.min_blob_area   = int(getattr(config, 'PLACEMENT_MIN_BLOB_AREA_PX', 50))
        self.max_blob_area   = int(getattr(config, 'PLACEMENT_MAX_BLOB_AREA_PX', 2000))
        self.r_assign        = float(getattr(config, 'PLACEMENT_R_ASSIGN', 0.15))
        self.canonical_w     = int(getattr(config, 'PLACEMENT_CANONICAL_W', 128))
        self.canonical_h     = int(getattr(config, 'PLACEMENT_CANONICAL_H', 64))
        self.canny_low       = int(getattr(config, 'PLACEMENT_CANNY_LOW', 40))
        self.canny_high      = int(getattr(config, 'PLACEMENT_CANNY_HIGH', 120))
        self.min_aspect      = float(getattr(config, 'PLACEMENT_MIN_ASPECT', 0.15))
        self.max_aspect      = float(getattr(config, 'PLACEMENT_MAX_ASPECT', 6.67))
        self.edge_expand_px  = int(getattr(config, 'EDGE_EXPAND_PX', 180))
        self.save_debug      = bool(getattr(config, 'PLACEMENT_SAVE_DEBUG_IMAGES', False))
        self.screenshot_dir  = Path(getattr(config, 'SCREENSHOT_DIR', 'output/screenshots'))

        # Per-session / per-cycle state
        self._S_prev: Optional[np.ndarray] = None   # 128×64 uint8 grayscale
        self._pour_index: int = 0
        self._locked_blobs: Dict[int, _BlobState] = {}
        self._next_blob_id: int = 1
        self._session_active: bool = False

    # ─────────────────────────── lifecycle ────────────────────────────────────

    def on_session_start(self, frame_rgba: Optional[np.ndarray],
                         trolley_bbox: Optional[Tuple[int, int, int, int]],
                         timestamp: float) -> None:
        """Reset session state and capture initial baseline snapshot."""
        self._S_prev = None
        self._pour_index = 0
        self._locked_blobs.clear()
        self._next_blob_id = 1
        self._session_active = True

        if frame_rgba is not None and trolley_bbox is not None:
            roi = self._extract_roi(frame_rgba, trolley_bbox)
            if roi is not None:
                self._S_prev = self._preprocess(roi)
                logger.debug("[placement] session_start: baseline snapshot captured")

    def on_session_end(self) -> None:
        self._session_active = False

    def on_cycle_reset(self) -> None:
        """Full reset on pouring cycle end / 5-min timeout."""
        self._S_prev = None
        self._pour_index = 0
        self._locked_blobs.clear()
        self._next_blob_id = 1
        self._session_active = False

    # ──────────────────────────── pour events ─────────────────────────────────

    def on_pour_start(self, frame_rgba: Optional[np.ndarray],
                      trolley_bbox: Optional[Tuple[int, int, int, int]],
                      timestamp: float) -> Optional[int]:
        """
        Capture pre-pour snapshot, detect new mould blob, lock it as a mould instance.
        Returns the assigned blob_id (int) or None if detection is unavailable.

        Always advances S_prev so future diffs stay current regardless of detection.
        """
        if not self._session_active:
            return None

        self._pour_index += 1

        if frame_rgba is None or trolley_bbox is None:
            logger.debug("[placement] pour_%d: no frame/trolley, detection skipped",
                         self._pour_index)
            return None

        roi = self._extract_roi(frame_rgba, trolley_bbox)
        if roi is None:
            return None
        S_N = self._preprocess(roi)

        blob_id = None
        if self._S_prev is not None:
            blobs = self._detect_new_mould(S_N, self._S_prev)
            if blobs:
                best = max(blobs, key=lambda b: b['area'])
                fh, fw = frame_rgba.shape[:2]
                cx_exp, cy_exp = self._to_expanded_norm(
                    best['cx_norm'], best['cy_norm'], trolley_bbox, fw, fh
                )
                state = _BlobState(
                    blob_id=self._next_blob_id,
                    cx_raw=best['cx_norm'],
                    cy_raw=best['cy_norm'],
                    cx_exp=cx_exp,
                    cy_exp=cy_exp,
                    area_norm=best['area'] / (self.canonical_w * self.canonical_h),
                    timestamp=timestamp,
                )
                self._locked_blobs[self._next_blob_id] = state
                blob_id = self._next_blob_id
                self._next_blob_id += 1
                logger.info(
                    "[placement] pour_%d: B%d at raw(%.3f,%.3f) exp(%.3f,%.3f) area=%d",
                    self._pour_index, blob_id,
                    best['cx_norm'], best['cy_norm'],
                    cx_exp, cy_exp, best['area'],
                )
            else:
                logger.info("[placement] pour_%d: no new blob detected", self._pour_index)

            if self.save_debug:
                self._save_debug_image(S_N, self._S_prev, self._pour_index, timestamp)
        else:
            logger.debug("[placement] pour_%d: no baseline yet (first pour of cycle)",
                         self._pour_index)

        self._S_prev = S_N
        return blob_id

    def on_pour_end(self, blob_id: Optional[int], duration_s: float) -> None:
        """Record a completed pour against the assigned blob."""
        if blob_id is None:
            return
        state = self._locked_blobs.get(blob_id)
        if state:
            state.pour_count += 1
            state.total_pour_s += duration_s

    # ────────────────────────── query ────────────────────────────────────────

    def get_poured_blob_count(self) -> int:
        """Number of mould instances that received at least one pour."""
        return sum(1 for b in self._locked_blobs.values() if b.pour_count >= 1)

    def get_result_dict(self) -> Dict:
        return {
            "predictive_mould_count": self.get_poured_blob_count(),
            "placement_blob_count": len(self._locked_blobs),
            "pour_index": self._pour_index,
            "blobs": [
                {
                    "blob_id": b.blob_id,
                    "cx_exp": round(b.cx_exp, 3),
                    "cy_exp": round(b.cy_exp, 3),
                    "area_norm": round(b.area_norm, 4),
                    "pour_count": b.pour_count,
                    "total_pour_s": round(b.total_pour_s, 1),
                }
                for b in self._locked_blobs.values()
            ],
        }

    # ───────────────────────── internal ──────────────────────────────────────

    @staticmethod
    def _to_grayscale(frame_rgba: np.ndarray) -> np.ndarray:
        if frame_rgba.ndim == 3 and frame_rgba.shape[2] >= 3:
            return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2GRAY)
        return frame_rgba

    def _extract_roi(self, frame_rgba: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Convert RGBA frame to grayscale, crop trolley bbox, resize to canonical grid."""
        gray = self._to_grayscale(frame_rgba)
        fh = gray.shape[0]
        fw = gray.shape[1] if gray.ndim == 2 else gray.shape[1]
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
        """Normalize local contrast to reduce lighting drift."""
        return cv2.equalizeHist(cv2.GaussianBlur(roi, (3, 3), 0))

    def _detect_new_mould(self, S_N: np.ndarray,
                          S_prev: np.ndarray) -> List[Dict]:
        """
        Two-channel detector:
          Channel 1 — signed intensity diff: objects that became brighter/appeared.
                       cv2.subtract clips negatives to 0 (fading glow → 0, new mould → positive).
          Channel 2 — edge addition diff: new rectangular contours regardless of temperature.
                       Sand moulds have sharp edges; diffuse glow does not.

        Returns list of {cx_norm, cy_norm, area, aspect}.
        """
        # Channel 1: new/brighter objects
        pos_diff = cv2.subtract(S_N, S_prev)
        _, intensity_mask = cv2.threshold(
            pos_diff, self.diff_thresh, 255, cv2.THRESH_BINARY
        )

        # Channel 2: new structural edges
        edges_prev = cv2.Canny(S_prev, self.canny_low, self.canny_high)
        edges_curr = cv2.Canny(S_N, self.canny_low, self.canny_high)
        new_edges = cv2.subtract(edges_curr, edges_prev)
        edge_dilated = cv2.dilate(new_edges, _KERNEL_5x5, iterations=2)
        _, edge_mask = cv2.threshold(edge_dilated, 10, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(intensity_mask, edge_mask)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, _KERNEL_3x3)

        n, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        blobs = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_blob_area or area > self.max_blob_area:
                continue
            w_px = max(int(stats[i, cv2.CC_STAT_WIDTH]), 1)
            h_px = max(int(stats[i, cv2.CC_STAT_HEIGHT]), 1)
            aspect = w_px / h_px
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            blobs.append({
                'cx_norm': float(centroids[i][0]) / self.canonical_w,
                'cy_norm': float(centroids[i][1]) / self.canonical_h,
                'area': area,
                'aspect': aspect,
            })
        return blobs

    def _to_expanded_norm(self, cx_raw: float, cy_raw: float,
                          trolley_bbox: Tuple[int, int, int, int],
                          frame_w: int, frame_h: int) -> Tuple[float, float]:
        """
        Convert raw-ROI centroid [0,1] to expanded-bbox normalized [0,1].
        Expanded bbox = trolley_bbox ± EDGE_EXPAND_PX, matching _normalize_mouth_position().
        """
        x1, y1, x2, y2 = trolley_bbox
        tw = max(x2 - x1, 1)
        th = max(y2 - y1, 1)
        ex = self.edge_expand_px
        # Pixel position in frame (relative to raw bbox)
        ax = x1 + cx_raw * tw
        ay = y1 + cy_raw * th
        # Expanded bbox dimensions
        ew = tw + 2 * ex
        eh = th + 2 * ex
        cx_exp = (ax - (x1 - ex)) / ew
        cy_exp = (ay - (y1 - ex)) / eh
        return cx_exp, cy_exp

    def _save_debug_image(self, S_N: np.ndarray, S_prev: np.ndarray,
                          pour_idx: int, timestamp: float) -> None:
        try:
            ts = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            pos_diff = cv2.subtract(S_N, S_prev)
            _, thr = cv2.threshold(pos_diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
            strip = np.hstack([S_prev, S_N, pos_diff, thr])
            path = self.screenshot_dir / f"placement_pour{pour_idx}_{ts}.jpg"
            cv2.imwrite(str(path), strip)
        except Exception as exc:
            logger.debug("[placement] debug image save failed: %s", exc)
