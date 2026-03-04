"""
Shared screenshot utilities for HiCon processors.

Provides common annotation patterns (header, footer, ROI overlay, save)
so each processor only adds its domain-specific drawings.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def prepare_frame(frame_rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA frame to BGR copy ready for annotation."""
    return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR).copy()


def add_header(annotated: np.ndarray, title: str, timestamp_str: str,
               extra_lines: Optional[List[str]] = None):
    """Add standard title, timestamp, and optional extra info lines.

    - Title at (10, 30): yellow, 0.9 scale, thickness 2
    - Timestamp at (10, 60): white, 0.6 scale, thickness 2
    - Extra lines starting at y=85, 25px apart: gray, 0.5 scale, thickness 1
    """
    cv2.putText(annotated, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(annotated, timestamp_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if extra_lines:
        y = 85
        for line in extra_lines:
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 25


def add_footer(annotated: np.ndarray, camera_id: str,
               status_text: Optional[str] = None):
    """Add camera ID and optional status bar at the bottom.

    - Status at (10, h-40): light gray, 0.6 scale, thickness 2
    - Camera at (w-200, h-15): teal, 0.5 scale, thickness 1
    """
    h, w = annotated.shape[:2]
    if status_text:
        cv2.putText(annotated, status_text, (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(annotated, f"CAM: {camera_id}", (w - 200, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)


def draw_roi_overlay(annotated: np.ndarray, points, color: Tuple[int, int, int],
                     label: Optional[str] = None, alpha: float = 0.15):
    """Draw a semi-transparent polygon ROI with optional centered label."""
    pts = np.array(points, dtype=np.int32)
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
    cv2.polylines(annotated, [pts], True, color, 2)
    if label:
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(annotated, label, (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def save(annotated: np.ndarray, prefix: str, tag: str,
         timestamp: datetime, screenshot_dir: Path) -> Optional[str]:
    """Save annotated screenshot and return file path, or None on error."""
    try:
        ts = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{tag}_{ts}.jpg"
        filepath = screenshot_dir / filename
        cv2.imwrite(str(filepath), annotated)
        logger.info(f"Saved screenshot: {filename}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
        return None
