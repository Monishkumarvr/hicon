"""
Shared screenshot utilities for HiCon processors.

Provides common annotation patterns (header, footer, ROI overlay, save)
so each processor only adds its domain-specific drawings.
"""

import cv2
import json
import numpy as np
import logging
import os
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Any

from utils.perf import timed_section

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WIDTH = int(os.getenv('HICON_SCREENSHOT_MAX_WIDTH', '1280'))
_DEFAULT_JPEG_QUALITY = int(os.getenv('HICON_SCREENSHOT_JPEG_QUALITY', '75'))
_DEFAULT_SAVE_RAW = os.getenv('HICON_SAVE_RAW_SCREENSHOTS', 'false').lower() == 'true'


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


def _to_coco_bbox(x1: int, y1: int, x2: int, y2: int) -> List[int]:
    """Convert (x1,y1,x2,y2) → COCO [x, y, w, h] with top-left origin."""
    return [x1, y1, x2 - x1, y2 - y1]


def _write_coco_sidecar(
    jpg_path: Path,
    annotated: np.ndarray,
    timestamp: datetime,
    annotations: List[dict],
    event_type: Optional[str],
    camera_id: Optional[str],
    categories: Optional[List[dict]],
    raw_filename: Optional[str] = None,
) -> None:
    """Write a COCO-format JSON sidecar file next to the saved JPEG.

    ``image.file_name`` points to the clean raw frame when available so the
    JSON can be used directly as a training annotation without overlay pixels.
    """
    h, w = annotated.shape[:2]
    coco = {
        "image": {
            "file_name": raw_filename or jpg_path.name,
            "width": w,
            "height": h,
            "date_captured": timestamp.isoformat(),
        },
        "annotations": annotations,
        "categories": categories or [],
        "event": {
            "event_type": event_type or "",
            "camera_id": camera_id or "",
        },
    }
    json_path = jpg_path.with_suffix(".json")
    with open(str(json_path), "w") as f:
        json.dump(coco, f, indent=2)
    logger.info(f"Saved COCO sidecar: {json_path.name}")


def _build_output_paths(
    prefix: str,
    tag: str,
    timestamp: datetime,
    screenshot_dir: Path,
) -> tuple[Path, str, Path]:
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{tag}_{ts}.jpg"
    filepath = screenshot_dir / filename
    raw_filename = f"{prefix}_{tag}_{ts}_raw.jpg"
    raw_filepath = screenshot_dir / raw_filename
    return filepath, raw_filename, raw_filepath


def _write_screenshot_bundle(
    filepath: Path,
    raw_filename: str,
    raw_filepath: Path,
    annotated: np.ndarray,
    timestamp: datetime,
    annotations: Optional[List[dict]] = None,
    *,
    raw_frame: Optional[np.ndarray] = None,
    event_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    categories: Optional[List[dict]] = None,
    max_width: int = _DEFAULT_MAX_WIDTH,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
    save_raw: bool = _DEFAULT_SAVE_RAW,
) -> Optional[str]:
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        source_h, source_w = annotated.shape[:2]
        scale = min(1.0, float(max_width) / max(source_w, 1))
        if scale < 1.0:
            output_size = (max_width, max(1, int(round(source_h * scale))))
            annotated = cv2.resize(annotated, output_size, interpolation=cv2.INTER_AREA)
            if raw_frame is not None:
                raw_frame = cv2.resize(raw_frame, output_size, interpolation=cv2.INTER_AREA)
            if annotations:
                scaled_annotations = []
                for annotation in annotations:
                    scaled = dict(annotation)
                    bbox = annotation.get('bbox')
                    if bbox and len(bbox) == 4:
                        scaled['bbox'] = [
                            round(float(value) * scale, 2) for value in bbox
                        ]
                    if 'area' in annotation:
                        scaled['area'] = round(float(annotation['area']) * scale * scale, 2)
                    scaled_annotations.append(scaled)
                annotations = scaled_annotations

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        cv2.imwrite(str(filepath), annotated, encode_params)
        logger.info(f"Saved screenshot: {filepath.name}")

        image_name = None
        if save_raw and raw_frame is not None:
            cv2.imwrite(str(raw_filepath), raw_frame, encode_params)
            logger.info(f"Saved raw screenshot: {raw_filename}")
            image_name = raw_filename

        if annotations:
            try:
                _write_coco_sidecar(
                    filepath,
                    annotated,
                    timestamp,
                    annotations,
                    event_type,
                    camera_id,
                    categories,
                    raw_filename=image_name,
                )
            except Exception as exc:
                logger.error(f"Error saving COCO sidecar: {exc}")
        return str(filepath)
    except Exception as exc:
        logger.error(f"Error saving screenshot: {exc}")
        return None


class AsyncScreenshotWriter:
    """Background screenshot writer for event-driven captures."""

    _SENTINEL = object()

    def __init__(self, maxsize: int = 20, *, max_width: int = _DEFAULT_MAX_WIDTH,
                 jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
                 save_raw: bool = _DEFAULT_SAVE_RAW):
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._max_width = int(max_width)
        self._jpeg_quality = int(jpeg_quality)
        self._save_raw = bool(save_raw)
        self._stopped = False
        self._thread = threading.Thread(
            target=self._worker,
            name="hicon-screenshot-writer",
            daemon=False,
        )
        self._thread.start()
        logger.info("AsyncScreenshotWriter initialized (maxsize=%d)", maxsize)

    def save(
        self,
        annotated: np.ndarray,
        prefix: str,
        tag: str,
        timestamp: datetime,
        screenshot_dir: Path,
        annotations: Optional[List[dict]] = None,
        *,
        raw_frame: Optional[np.ndarray] = None,
        event_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        categories: Optional[List[dict]] = None,
    ) -> Optional[str]:
        filepath, raw_filename, raw_filepath = _build_output_paths(
            prefix,
            tag,
            timestamp,
            screenshot_dir,
        )
        job = {
            "filepath": filepath,
            "raw_filename": raw_filename,
            "raw_filepath": raw_filepath,
            "annotated": annotated,
            "timestamp": timestamp,
            "annotations": annotations,
            "raw_frame": raw_frame,
            "event_type": event_type,
            "camera_id": camera_id,
            "categories": categories,
            "max_width": self._max_width,
            "jpeg_quality": self._jpeg_quality,
            "save_raw": self._save_raw,
        }

        if self._stopped:
            logger.warning("AsyncScreenshotWriter stopped; saving %s synchronously", filepath.name)
            return _write_screenshot_bundle(**job)

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            logger.warning("Screenshot queue full; dropping %s", filepath.name)
            return None
        return str(filepath)

    def _worker(self):
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                with timed_section("screenshot.write.async", threshold_ms=20.0, logger=logger):
                    _write_screenshot_bundle(**item)
            except Exception as exc:
                logger.error("AsyncScreenshotWriter failed: %s", exc, exc_info=True)
            finally:
                self._queue.task_done()

    def stop(self, timeout: float = 5.0, drain: bool = True) -> None:
        if self._stopped:
            return
        if drain:
            self._queue.join()
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("AsyncScreenshotWriter did not stop cleanly within %.1fs", timeout)
        self._stopped = True


def save(
    annotated: np.ndarray,
    prefix: str,
    tag: str,
    timestamp: datetime,
    screenshot_dir: Path,
    annotations: Optional[List[dict]] = None,
    *,
    raw_frame: Optional[np.ndarray] = None,
    event_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    categories: Optional[List[dict]] = None,
    writer: Optional[AsyncScreenshotWriter] = None,
) -> Optional[str]:
    """Save annotated screenshot and return file path, or None on error.

    If *raw_frame* is provided, a clean JPEG (no overlays) is saved as
    ``{prefix}_{tag}_{ts}_raw.jpg`` alongside the annotated one.

    If *annotations* is a non-empty list, a COCO-format JSON sidecar is written
    alongside the JPEG (same stem, ``.json`` extension). ``image.file_name``
    in the JSON points to the raw frame when available.
    """
    if writer is not None:
        return writer.save(
            annotated,
            prefix,
            tag,
            timestamp,
            screenshot_dir,
            annotations,
            raw_frame=raw_frame,
            event_type=event_type,
            camera_id=camera_id,
            categories=categories,
        )

    filepath, raw_filename, raw_filepath = _build_output_paths(
        prefix,
        tag,
        timestamp,
        screenshot_dir,
    )
    with timed_section("screenshot.write.sync", threshold_ms=20.0, logger=logger):
        return _write_screenshot_bundle(
            filepath,
            raw_filename,
            raw_filepath,
            annotated,
            timestamp,
            annotations,
            raw_frame=raw_frame,
            event_type=event_type,
            camera_id=camera_id,
            categories=categories,
            max_width=_DEFAULT_MAX_WIDTH,
            jpeg_quality=_DEFAULT_JPEG_QUALITY,
            save_raw=_DEFAULT_SAVE_RAW,
        )
