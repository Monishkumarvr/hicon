"""
HiCon Utility Functions
Common helper functions used across the pipeline
"""
import time
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


def get_mean_brightness(frame: np.ndarray, zone_points: np.ndarray) -> float:
    """
    Calculate mean brightness within polygon zone.
    Uses the same algorithm as user's deslagging script.

    Args:
        frame: Input frame (BGR or grayscale)
        zone_points: Polygon points defining the zone

    Returns:
        Mean brightness value (0-255)
    """
    # Create mask for polygon
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [zone_points], 255)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Get pixel values within polygon
    masked_values = gray[mask > 0]

    if len(masked_values) == 0:
        return 0.0

    # Return mean brightness
    return float(np.mean(masked_values))


def save_screenshot(frame: np.ndarray, prefix: str, screenshot_dir: Path) -> str:
    """
    Save screenshot and return path.

    Args:
        frame: Frame to save
        prefix: Filename prefix
        screenshot_dir: Directory to save screenshots

    Returns:
        Full path to saved screenshot
    """
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = screenshot_dir / filename
    cv2.imwrite(str(filepath), frame)
    return str(filepath)


def generate_sync_id(prefix: str) -> str:
    """
    Generate unique sync ID.

    Args:
        prefix: Prefix for the sync ID

    Returns:
        Unique sync ID string (format: prefix-timestamp-random)
    """
    return f"{prefix}-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"


def compute_bbox_scale(frame_shape: tuple, bbox: list, source_dims: tuple) -> tuple:
    """
    Compute scaling factor for bounding box coordinates.

    Args:
        frame_shape: Current frame shape (height, width, channels)
        bbox: Bounding box [x1, y1, x2, y2]
        source_dims: Original source dimensions (width, height)

    Returns:
        Tuple of (x_scale, y_scale)
    """
    if not source_dims or source_dims == (0, 0):
        return (1.0, 1.0)

    frame_h, frame_w = frame_shape[:2]
    source_w, source_h = source_dims

    x_scale = frame_w / source_w if source_w > 0 else 1.0
    y_scale = frame_h / source_h if source_h > 0 else 1.0

    return (x_scale, y_scale)


def scale_bbox(bbox: list, frame_shape: tuple, scale: tuple) -> list:
    """
    Scale bounding box coordinates.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        frame_shape: Current frame shape (height, width, channels)
        scale: Scaling factors (x_scale, y_scale)

    Returns:
        Scaled bounding box [x1, y1, x2, y2]
    """
    x_scale, y_scale = scale
    x1, y1, x2, y2 = bbox

    return [
        int(x1 * x_scale),
        int(y1 * y_scale),
        int(x2 * x_scale),
        int(y2 * y_scale)
    ]


def scale_point(point: tuple, frame_shape: tuple, scale: tuple) -> tuple:
    """
    Scale a single point coordinate.

    Args:
        point: Point (x, y)
        frame_shape: Current frame shape (height, width, channels)
        scale: Scaling factors (x_scale, y_scale)

    Returns:
        Scaled point (x, y)
    """
    x_scale, y_scale = scale
    x, y = point

    return (int(x * x_scale), int(y * y_scale))


def scale_polygon(polygon: list, frame_shape: tuple, scale: tuple) -> list:
    """
    Scale polygon coordinates.

    Args:
        polygon: List of points [(x1, y1), (x2, y2), ...]
        frame_shape: Current frame shape (height, width, channels)
        scale: Scaling factors (x_scale, y_scale)

    Returns:
        Scaled polygon points
    """
    x_scale, y_scale = scale

    return [
        (int(x * x_scale), int(y * y_scale))
        for x, y in polygon
    ]
