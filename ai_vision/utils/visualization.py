"""
Visualization utilities for HiCon - Zone and detection rendering.
Provides zone overlay, detection bounding boxes, and labels for debugging.
"""
import cv2
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def draw_pan_zones(
    frame: np.ndarray,
    zones: Dict,
    alpha: float = 0.2,
    draw_labels: bool = True
) -> np.ndarray:
    """
    Draw all pan zones with alpha-blended overlay and labels.

    Args:
        frame: Input frame (BGR format)
        zones: Pan zones dict from zone_loader (furnace_name → {points, threshold, color})
        alpha: Transparency (0-1, default 0.2)
        draw_labels: Whether to draw zone name labels (default True)

    Returns:
        Frame with zone overlays
    """
    if not zones:
        logger.warning("No zones provided for visualization")
        return frame

    overlay = frame.copy()

    for furnace_name, zone_config in zones.items():
        points = zone_config.get('points')
        if points is None or len(points) < 3:
            logger.warning(f"Invalid zone points for {furnace_name}")
            continue

        # Get color (default to magenta if missing)
        # zones.json stores RGB, OpenCV needs BGR
        color_rgb = zone_config.get('color', [255, 0, 255])
        color_bgr = tuple(reversed(color_rgb))

        # Convert points to numpy array
        pts = np.array(points, np.int32).reshape((-1, 1, 2))

        # Fill polygon
        cv2.fillPoly(overlay, [pts], color_bgr)

        # Draw boundary
        cv2.polylines(overlay, [pts], True, color_bgr, 3)

        # Draw zone label at centroid
        if draw_labels:
            pts_flat = pts.reshape((-1, 2))
            M = cv2.moments(pts_flat)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Get threshold for label
                threshold = zone_config.get('threshold', 0.5)

                # Label text
                label = f"{furnace_name}"
                sublabel = f"thresh: {threshold:.2f}"

                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Main label size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Label background rectangle
                bg_pt1 = (cx - text_width // 2 - 5, cy - text_height // 2 - 5)
                bg_pt2 = (cx + text_width // 2 + 5, cy + text_height // 2 + baseline + 5)
                cv2.rectangle(overlay, bg_pt1, bg_pt2, color_bgr, -1)

                # Draw label text
                text_pos = (cx - text_width // 2, cy + text_height // 2)
                cv2.putText(overlay, label, text_pos, font, font_scale, (255, 255, 255), thickness)

                # Draw sublabel (threshold)
                (sub_width, sub_height), sub_baseline = cv2.getTextSize(
                    sublabel, font, 0.5, 1
                )
                sub_y = cy + text_height // 2 + sub_height + 10
                sub_x = cx - sub_width // 2

                # Sublabel background
                sub_bg_pt1 = (sub_x - 3, sub_y - sub_height - 3)
                sub_bg_pt2 = (sub_x + sub_width + 3, sub_y + sub_baseline + 3)
                cv2.rectangle(overlay, sub_bg_pt1, sub_bg_pt2, (0, 0, 0), -1)

                # Draw sublabel text
                cv2.putText(overlay, sublabel, (sub_x, sub_y), font, 0.5, (255, 255, 255), 1)

    # Blend with alpha
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return result


def draw_pan_detections(
    frame: np.ndarray,
    pans: List[Dict],
    draw_center: bool = True,
    draw_confidence: bool = True
) -> np.ndarray:
    """
    Draw pan bounding boxes with track IDs and confidence.

    Args:
        frame: Input frame (BGR format)
        pans: List of pan detection dicts with {bbox, track_id, confidence}
        draw_center: Whether to draw center point marker (default True)
        draw_confidence: Whether to show confidence value (default True)

    Returns:
        Frame with detection overlays
    """
    if not pans:
        return frame

    result = frame.copy()

    for pan in pans:
        bbox = pan.get('bbox')
        track_id = pan.get('track_id')
        conf = pan.get('confidence', 0.0)

        if not bbox or len(bbox) < 4:
            logger.warning(f"Invalid bbox for pan track_id {track_id}")
            continue

        x1, y1, x2, y2 = bbox

        # Draw bounding box (green)
        color = (0, 255, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Build label
        if draw_confidence:
            label = f"Pan {track_id}: {conf:.2f}"
        else:
            label = f"Pan {track_id}"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Label background rectangle
        label_bg_pt1 = (x1, y1 - text_height - baseline - 5)
        label_bg_pt2 = (x1 + text_width + 5, y1)
        cv2.rectangle(result, label_bg_pt1, label_bg_pt2, (0, 0, 0), -1)

        # Draw label text
        label_pos = (x1 + 2, y1 - baseline - 2)
        cv2.putText(result, label, label_pos, font, font_scale, color, thickness)

        # Draw center point
        if draw_center:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(result, (cx, cy), 5, (0, 255, 255), -1)  # Cyan center
            cv2.circle(result, (cx, cy), 7, (0, 0, 0), 1)  # Black outline

    return result


def draw_deslagging_zones(
    frame: np.ndarray,
    zones: Dict,
    brightness_values: Dict[int, float],
    alpha: float = 0.3,
    triggered_zone_id: int = None,
    draw_labels: bool = True
) -> np.ndarray:
    """
    Draw deslagging zones with brightness values overlay.

    Args:
        frame: Input frame (BGR format)
        zones: Deslagging zones dict from zone_loader {zone_id: {points, threshold, name, ...}}
        brightness_values: Current brightness per zone {zone_id: brightness_value}
        alpha: Transparency (0-1, default 0.3)
        triggered_zone_id: Zone ID that triggered event (highlighted differently)
        draw_labels: Whether to draw zone labels (default True)

    Returns:
        Frame with zone overlays and brightness values
    """
    if not zones:
        logger.warning("No deslagging zones provided for visualization")
        return frame

    overlay = frame.copy()

    for zone_id, zone_config in zones.items():
        points = zone_config.get('points')
        if points is None or len(points) < 3:
            logger.warning(f"Invalid zone points for zone {zone_id}")
            continue

        # Choose color based on triggered state
        if zone_id == triggered_zone_id:
            # Triggered zone: Red (BGR)
            color_bgr = (0, 0, 255)
            zone_alpha = 0.5  # More opaque for triggered zone
        else:
            # Normal zone: Orange (BGR)
            color_bgr = (0, 165, 255)
            zone_alpha = alpha

        # Convert points to numpy array
        pts = np.array(points, np.int32).reshape((-1, 1, 2))

        # Fill polygon on overlay
        cv2.fillPoly(overlay, [pts], color_bgr)

        # Draw boundary
        cv2.polylines(overlay, [pts], True, color_bgr, 3)

        if draw_labels:
            # Calculate centroid for label placement
            pts_flat = np.array(points, np.int32)
            M = cv2.moments(pts_flat)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Get zone info
                zone_name = zone_config.get('name', f'Zone {zone_id}')
                threshold = zone_config.get('brightness_threshold', 160)
                current_brightness = brightness_values.get(zone_id, 0.0)

                # Label: Zone name
                label = f"{zone_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw background rectangle for readability
                bg_x1 = cx - text_width // 2 - 5
                bg_y1 = cy - text_height - 10
                bg_x2 = cx + text_width // 2 + 5
                bg_y2 = cy + 10
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

                # Draw zone name
                cv2.putText(
                    overlay, label,
                    (cx - text_width // 2, cy - 5),
                    font, font_scale, (255, 255, 255), thickness
                )

                # Sublabel: Brightness value + threshold
                sublabel = f"Brightness: {current_brightness:.1f} / {threshold}"
                (sub_width, sub_height), _ = cv2.getTextSize(
                    sublabel, font, 0.5, 1
                )

                # Draw sublabel background
                sub_bg_y1 = cy + 5
                sub_bg_y2 = cy + sub_height + 15
                cv2.rectangle(
                    overlay,
                    (cx - sub_width // 2 - 5, sub_bg_y1),
                    (cx + sub_width // 2 + 5, sub_bg_y2),
                    (0, 0, 0), -1
                )

                # Draw sublabel
                cv2.putText(
                    overlay, sublabel,
                    (cx - sub_width // 2, cy + sub_height + 10),
                    font, 0.5, (255, 255, 255), 1
                )

    # Blend overlay with original frame (use triggered zone's alpha if any)
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return result


def draw_ppe_zone(
    frame: np.ndarray,
    ppe_zone: np.ndarray,
    alpha: float = 0.15,
    draw_label: bool = True
) -> np.ndarray:
    """
    Draw PPE zone with alpha-blended overlay.

    Args:
        frame: Input frame (BGR format)
        ppe_zone: Numpy array of PPE zone polygon points (from zones.json)
        alpha: Transparency (0-1, default 0.15)
        draw_label: Whether to draw zone label (default True)

    Returns:
        Frame with PPE zone overlay
    """
    if ppe_zone is None or len(ppe_zone) < 3:
        logger.warning("Invalid PPE zone points")
        return frame

    overlay = frame.copy()

    # zones.json stores RGB: [255, 128, 0] (Orange)
    # Convert to BGR for OpenCV: (0, 128, 255)
    color_bgr = (0, 128, 255)

    # Fill polygon
    cv2.fillPoly(overlay, [ppe_zone], color_bgr)

    # Draw boundary
    cv2.polylines(overlay, [ppe_zone], True, color_bgr, 3)

    # Draw zone label at centroid
    if draw_label:
        M = cv2.moments(ppe_zone)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Label text
            label = "PPE ZONE"

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Label background rectangle
            bg_pt1 = (cx - text_width // 2 - 5, cy - text_height // 2 - 5)
            bg_pt2 = (cx + text_width // 2 + 5, cy + text_height // 2 + baseline + 5)
            cv2.rectangle(overlay, bg_pt1, bg_pt2, color_bgr, -1)

            # Draw label text
            text_pos = (cx - text_width // 2, cy + text_height // 2)
            cv2.putText(overlay, label, text_pos, font, font_scale, (255, 255, 255), thickness)

    # Blend with alpha
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return result
