"""
Improved Pouring Processor Overlay — Clean Panel Design

This is a REFERENCE implementation showing how to redesign
_add_inference_display_meta() for cleaner, notebook-style annotations.

Key improvements:
1. Per-mould timing breakdown (like notebook)
2. Clean vertical panel layout
3. Larger, more visible probe dots
4. Color-coded active vs completed moulds
5. Optional debug mode toggle

To integrate into pouring_processor.py:
- Add self.mould_completed_times = {} to __init__
- Update self.mould_completed_times on pour completion
- Replace existing _add_inference_display_meta with this version
"""

import pyds
from datetime import datetime


def _add_inference_display_meta_v2(self, batch_meta, frame_meta, mouths, trolleys,
                                    target_trolley, timestamp, datetime_obj):
    """
    Attach clean, notebook-style overlay to nvosd.

    Layout:
    ┌────────────────────────────────┐
    │ POURING | 14:22:15              │
    │                                │
    │ Trolley #5 [LOCKED]            │
    │   Total Moulds: 3              │
    │   Mould #1: 12.3s ✓            │
    │   Mould #2:  8.7s ✓            │
    │   Mould #3: 15.2s ●            │
    │                                │
    │ Session: 45s | Cycle: 1h 12m   │
    └────────────────────────────────┘
    """
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            return

        # ----------------------------
        # 1. Build text lines
        # ----------------------------
        lines = []

        # Header
        lines.append(f"POURING | {datetime_obj.strftime('%H:%M:%S')}")
        lines.append("")  # blank line

        # Active trolley
        if self.trolley_locked and self.locked_trolley_id is not None:
            lines.append(f"Trolley #{self.locked_trolley_id} [LOCKED]")
            lines.append(f"  Total Moulds: {self.mould_count}")

            # Per-mould times (from mould_completed_times dict)
            for mid in sorted(self.mould_completed_times.keys()):
                frames = self.mould_completed_times[mid]
                time_s = frames / self.fps
                suffix = " ✓"  # checkmark for completed
                lines.append(f"  Mould #{mid}: {time_s:.1f}s{suffix}")

            # Active pour (not yet in mould_completed_times)
            if self.pour_active and self._pour_start_time is not None:
                active_s = timestamp - self._pour_start_time
                next_mid = self.mould_count + 1
                lines.append(f"  Mould #{next_mid}: {active_s:.1f}s ●")  # dot for active

        else:
            lines.append("No Active Trolley")

        lines.append("")  # blank line

        # Footer stats (collapsed)
        session_age = (timestamp - self.session_start_time) if self.session_start_time else 0.0
        cycle_age = (timestamp - self.cycle_start_time) if self.cycle_start_time else 0.0

        session_str = f"{int(session_age)}s"
        if cycle_age >= 3600:
            cycle_str = f"{int(cycle_age // 3600)}h {int((cycle_age % 3600) // 60)}m"
        elif cycle_age >= 60:
            cycle_str = f"{int(cycle_age // 60)}m"
        else:
            cycle_str = f"{int(cycle_age)}s"

        lines.append(f"Session: {session_str} | Cycle: {cycle_str}")

        # Optional debug line (if env var HICON_DEBUG_OVERLAY=true)
        if getattr(self.config, 'DEBUG_OVERLAY', False):
            absence = (timestamp - self.mouth_last_seen_in_trolley) if self.mouth_last_seen_in_trolley else 0.0
            brightness = self._last_probe_brightness if self._last_probe_brightness is not None else 0
            lines.append(f"DEBUG | B:{brightness:.0f} ABSENCE:{absence:.1f}s CLUSTERS:{self.clustered_mould_count}")

        # ----------------------------
        # 2. Render multi-line text
        # ----------------------------
        # Calculate scale factor for downscaled recording
        scale_up = 1.0
        try:
            target_w = int(getattr(self.config, "INFERENCE_VIDEO_WIDTH", 0) or 0)
            if self._frame_w and target_w and target_w < self._frame_w:
                scale_up = min(2.5, self._frame_w / float(target_w))
        except Exception:
            scale_up = 1.0

        line_height = max(18, int(round(18 * scale_up)))
        font_size = max(12, int(round(12 * scale_up)))

        num_lines = min(len(lines), 16)  # Max 16 labels per display_meta
        display_meta.num_labels = num_lines

        for i in range(num_lines):
            txt = display_meta.text_params[i]
            txt.display_text = lines[i]
            txt.x_offset = 10
            txt.y_offset = 15 + i * line_height

            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = font_size

            # Color coding
            if "●" in lines[i]:  # Active mould
                txt.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)  # Green
            elif "✓" in lines[i]:  # Completed mould
                txt.font_params.font_color.set(0.8, 0.8, 0.8, 1.0)  # Light gray
            elif "[LOCKED]" in lines[i]:  # Locked trolley
                txt.font_params.font_color.set(0.0, 1.0, 1.0, 1.0)  # Cyan
            else:
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)  # White

            # Semi-transparent black background for readability
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)

        # ----------------------------
        # 3. Draw probe dot (large circle)
        # ----------------------------
        if self._last_probe_base is not None and display_meta.num_circles < 16:
            base_x, base_y = self._last_probe_base
            probe_on = (self._last_probe_brightness or 0.0) > self.brightness_start

            circle = display_meta.circle_params[0]
            circle.xc = int(base_x)
            circle.yc = int(base_y)
            circle.radius = max(10, int(round(10 * scale_up)))  # Larger, more visible

            if probe_on:
                circle.circle_color.set(0.0, 1.0, 0.0, 1.0)  # Green border
                circle.has_bg_color = 1
                circle.bg_color.set(0.0, 1.0, 0.0, 0.85)  # Green fill
            else:
                circle.circle_color.set(1.0, 0.0, 0.0, 1.0)  # Red border
                circle.has_bg_color = 1
                circle.bg_color.set(1.0, 0.0, 0.0, 0.85)  # Red fill

            display_meta.num_circles = 1

            # Brightness value label next to dot
            if display_meta.num_labels < 16:
                brightness_label = display_meta.text_params[display_meta.num_labels]
                brightness_label.display_text = f"B:{self._last_probe_brightness:.0f}" if self._last_probe_brightness is not None else "B:--"
                brightness_label.x_offset = int(base_x + 20)
                brightness_label.y_offset = int(base_y - 5)
                brightness_label.font_params.font_name = "Serif"
                brightness_label.font_params.font_size = max(10, int(round(10 * scale_up)))
                brightness_label.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                brightness_label.set_bg_clr = 1
                brightness_label.text_bg_clr.set(0.0, 0.0, 0.0, 0.75)
                display_meta.num_labels += 1

        # ----------------------------
        # 4. Draw expanded trolley bbox (optional dashed outline)
        # ----------------------------
        if self.trolley_locked and self.locked_trolley_bbox and display_meta.num_rects < 12:
            x1, y1, x2, y2 = self.locked_trolley_bbox
            ey1 = max(0, y1 - self.edge_expand)

            rect = display_meta.rect_params[0]
            rect.left = int(max(0, x1))
            rect.top = int(max(0, ey1))
            rect.width = int(max(1, x2 - x1))
            rect.height = int(max(1, y2 - ey1))
            rect.border_width = max(1, int(round(1 * scale_up)))
            rect.border_color.set(0.0, 1.0, 0.0, 0.5)  # Semi-transparent green
            rect.has_bg_color = 0  # No fill, just outline
            display_meta.num_rects = 1

        # Attach to batch
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in inference overlay: {e}", exc_info=True)


# ----------------------------
# Additional changes needed in __init__
# ----------------------------
def __init__(self, ...):
    # ... existing init code ...

    # NEW: Add per-mould timing tracker
    self.mould_completed_times = {}  # mould_id → total frames accumulated
    self._pour_start_time = None  # Timestamp when current pour started

    # NEW: FPS for time conversion (from config or default)
    self.fps = getattr(config, 'RTSP_FPS', 25.0)


# ----------------------------
# Update _end_pour to track mould times
# ----------------------------
def _end_pour(self):
    """Called when pour ends (brightness drops below threshold for K_OFF frames)."""
    if not self.pour_active or self.current_pour_start is None:
        return

    pour_duration = self.frame_count - self.current_pour_start

    # Check minimum duration
    if pour_duration < self.min_pour_duration:
        logger.info(f"[pouring] Pour too short ({pour_duration} frames), discarding")
        self.pour_active = False
        self.current_pour_start = None
        self._pour_start_time = None
        return

    # Valid pour — increment mould count and track time
    self.mould_count += 1
    self.mould_completed_times[self.mould_count] = pour_duration

    logger.info(f"[pouring] Mould #{self.mould_count} completed: {pour_duration/self.fps:.1f}s")

    # ... rest of existing _end_pour logic ...

    self.pour_active = False
    self.current_pour_start = None
    self._pour_start_time = None


# ----------------------------
# Update _start_pour to track start time
# ----------------------------
def _start_pour(self, timestamp):
    """Called when pour starts (brightness exceeds threshold for K_ON frames)."""
    self.pour_active = True
    self.current_pour_start = self.frame_count
    self._pour_start_time = timestamp  # Store wall-clock time for live overlay
    logger.info(f"[pouring] Pour started at frame {self.frame_count}")

    # ... rest of existing _start_pour logic ...


# ----------------------------
# USAGE NOTES
# ----------------------------
"""
1. Copy _add_inference_display_meta_v2 into processors/pouring_processor.py
   and rename to _add_inference_display_meta (replace existing)

2. Add to __init__:
   - self.mould_completed_times = {}
   - self._pour_start_time = None
   - self.fps = getattr(config, 'RTSP_FPS', 25.0)

3. Update _end_pour to populate mould_completed_times dict

4. Update _start_pour to set _pour_start_time = timestamp

5. Add to config.py:
   DEBUG_OVERLAY = os.getenv('HICON_DEBUG_OVERLAY', 'false').lower() == 'true'

6. Test on recorded video, verify:
   - Per-mould times appear in real time
   - Active mould shows green dot
   - Completed moulds show checkmark
   - Probe dot is large and visible
   - No overlapping text
"""
