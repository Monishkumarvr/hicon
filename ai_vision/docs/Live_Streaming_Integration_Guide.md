

# HiCon Live Streaming Integration Guide

**Date:** 2026-02-17
**Goal:** Add MJPEG live streaming + improved annotations to HiCon pipeline

---

## Overview

This guide implements:
1. **Clean notebook-style overlays** with per-mould timing breakdown
2. **MJPEG HTTP streaming** for live browser monitoring (< 200ms latency)
3. **Larger, more visible probe dots** (12px circles instead of small rects)
4. **Color-coded status** (green for active, gray for completed)

**Estimated Time:** 2-3 hours for full implementation + testing

---

## Phase 1: Install Dependencies

### 1.1. Install Flask for MJPEG Server

```bash
cd /home/hicon/hicon/ai_vision
pip3 install Flask==3.0.0
```

**Verify:**
```bash
python3 -c "from flask import Flask; print('Flask OK')"
```

---

## Phase 2: Update Pouring Processor (Improved Overlays)

### 2.1. Add Per-Mould Timing Tracking

**File:** `processors/pouring_processor.py`

**Add to `__init__` (around line 80):**
```python
# Per-mould timing tracker (incremental, like notebook)
self.mould_completed_times = {}  # mould_id → total frames
self._pour_start_time = None     # Wall-clock time for live overlay
self.fps = fps or getattr(config, 'RTSP_FPS', 25.0)
```

---

### 2.2. Update `_start_pour` to Track Start Time

**Find `_start_pour` method (around line 520):**

**Add BEFORE the existing logger.info line:**
```python
def _start_pour(self, timestamp):
    """Start a new pour."""
    self.pour_active = True
    self.current_pour_start = self.frame_count
    self._pour_start_time = timestamp  # ← ADD THIS LINE
    logger.info(f"[pouring] Pour started at frame {self.frame_count}")
    # ... rest of existing code ...
```

---

### 2.3. Update `_end_pour` to Track Mould Times

**Find `_end_pour` method (around line 560):**

**REPLACE the section after duration check with:**
```python
def _end_pour(self):
    """End current pour."""
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
    self.mould_completed_times[self.mould_count] = pour_duration  # ← ADD THIS LINE

    logger.info(f"[pouring] Mould #{self.mould_count} completed: {pour_duration/self.fps:.1f}s")

    # Screenshot (existing code continues below)
    # ... rest of existing _end_pour code ...

    self.pour_active = False
    self.current_pour_start = None
    self._pour_start_time = None  # ← ADD THIS LINE
```

---

### 2.4. Replace `_add_inference_display_meta` with Clean Panel Version

**Find `_add_inference_display_meta` method (around line 1173):**

**REPLACE ENTIRE METHOD with:**
```python
def _add_inference_display_meta(self, batch_meta, frame_meta, mouths, trolleys,
                                target_trolley, timestamp, datetime_obj):
    """Attach clean notebook-style overlay to nvosd."""
    try:
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if not display_meta:
            return

        # Build text lines (clean vertical panel)
        lines = []
        lines.append(f"POURING | {datetime_obj.strftime('%H:%M:%S')}")
        lines.append("")  # blank line

        # Active trolley info
        if self.trolley_locked and self.locked_trolley_id is not None:
            lines.append(f"Trolley #{self.locked_trolley_id} [LOCKED]")
            lines.append(f"  Total Moulds: {self.mould_count}")

            # Per-mould times (completed moulds)
            for mid in sorted(self.mould_completed_times.keys()):
                frames = self.mould_completed_times[mid]
                time_s = frames / self.fps
                lines.append(f"  Mould #{mid}: {time_s:.1f}s \u2713")  # checkmark

            # Active pour (not yet in mould_completed_times)
            if self.pour_active and self._pour_start_time is not None:
                active_s = timestamp - self._pour_start_time
                next_mid = self.mould_count + 1
                lines.append(f"  Mould #{next_mid}: {active_s:.1f}s \u25CF")  # dot
        else:
            lines.append("No Active Trolley")

        lines.append("")  # blank line

        # Footer stats
        session_age = (timestamp - self.session_start_time) if self.session_start_time else 0.0
        cycle_age = (timestamp - self.cycle_start_time) if self.cycle_start_time else 0.0
        session_str = f"{int(session_age)}s"
        cycle_str = f"{int(cycle_age // 60)}m" if cycle_age >= 60 else f"{int(cycle_age)}s"
        lines.append(f"Session: {session_str} | Cycle: {cycle_str}")

        # Calculate scale for downscaled recording
        scale_up = 1.0
        try:
            target_w = int(getattr(self.config, "INFERENCE_VIDEO_WIDTH", 0) or 0)
            if self._frame_w and target_w and target_w < self._frame_w:
                scale_up = min(2.5, self._frame_w / float(target_w))
        except Exception:
            scale_up = 1.0

        line_height = max(18, int(round(18 * scale_up)))
        font_size = max(12, int(round(12 * scale_up)))

        # Render multi-line text
        num_lines = min(len(lines), 16)
        display_meta.num_labels = num_lines

        for i in range(num_lines):
            txt = display_meta.text_params[i]
            txt.display_text = lines[i]
            txt.x_offset = 10
            txt.y_offset = 15 + i * line_height
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = font_size

            # Color coding
            if "\u25CF" in lines[i]:  # Active mould (dot)
                txt.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            elif "\u2713" in lines[i]:  # Completed mould (checkmark)
                txt.font_params.font_color.set(0.8, 0.8, 0.8, 1.0)
            elif "[LOCKED]" in lines[i]:
                txt.font_params.font_color.set(0.0, 1.0, 1.0, 1.0)
            else:
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)

        # Draw large probe dot (circle)
        if self._last_probe_base is not None and display_meta.num_circles < 16:
            base_x, base_y = self._last_probe_base
            probe_on = (self._last_probe_brightness or 0.0) > self.brightness_start

            circle = display_meta.circle_params[0]
            circle.xc = int(base_x)
            circle.yc = int(base_y)
            circle.radius = max(10, int(round(10 * scale_up)))

            if probe_on:
                circle.circle_color.set(0.0, 1.0, 0.0, 1.0)
                circle.has_bg_color = 1
                circle.bg_color.set(0.0, 1.0, 0.0, 0.85)
            else:
                circle.circle_color.set(1.0, 0.0, 0.0, 1.0)
                circle.has_bg_color = 1
                circle.bg_color.set(1.0, 0.0, 0.0, 0.85)

            display_meta.num_circles = 1

            # Brightness label next to dot
            if display_meta.num_labels < 16:
                b_label = display_meta.text_params[display_meta.num_labels]
                b_val = f"B:{self._last_probe_brightness:.0f}" if self._last_probe_brightness is not None else "B:--"
                b_label.display_text = b_val
                b_label.x_offset = int(base_x + 20)
                b_label.y_offset = int(base_y - 5)
                b_label.font_params.font_name = "Serif"
                b_label.font_params.font_size = max(10, int(round(10 * scale_up)))
                b_label.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                b_label.set_bg_clr = 1
                b_label.text_bg_clr.set(0.0, 0.0, 0.0, 0.75)
                display_meta.num_labels += 1

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    except Exception as e:
        logger.error(f"Error in inference overlay: {e}", exc_info=True)
```

---

## Phase 3: Integrate MJPEG Server

### 3.1. Add MJPEG Server to Pipeline

**File:** `hicon_pipeline.py`

**Add imports at top:**
```python
from streaming.mjpeg_server import MJPEGServer
```

**Add after config loading (around line 50):**
```python
# Initialize MJPEG server for live streaming
mjpeg_server = None
if config.ENABLE_LIVE_STREAM:
    mjpeg_server = MJPEGServer(
        host=config.LIVE_STREAM_HOST,
        port=config.LIVE_STREAM_PORT,
        jpeg_quality=config.LIVE_STREAM_QUALITY,
        max_fps=config.LIVE_STREAM_FPS
    )
    mjpeg_server.register_stream(0)  # Process camera
    mjpeg_server.register_stream(1)  # Pyrometer camera
    mjpeg_server.start()
    logger.info(f"Live streaming enabled: http://{config.LIVE_STREAM_HOST}:{config.LIVE_STREAM_PORT}/")
```

---

### 3.2. Extract Annotated Frames from OSD Probe

**Find the OSD sink pad probe (where `pouring_processor.process_frame` is called):**

**Add AFTER `pouring_processor.process_frame()` call:**
```python
# Extract annotated frame for live streaming
if mjpeg_server is not None:
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_rgba = np.array(n_frame, copy=True, order='C')
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        mjpeg_server.update_frame(stream_id=0, frame_bgr=frame_bgr)
        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    except Exception as e:
        logger.error(f"Error extracting frame for streaming: {e}")
```

**Do the same for Stream 1 pyrometer probe** (if needed).

---

### 3.3. Add Config Variables

**File:** `config.py`

**Add after existing feature flags (around line 200):**
```python
# =============================================================================
# LIVE STREAMING CONFIGURATION
# =============================================================================

ENABLE_LIVE_STREAM = os.getenv('HICON_ENABLE_LIVE_STREAM', 'false').lower() == 'true'
LIVE_STREAM_HOST = os.getenv('HICON_LIVE_STREAM_HOST', '0.0.0.0')
LIVE_STREAM_PORT = int(os.getenv('HICON_LIVE_STREAM_PORT', '8080'))
LIVE_STREAM_QUALITY = int(os.getenv('HICON_LIVE_STREAM_QUALITY', '85'))  # JPEG quality 0-100
LIVE_STREAM_FPS = int(os.getenv('HICON_LIVE_STREAM_FPS', '15'))  # Max FPS for stream
```

---

### 3.4. Update .env for Testing

**File:** `.env`

**Add:**
```bash
# Live streaming
HICON_ENABLE_LIVE_STREAM=true
HICON_LIVE_STREAM_HOST=0.0.0.0
HICON_LIVE_STREAM_PORT=8080
HICON_LIVE_STREAM_QUALITY=85
HICON_LIVE_STREAM_FPS=15
```

---

## Phase 4: Testing

### 4.1. Syntax Check

```bash
cd /home/hicon/hicon/ai_vision
python3 -c "from processors.pouring_processor import PouringProcessor; print('Pouring processor: OK')"
python3 -c "from streaming.mjpeg_server import MJPEGServer; print('MJPEG server: OK')"
python3 -c "import hicon_pipeline; print('Pipeline imports: OK')"
```

---

### 4.2. Run Pipeline with Live Streaming

```bash
python3 hicon_pipeline.py
```

**Expected output:**
```
[INFO] MJPEG server started: http://0.0.0.0:8080/
[INFO]   Index page: http://0.0.0.0:8080/
[INFO]   Stream 0: http://0.0.0.0:8080/stream0
[INFO]   Stream 1: http://0.0.0.0:8080/stream1
[INFO] Pipeline PLAYING
```

---

### 4.3. Open Browser

**On same machine:**
```
http://localhost:8080/
```

**From another device on local network:**
```
http://<jetson-ip>:8080/
```

**Expected:**
- Index page with two streams (Process + Pyrometer)
- Live MJPEG video with clean panel overlays
- Per-mould timing visible in real time
- Larger green probe dot when pouring

---

### 4.4. Verify Overlays

**Check that you see:**
1. ✅ Top-left panel with vertical stack layout
2. ✅ "Trolley #X [LOCKED]" when session active
3. ✅ Per-mould times: "Mould #1: 12.3s ✓"
4. ✅ Active mould highlighted in green with dot: "Mould #3: 5.2s ●"
5. ✅ Large green circle (12px radius) at probe position when pouring
6. ✅ Brightness value "B:245" next to probe dot
7. ✅ No overlapping text or clutter

---

## Phase 5: Performance Tuning

### 5.1. Check CPU/GPU Load

```bash
tegrastats --interval 1000
```

**Baseline (no streaming):**
- CPU: ~60-80%
- GPU: ~50-70%
- RAM: ~4-5 GB

**With MJPEG streaming (15 FPS):**
- CPU: +5-10% (JPEG encoding)
- GPU: no change
- RAM: +100-200 MB

**If CPU too high:**
- Reduce `LIVE_STREAM_FPS` to 10 or 5
- Reduce `LIVE_STREAM_QUALITY` to 70

---

### 5.2. Check Network Bandwidth

**At 1920×1080, 85% JPEG quality, 15 FPS:**
- ~1.5-2.5 Mbps per stream
- Total: ~3-5 Mbps for both streams

**For remote monitoring over slow network:**
- Add downscaling before MJPEG encode:
  ```python
  frame_bgr_small = cv2.resize(frame_bgr, (1280, 720))
  mjpeg_server.update_frame(stream_id=0, frame_bgr=frame_bgr_small)
  ```

---

## Phase 6: Optional Enhancements

### 6.1. Add Authentication (Basic Auth)

**Modify `mjpeg_server.py`:**
```python
from flask import request, Response
from functools import wraps

def check_auth(username, password):
    return username == 'admin' and password == 'your-password'

def authenticate():
    return Response('Authentication required', 401,
                    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Add @requires_auth decorator to routes:
@app.route('/stream<int:stream_id>')
@requires_auth
def stream(stream_id):
    # ... existing code ...
```

---

### 6.2. Add Timestamp Overlay (Server-Side)

**In `_generate_mjpeg` method:**
```python
# Before encoding JPEG
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
```

---

### 6.3. Add FPS Counter

**In `_generate_mjpeg` method:**
```python
frame_times = deque(maxlen=30)
frame_times.append(time.time())
if len(frame_times) > 1:
    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
    cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

---

## Troubleshooting

### Issue: "Flask not found"
**Solution:**
```bash
pip3 install Flask==3.0.0
```

---

### Issue: Browser shows "No image" or black screen
**Check:**
1. Pipeline is running: `ps aux | grep hicon_pipeline`
2. MJPEG server started: Look for "MJPEG server started" in logs
3. Firewall allows port 8080: `sudo ufw allow 8080/tcp`
4. Frames are being extracted: Add debug log in update_frame()

---

### Issue: High CPU usage
**Solutions:**
1. Reduce FPS: `HICON_LIVE_STREAM_FPS=10`
2. Reduce quality: `HICON_LIVE_STREAM_QUALITY=70`
3. Downscale frames before encoding (see Phase 5.2)

---

### Issue: Overlays not appearing
**Check:**
1. `mould_completed_times` dict is being populated (add logger.debug)
2. `_add_inference_display_meta` is being called (add logger at method start)
3. `display_meta` is not None
4. `batch_meta` is being passed to processor

---

### Issue: Probe dot not visible
**Check:**
1. `_last_probe_base` is being set in `_update_pour`
2. `_last_probe_brightness` is being updated
3. `display_meta.num_circles` is incremented
4. Circle radius is large enough (should be ≥10px)

---

## Success Criteria

✅ Pipeline runs without errors
✅ MJPEG streams accessible in browser
✅ Per-mould timing visible in real time
✅ Probe dot large and clearly visible
✅ No overlapping text or clutter
✅ Active mould highlighted in green
✅ Completed moulds show checkmark
✅ CPU usage < 90% with streaming enabled
✅ Latency < 300ms from live event to browser display

---

## Rollback Plan

If issues occur, revert changes:

```bash
cd /home/hicon/hicon
git diff ai_vision/processors/pouring_processor.py > /tmp/pouring_changes.patch
git diff ai_vision/hicon_pipeline.py > /tmp/pipeline_changes.patch

# To rollback:
git checkout ai_vision/processors/pouring_processor.py
git checkout ai_vision/hicon_pipeline.py

# Disable streaming:
export HICON_ENABLE_LIVE_STREAM=false
python3 ai_vision/hicon_pipeline.py
```

---

**Estimated Total Time:** 2-3 hours
**Priority:** High (enables live verification of pouring logic)
**Risk:** Low (streaming is optional, can be disabled via env var)
