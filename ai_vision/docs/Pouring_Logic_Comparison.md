# Pouring Logic Comparison: Notebook vs DeepStream

**Date:** 2026-02-17
**Purpose:** Compare standalone notebook inference with production DeepStream pipeline

---

## 1. Annotation Quality: Notebook vs DeepStream

### Notebook Annotations (Clean, User-Friendly)

**Top-Left Info Panel:**
```
┌─────────────────────────────────┐
│ Trolley #5                      │
│   Total Moulds: 3               │
│   Mould #1: 12.3s               │
│   Mould #2: 8.7s                │
│   Mould #3: 15.2s (pouring)  ◄─ GREEN for active
│                                 │
│ Trolley #7                      │
│   Total Moulds: 2               │
│   Mould #1: 10.5s               │
│   Mould #2: 6.8s                │
└─────────────────────────────────┘
```

**Bounding Boxes:**
- Trolley: Yellow `(0,255,255)`, label `T5 (3M)` (mould count in label)
- Trolley (pouring): Green `(0,255,0)`, thicker line
- Mouth: Magenta `(255,0,255)`, label `M123`

**Pour Indicator:**
- **Visible dot** 50px below mouth bbox bottom
- Green filled circle (10px radius) with white border
- Only appears when `brightness >= TH_ON (240)`
- **Position:** Horizontal center of mouth bbox, vertical = `y2 + 50px`

**Panel Features:**
- **Semi-transparent black background** (0.6 alpha)
- **Dynamic height** based on number of active trolleys + moulds
- **Per-mould timing** updated every frame
- **Active mould highlighted in green** with "(pouring)" suffix
- **No overlapping text** — clean vertical stack

---

### DeepStream Annotations (Current State — Cluttered)

**Top Status Line:**
```
POURING INFERENCE | 2026-02-17 14:22:15 | SESSION:ON POUR:ON MOULDS:3 CLUSTERS:2 B:245 TARGET_T:5 LOCK_T:5 CYCLE_AGE:45.2s ABSENCE:0.1s
```

**Problems:**
1. **Too much technical info** (TARGET_T, CYCLE_AGE, ABSENCE — debug metrics)
2. **Single long line** wraps or truncates at low resolutions
3. **No visual hierarchy** — all text same weight
4. **No per-mould timing breakdown** — only total count
5. **Probe points as small rectangles** — hard to see at 640×360 recording resolution
6. **"POURING ACTIVE" banner** overlaps with status line

**What's Missing:**
- No info panel showing per-mould pour times
- No clear "currently pouring" visual indicator
- No trolley-specific breakdown when multiple trolleys present
- Probe dots not visible enough (should be larger circles, not small rects)

---

## 2. Logic Comparison: Session/Pour/Mould Detection

### Core Algorithm (Both Identical)

| Subsystem | Notebook | DeepStream | Match? |
|-----------|----------|------------|--------|
| **Session Enter** | Mouth center in expanded trolley ≥1.0s | Same | ✅ |
| **Session Exit** | Mouth absent >0.8s + 1.5s | Same | ✅ |
| **Pour ON** | Brightness >240 for 0.25s | Same | ✅ |
| **Pour OFF** | Brightness <180 for 1.0s | Same | ✅ |
| **Min Pour Duration** | 2.0s before mould switch | Same | ✅ |
| **Trolley Locking** | Lock on first pour start | Same | ✅ |
| **EDGE_EXPAND** | 200px top expansion | Same | ✅ |
| **Probe Position** | 50px below mouth bbox bottom, 3 offsets | Same | ✅ |
| **HSV-V Sampling** | Use V channel from HSV | Same | ✅ |

### Key Differences

#### 1. **Mould Counting Strategy**

**Notebook:**
```python
# Real-time incremental counting
st.current_mould_id += 1  # Increment on each completed pour
st.mould_completed_times[st.current_mould_id] = frames
trolley_id_to_count[tid] = st.current_mould_id  # Update UI immediately
```

**DeepStream:**
```python
# Clustering-based final count
split_segs = split_segment_by_motion(segments, D_SPLIT, T_HOLD)
clusters = build_clusters(split_segs, MIN_POUR_FRAMES, R_CLUSTER, R_MERGE)
mould_count = len([c for c in clusters if c.total_frames() >= MIN_CLUSTER_POUR])
```

**Impact:**
- **Notebook:** Shows mould count incrementally during session (user sees count go 1→2→3 live)
- **DeepStream:** Shows final clustered count only after session ends (count jumps from 0 to 3)

**Recommendation:** **Adopt notebook's incremental approach for live UI**, keep clustering for final DB record

---

#### 2. **Per-Mould Timing Tracking**

**Notebook:**
```python
st.mould_completed_times = {
    1: 308,  # frames
    2: 217,
    3: 380
}
# Updated incrementally as each pour completes
```

**DeepStream:**
```python
# Only aggregate stats available:
st.completed = [seg1, seg2, seg3, ...]  # All segments lumped together
# No per-mould breakdown until clustering at session end
```

**Impact:**
- **Notebook:** Can display "Mould #2: 8.7s" in real time
- **DeepStream:** Only shows total mould count, no per-mould times

**Recommendation:** **Add `mould_completed_times` dict to DeepStream processor**, update on each pour completion

---

#### 3. **CSV Incremental Writing**

**Notebook:**
```python
csv_writer.writerow([tid, st.current_mould_id, f"{frames/fps:.2f}"])
csv_file.flush()  # Write immediately on each pour completion
```

**DeepStream:**
```python
# No CSV output — only SQLite DB writes at session end
```

**Impact:**
- **Notebook:** CSV file updates incrementally during session (useful for live monitoring)
- **DeepStream:** Data only written to DB after session finishes

**Recommendation:** **Optional** — add incremental CSV for debugging, but SQLite is sufficient for production

---

#### 4. **Trolley Reappearance Handling**

**Notebook:**
```python
# Reset count if trolley reappears after >5s absence
if st.last_disappeared_f >= 0 and (frame_idx - st.last_disappeared_f) > sec_to_frames(5.0, fps):
    trolley_id_to_count[tid] = 0
    st.final_clustered_count = -1
```

**DeepStream:**
```python
# No explicit "new trolley" detection — relies on tracker ID assignment
# If same tracker ID reappears, mould count continues incrementing
```

**Impact:**
- **Notebook:** Distinguishes "same trolley returning" from "new cycle for same trolley"
- **DeepStream:** May accumulate counts across disconnected sessions if tracker ID persists

**Recommendation:** **Add reappearance timeout** (5s) to reset mould count for "new cycles"

---

#### 5. **Mould Anchor Update Strategy**

**Notebook:**
```python
# Anchor set at pour start, updated ONLY on confirmed mould switch
st.mould_anchor_pt = mouth_pt_norm  # Set once
# ... later, on switch:
st.mould_anchor_pt = mouth_pt_norm  # Update to new mould position
```

**DeepStream:**
```python
# Similar approach (anchor-based displacement)
# Both use anchor set at pour start, updated on mould switch
```

**Match:** ✅ Both implementations use same anchor strategy

---

## 3. Annotation Design Recommendations

### Proposed Overlay Layout (Clean + Informative)

**Top-Left Info Panel (Notebook Style):**
```
┌────────────────────────────────────┐
│ POURING SYSTEM | 14:22:15          │  ← Timestamp, no debug info
│                                    │
│ Trolley #5 [LOCKED]                │  ← Active trolley indicator
│   Total Moulds: 3                  │
│   Mould #1: 12.3s ✓                │  ← Checkmark for completed
│   Mould #2:  8.7s ✓                │
│   Mould #3: 15.2s ●                │  ← Dot for actively pouring
│                                    │
│ Session: 45.2s | Cycle: 1h 12m     │  ← Collapsed stats
└────────────────────────────────────┘
```

**Visual Hierarchy:**
1. **Timestamp + system name** (top line, small font)
2. **Active trolley** (bold, with [LOCKED] indicator)
3. **Per-mould times** (main content, green for active)
4. **Collapsed stats** (bottom line, minimal)

**Bounding Boxes:**
- **Trolley (locked):** Green, thick (3px), label `T5 [3M]`
- **Trolley (unlocked):** Yellow, thin (1px), label `T7`
- **Mouth:** Cyan, medium (2px), label `M123`
- **Expanded region:** Dashed green outline (low alpha)

**Pour Indicator (Large Visible Dot):**
- **Position:** 50px below mouth bbox bottom, horizontally centered
- **Style:** Green filled circle (12px radius), white border (2px)
- **Visibility:** Only when `brightness >= 240`
- **Label:** `B:245` next to dot

---

### Debug Mode Toggle (Optional)

**Production Mode (Default):**
- Info panel with per-mould times
- Clean timestamp
- No technical metrics

**Debug Mode (via env var `HICON_DEBUG_OVERLAY=true`):**
- Add bottom status bar:
  ```
  DEBUG | TARGET_T:5 LOCK_T:5 B:245 ABSENCE:0.1s CYCLE:45.2s CLUSTERS:2
  ```

---

## 4. Implementation Changes Required

### 4.1. Add Per-Mould Timing to DeepStream Processor

**File:** `processors/pouring_processor.py`

**Add to `PouringProcessor.__init__`:**
```python
self.mould_completed_times = {}  # mould_id → total frames
```

**Modify `_end_pour` (on pour completion):**
```python
if pour_duration >= self.min_pour_duration:
    self.mould_count += 1
    self.mould_completed_times[self.mould_count] = pour_duration
    logger.info(f"[pouring] Mould #{self.mould_count} completed: {pour_duration/fps:.1f}s")
```

**Modify `_end_session`:**
```python
# Clustering still runs for final DB record validation
# But mould_completed_times dict already has incremental data
```

---

### 4.2. Redesign `_add_inference_display_meta` for Clean Panel

**Replace current implementation with:**
```python
def _add_inference_display_meta(self, batch_meta, frame_meta, mouths, trolleys, ...):
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    if not display_meta:
        return

    # Panel layout (vertical stack)
    lines = [
        f"POURING | {datetime_obj.strftime('%H:%M:%S')}",
        "",  # blank line
        f"Trolley #{self.locked_trolley_id} [LOCKED]" if self.trolley_locked else "No Active Trolley",
        f"  Total Moulds: {self.mould_count}",
    ]

    # Per-mould times (sorted by mould_id)
    for mid in sorted(self.mould_completed_times.keys()):
        frames = self.mould_completed_times[mid]
        is_active = (self.pour_active and mid == self.mould_count)
        suffix = " ●" if is_active else " ✓"
        lines.append(f"  Mould #{mid}: {frames/fps:.1f}s{suffix}")

    # Active pour (not yet in mould_completed_times)
    if self.pour_active and self.current_pour_start_time:
        active_s = timestamp - self.current_pour_start_time
        lines.append(f"  Mould #{self.mould_count + 1}: {active_s:.1f}s ●")

    lines.append("")  # blank line
    lines.append(f"Session: {session_age:.0f}s | Cycle: {cycle_age_formatted}")

    # Render multi-line text
    for i, line in enumerate(lines[:MAX_LABELS]):
        txt = display_meta.text_params[i]
        txt.display_text = line
        txt.x_offset = 10
        txt.y_offset = 20 + i * 22  # 22px line height
        txt.font_params.font_size = 14
        # ... color, bg, etc.

    display_meta.num_labels = len(lines)
```

---

### 4.3. Improve Probe Dot Visibility

**Replace small rectangles with larger circles:**
```python
# In _add_inference_display_meta, for probe points:
if self._last_probe_base is not None:
    base_x, base_y = self._last_probe_base
    # Draw ONE large circle at base position (not 3 small rects)
    circle = display_meta.circle_params[0]
    circle.xc = int(base_x)
    circle.yc = int(base_y)
    circle.radius = 12  # Larger, more visible
    circle.circle_color.set(0.0, 1.0, 0.0, 1.0) if probe_on else (1.0, 0.0, 0.0, 1.0)
    circle.has_bg_color = 1
    circle.bg_color.set(0.0, 1.0, 0.0, 0.8) if probe_on else (1.0, 0.0, 0.0, 0.8)
    display_meta.num_circles = 1
```

---

## 5. Live Streaming Architecture

### Option 1: MJPEG HTTP Stream (Simplest)

**Pros:**
- Trivial to implement (Flask + cv2.imencode)
- No browser plugins, works everywhere
- Low latency (<200ms)

**Cons:**
- High bandwidth (~2-5 Mbps for 1920×1080)
- No adaptive bitrate

**Implementation:**
```python
# Add to hicon_pipeline.py
from flask import Flask, Response
import threading

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

def mjpeg_generator():
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/stream0')
def stream0():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# In pad probe callback:
with frame_lock:
    latest_frame = annotated_frame.copy()

# Start Flask in background thread
threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080), daemon=True).start()
```

**HTML Viewer:**
```html
<img src="http://jetson-ip:8080/stream0" width="1920" height="1080">
```

---

### Option 2: WebRTC (Low Latency, Production Grade)

**Pros:**
- Ultra-low latency (<100ms)
- Adaptive bitrate
- H.264 hardware encoding on Jetson

**Cons:**
- Complex setup (signaling server, STUN/TURN)
- Requires aiortc or GStreamer webrtcbin

**Implementation (GStreamer WebRTC):**
```python
# Add WebRTC branch after OSD
webrtc_pipeline = (
    "... nvosd ! "
    "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
    "nvv4l2h264enc bitrate=2000000 ! "
    "h264parse ! rtph264pay config-interval=1 pt=96 ! "
    "webrtcbin name=webrtc"
)
```

**Requires:**
- Signaling server (WebSocket, e.g., Python `websockets` library)
- STUN server for NAT traversal
- Browser client using WebRTC API

---

### Option 3: HLS/DASH (Adaptive Streaming)

**Pros:**
- Adaptive bitrate (multi-quality)
- CDN-compatible
- Works on all browsers

**Cons:**
- High latency (2-10 seconds due to segment buffering)
- Complex setup (requires nginx-rtmp or similar)

**Not recommended for real-time monitoring**

---

## 6. Recommended Implementation Plan

### Phase 1: Improve Annotations (1-2 days)
1. Add `mould_completed_times` dict to `PouringProcessor`
2. Rewrite `_add_inference_display_meta` with clean panel layout
3. Replace probe rectangles with larger circles
4. Test on recorded video, verify no overlap

### Phase 2: MJPEG Streaming (1 day)
1. Add Flask MJPEG endpoint to `hicon_pipeline.py`
2. Extract annotated frame from OSD probe
3. Test Chrome/Firefox access from local network
4. Measure bandwidth + latency

### Phase 3: WebRTC (Optional, 3-5 days)
1. Set up GStreamer webrtcbin branch
2. Implement WebSocket signaling server
3. Create browser WebRTC client
4. Deploy + test on Jetson

---

## 7. Quick Wins for Immediate Improvement

**Without any streaming setup, improve recorded video quality:**

1. **Increase inference video resolution:**
   ```python
   # In config.py
   INFERENCE_VIDEO_WIDTH = 1280  # Up from 640
   INFERENCE_VIDEO_HEIGHT = 720  # Up from 360
   ```

2. **Redesign overlay for clarity:**
   - Remove debug metrics from default view
   - Add per-mould timing panel
   - Enlarge probe dot to 12px radius circle

3. **Add color-coded status:**
   - Green trolley bbox when pouring
   - Red trolley bbox when session ended
   - Cyan mouth bbox always

**Result:** Recorded inference video becomes **usable for verification** without needing live stream.

---

**Next Steps:**
1. Review this comparison document
2. Decide: MJPEG (fast) or WebRTC (production-grade) for live streaming?
3. Approve annotation redesign
4. Implement Phase 1 (annotations) first, then streaming
