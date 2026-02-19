# PRD: HiCon Pipeline GPU Load Visualizer
**Version:** 1.0
**Date:** 2026-02-19
**Target:** Interactive browser-based flowchart simulation of the HiCon DeepStream 7.1 pipeline, showing GPU/CPU load per stage based on real model and hardware properties.

---

## 1. Purpose & Goal

Build a **self-contained, single-page HTML/JS web app** (no backend needed) that visually simulates the HiCon AI Vision System pipeline as an **animated flowchart**. The visualization must communicate:

- What processing stages exist (decode, inference, tracking, brightness analysis, etc.)
- Whether each stage runs on **GPU** or **CPU**
- The **estimated load** (latency, memory, utilization) each stage places on the hardware
- The **data flow** (frame path) through both Stream 0 and Stream 1
- Which stages are **concurrent** vs. **sequential**
- How **cloud sync** and **DB writes** hang off the main pipeline

This is a **read-only diagnostic/planning tool** — it does not connect to the live Jetson. All load numbers are derived from the project's own documentation.

---

## 2. Reference Architecture (Source of Truth)

The two-stream pipeline from CLAUDE.md and the codebase exploration is the exact topology to render:

```
Stream 0 (Process Camera)
  rtspsrc_0 → rtph264depay → h264parse → nvv4l2decoder → nvvideoconvert
    → nvstreammux ──────────────────────┐
                                        ↓
                                   nvinfer (GIE-1, best_pouring_hicon_v1_930, 640×640 FP16)
                                        ↓
                                   nvtracker (NvDCF)
                                        ↓
                                   nvosd
                                        ↓ (post-OSD probe)
                              ┌─────── tee ──────────────────┐
                              ↓ (display)                     ↓ (optional recording)
                         fakesink_0                    nvjpegenc → matroskamux → filesink
                              │
                    [osd_sink_pad_probe]
                         ├── pouring_processor (CPU)
                         │     ├─ session_manager
                         │     ├─ pour_detector
                         │     └─ mould_counter
                         └── brightness_processor (CPU)
                               ├─ tapping_tracker
                               ├─ deslagging_tracker
                               └─ spectro_tracker

Stream 1 (Pyrometer Camera)
  rtspsrc_1 → rtph264depay → h264parse → nvv4l2decoder → nvvideoconvert
    → nvstreammux ──────────────────────┐
                                        ↓
                                   nvinfer (GIE-2, best_pyro_rod_v1, 1280×1280 FP16)
                                        ↓ (post-nvinfer probe)
                                   pyrometer_processor (CPU)
                                        ↓
                                   [nvvideoconvert → nvosd → fakesink_1]

Shared Side-channels
  db_manager (SQLite, CPU) ← events from pouring + brightness + pyrometer processors
  sync_manager (background thread, CPU) → AGNI API (POST /melting, POST /pouring)
```

---

## 3. Visual Design Requirements

### 3.1 Layout

- **Two swim-lanes** running left-to-right: Stream 0 (top lane) and Stream 1 (bottom lane)
- A **shared hardware bar** at the top: Jetson Orin Nano 8GB stats (total GPU %, total mem)
- A **side panel** on the right: aggregate resource gauges (GPU Util %, CPU Util %, RAM GB)
- A **legend** explaining color codes and icon meanings

### 3.2 Node Appearance

Each pipeline stage is a **node card** with:

| Field | Content |
|-------|---------|
| **Title** | GStreamer element or processor name (e.g., `nvv4l2decoder`, `nvinfer [GIE-1]`) |
| **Subtitle** | Short role description (e.g., "H.264 HW Decode", "Pouring YOLO FP16") |
| **Hardware badge** | `GPU` (blue) or `CPU` (green) pill |
| **Latency chip** | Estimated per-frame processing time (e.g., `~1 ms`, `~12 ms`) |
| **Memory bar** | Horizontal mini-bar showing this stage's VRAM/RAM footprint |
| **Load ring** | Circular progress ring showing % utilization (color: green < 50%, yellow 50-80%, red > 80%) |

### 3.3 Edges (Connections)

- Animated dashed lines showing data flowing left-to-right at ~30 FPS
- Line color: **blue** for GPU-to-GPU (NVMM buffer), **gray** for CPU-to-CPU, **orange** for GPU-to-CPU handoff (frame extraction via `get_nvds_buf_surface`)
- Label on edge: buffer format in transit (e.g., `NV12 (NVMM)`, `RGBA (NVMM)`, `NumPy RGBA CPU`)

### 3.4 Color Scheme

| Category | Color |
|----------|-------|
| GPU element | `#1e40af` (dark blue) background, white text |
| CPU element | `#065f46` (dark green) background, white text |
| NVDEC hardware | `#6b21a8` (purple) |
| State machine / probe logic | `#92400e` (amber) |
| Storage / side-channel | `#374151` (gray) |
| Critical path highlight | `#dc2626` (red) outline |
| Edges: GPU-GPU | `#3b82f6` (blue animated dash) |
| Edges: CPU-CPU | `#9ca3af` (gray) |
| Edges: GPU→CPU boundary | `#f59e0b` (orange) |

---

## 4. Pipeline Nodes — Complete Specification

### Stream 0 Nodes (in order)

| # | Node ID | Label | Type | Latency | Memory | Utilization | Notes |
|---|---------|-------|------|---------|--------|-------------|-------|
| 1 | `rtsp_src_0` | rtspsrc (Stream 0) | CPU | ~0 ms | 10 MB | 2% CPU | RTSP TCP, retry=65535, 60s timeout |
| 2 | `rtp_depay_0` | rtph264depay | CPU | <1 ms | 5 MB | 1% CPU | H.264 RTP depayload |
| 3 | `h264_parse_0` | h264parse | CPU | <1 ms | 5 MB | 1% CPU | Byte-stream align |
| 4 | `nvv4l2dec_0` | nvv4l2decoder | NVDEC | ~1 ms | 50 MB VRAM | 15% GPU | HW H.264 decode → NV12 NVMM |
| 5 | `nvvconv_0a` | nvvideoconvert (→RGBA) | GPU | ~1 ms | 25 MB VRAM | 5% GPU | NV12 → RGBA for streammux |
| 6 | `streammux_0` | nvstreammux (Stream 0) | GPU | ~1 ms | 30 MB VRAM | 3% GPU | batch=1, 1920×1080, 40ms push timeout |
| 7 | `nvinfer_1` | **nvinfer [GIE-1]** | GPU | ~12 ms | **800 MB VRAM** | **45% GPU** | best_pouring_hicon_v1_930, 640×640 FP16, 2 classes |
| 8 | `nvtracker` | nvtracker (NvDCF) | GPU | ~7 ms | 100 MB VRAM | 20% GPU | maxShadowAge=600f, 2 objects (mouth+trolley) |
| 9 | `nvosd_0` | nvosd (OSD render) | GPU | ~5 ms | 50 MB VRAM | 10% GPU | DS-native bbox/text/custom overlays |
| 10 | `tee_0` | tee (display/record split) | GPU | <1 ms | 5 MB VRAM | 1% GPU | Optional recording branch |
| 11 | `probe_s0` | **osd_sink_pad_probe** | CPU | ~3 ms | 200 MB RAM | 15% CPU | Pouring + brightness processing |
| 11a | `pouring_proc` | pouring_processor | CPU | ~2 ms | 80 MB RAM | 8% CPU | Session + pour + mould state machines |
| 11b | `brightness_proc` | brightness_processor | CPU | ~3 ms | 120 MB RAM | 7% CPU | Tapping/Deslagging/Spectro (NumPy ROI) |
| 12 | `fakesink_0` | fakesink (display) | CPU | <1 ms | 2 MB | 1% CPU | Drop frames (no display hardware) |

### Stream 0 — Optional Recording Branch (conditional)

| # | Node ID | Label | Type | Latency | Memory | Notes |
|---|---------|-------|------|---------|--------|-------|
| R1 | `nvjpegenc` | nvjpegenc (quality=85) | GPU | ~5 ms | 30 MB VRAM | GPU JPEG encode @ 10 FPS 640×360 |
| R2 | `matroskamux` | matroskamux (MKV) | CPU | ~2 ms | 10 MB RAM | Container mux for MKV streaming write |
| R3 | `filesink` | filesink (MKV output) | CPU | ~1 ms | 2 MB | Disk write to output/videos/inference/ |

### Stream 1 Nodes (in order)

| # | Node ID | Label | Type | Latency | Memory | Utilization | Notes |
|---|---------|-------|------|---------|--------|-------------|-------|
| 1 | `rtsp_src_1` | rtspsrc (Stream 1) | CPU | ~0 ms | 10 MB | 2% CPU | RTSP TCP, same resilience as Stream 0 |
| 2 | `rtp_depay_1` | rtph264depay | CPU | <1 ms | 5 MB | 1% CPU | |
| 3 | `h264_parse_1` | h264parse | CPU | <1 ms | 5 MB | 1% CPU | |
| 4 | `nvv4l2dec_1` | nvv4l2decoder | NVDEC | ~1 ms | 50 MB VRAM | 15% GPU | Parallel to Stream 0 NVDEC |
| 5 | `nvvconv_1a` | nvvideoconvert (→NV12) | GPU | ~1 ms | 25 MB VRAM | 5% GPU | |
| 6 | `streammux_1` | nvstreammux (Stream 1) | GPU | ~1 ms | 30 MB VRAM | 3% GPU | Separate mux for Stream 1 |
| 7 | `nvinfer_2` | **nvinfer [GIE-2]** | GPU | ~18 ms | **600 MB VRAM** | **35% GPU** | best_pyro_rod_v1, 1280×1280 FP16, 1 class (rod) |
| 8 | `probe_s1` | **post-nvinfer probe** | CPU | ~1 ms | 30 MB RAM | 5% CPU | pyrometer_processor zone check + temporal |
| 9 | `pyro_proc` | pyrometer_processor | CPU | ~1 ms | 30 MB RAM | 4% CPU | Zone polygon check, 10-frame temporal |
| 10 | `nvvconv_1b` | nvvideoconvert (→RGBA) | GPU | ~1 ms | 25 MB VRAM | 3% GPU | |
| 11 | `nvosd_1` | nvosd (Stream 1) | GPU | ~5 ms | 30 MB VRAM | 8% GPU | |
| 12 | `fakesink_1` | fakesink (Stream 1) | CPU | <1 ms | 2 MB | 1% CPU | |

### Shared Side-Channel Nodes

| # | Node ID | Label | Type | Notes |
|---|---------|-------|------|-------|
| S1 | `db_manager` | SQLite DB (hicon.db) | CPU | Event-driven writes; 7-day retention; tables: melting_events, heat_cycles, pouring_events |
| S2 | `sync_manager` | Cloud Sync (30s interval) | CPU | Background thread; HMAC-SHA256 POST to AGNI API; /melting + /pouring endpoints; 3 retries |
| S3 | `state_machines` | State Machine Layer | CPU | BrightnessTracker × 3, PouringTracker, HeatCycleManager, PyrometerProcessor |
| S4 | `screenshot_saver` | Event Screenshot (JPEG) | CPU | OpenCV JPEG on event transitions; output/screenshots/ |

---

## 5. Hardware Budget Panel

Display a persistent panel (top or sidebar) showing the aggregate hardware state:

### Jetson Orin Nano 8GB — System Resources

| Resource | Baseline (Idle) | Under Load (Both Streams Active) | Budget Limit |
|----------|----------------|----------------------------------|-------------|
| GPU Utilization | 5% | **50–60%** | 100% (thermal throttle at >80%) |
| CPU Utilization | 10% | **25–35%** | 100% (6 cores available) |
| VRAM Used (GPU) | 500 MB | **~2.0–2.5 GB** | 6 GB (shared with CPU RAM) |
| RAM Used (CPU) | 1.5 GB | **~4.2–4.5 GB** | 8 GB total (shared) |
| Power Draw | 7W | **12–15W** | 15W (Orin Nano limit) |
| Thermal | 40°C | **55–65°C** | 80°C (throttle point) |

### Critical Memory Budget Breakdown (Pie Chart or Stacked Bar)

Show as a **stacked horizontal bar** or **donut chart**:

```
OS + System:               1.5 GB (19%)
Pouring Model + Engine:    0.8 GB (10%)
Pyrometer Model + Engine:  0.6 GB  (8%)
NvDCF Tracker State:       0.1 GB  (1%)
Frame Buffers (NV12/RGBA): 0.4 GB  (5%)
CPU NumPy Buffers:         0.2 GB  (3%)
SQLite + Python Runtime:   0.3 GB  (4%)
Headroom (Free):           3.7 GB (46%) ← show in green
```

---

## 6. Simulation Modes

The app should have a **mode toggle** (tab or dropdown) to switch between:

### Mode A: Static Architecture View
- All nodes rendered in their pipeline positions
- Load numbers shown as fixed values from the table above
- No animation — good for screenshots / documentation

### Mode B: Animated Flow Simulation
- Animated "data packets" (small dots or dashes) flow along edges at speed proportional to frame rate
- Each node briefly "pulses" (brightness flash) when a packet passes through
- GPU nodes show active color when processing
- Timeline at the bottom shows the per-frame latency waterfall:
  ```
  0 ms ──── decode(1ms) ──── nvinfer(12ms) ──── track(7ms) ──── osd(5ms) ──── probe(3ms) → 28 ms total
  ```
- Allow user to click **"Simulate Tapping Event"**, **"Simulate Pouring Event"**, or **"Simulate Rod Insertion"** to see how event data flows from GPU probe → CPU state machine → DB → cloud sync

### Mode C: GPU Load Heatmap
- Render each node as a **heat-colored rectangle** (cool blue = low load, warm red = high load)
- Tooltip on hover shows: latency, memory, utilization %, data format in/out
- Highlight the **critical path** (longest latency chain) with a red outline
- Show "GPU budget consumed by model" — annotate nvinfer nodes prominently as the biggest consumers

---

## 7. Interactive Features

### 7.1 Node Hover Tooltip
On hover over any node, show a tooltip card:
```
┌─────────────────────────────────┐
│  nvinfer [GIE-1]                │
│  best_pouring_hicon_v1_930      │
├─────────────────────────────────┤
│  Input:  RGBA 640×640 FP16      │
│  Output: NvDsObjectMeta         │
│  Classes: ladle_mouth, trolley  │
│  Confidence threshold: 0.25     │
│  NMS IoU: 0.45                  │
│  Interval: every frame          │
├─────────────────────────────────┤
│  GPU: ~45% utilization          │
│  VRAM: 800 MB                   │
│  Latency: ~12 ms/frame          │
│  Batch: 1                       │
│  Precision: FP16 TensorRT       │
└─────────────────────────────────┘
```

### 7.2 Edge Hover Tooltip
On hover over any edge/connection:
```
nvv4l2decoder → nvvideoconvert
Buffer format: NV12 (NVMM)
Stays in GPU memory (zero-copy)
Throughput: 1920×1080 @ 30 FPS
```

```
nvosd → [osd_sink_pad_probe]
Boundary: GPU → CPU (frame extraction)
Method: pyds.get_nvds_buf_surface()
Format: RGBA uint8 → NumPy array
⚠ MANDATORY: unmap_nvds_buf_surface() after use
Cost: ~200 MB RAM copy per frame
```

### 7.3 Event Simulation (Click Triggers)
Three buttons to animate event flow:

**"Tapping Event":**
- Highlight: brightness_proc → tapping_tracker → ACTIVE state → state_machines → db_manager (write) → sync_manager (POST /melting)
- Shows the tapping quad ROI polygon as an inset diagram on the brightness_proc node

**"Pouring Session":**
- Highlight: nvinfer_1 → nvtracker → probe_s0 → pouring_proc → session_manager (ACTIVE) → pour_detector (START) → mould_counter → heat_cycle_manager → db_manager → sync_manager (POST /pouring)
- Animate multiple "mould count" events

**"Rod Insertion (Pyrometer)":**
- Highlight: nvinfer_2 → probe_s1 → pyro_proc → ACTIVE (10 consecutive frames) → db_manager → sync_manager (POST /melting)

### 7.4 Toggle: Show/Hide Optional Nodes
- Checkbox: **"Recording branch"** — show/hide nvjpegenc + matroskamux + filesink
- Checkbox: **"Pyrometer stream"** — collapse Stream 1 to a single summary block
- Checkbox: **"Side channels"** — show/hide DB + sync + state machine nodes

### 7.5 Stats Panel Live Slider (Demo Mode)
Sliders to simulate "what if" GPU load scenarios:
- **Model precision**: FP16 (current) vs FP32 — show VRAM and latency increase
- **Batch size**: 1 (current) → 2 → 4 — show throughput vs latency tradeoff
- **Recording enabled**: toggle shows ~10% GPU increase for nvjpegenc

---

## 8. Annotations and Callouts

Add these labeled callout boxes (sticky notes) near relevant nodes:

| Callout | Near Node | Text |
|---------|-----------|------|
| ⚠ Memory Leak Risk | nvv4l2dec_0 / probe boundary | `ALWAYS call unmap_nvds_buf_surface() after get_nvds_buf_surface() — missing this crashes pipeline in minutes` |
| ❄ No CuPy on Jetson | brightness_proc | `CuPy not available in DeepStream on Jetson — brightness analysis uses NumPy CPU (2-4 ms/frame, acceptable)` |
| 🔒 GPU → CPU Boundary | osd_sink_pad_probe | `Frame extracted here from GPU NVMM to CPU RAM via pyds. Orange edge = memory bandwidth cost` |
| 🎯 Critical Path | nvinfer_1 | `Largest GPU consumer: 45% for 640×640 FP16 inference. Dominates frame latency budget` |
| 🔥 Latency Budget | fakesink_0 | `Total budget: 33 ms @ 30 FPS. Consumed: ~28 ms. Margin: ~5 ms for variance` |
| 📦 Custom Parser | nvinfer_2 | `YOLO26 end-to-end (batch,300,6) format. libnvdsinfer_custom_impl_Yolo.so auto-detects output shape` |
| 🌡 Thermal Watch | hardware_panel | `Sustained GPU > 80% → thermal throttle on Orin Nano 15W. Monitor via tegrastats` |

---

## 9. Technical Stack

### Must-Have (for Claude to implement):
- **Pure HTML + CSS + JavaScript** — single `index.html` file, no build step, no server needed
- Open `index.html` directly in a browser on any machine (not just Jetson)
- Use **SVG** for pipeline graph rendering (not Canvas) — nodes as SVG `foreignObject` with HTML content inside, edges as SVG `path` with animated `stroke-dashoffset`
- Use **CSS animations** for flow animation (no heavy JS animation loop needed)
- Responsive layout: minimum 1400px wide, scrollable horizontally on smaller screens

### Recommended Libraries (load from CDN):
- `Alpine.js` (v3) for reactive state and mode switching — lightweight, no build step
- No Cytoscape.js, no D3.js, no React — keep it simple and auditable
- Chart.js (v4) for the memory donut chart and latency waterfall bar chart

### File Structure:
```
hicon-pipeline-visualizer/
└── index.html          ← Everything in one file (inline CSS + JS)
```

---

## 10. Exact Layout Specification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HiCon Pipeline Visualizer  [Static | Animated | Heatmap]   [Event: ▾]     │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Hardware: Jetson Orin Nano 8GB ──────────────────────────────────────┐  │
│ │ GPU: ████████░░░░ 55%  CPU: █████░░░░░ 30%  RAM: ██████░░░░ 4.3/8GB │  │
│ └────────────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────┬──────────────────────────────────────────┤
│  Stream 0 (Process Camera)       │  Stream 1 (Pyrometer Camera)            │
│                                  │                                          │
│  [rtsp_0]→[depay]→[h264]→        │  [rtsp_1]→[depay]→[h264]→              │
│  [NVDEC]→[conv]→[mux]→           │  [NVDEC]→[conv]→[mux]→                 │
│  [nvinfer GIE-1 ★]→[tracker]→    │  [nvinfer GIE-2 ★]→                    │
│  [osd]→[tee]→[sink]              │  [conv]→[osd]→[sink]                   │
│         ↓                        │       ↓                                  │
│  [Probe: Pouring + Brightness]   │  [Probe: Pyrometer]                     │
│   └─[state machines]             │   └─[state machines]                    │
│                                  │                                          │
├──────────────────────────────────┴──────────────────────────────────────────┤
│  Side Channels:  [SQLite DB] ←──── events ────→ [Sync Manager] → AGNI API  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Latency Waterfall (per frame):                                             │
│  decode[1ms] nvinfer1[12ms] track[7ms] osd[5ms] probe[3ms] = 28ms / 33ms  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Data Constants (Hardcoded in JS)

Provide Claude with these exact constants to embed as the data source:

```javascript
const PIPELINE_DATA = {
  hardware: {
    name: "Jetson Orin Nano 8GB",
    totalRAM_GB: 8,
    powerLimit_W: 15,
    thermalLimit_C: 80,
    hasNVDLA: false,
    hasHWEncoder: false,
    hasCuPy: false,
    cudaCores: 1024,
    nvdecInstances: 1
  },
  models: {
    pouring: {
      name: "best_pouring_hicon_v1_930",
      format: "TensorRT FP16",
      inputSize: "640×640",
      classes: ["ladle_mouth", "trolley"],
      batchSize: 1,
      vram_MB: 800,
      latency_ms: 12,
      gpuUtil_pct: 45,
      outputFormat: "YOLO 6-ch [x,y,w,h,conf,class]",
      parser: "NvDsInferParseYolo (6-ch auto-detect)"
    },
    pyrometer: {
      name: "best_pyro_rod_v1",
      format: "TensorRT FP16",
      inputSize: "1280×1280",
      classes: ["rod"],
      batchSize: 1,
      vram_MB: 600,
      latency_ms: 18,
      gpuUtil_pct: 35,
      outputFormat: "YOLO26 end-to-end (batch,300,6) [x1,y1,x2,y2,conf,class_id]",
      parser: "NvDsInferParseYolo (end-to-end auto-detect)"
    }
  },
  memoryBudget: [
    { label: "OS + System",           gb: 1.5, color: "#6b7280" },
    { label: "Pouring Model (TRT)",    gb: 0.8, color: "#1e40af" },
    { label: "Pyrometer Model (TRT)",  gb: 0.6, color: "#1e3a8a" },
    { label: "NvDCF Tracker",          gb: 0.1, color: "#3730a3" },
    { label: "Frame Buffers (NVMM)",   gb: 0.4, color: "#6d28d9" },
    { label: "NumPy CPU Buffers",      gb: 0.2, color: "#065f46" },
    { label: "SQLite + Python",        gb: 0.3, color: "#374151" },
    { label: "Free Headroom",          gb: 3.7, color: "#d1fae5" }
  ],
  stream0_nodes: [
    { id: "rtsp0",        label: "rtspsrc",           sub: "Stream 0 RTSP",        hw: "CPU",   lat_ms: 0,  mem_mb: 10,  util_pct: 2  },
    { id: "depay0",       label: "rtph264depay",      sub: "H.264 RTP Depayload",  hw: "CPU",   lat_ms: 0,  mem_mb: 5,   util_pct: 1  },
    { id: "parse0",       label: "h264parse",         sub: "Bytestream Align",     hw: "CPU",   lat_ms: 0,  mem_mb: 5,   util_pct: 1  },
    { id: "nvdec0",       label: "nvv4l2decoder",     sub: "HW H.264 Decode",      hw: "NVDEC", lat_ms: 1,  mem_mb: 50,  util_pct: 15 },
    { id: "conv0a",       label: "nvvideoconvert",    sub: "NV12 → RGBA",          hw: "GPU",   lat_ms: 1,  mem_mb: 25,  util_pct: 5  },
    { id: "mux0",         label: "nvstreammux",       sub: "Batch Mux (1920×1080)",hw: "GPU",   lat_ms: 1,  mem_mb: 30,  util_pct: 3  },
    { id: "nvinfer1",     label: "nvinfer [GIE-1]",   sub: "Pouring YOLO FP16",    hw: "GPU",   lat_ms: 12, mem_mb: 800, util_pct: 45, critical: true },
    { id: "tracker",      label: "nvtracker",         sub: "NvDCF Tracking",       hw: "GPU",   lat_ms: 7,  mem_mb: 100, util_pct: 20 },
    { id: "osd0",         label: "nvosd",             sub: "OSD Overlay Render",   hw: "GPU",   lat_ms: 5,  mem_mb: 50,  util_pct: 10 },
    { id: "tee0",         label: "tee",               sub: "Display/Record Split", hw: "GPU",   lat_ms: 0,  mem_mb: 5,   util_pct: 1  },
    { id: "probe0",       label: "osd_sink_pad_probe",sub: "Pouring + Brightness", hw: "CPU",   lat_ms: 3,  mem_mb: 200, util_pct: 15, boundary: true },
    { id: "pour_proc",    label: "pouring_processor", sub: "Session/Pour/Mould",   hw: "CPU",   lat_ms: 2,  mem_mb: 80,  util_pct: 8  },
    { id: "bright_proc",  label: "brightness_processor", sub: "Tap/Deslag/Spectro NumPy", hw: "CPU", lat_ms: 3, mem_mb: 120, util_pct: 7 },
    { id: "sink0",        label: "fakesink",          sub: "Display Discard",      hw: "CPU",   lat_ms: 0,  mem_mb: 2,   util_pct: 1  }
  ],
  stream0_recording: [
    { id: "jpegenc",  label: "nvjpegenc",    sub: "GPU JPEG Encode (85%)", hw: "GPU", lat_ms: 5, mem_mb: 30, util_pct: 8 },
    { id: "mkvmux",   label: "matroskamux",  sub: "MKV Container",         hw: "CPU", lat_ms: 2, mem_mb: 10, util_pct: 3 },
    { id: "filesink", label: "filesink",     sub: "Disk Write (MKV)",       hw: "CPU", lat_ms: 1, mem_mb: 2,  util_pct: 1 }
  ],
  stream1_nodes: [
    { id: "rtsp1",    label: "rtspsrc",        sub: "Stream 1 RTSP",         hw: "CPU",   lat_ms: 0,  mem_mb: 10,  util_pct: 2  },
    { id: "depay1",   label: "rtph264depay",   sub: "H.264 RTP Depayload",   hw: "CPU",   lat_ms: 0,  mem_mb: 5,   util_pct: 1  },
    { id: "parse1",   label: "h264parse",      sub: "Bytestream Align",      hw: "CPU",   lat_ms: 0,  mem_mb: 5,   util_pct: 1  },
    { id: "nvdec1",   label: "nvv4l2decoder",  sub: "HW H.264 Decode",       hw: "NVDEC", lat_ms: 1,  mem_mb: 50,  util_pct: 15 },
    { id: "conv1a",   label: "nvvideoconvert", sub: "NV12 → RGBA",           hw: "GPU",   lat_ms: 1,  mem_mb: 25,  util_pct: 5  },
    { id: "mux1",     label: "nvstreammux",    sub: "Batch Mux (1920×1080)", hw: "GPU",   lat_ms: 1,  mem_mb: 30,  util_pct: 3  },
    { id: "nvinfer2", label: "nvinfer [GIE-2]",sub: "Pyrometer YOLO26 FP16", hw: "GPU",   lat_ms: 18, mem_mb: 600, util_pct: 35, critical: true },
    { id: "probe1",   label: "post-nvinfer probe", sub: "Pyrometer Zone+Temporal", hw: "CPU", lat_ms: 1, mem_mb: 30, util_pct: 5, boundary: true },
    { id: "pyro_proc",label: "pyrometer_processor", sub: "Zone Polygon + 10f Temporal", hw: "CPU", lat_ms: 1, mem_mb: 30, util_pct: 4 },
    { id: "conv1b",   label: "nvvideoconvert", sub: "Format Convert",        hw: "GPU",   lat_ms: 1,  mem_mb: 25,  util_pct: 3  },
    { id: "osd1",     label: "nvosd",          sub: "OSD Stream 1",          hw: "GPU",   lat_ms: 5,  mem_mb: 30,  util_pct: 8  },
    { id: "sink1",    label: "fakesink",       sub: "Stream 1 Discard",      hw: "CPU",   lat_ms: 0,  mem_mb: 2,   util_pct: 1  }
  ],
  side_channels: [
    { id: "db",       label: "SQLite (hicon.db)", sub: "melting/heat/pouring tables", hw: "CPU", mem_mb: 50 },
    { id: "sync",     label: "sync_manager",      sub: "30s → AGNI API HMAC-SHA256",  hw: "CPU", mem_mb: 10 },
    { id: "states",   label: "State Machines",    sub: "BrightnessTracker×3 + PouringTracker + HeatCycleManager", hw: "CPU", mem_mb: 20 },
    { id: "shots",    label: "screenshot_saver",  sub: "OpenCV JPEG on event",        hw: "CPU", mem_mb: 15 }
  ]
};
```

---

## 12. Latency Waterfall Chart Specification

A horizontal bar chart (Gantt-style) at the bottom of the page showing time breakdown for a single frame through the **critical path** (Stream 0 pouring inference path, worst case):

```
Frame time budget: 33.3 ms (30 FPS)

 0          5          10         15         20         25         30 ms
 │          │          │          │          │          │          │
 ██ NVDEC(1)│████████████ nvinfer(12)│███████ track(7)│█████ osd(5)│███ probe(3)
            ↑GPU start                                              ↑GPU→CPU

 Total critical path: 28 ms  │  Margin: 5 ms  │  GPU: 25 ms  │  CPU: 3 ms
```

The chart must:
- Show each stage as a colored segment (GPU stages blue, CPU green)
- Show the 33.3 ms budget line as a dashed red vertical line
- Show actual consumed time (28 ms) vs budget
- On hover show stage name + time

---

## 13. Acceptance Criteria

The delivered `index.html` is considered complete when:

1. **All pipeline nodes** from Section 4 are rendered with correct labels, HW badge, latency chip, and memory bar
2. **GPU/CPU color coding** is applied consistently across all nodes
3. **Animated edges** show correct buffer format labels and GPU↔CPU boundary highlights
4. **Hardware budget panel** shows RAM donut chart + GPU/CPU utilization bars
5. **Three view modes** (Static / Animated / Heatmap) are implemented and switchable
6. **Three event simulations** (Tapping / Pouring / Rod Insertion) animate the correct node paths
7. **Node hover tooltips** show the spec from Section 7.1 format
8. **Latency waterfall chart** from Section 12 is present and readable
9. **Callout annotations** from Section 8 are rendered near their target nodes
10. **Recording branch toggle** shows/hides the optional recording sub-pipeline
11. The page opens in Chrome/Firefox without a server (`file://` protocol works)
12. The page is usable at 1440px wide without horizontal scroll on the main content area

---

## 14. Out of Scope

- No live data connection to the Jetson
- No real-time telemetry (tegrastats integration)
- No editing of pipeline topology
- No actual video frames or detection overlays
- No authentication or multi-user support
- No PPE detection (not part of HiCon)

---

## 15. Suggested Prompt for Claude

Use this prompt to initiate the build with Claude:

> "Build the HiCon Pipeline GPU Load Visualizer as described in HiCon_GPU_Load_Visualizer_PRD.md. Create a single self-contained `index.html` file. Use Alpine.js v3 (CDN) for reactivity and Chart.js v4 (CDN) for charts. Render the pipeline as SVG with foreignObject nodes. Implement all three modes (Static, Animated, Heatmap), node hover tooltips, event simulation buttons, and the latency waterfall chart. Use the PIPELINE_DATA constants from Section 11 as the data source. Do not use React, Vue, or any build tools — the file must open directly from disk in a browser."

---

*This PRD was auto-generated from the HiCon codebase analysis on 2026-02-19.*
