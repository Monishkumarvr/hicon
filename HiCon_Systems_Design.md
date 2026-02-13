# HiCon — Systems Design Document

## 1. Executive Summary

HiCon is a 2-camera edge AI vision system deployed on NVIDIA Jetson Orin Nano 8GB for induction furnace process monitoring at steel/foundry plants. Compared with the earlier 3-camera PPE-focused baseline, HiCon focuses exclusively on **furnace operations** — no PPE detection is performed. The system monitors tapping, pouring, spectrometry, and deslagging through one camera, and pyrometer rod insertion through a second camera, synchronizing all events to the AGNI cloud backend.

**Key Differences from Legacy Baseline:**

| Aspect | Legacy Baseline | HiCon |
|--------|----------------|-------|
| Cameras | 3 (2× PPE + 1× Pouring) | 2 (1× Process + 1× Pyrometer) |
| PPE Detection | Yes (Person + Helmet) | **No** |
| Pouring Detection | Yes (Mould-Ladle OBB) | Yes (YOLO + ByteTrack + brightness) |
| Tapping Detection | No | **Yes** (CuPy brightness, quad ROI) |
| Deslagging Detection | No | **Yes** (CuPy brightness, polygon ROI) |
| Spectrometry Capture | No | **Yes** (event capture — design only) |
| Pyrometer Detection | No | **Yes** (YOLO26 rod detection, TensorRT) |
| Decode Pipeline | DeepStream GStreamer | **FFmpeg NVDEC** (standalone per-detector) |
| ML Inference | DeepStream nvinfer | **Direct TensorRT** (end-to-end YOLO) |
| Tracking | NvDCF tracker | **ByteTrack** (pouring only) |
| GPU Processing | NvBufSurface RGBA | **CuPy** GPU arrays (brightness detectors) |
| Cloud APIs | `/safety`, `/pouring` | `/melting`, `/pouring` (extended) |

**Key Metrics:**
- 2 concurrent RTSP streams at 1920×1080 / 30 FPS
- Pyrometer detector: 250+ FPS inference throughput
- Pouring system: 20–30 FPS on RTX 3060+ (~2–3 GB GPU memory)
- Brightness detectors: real-time at native frame rate (minimal GPU overhead)
- 7-day local data retention with automatic cloud sync every 30 seconds
- Zero-downtime RTSP reconnection with 10-minute stall auto-recovery

---

## 2. System Context

```
┌──────────────────────────────────────────────────────────────────────┐
│                          PLANT FLOOR                                 │
│                                                                      │
│   ┌───────────────────────────┐   ┌───────────────────────────┐      │
│   │   Camera 1 (Process)      │   │   Camera 2 (Pyrometer)    │      │
│   │   Tapping / Pouring /     │   │   Rod insertion monitoring │      │
│   │   Spectro / Deslagging    │   │   (YOLO26 object detect)  │      │
│   └─────────────┬─────────────┘   └─────────────┬─────────────┘      │
│                 │RTSP                            │RTSP                │
│                 └──────────┬─────────────────────┘                    │
│                            │                                          │
│                 ┌──────────▼──────────────┐                           │
│                 │  JETSON ORIN NANO 8GB    │                           │
│                 │  ┌───────────────────┐  │                           │
│                 │  │  HiCon Detectors  │  │                           │
│                 │  │  (Standalone Py)  │  │     ┌──────────────────┐  │
│                 │  └────────┬──────────┘  │     │                  │  │
│                 │           │             │     │  Cloud Backend   │  │
│                 │  ┌────────▼──────────┐  │     │  (AGNI API)      │  │
│                 │  │  SQLite (7-day)   │──┼────►│                  │  │
│                 │  └───────────────────┘  │HTTPS│  POST /melting   │  │
│                 │                         │HMAC │  POST /pouring   │  │
│                 └─────────────────────────┘     └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

**External Interfaces:**
- **Inbound:** 2 × RTSP H.264 streams (Baseline profile preferred for HW decode)
- **Outbound:** HTTPS POST to AGNI cloud API with HMAC-SHA256 authentication
- **Local Storage:** SQLite database + JPEG screenshots + MP4 video recordings + CSV/JSON output files

---

## 3. Hardware Platform

| Component | Specification |
|-----------|--------------|
| SoC | NVIDIA Jetson Orin Nano 8GB |
| GPU | Ampere (1024 CUDA cores, 32 Tensor cores, compute 8.7) |
| CPU | 6-core Arm Cortex-A78AE v8.2, 1.5 GHz |
| Memory | 8 GB 128-bit LPDDR5 (68 GB/s bandwidth) |
| Storage | NVMe SSD (M.2 Key M) + microSD slot |
| Power | 7W–15W (configurable) |
| AI Performance | Up to 40 TOPS |
| Video Decode | HW: 1×4K60, 2×4K30, 5×1080p60 (H.265) |
| Video Encode | **Software-only** (1080p30 via CPU — no HW encoder) |
| SDK | JetPack 6.2.1 (L4T R36.4.7) |
| AI Stack | TensorRT, CUDA 12.6, CuPy, FFmpeg (NVDEC) |
| DLA | **None** (no NVDLA on Orin Nano) |

**Resource Budget (HiCon):**
- GPU: ~50% utilization (2 streams × decode + inference, no PPE models)
- RAM: ~4 GB (models + CuPy buffers + ByteTrack state + OS)
- Pouring system alone: ~2–3 GB GPU memory (YOLO + ByteTrack)

**Hardware Constraints vs Orin NX:**
- **No hardware video encoder:** MP4 recording uses software encoding (CPU-bound at 1080p30), which consumes ~1–2 CPU cores. Recording from both cameras simultaneously may require reduced resolution or frame rate.
- **No NVDLA:** All inference runs on GPU CUDA cores only — no offloading to DLA accelerators.
- **Lower power ceiling:** 15W max vs 25W on Orin NX — sustained GPU-heavy workloads may throttle at thermal limits.
- **DeepStream CuPy limitation:** `pyds.get_nvds_buf_surface_gpu()` is **x86/dGPU only** — NOT available on Jetson. On Jetson, DeepStream exposes frames via `get_nvds_buf_surface()` → CPU NumPy (requires `unmap_nvds_buf_surface()` after). CuPy GPU processing works only in standalone mode (FFmpeg NVDEC → CuPy), NOT through DeepStream's Python buffer API. If migrating to a DeepStream pipeline, brightness detectors must either (a) use NumPy on CPU, or (b) implement a custom C/CUDA GStreamer plugin using EGLImage for zero-copy GPU access.

---

## 4. Camera Assignments

| Camera | Stream ID | Purpose | Detection Method | Frame Rate |
|--------|-----------|---------|-----------------|------------|
| Process Camera | Stream 0 | Tapping, Pouring, Spectrometry, Deslagging | YOLO TensorRT (pouring) + CuPy brightness (tapping, deslagging) | 30 FPS |
| Pyrometer Camera | Stream 1 | Pyrometer rod insertion detection | YOLO26 TensorRT end-to-end, 1280×1280, FP16 | 30 FPS |

**Camera 1 (Process Camera)** handles four distinct detection tasks from a single RTSP stream. Pouring uses ML-based object detection with tracking; tapping and deslagging use GPU-accelerated brightness analysis with CuPy. Spectrometry is event-capture only (design phase).

**Camera 2 (Pyrometer Camera)** runs a dedicated YOLO26 end-to-end model to detect pyrometer rod insertion into the furnace. No OCR or temperature reading — the detector identifies when the rod is present in a defined zone.

---

## 5. Detector Architecture

### 5.1 Architecture Overview — Standalone Detectors (NOT DeepStream)

Unlike the previous baseline which uses a DeepStream GStreamer pipeline, HiCon implements **standalone Python detectors** that each manage their own decode → process → output pipeline. This provides per-detector isolation, simpler debugging, and independent restart capability.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     STREAM 0 (Process Camera)                        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  Pouring System (pouring_system.py)                        │      │
│  │  RTSP → FFmpeg NVDEC → TensorRT YOLO → ByteTrack          │      │
│  │  → Session Mgmt → Pour Detection → Mould Counting          │      │
│  │  → CSV (incremental) + JSON (batch)                        │      │
│  └────────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  Tapping Detector (molten_flow_detector.py)                │      │
│  │  RTSP → FFmpeg NVDEC → CuPy GPU brightness analysis       │      │
│  │  → Quad ROI → Y>180 threshold → dual hysteresis            │      │
│  │  → IDLE↔ACTIVE state machine                               │      │
│  └────────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  Deslagging Detector (deslagging_detector.py)              │      │
│  │  RTSP → FFmpeg NVDEC → CuPy GPU brightness analysis       │      │
│  │  → Polygon ROI (6-pt) → Y>250 threshold → 0.01 ratio      │      │
│  │  → IDLE↔ACTIVE state machine                               │      │
│  └────────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  Spectro Processor (spectro_processor.py) [DESIGN ONLY]   │      │
│  │  Event capture — triggers on external signal               │      │
│  └────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     STREAM 1 (Pyrometer Camera)                      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  Pyrometer Rod Detector (pyrometer_rod_detector.py)        │      │
│  │  3-stage threaded pipeline:                                │      │
│  │    VideoDecoder → InferenceEngine → PostProcessor          │      │
│  │  RTSP → FFmpeg NVDEC → CuPy preprocess → TensorRT YOLO26  │      │
│  │  → end-to-end (1280×1280, FP16, batch=2) → zone check     │      │
│  │  → 10-frame temporal → event start/end                     │      │
│  └────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Decode Pipeline — FFmpeg NVDEC (All Detectors)

All detectors use the same FFmpeg-based hardware decode approach instead of DeepStream/GStreamer:

```
RTSP Stream
  │
  ▼
FFmpeg subprocess (ffmpeg -hwaccel cuda -c:v h264_cuvid ...)
  │
  ▼
Raw NV12 frames (GPU memory via NVDEC)
  │
  ▼
CuPy GPU array conversion (zero-copy where possible)
  │
  ▼
Per-detector processing (brightness analysis OR TensorRT inference)
```

**Advantages over DeepStream:**
- Simpler per-detector lifecycle management
- Independent restart on failure (one detector crash doesn't kill others)
- Easier threshold tuning and debugging per algorithm
- No GStreamer element dependency chain

### 5.3 Model Configuration (Actual Deployed Models)

| Model | Detector | Input | Output Shape | Classes | Threshold | Batch | Precision |
|-------|----------|-------|-------------|---------|-----------|-------|-----------|
| best_pouring_hicon_v1_930 | Pouring System | 640×640 | Standard YOLO | ladle_mouth, trolley | mouth≥0.4, trolley≥0.25 | 1 | FP16 |
| best_pyro_rod_v1 (YOLO26) | Pyrometer Rod | 1280×1280 | (batch, 300, 6) end-to-end | rod class(es) | ≥0.25 | 2 | FP16 |
| *None* | Tapping | N/A | Brightness-based | N/A | Y>180 | N/A | N/A |
| *None* | Deslagging | N/A | Brightness-based | N/A | Y>250 | N/A | N/A |

**Notes:**
- Person and Helmet models from the previous PPE system are **NOT loaded** — no PPE detection
- Tapping and Deslagging use **no ML model** — pure CuPy GPU brightness analysis
- YOLO26 uses end-to-end TensorRT (NMS built into model), output is `[x1,y1,x2,y2,conf,class_id]`
- Pouring YOLO uses standard TensorRT with external ByteTrack tracking

---

## 6. Software Architecture

### 6.1 Module Dependency Graph

```
hicon_orchestrator.py  (Process supervisor / launcher)
  │
  ├── detectors/
  │   ├── pouring_system.py              YOLO TRT + ByteTrack (standalone)
  │   │   ├── TensorRT inference engine
  │   │   ├── ByteTrack multi-object tracker
  │   │   ├── Session manager (ladle-in-trolley lifecycle)
  │   │   ├── Pour detector (brightness probe below mouth)
  │   │   ├── Mould counter (anchor-based motion + clustering)
  │   │   └── Output: CSV (incremental) + JSON (batch)
  │   │
  │   ├── molten_flow_detector.py        Tapping (CuPy brightness)
  │   │   ├── FFmpeg NVDEC decode
  │   │   ├── CuPy quad-ROI masking
  │   │   ├── YUV Y-channel brightness
  │   │   └── Dual-hysteresis state machine
  │   │
  │   ├── deslagging_detector.py         Deslagging (CuPy brightness)
  │   │   ├── FFmpeg NVDEC decode
  │   │   ├── CuPy polygon-ROI masking (6 vertices)
  │   │   ├── YUV Y-channel brightness
  │   │   └── White-ratio state machine
  │   │
  │   ├── pyrometer_rod_detector.py      YOLO26 TRT rod detection
  │   │   ├── VideoDecoder thread (FFmpeg NVDEC + CuPy)
  │   │   ├── InferenceEngine thread (TensorRT FP16, batch=2)
  │   │   ├── PostProcessor thread (zone check + temporal)
  │   │   └── Double-buffered, pinned-memory pipeline
  │   │
  │   └── spectro_processor.py           [Design only — event capture]
  │
  ├── db_manager.py                      SQLite abstraction (extended tables)
  │
  ├── sync/
  │   ├── api_client.py                  HMAC-authenticated HTTP client
  │   └── sync_manager.py               Background sync thread (30s)
  │
  ├── config.py                          Environment variable config
  └── utils.py                           Shared utilities
```

### 6.2 Data Flow — Per-Detector Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  BRIGHTNESS DETECTORS (Tapping + Deslagging)                    │
│                                                                  │
│  FFmpeg subprocess (NVDEC HW decode)                            │
│    │                                                             │
│    ▼                                                             │
│  Raw YUV frame → CuPy GPU array                                │
│    │                                                             │
│    ▼                                                             │
│  Extract Y channel (luminance only — no color conversion)       │
│    │                                                             │
│    ▼                                                             │
│  Apply ROI mask (quad or polygon → CuPy boolean mask)           │
│    │                                                             │
│    ▼                                                             │
│  Threshold: Y > threshold → binary "white" pixels               │
│    │                                                             │
│    ▼                                                             │
│  Compute white_ratio = count(white) / count(ROI pixels)         │
│    │                                                             │
│    ▼                                                             │
│  Temporal state machine (frame counters for hysteresis)         │
│    │                                                             │
│    ├── Event START → timestamp + optional screenshot/video      │
│    └── Event END   → timestamp + duration → SQLite + sync       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  POURING SYSTEM (ML-based)                                      │
│                                                                  │
│  FFmpeg subprocess (NVDEC HW decode)                            │
│    │                                                             │
│    ▼                                                             │
│  Frame → TensorRT YOLO inference (640×640 FP16)                │
│    │                                                             │
│    ▼                                                             │
│  Detections: ladle_mouth (conf≥0.4) + trolley (conf≥0.25)      │
│    │                                                             │
│    ▼                                                             │
│  ByteTrack multi-object tracking (persistent IDs)              │
│    │                                                             │
│    ├── Session Manager                                          │
│    │     ├── mouth bbox center inside trolley bbox?             │
│    │     ├── IN for ≥1.0s → session START                       │
│    │     └── OUT for ≥1.5s → session END                        │
│    │                                                             │
│    ├── Pour Detector                                            │
│    │     ├── Probe point: 50px below mouth center, 6px radius  │
│    │     ├── Brightness >230 sustained 0.25s → pour START       │
│    │     ├── Brightness <180 sustained 1.0s → pour END          │
│    │     └── Min duration 2.0s filter                           │
│    │                                                             │
│    └── Mould Counter                                            │
│          ├── Anchor-based motion detection per track            │
│          ├── Displacement >0.15 threshold, sustained ≥1.5s     │
│          ├── Motion segmentation + spatial clustering           │
│          │   (R_CLUSTER=0.08, R_MERGE=0.05)                    │
│          └── Incremental count → CSV + JSON output             │
│                                                                  │
│  Output: CSV (real-time per-event) + JSON (end-of-video batch) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PYROMETER ROD DETECTOR (3-stage threaded pipeline)             │
│                                                                  │
│  Thread 1: VideoDecoder                                         │
│    ├── FFmpeg NVDEC subprocess                                  │
│    ├── Raw frame → CuPy GPU array                               │
│    ├── Resize + normalize + pad to 1280×1280                    │
│    └── Batched upload (batch=2) via double-buffered queues     │
│                                                                  │
│  Thread 2: InferenceEngine                                      │
│    ├── TensorRT YOLO26 end-to-end (FP16)                       │
│    ├── Input: (2, 3, 1280, 1280) contiguous GPU tensor         │
│    ├── Output: (batch, 300, 6) = [x1,y1,x2,y2,conf,class_id] │
│    ├── GPU-to-GPU copy (no host roundtrip)                     │
│    └── Pinned memory for results transfer                      │
│                                                                  │
│  Thread 3: PostProcessor                                        │
│    ├── Filter: confidence ≥ 0.25                                │
│    ├── Zone check: bbox top-left AND bottom-center in polygon  │
│    │   Zone polygon: [(303,151), (303,568), (724,568), (724,151)]│
│    ├── Temporal filter: 10 consecutive in-zone frames → START  │
│    │                     10 consecutive absent frames → END     │
│    └── Event output: start/end timestamps + metadata           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Detection Algorithms

### 7.1 Pouring Detection (YOLO + ByteTrack + Brightness Probe)

The pouring system is the most complex detector, combining ML object detection, multi-object tracking, brightness-based pour sensing, and motion-based mould counting.

```
                YOLO TensorRT (640×640 FP16)
                        │
                        ▼
            ┌───────────────────────┐
            │  Detected Objects     │
            │  • ladle_mouth ≥ 0.4  │
            │  • trolley ≥ 0.25     │
            └───────────┬───────────┘
                        │
                        ▼
               ByteTrack Tracker
              (persistent track IDs)
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   Session Manager  Pour Detector  Mould Counter
```

**7.1.1 Session Management (Ladle-in-Trolley Lifecycle)**

```
  ┌─────────────────────────────────────────┐
  │           trolley bbox                   │
  │                                          │
  │      ┌──────────────┐                    │
  │      │  ladle_mouth  │ ← bbox center     │
  │      └──────┬───────┘   tested against   │
  │             │            trolley bounds   │
  │             │                             │
  └─────────────┼─────────────────────────────┘
                │
     mouth center INSIDE trolley?
          │              │
         Yes             No
          │              │
     Timer ≥1.0s     Timer ≥1.5s
          │              │
          ▼              ▼
   SESSION START    SESSION END
```

- **Start condition:** ladle_mouth bbox center is inside trolley bbox for ≥1.0 seconds continuously
- **End condition:** ladle_mouth bbox center is absent from trolley bbox for ≥1.5 seconds
- Sessions encapsulate all pour events and mould counts for a single trolley visit

**7.1.2 Pour Detection (Sub-Mouth Brightness Probe)**

```
       ladle_mouth bbox
            │
     bottom center point
            │
            ▼  50px below
         ●──────● ← 6px radius circular probe
            │
    Probe region brightness
            │
    ┌───────┴───────┐
    │               │
  >230            <180
  (0.25s)         (1.0s)
    │               │
    ▼               ▼
  POUR START     POUR END
  (if duration ≥ 2.0s, emit event)
```

- **Probe geometry:** Circular region, radius=6px, centered 50px below ladle_mouth bottom-center
- **Start:** Average brightness in probe > 230, sustained for 0.25 seconds
- **End:** Average brightness in probe < 180, sustained for 1.0 seconds
- **Minimum duration:** 2.0 seconds (shorter detections are filtered as noise)
- **Output:** Pour start/end timestamps per session

**7.1.3 Mould Counting (Anchor-Based Motion + Spatial Clustering)**

```
  Per tracked trolley:
    anchor_position ← initial bbox center when first tracked
    current_position ← current bbox center
    displacement = |current - anchor| / frame_diagonal

    displacement > 0.15 for ≥1.5s?
          │              │
         Yes             No
          │              │
    Mark "moved"    Keep watching
          │
          ▼
    Motion Segmentation
          │
          ▼
    Spatial Clustering
    ├── R_CLUSTER = 0.08 (cluster formation radius)
    ├── R_MERGE = 0.05 (cluster merge radius)
    └── Each stable cluster = 1 mould position
          │
          ▼
    Increment mould_count
```

- **Anchor-based motion:** Each tracked object gets an anchor at first detection; displacement measured as normalized Euclidean distance
- **Movement threshold:** displacement > 0.15 (relative to frame diagonal), sustained ≥1.5 seconds
- **Clustering:** Agglomerative approach with R_CLUSTER=0.08 for initial cluster assignment and R_MERGE=0.05 for merging nearby clusters
- **Output:** Incremental mould count per session, written to CSV in real-time and JSON at end-of-video

**7.1.4 Pouring Output Formats**

Two output formats are produced simultaneously:

- **Incremental CSV** (real-time): Appended after each pour event — one row per pour with session ID, timestamps, mould count at time of pour
- **Batch JSON** (end-of-video): Complete summary with all sessions, pours, and final mould counts — written when video/stream processing completes

### 7.2 Tapping Detection (CuPy GPU Brightness — Quad ROI)

Tapping monitors molten metal flow from the furnace tap hole using brightness analysis on the YUV luminance channel.

```
  Frame (YUV from FFmpeg NVDEC)
    │
    ▼
  Extract Y channel → CuPy GPU array
    │
    ▼
  Apply quadrilateral ROI mask (4 vertices)
  ┌─────────────────────────────────┐
  │  (x1,y1)────────────(x2,y2)    │
  │  │    Tapping Zone           │  │
  │  │    (furnace tap area)     │  │
  │  (x4,y4)────────────(x3,y3)    │
  └─────────────────────────────────┘
    │
    ▼
  Threshold: Y > 180 → binary white pixels
    │
    ▼
  white_ratio = count(white pixels) / count(ROI pixels)
    │
    ▼
  Dual Hysteresis State Machine
    │
    ├── white_ratio ≥ 0.80 for 10 consecutive frames → ACTIVE (tapping start)
    │
    └── white_ratio < 0.60 for 20 consecutive frames → IDLE (tapping end)
```

**Key Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| ROI shape | Quadrilateral (4 points) | Matches furnace tap hole area |
| Brightness threshold | Y > 180 | Detects hot/molten metal glow |
| Start ratio | white_ratio ≥ 0.80 | High threshold to avoid false starts |
| Start frames | 10 consecutive | ~0.33s at 30 FPS confirmation |
| End ratio | white_ratio < 0.60 | Lower threshold for hysteresis gap |
| End frames | 20 consecutive | ~0.67s at 30 FPS grace period |

**State Machine:**

```
              white_ratio ≥ 0.80
              (10 frames)
    ┌──────┐ ─────────────► ┌────────┐
    │ IDLE │                │ ACTIVE │
    └──────┘ ◄───────────── └────────┘
              white_ratio < 0.60
              (20 frames)
```

- **IDLE → ACTIVE:** Sustained high brightness (≥80% of ROI pixels above Y=180 for 10 frames)
- **ACTIVE → IDLE:** Brightness drops below 60% threshold for 20 frames
- **Dual hysteresis** prevents rapid state oscillation at boundary conditions
- **Events:** Tapping start timestamp, tapping end timestamp, duration (HH:MM:SS), optional video recording

### 7.3 Deslagging Detection (CuPy GPU Brightness — Polygon ROI)

Deslagging monitors slag removal from the furnace surface, detecting the characteristic bright/white visual signature of hot slag being scraped off.

```
  Frame (YUV from FFmpeg NVDEC)
    │
    ▼
  Extract Y channel → CuPy GPU array
    │
    ▼
  Apply polygon ROI mask (6 vertices)
  ┌─────────────────────────────────┐
  │      (x1,y1)                    │
  │     /        \                  │
  │  (x6,y6)   (x2,y2)            │
  │  │    Deslagging Zone      │   │
  │  (x5,y5)   (x3,y3)            │
  │     \        /                  │
  │      (x4,y4)                    │
  └─────────────────────────────────┘
    │
    ▼
  Threshold: Y > 250 → binary white pixels
    │
    ▼
  white_ratio = count(white pixels) / count(ROI pixels)
    │
    ▼
  Simple Hysteresis State Machine
    │
    ├── white_ratio ≥ 0.01 for 10 consecutive frames → ACTIVE (deslagging start)
    │
    └── white_ratio < 0.01 for 15 consecutive frames → IDLE (deslagging end)
```

**Key Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| ROI shape | Polygon (6 points) | Irregular shape matching furnace mouth/surface |
| Brightness threshold | Y > 250 | Very high — only near-white pixels (hot slag) |
| Start ratio | white_ratio ≥ 0.01 | Very low — even tiny bright area triggers |
| Start frames | 10 consecutive | ~0.33s at 30 FPS confirmation |
| End ratio | white_ratio < 0.01 | Same threshold (no hysteresis gap) |
| End frames | 15 consecutive | ~0.50s at 30 FPS grace period |

**Comparison — Tapping vs Deslagging:**

| Aspect | Tapping | Deslagging |
|--------|---------|------------|
| ROI shape | Quadrilateral (4 pts) | Polygon (6 pts) |
| Y threshold | > 180 (moderate glow) | > 250 (near-white only) |
| Start white_ratio | ≥ 0.80 (most of ROI) | ≥ 0.01 (tiny bright area) |
| End white_ratio | < 0.60 (hysteresis) | < 0.01 (same as start) |
| Start frames | 10 | 10 |
| End frames | 20 | 15 |
| Visual signature | Molten stream fills ROI | Small bright spots on surface |

Both detectors share the same FFmpeg NVDEC → CuPy → Y-channel → threshold → white_ratio → state machine architecture, but are tuned for their specific visual characteristics.

### 7.4 Pyrometer Rod Detection (YOLO26 End-to-End TensorRT — 3-Stage Pipeline)

The pyrometer detector identifies when a measurement rod is inserted into the furnace, using a high-performance threaded inference pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│  Thread 1: VideoDecoder                                         │
│                                                                  │
│  RTSP → FFmpeg (NVDEC) → Raw frames → CuPy GPU array           │
│    │                                                             │
│    ▼                                                             │
│  Preprocess on GPU (CuPy):                                      │
│    ├── Resize to 1280×1280 (letterbox/pad)                      │
│    ├── Normalize to [0,1] float32                                │
│    ├── CHW transpose                                             │
│    └── Batch accumulate (batch_size=2)                          │
│    │                                                             │
│    ▼                                                             │
│  Double-buffered queue → Thread 2                               │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Thread 2: InferenceEngine                                      │
│                                                                  │
│  TensorRT YOLO26 end-to-end model                              │
│    ├── Input:  (2, 3, 1280, 1280) — FP16 contiguous GPU tensor │
│    ├── Output: (batch, 300, 6) per batch item                   │
│    │           [x1, y1, x2, y2, confidence, class_id]           │
│    ├── NMS built into model (no external post-processing)       │
│    ├── GPU-to-GPU copy (no host roundtrip for input)           │
│    └── Pinned memory for output transfer                       │
│    │                                                             │
│    ▼                                                             │
│  Results queue → Thread 3                                       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Thread 3: PostProcessor                                        │
│                                                                  │
│  For each detection with confidence ≥ 0.25:                     │
│    │                                                             │
│    ▼                                                             │
│  Zone Check (both conditions must be true):                     │
│    ├── bbox top-left corner (x1, y1) inside polygon            │
│    └── bbox bottom-center ((x1+x2)/2, y2) inside polygon      │
│                                                                  │
│  Zone polygon vertices:                                         │
│    [(303, 151), (303, 568), (724, 568), (724, 151)]            │
│                                                                  │
│    │                                                             │
│    ▼                                                             │
│  Temporal Filter:                                               │
│    ├── Rod IN zone: 10 consecutive frames → EVENT START        │
│    └── Rod OUT of zone: 10 consecutive frames → EVENT END      │
│    │                                                             │
│    ▼                                                             │
│  Output: event timestamps + metadata                            │
└─────────────────────────────────────────────────────────────────┘
```

**GPU Optimization Techniques:**

| Technique | Implementation | Benefit |
|-----------|---------------|---------|
| CuPy preprocessing | All resize/normalize/transpose on GPU | Eliminates CPU↔GPU transfer for preprocessing |
| Batched upload | Accumulate 2 frames before inference | Better GPU utilization per TRT call |
| Double buffering | Two input buffers alternate fill/infer | Overlaps decode with inference |
| GPU-to-GPU copy | Input tensor stays on GPU throughout | No host memory roundtrip |
| Pinned memory | Output results use CUDA pinned memory | Faster GPU→CPU transfer for results |
| End-to-end NMS | NMS baked into TensorRT model | No post-processing overhead |

**Performance:** 250+ FPS inference throughput (significantly faster than camera frame rate, leaving headroom for batch processing and other GPU tasks).

### 7.5 Spectrometry Processor (Design Phase — Not Yet Implemented)

The spectrometry processor is designed for event capture only — it will record spectrometry readings (C%, Si%, Mn%) when triggered by an external signal or manual input. No ML model or brightness detection is used; the processor simply captures the event timestamp, associates it with the current heat, and stores the readings for cloud sync.

---

## 8. Data Architecture

### 8.1 Database Schema (SQLite — Extended for HiCon)

```sql
-- Melting Events (NEW — replaces safety_violations)
CREATE TABLE melting_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sync_id             TEXT UNIQUE NOT NULL,
    customer_id         TEXT NOT NULL,
    slno                TEXT NOT NULL,
    date                TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    camera_id           TEXT NOT NULL,
    location            TEXT NOT NULL,
    event_type          TEXT NOT NULL,           -- 'tapping', 'spectro', 'deslagging'
    pyrometer_temp      REAL,                    -- °C from pyrometer camera
    spectro_c           REAL,                    -- Carbon %
    spectro_si          REAL,                    -- Silicon %
    spectro_mn          REAL,                    -- Manganese %
    heat_no             TEXT,                    -- Heat number
    screenshot_path     TEXT,
    start_time          TEXT,                    -- ISO8601
    end_time            TEXT,                    -- ISO8601
    duration            TEXT,                    -- "HH:MM:SS"
    synced              INTEGER DEFAULT 0,
    sync_attempts       INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL
);

-- Pouring Events (Extended from legacy baseline)
CREATE TABLE pouring_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    sync_id                 TEXT UNIQUE NOT NULL,
    customer_id             TEXT NOT NULL,
    slno                    TEXT NOT NULL,
    date                    TEXT NOT NULL,
    camera_id               TEXT NOT NULL,
    location                TEXT NOT NULL,
    heat_no                 TEXT,                    -- NEW: Heat number
    mould_count             INTEGER,                 -- NEW: Total moulds in heat
    pouring_start_time      TEXT NOT NULL,
    pouring_end_time        TEXT,
    total_pouring_time      TEXT,
    mould_wise_pouring_time TEXT,                    -- JSON string
    video_path              TEXT,
    screenshot_path         TEXT,
    synced                  INTEGER DEFAULT 0,
    sync_attempts           INTEGER DEFAULT 0,
    cloud_response          TEXT,
    created_at              TEXT NOT NULL
);

-- Indexes
CREATE INDEX idx_melting_synced ON melting_events(synced);
CREATE INDEX idx_melting_date ON melting_events(date);
CREATE INDEX idx_pouring_synced ON pouring_events(synced);
CREATE INDEX idx_pouring_date ON pouring_events(date);
```

### 8.2 Pouring System Local Output (CSV + JSON)

In addition to SQLite/cloud sync, the pouring system produces local output files:

**Incremental CSV (real-time):** Appended after each event as it occurs.
```
session_id, event_type, timestamp, mould_count, pour_start, pour_end, pour_duration
S001, session_start, 2026-02-09 14:30:00, 0, , ,
S001, pour, 2026-02-09 14:30:15, 1, 14:30:12, 14:30:15, 3.2
S001, pour, 2026-02-09 14:31:20, 2, 14:31:17, 14:31:20, 3.1
S001, session_end, 2026-02-09 14:32:35, 2, , ,
```

**Batch JSON (end-of-video):** Complete session summary written once at stream/video end.
```json
{
  "sessions": [
    {
      "session_id": "S001",
      "start_time": "14:30:00",
      "end_time": "14:32:35",
      "total_pours": 2,
      "mould_count": 2,
      "pours": [
        {"start": "14:30:12", "end": "14:30:15", "duration": 3.2},
        {"start": "14:31:17", "end": "14:31:20", "duration": 3.1}
      ]
    }
  ]
}
```

### 8.3 Data Lifecycle

Same sync lifecycle: SQLite INSERT → 30s sync cycle → HMAC-authenticated POST → 7-day cleanup.

---

## 9. Cloud Sync Protocol (AGNI API)

### 9.1 Authentication

Authentication uses HMAC-SHA256 on request body → `X-HMAC-Signature` header.

### 9.2 API Payloads

**Melting Events (NEW):**
```json
POST /api/v1/melting
{
  "items": [
    {
      "sync_id": "hicon-melting-1706000000123",
      "customer_id": "HICON_001",
      "slno": "0001",
      "date": "2026-02-09",
      "timestamp": "2026-02-09 14:30:00",
      "camera_id": "Process_Camera",
      "location": "Induction Furnace 1",
      "event_type": "tapping",
      "pyrometer_temp": 1485.0,
      "spectro_c": 3.45,
      "spectro_si": 2.10,
      "spectro_mn": 0.65,
      "heat_no": "H2026-0209-001",
      "screenshot": "<base64 JPEG>"
    }
  ]
}
```

**Pouring Events (Extended):**
```json
POST /api/v1/pouring
{
  "items": [
    {
      "sync_id": "hicon-pouring-1706000000456",
      "customer_id": "HICON_001",
      "slno": "0015",
      "date": "2026-02-09",
      "camera_id": "Process_Camera",
      "location": "Induction Furnace 1",
      "heat_no": "H2026-0209-001",
      "mould_count": 12,
      "pouring_start_time": "2026-02-09 14:30:00",
      "pouring_end_time": "2026-02-09 14:32:35",
      "total_pouring_time": "00:02:35",
      "mould_wise_pouring_time": [
        {"mould_id": "M001", "start": "14:30:00", "end": "14:31:15", "duration": "75"},
        {"mould_id": "M002", "start": "14:31:20", "end": "14:32:35", "duration": "75"}
      ],
      "screenshot": "<base64 JPEG>"
    }
  ]
}
```

### 9.3 AGNI Field Mapping (New vs Inherited)

| Field | API | Status | Source |
|-------|-----|--------|--------|
| customer_id | Both | Inherited | Config |
| slno | Both | Inherited | Auto-increment |
| date, timestamp | Both | Inherited | System clock |
| camera_id | Both | Inherited | Config |
| location | Both | Inherited | Config |
| screenshot | Both | Inherited | Frame capture → JPEG |
| **pyrometer_temp** | Melting | **NEW** | Pyrometer Camera (rod detection event) |
| **spectro_c** | Melting | **NEW** | Spectro event capture |
| **spectro_si** | Melting | **NEW** | Spectro event capture |
| **spectro_mn** | Melting | **NEW** | Spectro event capture |
| **heat_no** | Both | **NEW** | Heat tracking state machine |
| **mould_count** | Pouring | **NEW** | Mould counter (anchor + clustering) |

---

## 10. Configuration Management

### 10.1 Key Configuration (HiCon-specific)

| Category | Variables | Example |
|----------|-----------|---------|
| Streams | `HICON_RTSP_STREAM_0`, `HICON_RTSP_STREAM_1` | `rtsp://192.168.1.100:554/process` |
| Identity | `HICON_CUSTOMER_ID` | `HICON_001` |
| Camera IDs | `HICON_CAMERA_ID_PROCESS`, `HICON_CAMERA_ID_PYRO` | `Process_Camera`, `Pyrometer_Camera` |
| Cloud | `HICON_HMAC_SECRET`, `HICON_ENABLE_SYNC` | secret key, `true/false` |
| API URLs | `HICON_MELTING_API_URL`, `HICON_POURING_API_URL` | AGNI endpoints |
| Paths | `HICON_BASE_DIR`, `HICON_DATA_DIR` | `/home/hicon/hicon/ai_vision` |

### 10.2 Detector Parameters (Actual Tuned Values)

**Tapping Detector (molten_flow_detector.py):**
```json
{
  "tapping": {
    "roi_type": "quadrilateral",
    "roi_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "brightness_channel": "Y (YUV luminance)",
    "brightness_threshold": 180,
    "start_white_ratio": 0.80,
    "start_frame_count": 10,
    "end_white_ratio": 0.60,
    "end_frame_count": 20
  }
}
```

**Deslagging Detector (deslagging_detector.py):**
```json
{
  "deslagging": {
    "roi_type": "polygon",
    "roi_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5], [x6,y6]],
    "brightness_channel": "Y (YUV luminance)",
    "brightness_threshold": 250,
    "start_white_ratio": 0.01,
    "start_frame_count": 10,
    "end_white_ratio": 0.01,
    "end_frame_count": 15
  }
}
```

**Pouring System:**
```json
{
  "pouring": {
    "model_input_size": [640, 640],
    "model_precision": "FP16",
    "ladle_mouth_confidence": 0.40,
    "trolley_confidence": 0.25,
    "session_start_duration": 1.0,
    "session_end_duration": 1.5,
    "pour_probe_offset_px": 50,
    "pour_probe_radius_px": 6,
    "pour_brightness_start": 230,
    "pour_brightness_end": 180,
    "pour_start_duration": 0.25,
    "pour_end_duration": 1.0,
    "pour_min_duration": 2.0,
    "mould_displacement_threshold": 0.15,
    "mould_sustained_duration": 1.5,
    "cluster_r_cluster": 0.08,
    "cluster_r_merge": 0.05
  }
}
```

**Pyrometer Rod Detector:**
```json
{
  "pyrometer": {
    "model": "YOLO26 end-to-end",
    "model_input_size": [1280, 1280],
    "model_precision": "FP16",
    "batch_size": 2,
    "confidence_threshold": 0.25,
    "zone_polygon": [[303,151], [303,568], [724,568], [724,151]],
    "zone_check": "top-left AND bottom-center in polygon",
    "temporal_in_frames": 10,
    "temporal_out_frames": 10
  }
}
```

---

## 11. Heat-Level State Machine (NEW)

HiCon tracks operations at the **heat level** — a complete furnace cycle from charge to final pour:

```
                    ┌──────────┐
                    │   IDLE   │
                    └────┬─────┘
                         │  Furnace charged / heat starts
                         ▼
                    ┌──────────┐
              ┌────►│ MELTING  │◄──── Pyrometer rod insertions
              │     └────┬─────┘      detected during this phase
              │          │
              │          │  Spectro sample taken
              │          ▼
              │     ┌──────────┐
              │     │ SPECTRO  │───── spectro_c, spectro_si, spectro_mn
              │     └────┬─────┘      recorded
              │          │
              │          │  Temperature OK / deslagging starts
              │          ▼
              │     ┌──────────────┐
              │     │ DESLAGGING   │── Deslagging duration tracked
              │     └────┬─────────┘   (CuPy brightness, polygon ROI)
              │          │
              │          │  Deslagging complete / tapping starts
              │          ▼
              │     ┌──────────┐
              │     │ TAPPING  │───── Tapping start/end timestamps
              │     └────┬─────┘      (CuPy brightness, quad ROI)
              │          │
              │          │  Metal in ladle → pouring starts
              │          ▼
              │     ┌──────────┐
              │     │ POURING  │───── Session tracking, pour detection,
              │     └────┬─────┘      mould counting (YOLO + ByteTrack)
              │          │
              │          │  All moulds filled / ladle empty
              │          ▼
              │     ┌──────────┐
              └─────│  RESET   │───── Heat complete → increment heat_no
                    └──────────┘      Sync all accumulated data
```

---

## 12. Resilience & Fault Tolerance

Adapted from the earlier baseline for standalone detector architecture:

| Failure | Detection | Recovery | Downtime |
|---------|-----------|----------|----------|
| Single detector crash | Process monitor / heartbeat | Auto-restart individual detector | ~2s for that detector |
| FFmpeg decode stall | No frames > 60s | Kill + restart FFmpeg subprocess | ~3s |
| TensorRT engine fail | Inference exception | Reload .engine file + restart | ~10s |
| Both streams stalled | Health check > 10 min | Full pipeline restart via systemd | ~15s |
| Cloud API down | HTTP 4xx/5xx | Retry up to 5×, buffer locally | 0s (local buffer) |
| Disk full | Insert failure | 7-day cleanup runs on INSERT | 0s (auto-cleanup) |
| GPU OOM | CUDA allocation error | systemd restart | ~15s |
| Power loss | N/A | systemd auto-start on boot | Boot time (~30s) |

**Advantage of standalone detector architecture:** A crash in one detector (e.g., pyrometer) does not affect other detectors (tapping, deslagging, pouring). Each detector manages its own FFmpeg subprocess and can be restarted independently.

---

## 13. Deployment Architecture

### 13.1 Service Management

```
┌──────────────────────────────────────────┐
│          systemd                          │
│                                          │
│  hicon-vision.service                    │
│  ├── Type=simple                         │
│  ├── Restart=always                      │
│  ├── RestartSec=10                       │
│  ├── User=hicon                          │
│  ├── WorkingDirectory=/home/hicon/hicon/ │
│  │   ai_vision                           │
│  ├── ExecStart=python3 hicon_pipeline.py │
│  ├── EnvironmentFile=.env                │
│  └── StandardOutput=append:logs/         │
│       pipeline.log                       │
└──────────────────────────────────────────┘
```

### 13.2 Filesystem Layout

```
/home/hicon/hicon/
├── ai_vision/                    # Application root
│   ├── hicon_pipeline.py         # Entry point / orchestrator
│   ├── config.py                 # Configuration
│   ├── .env                      # Environment overrides
│   │
│   ├── detectors/                # Standalone detector modules
│   │   ├── pouring_system.py     # YOLO TRT + ByteTrack
│   │   ├── molten_flow_detector.py  # Tapping (CuPy brightness)
│   │   ├── deslagging_detector.py   # Deslagging (CuPy brightness)
│   │   ├── pyrometer_rod_detector.py  # YOLO26 TRT rod detection
│   │   └── spectro_processor.py  # [Design only]
│   │
│   ├── models/                   # TensorRT engine files
│   │   ├── best_pouring_hicon_v1_930.engine  # 640×640 FP16 (from best_pouring_hicon_v1_930.pt)
│   │   └── best_pyro_rod_v1.engine  # 1280×1280 FP16 end-to-end (from best_pyro_rod_v1.pt)
│   │
│   ├── db_manager.py             # SQLite abstraction
│   ├── sync/                     # Cloud sync (HMAC + HTTP)
│   │   ├── api_client.py
│   │   └── sync_manager.py
│   │
│   ├── data/                     # SQLite database
│   │   └── hicon.db
│   ├── output/
│   │   ├── screenshots/          # JPEG captures
│   │   ├── videos/               # MP4 recordings
│   │   ├── csv/                  # Incremental pouring CSV
│   │   └── json/                 # Batch pouring JSON
│   └── logs/
│       └── pipeline.log
│
└── [No DeepStream-Yolo dependency — standalone TensorRT]
```

---

## 14. Performance Characteristics

### 14.1 Per-Detector Performance

```
PYROMETER ROD DETECTOR:
  3-stage threaded pipeline
  Inference: 250+ FPS throughput (TensorRT YOLO26 FP16, batch=2)
  Decode bottleneck: ~30 FPS (camera rate) — inference has massive headroom
  GPU memory: ~1 GB (model + buffers + CuPy preprocess)

POURING SYSTEM:
  YOLO TRT + ByteTrack
  End-to-end: 20–30 FPS on RTX 3060+
  GPU memory: ~2–3 GB (model + tracker state + frame buffers)

TAPPING DETECTOR:
  CuPy brightness analysis (no ML model)
  Throughput: native frame rate (30 FPS) — negligible GPU compute
  GPU memory: ~100 MB (CuPy arrays + ROI mask)

DESLAGGING DETECTOR:
  CuPy brightness analysis (no ML model)
  Throughput: native frame rate (30 FPS) — negligible GPU compute
  GPU memory: ~100 MB (CuPy arrays + ROI mask)
```

**Total GPU Memory Budget (Orin Nano 8GB — shared CPU/GPU):**
- Pyrometer model + buffers: ~1.0 GB
- Pouring model + ByteTrack: ~2.5 GB
- Brightness detectors (×2): ~0.2 GB
- FFmpeg NVDEC (×2 streams): ~0.3 GB
- OS + CPU overhead (shared memory): ~2.0 GB
- **Total: ~6.0 GB** of 8 GB shared (75% utilization — tight budget)

**⚠ Orin Nano Constraint — Software Video Encoding:**
MP4 recording of pouring/tapping events uses software H.264 encoding (no HW encoder on Orin Nano). Each concurrent encode at 1080p30 consumes ~1 CPU core out of 6 available. Mitigations:
- Record at reduced resolution (720p) or frame rate (15 FPS) to lower CPU load
- Record only one stream at a time (pouring takes priority)
- Use FFmpeg `-preset ultrafast` to minimize encoding CPU cost

### 14.2 Storage Budget (7-Day Retention)

| Artifact | Size/Event | Events/Day (est.) | 7-Day Total |
|----------|-----------|-------------------|-------------|
| Melting screenshots | ~80 KB | ~30 | ~17 MB |
| Pouring screenshots | ~80 KB | ~50 | ~28 MB |
| Pouring videos | ~50 MB | ~20 | ~7 GB |
| Pouring CSV/JSON | ~10 KB | ~50 | ~3.5 MB |
| SQLite database | ~1 KB/row | ~100 | ~700 KB |
| **Total** | | | **~7.05 GB** |

---

## 15. Security

| Layer | Mechanism |
|-------|-----------|
| API Authentication | HMAC-SHA256 on request body |
| Transport | HTTPS (TLS 1.2+) |
| Local Storage | SQLite (no encryption — physical access assumed secure) |
| RTSP Streams | Isolated plant network (no auth on streams) |
| Service Isolation | Dedicated `hicon` user, no root access needed |
| Secret Management | Environment variable (`HICON_HMAC_SECRET`), not in code |

---

## 16. Implementation Status

### Implemented (Production-Ready)

- [x] **Tapping Detector** — CuPy GPU brightness, quad ROI, dual hysteresis (0.80/0.60), temporal filtering (10/20 frames)
- [x] **Deslagging Detector** — CuPy GPU brightness, 6-point polygon ROI, white_ratio 0.01, temporal filtering (10/15 frames)
- [x] **Pyrometer Rod Detector** — YOLO26 TensorRT end-to-end, 3-stage threaded pipeline, 250+ FPS, zone detection, 10-frame temporal
- [x] **Pouring System** — YOLO TensorRT + ByteTrack, session management, brightness-based pour detection, anchor-based mould counting with spatial clustering, CSV + JSON output

### Design Phase (Not Yet Implemented)

- [ ] **Spectro Processor** — Event capture only (external trigger, store readings)
- [ ] **Heat-level state machine** — Orchestration across all detector events
- [ ] **AGNI cloud sync integration** — Melting + extended pouring API payloads
- [ ] **Orchestrator** — Process supervisor to launch/monitor all detectors
- [ ] **Site deployment** — Camera installation, zone calibration, threshold tuning with live data
