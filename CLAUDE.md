# CLAUDE.md — HiCon AI Vision System

## Project Overview
HiCon is a 3-camera edge AI vision system for induction furnace monitoring on **Jetson Orin Nano 8GB**, built on **DeepStream 7.1**. No PPE detection — furnace operations only.
- **Camera 0 (Process):** Tapping, pouring, deslagging, spectrometry — single RTSP decode shared by all detectors
- **Camera 1 (Pyrometer):** Rod insertion detection via YOLO26
- **Camera 2 (Pouring2):** Second pouring detection angle
- **Pipeline:** DeepStream 7.1 GStreamer (single Python process, single decode per stream)
- **Cloud:** AGNI API (HMAC-SHA256 authenticated HTTPS POST)
- **Design doc:** See `HiCon_Systems_Design.md` for full architecture and data flows
- **Architecture decision:** See `HiCon_Architecture_Comparison.md` for standalone vs DeepStream analysis
- **Technical architecture:** See `HiCon_Technical_Architecture.md` for deep codebase analysis, design reasoning, and performance data

## Implementation Status (Updated: 2026-02-18)

### Implemented and verified in code
- **Brightness stack unified:** tapping + deslagging + spectro all run in `brightness_processor.py` using shared CPU frame extraction.
- **Spectro detector integrated:** ROI in `configs/zones.json`, thresholds in config, and tracking in `BrightnessTracker`.
- **Spectro false-positive filter implemented:** `BrightnessTracker(max_white_ratio=0.20)` discards events that spike above cap.
- **Cross-suppression active:** deslagging and spectro checks are suppressed during tapping or active pouring cycle windows.
- **Heat-cycle aggregation expanded:** `tapping_events`, `deslagging_events`, and `spectro_events` are aggregated into `HeatCycleManager` and persisted in `heat_cycles`.
- **DB migration added:** `heat_cycles` now includes `tapping_events`, `deslagging_events`, `spectro_events` JSON columns.
- **Pouring parity updates applied:** top-only trolley edge expand, HSV-V probe brightness sampling, probe points on all pouring event screenshots.
- **MIN_CLUSTER_POUR_S implemented:** mould clusters with low cumulative pour time are filtered before final mould count.
- **DS-native inference overlays added:** custom status/probe overlays are attached via `nvds_acquire_display_meta_from_pool` and rendered by `nvosd`.
- **Brightness OSD overlays added:** tapping/deslagging/spectro status + ROI bounds are rendered on Stream 0 inference video.
- **DS-native recording branch added:** post-OSD tee branch configured through `RecordingManager`.
- **Automated tests added:** tracker max-threshold discard, pouring session/cycle transitions, probe-point screenshot rendering, DS-native overlay and recording smoke coverage.
- **Trolley relock behavior added:** when locked trolley ID disappears, padding re-locks to the best match (IoU/ confidence) so expand follows the active trolley.
- **API payloads aligned to new format:** `/pouring` and `/melting` payloads are built from `heat_cycles` with `mould_count`, `heat_no`, and cycle-level flags.
- **Furnace ID config added:** `HICON_FURNACE_ID` (defaults to `LOCATION`) used in melting payloads.
- **Quadrant-based direction guard implemented:** Mould split detection uses quadrant tracking (Q1–Q4) instead of axis+sign, fixing spurious resets during diagonal trolley movements. Improved mould count accuracy from ~85% to ~98%.
- **Split re-arm baseline gate added:** After a mould split, displacement must drop below threshold for 0.5s before next split can fire. Reduced false split rate from ~12% to <2%.
- **Split cooldown added:** 1.5s time-based cooldown between splits prevents double-counting.
- **Hybrid OR split trigger:** `magnitude_ok OR axis_only_ok` allows both diagonal and axis-dominant movements to trigger splits.
- **MJPEG streaming server added:** `streaming/mjpeg_server.py` for live inference video (optional, disabled by default).
- **Screenshot utilities added:** `utils/screenshot.py` with `prepare_frame`, `add_header`, `add_footer`, `save` for event captures with probe point overlays.

### Changes made but still requiring live validation closure
- DS-native recording path has been actively iterated to address zero-byte outputs.
- Current implementation uses `tee -> queue -> nvjpegenc -> matroskamux -> filesink` (MKV output), with branch buffer diagnostics in logs.
- Keep live validation as mandatory after each pipeline/release change on target Jetson.

## Hardware Constraints (Orin Nano 8GB)
- **No robust HW H.264 encode path available in this deployment** — avoid heavy CPU MP4 paths when possible
- **No NVDLA** — all inference on GPU CUDA cores only via `nvinfer`
- **7W–15W power** — may thermal-throttle under sustained load; monitor with `tegrastats`
- **Shared 8GB memory** — budget ~6GB for pipeline + models + OS
- **CuPy NOT available in DeepStream on Jetson** — `pyds.get_nvds_buf_surface_gpu()` is x86-only. Brightness detectors use `pyds.get_nvds_buf_surface()` → **NumPy on CPU** (+ mandatory `unmap_nvds_buf_surface()` on Jetson to prevent memory leak)
- **Event screenshots still use SW JPEG via OpenCV** (separate from DS recording branch)

## Pipeline Architecture (Hybrid DeepStream 7.1)
```
Stream 0 (Process Camera — single NVDEC decode)
  rtspsrc → nvv4l2decoder → nvvideoconvert(RGBA) → streammux
    ├─→ nvinfer (best_pouring_hicon_v1_930, GIE-1, 640×640 FP16)
    │     └─→ nvtracker (NvDCF or ByteTrack)
    │           └─→ nvosd
    │                 └─→ PROBE [osd_sink_pad]: pouring_processor()
    │                       ├── Session management (mouth-in-trolley)
    │                       ├── Pour detection (brightness probe below mouth)
    │                       ├── Mould counting (anchor + clustering)
    │                       └── Custom OSD overlay metadata for inference recording
    │
    │                 └─→ post-OSD convert/caps → tee
    │                       ├── display queue → sink_0
    │                       └── RecordingManager branch (DS-native inference video)
    │
    └─→ PROBE [post-decode or tee branch]: brightness_processor()
          ├── get_nvds_buf_surface() → NumPy (CPU)
          ├── Tapping check (quad ROI, Y>180, white_ratio)
          ├── Deslagging check (polygon ROI, Y>250, white_ratio)
          ├── Spectro check (polygon ROI, Y>250, white_ratio + max ratio discard)
          ├── DS-native OSD overlay (tapping/deslagging/spectro status + ROI bounds)
          └── unmap_nvds_buf_surface()  ← MANDATORY on Jetson

Stream 1 (Pyrometer Camera — single NVDEC decode)
  rtspsrc → nvv4l2decoder → nvvideoconvert(RGBA) → streammux
    └─→ nvinfer (best_pyro_rod_v1, GIE-2, 1280×1280 FP16, custom parser)
          └─→ PROBE [post-nvinfer]: pyrometer_processor()
                ├── Filter detections: confidence ≥ 0.25
                ├── Zone check: top-left AND bottom-center in polygon
                └── Temporal: 10 frames in/out for event start/end
```

## Project Structure
```
ai_vision/
├── hicon_pipeline.py               # Main entry — builds & runs DeepStream pipeline
├── config.py                       # Env var config + zone/threshold constants
├── .env                            # Runtime env overrides (RTSP URLs, HMAC, etc.)
│
├── pipeline/
│   ├── gst_builder.py              # GStreamer element creation (2-stream)
│   ├── bus_handler.py              # GStreamer bus message handling + error recovery
│   └── recording.py                # DS-native inference recording branch (post-OSD tee)
│
├── processors/                     # Pad probe callbacks (all run in single process)
│   ├── pouring_processor.py        # nvinfer detections → session/pour/mould logic
│   ├── brightness_processor.py     # NumPy CPU brightness for tapping + deslagging + spectro
│   └── pyrometer_processor.py      # nvinfer detections → zone check + temporal
│
├── state/
│   ├── brightness_tracker.py       # IDLE↔ACTIVE state machine (tapping & deslagging & spectro)
│   └── heat_cycle_manager.py       # Heat-level aggregation (tapping + pouring + deslagging + spectro)
│
├── configs/
│   ├── config_pouring_pgie.txt     # nvinfer config: best_pouring_hicon_v1_930
│   ├── config_pyrometer_pgie.txt   # nvinfer config: best_pyro_rod_v1
│   ├── config_tracker.txt          # nvtracker config (NvDCF or DeepSORT)
│   ├── labels_pouring.txt          # Class labels: ladle_mouth, trolley
│   ├── labels_pyrometer.txt        # Class labels: rod
│   └── zones.json                  # ROI polygons, thresholds, temporal params
│
├── custom_parsers/
│   └── nvdsinfer_custom_impl_Yolo/ # Custom parser shared library
│       ├── Makefile
│       └── nvdsinfer_yolo.cpp      # Handles standard YOLO + YOLO26 end-to-end output
│       → builds: libnvdsinfer_custom_impl_Yolo.so
│
├── models/
│   ├── best_pouring_hicon_v1_930.pt      # Source weights (pouring)
│   ├── best_pyro_rod_v1.pt              # Source weights (pyrometer)
│   └── onnx/                            # ONNX exports + .engine files (auto-generated)
│       ├── best_pouring_hicon_v1_930.onnx
│       ├── best_pouring_hicon_v1_930.engine
│       ├── best_pyro_rod_v1.onnx
│       └── best_pyro_rod_v1.engine
│
├── utils/
│   ├── utils.py                    # Shared helpers (sync ID generation, etc.)
│   ├── zone_loader.py              # Load ROI polygons from zones.json
│   ├── screenshot.py               # Event screenshot capture (prepare, header/footer, save)
│   └── visualization.py            # Overlay drawing utilities
│
├── streaming/
│   └── mjpeg_server.py             # Optional MJPEG HTTP server for live inference video
│
├── db_manager.py                   # SQLite (melting_events + pouring_events + heat_cycles)
├── sync/
│   ├── api_client.py               # HMAC-SHA256 HTTP client
│   └── sync_manager.py             # 30s background sync thread
│
├── tools/                          # Standalone offline tools (NOT production pipeline)
│   ├── molten_flow_detector.py     # Standalone tapping detector (FFmpeg+CuPy)
│   ├── deslagging_detector.py      # Standalone deslagging detector (FFmpeg+CuPy)
│   ├── pyrometer_rod_detector.py   # Standalone pyrometer detector (direct TRT)
│   └── pouring_system.py           # Standalone pouring system (direct TRT+ByteTrack)
│
├── data/hicon.db                   # SQLite database
├── output/
│   ├── screenshots/                # JPEG event captures
│   ├── videos/                     # Inference recordings (DS branch output)
│   ├── csv/                        # Incremental pouring CSV
│   └── json/                       # Batch pouring JSON
└── logs/pipeline.log
```

## Models & nvinfer Config

| Model | nvinfer Config | GIE ID | Input | Output | Custom Parser |
|-------|---------------|--------|-------|--------|---------------|
| `best_pouring_hicon_v1_930` | `config_pouring_pgie.txt` | 1 | 640×640 FP16 | Standard YOLO `[x,y,w,h,conf,class]` | `libnvdsinfer_custom_impl_Yolo.so` (6-ch auto-detect) |
| `best_pyro_rod_v1` | `config_pyrometer_pgie.txt` | 2 | 1280×1280 FP16 | End-to-end `(batch,300,6)` `[x1,y1,x2,y2,conf,class_id]` | `libnvdsinfer_custom_impl_Yolo.so` (end-to-end path) |

**nvinfer config essentials:**
```ini
# config_pouring_pgie.txt
[property]
gpu-id=0
net-scale-factor=0.00392156862  # 1/255
model-engine-file=models/onnx/best_pouring_hicon_v1_930.engine
batch-size=1
network-mode=2                   # FP16
num-detected-classes=2           # ladle_mouth, trolley
process-mode=1                   # primary
custom-lib-path=custom_parsers/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
parse-bbox-func-name=NvDsInferParseYolo

[class-attrs-all]
pre-cluster-threshold=0.25
```

**Custom parser must handle two output formats:**
- 6-channel: `[x, y, w, h, conf, class]` → standard YOLO (pouring model)
- End-to-end: `(batch, 300, 6)` = `[x1, y1, x2, y2, conf, class_id]` → YOLO26 (pyrometer model)
- Auto-detect by output tensor shape in the parser

**TensorRT engines must be built ON the target Jetson** — not portable across GPU architectures.
```bash
# Export on Jetson (generates .onnx then .engine via nvinfer on first run)
yolo export model=best_pouring_hicon_v1_930.pt format=onnx imgsz=640
yolo export model=best_pyro_rod_v1.pt format=onnx imgsz=1280
# nvinfer auto-builds .engine from .onnx on first pipeline launch
```

## Detection Logic in Probe Callbacks

### Brightness Processor (tapping + deslagging + spectro) — CPU NumPy
Runs in `osd_sink_pad` probe or a dedicated `tee` branch probe on Stream 0.
```python
# PATTERN: get frame → crop ROI → threshold → ratio → state machine
n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
frame = np.array(n_frame, copy=True, order='C')  # RGBA uint8

# Extract approximate luminance from RGBA (R channel or grayscale convert)
gray = frame[:, :, 0]  # or cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

# Tapping: quad ROI mask, Y > 180, ratio ≥ 0.80 start / < 0.60 end
# Deslagging: polygon ROI mask, Y > 250, ratio ≥ 0.01 start / < 0.01 end
# Spectro: polygon ROI mask, Y > 250, ratio ≥ 0.03 start / < 0.03 end
#          discard if ratio crosses max_white_ratio (0.20) during ACTIVE

pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)  # MANDATORY on Jetson
```
**CRITICAL:** Always call `unmap_nvds_buf_surface()` after `get_nvds_buf_surface()` on Jetson. Missing this causes memory leak that kills the pipeline within minutes.

**CPU budget:** ~2–4 ms/frame for both tapping + deslagging (well within 33ms @ 30FPS).

### Pouring Processor — nvinfer detections
Runs in probe after `nvinfer` + `nvtracker` on Stream 0. Reads `NvDsObjectMeta` from batch meta.
```python
# PATTERN: iterate detections → apply business logic
l_obj = frame_meta.obj_meta_list
while l_obj is not None:
    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
    if obj_meta.class_id == CLASS_LADLE_MOUTH and obj_meta.confidence >= 0.4:
        # session management, pour detection, mould counting
    elif obj_meta.class_id == CLASS_TROLLEY and obj_meta.confidence >= 0.25:
        # trolley tracking
    l_obj = l_obj.next
```

**Confidence thresholds applied in probe** (not in nvinfer config) for finer control:
- ladle_mouth ≥ 0.4, trolley ≥ 0.25

**Pouring sub-systems (same logic as standalone, now in probe):**
1. **Session manager:** mouth center inside trolley bbox (top-expanded by EDGE_EXPAND_PX) ≥1.0s start; end after mouth-missing tolerance + sustained absence window
2. **Pour detector:** probe baseline 50px below mouth bottom, offsets `[(20,0),(30,0),(40,0)]`, HSV-V channel sampling, >230 (0.25s) start / <180 (1.0s) end, min 2.0s
3. **Mould counter:** anchor displacement >0.15, sustained ≥1.5s (38 frames @ 25fps), quadrant-based direction guard (Q1–Q4), re-arm baseline gate (0.5s), split cooldown (1.5s), clustering (`R_CLUSTER=0.08`, `R_MERGE=0.05`) plus `MIN_CLUSTER_POUR_S` filter
4. **Cycle timeout:** locked trolley cycle reset after 300s mouth absence

**Mould split direction consistency:**
- Tracks displacement quadrant: Q1 (right-up), Q2 (left-up), Q3 (left-down), Q4 (right-down)
- Hold counter increments while displacement stays in same quadrant
- Resets only on true direction reversal (quadrant change), NOT on dominant-axis oscillation
- This allows diagonal trolley movements (where dx≈dy causes axis flipping) to accumulate smoothly

### Pyrometer Processor — nvinfer detections (custom parser)
Runs in probe after `nvinfer` on Stream 1. Custom parser converts YOLO26 end-to-end output to `NvDsObjectMeta`.
- Filter: confidence ≥ 0.25
- Zone check: bbox top-left `(x1,y1)` AND bottom-center `((x1+x2)/2, y2)` inside polygon `[(303,151),(303,568),(724,568),(724,151)]`
- Temporal: 10 consecutive in-zone frames → EVENT START, 10 consecutive absent → EVENT END

## Coding Conventions
- **Single process:** All probe callbacks run in one Python process — keep probes fast (<10ms)
- **NumPy for brightness:** Use `pyds.get_nvds_buf_surface()` → NumPy on CPU (CuPy not available on Jetson DeepStream)
- **Always unmap:** Every `get_nvds_buf_surface()` call MUST have matching `unmap_nvds_buf_surface()` on Jetson
- **nvinfer for ML:** Use DeepStream `nvinfer` element — do NOT use direct `trt.Runtime` in pipeline code
- **Custom parser:** All YOLO output parsing in `libnvdsinfer_custom_impl_Yolo.so` (C++) — auto-detects format by output shape
- **Confidence in probes:** Apply per-class confidence thresholds in Python probes, set `pre-cluster-threshold` low in nvinfer config
- **State machines:** Frame-counter based (not timer-based) — count consecutive frames meeting condition
- **Timestamps:** ISO8601 for storage, HH:MM:SS for durations
- **Logging:** Python `logging`, INFO default, DEBUG for dev; GStreamer debug via `GST_DEBUG=3`
- **Error handling:** Wrap each probe callback in try/except — never let an exception crash the pipeline

## Building the Custom Parser
```bash
cd custom_parsers/nvdsinfer_custom_impl_Yolo/
# Requires: CUDA, TensorRT headers (from JetPack)
export CUDA_VER=12.6
make
# Output: libnvdsinfer_custom_impl_Yolo.so
# Referenced by nvinfer config: custom-lib-path=...
```

## Environment Setup
```bash
# JetPack 6.2.1 (L4T R36.4.7) provides: DeepStream 7.1, TensorRT, CUDA 12.6
# DeepStream Python bindings
pip3 install pyds                  # or from JetPack DeepStream package
pip3 install ultralytics           # YOLO export tooling (dev/export only)
pip3 install numpy                 # Frame processing in probes (pre-installed)
# Do NOT install CuPy for pipeline code — it's unused in DeepStream on Jetson
# Do NOT pip install tensorrt — use JetPack system TensorRT
```

## Running
```bash
# Production
python3 hicon_pipeline.py

# With debug logging
GST_DEBUG=3 python3 hicon_pipeline.py

# Monitor resources
tegrastats --interval 1000

# Service (systemd)
sudo systemctl start hicon-vision
sudo systemctl status hicon-vision
journalctl -u hicon-vision -f
```

## Testing & Calibration
```bash
# Offline standalone tools (use for threshold tuning on recorded video)
# These use FFmpeg+CuPy directly — NOT DeepStream
python3 tools/molten_flow_detector.py --input test_tapping.mp4 --output results/
python3 tools/deslagging_detector.py --input test_deslagging.mp4 --output results/
python3 tools/pyrometer_rod_detector.py --input test_pyro.mp4 --output results/
python3 tools/pouring_system.py --input test_pouring.mp4 --output results/

# Pipeline test with local video file (replace RTSP with filesrc)
python3 hicon_pipeline.py --source0 test_process.mp4 --source1 test_pyro.mp4
```

## What NOT to Change (Site-Calibrated)
- ROI polygons/quadrilaterals in `zones.json` — calibrated per camera installation
- Brightness thresholds (Y>180 tapping, Y>250 deslagging) — tuned for specific furnace lighting
- White_ratio thresholds and frame counts — tuned to avoid false triggers at site
- Spectro thresholds (`start/end ratio`, `max_white_ratio`) — tuned for false-positive rejection
- Zone polygon for pyrometer `[(303,151),(303,568),(724,568),(724,151)]` — calibrated to camera angle
- HMAC secret — stays in `.env`, never in code
- nvinfer `model-engine-file` paths — engines are device-specific

## RTSP Resilience (DeepStream built-in)
- `rtspsrc` auto-reconnects on stream drop
- Pipeline stall detection: bus error messages → per-source restart
- Full pipeline watchdog: no frames > 10 min → systemd restart
- Recording: post-OSD tee branch managed by `RecordingManager` for inference video capture

## Camera Hardware (Updated 2026-05-04)

**Current cameras:** 3× **Hikvision DS-2CD2043G2-LI2U** (4MP ColorVu bullet, ISAPI)

| Stream | IP | Role | Firmware | Serial | MAC |
|--------|-----|------|----------|--------|-----|
| 0 | 192.168.28.119 | Process (POURING) | V5.7.23 (build 260320) | ...FW8319581 | bc:29:78:85:8a:85 |
| 1 | 192.168.28.172 | Pyrometer (FURNACE) | V5.7.23 (build 260320) | ...FR7129271 | bc:29:78:53:27:40 |
| 2 | 192.168.28.174 | Pouring2 | V5.7.23 (build 260320) | ...FR7128559 | bc:29:78:53:24:78 |

- **Credentials:** `admin` / `india@789` (URL-encoded: `india%40789`)
- **RTSP URL (sub-stream):** `rtsp://admin:india%40789@{IP}:554/Streaming/Channels/102`
- **RTSP URL (main-stream):** `rtsp://admin:india%40789@{IP}:554/Streaming/Channels/101`
- **Stream 0 uses main-stream (Channels/101):** H.265, 2688×1520 @ 25fps — mux scaled to 1600×900
- **Streams 1 & 2 use sub-stream (Channels/102):** H.265, 1280×720 @ 25fps
- **ONVIF:** Disabled in camera settings (returns 404)
- **ISAPI:** Available on HTTP :80 and HTTPS :443 on all cameras

**Stream configs (confirmed via ISAPI 2026-05-04):**
- Stream 0 CH101 (192.168.28.119): 2688×1520, 3072 kbps VBR (avg 1536), **audio OFF**, **SmartCodec OFF** — CH102 tested and showed identical reconnect delays; bottleneck is camera-side reconnect speed, not resolution
- Stream 1 CH102 (192.168.28.172): 1280×720, 320 kbps VBR, audio ON, SmartCodec off
- Stream 2 CH102 (192.168.28.174): 1280×720, 512 kbps VBR, audio OFF, SmartCodec off

**Previous cameras (replaced 2026-03-20):** 3× CP Plus (Dahua OEM):
- 192.168.28.155: CP-UNC-TC41L5C-VMD-LQ (fw 2.860.00AT001.0.R)
- 192.168.28.152: CP-UNC-TA41L3C-D-LQ (fw 2.860.00AT002.0.R)
- 192.168.28.162: CP-UNC-TC41L5C-VMD-LQ (fw 2.860.00AT001.0.R)
- CP Plus cameras had firmware bug: TCP RTSP sessions killed every ~3-5 min
- Segment buffer workaround was built but is now **disabled** (not needed with Hikvision cameras)

**NVR:** CP Plus NVR at 192.168.28.6 (password `NVR@321#`) — tested and **rejected** for relay
(March 19: 12-min soak showed NVR relay is worse than direct camera connection)

**RTSP drop root cause (confirmed 2026-05-04):** Drops were caused by `gst_builder.py` silently
overriding stream 0's nvurisrcbin transport from `tcp` → `auto` (UDP multi). UDP has no
retransmission — inference stalls caused permanent packet loss and session death.
Fix: `HICON_STREAM_0_NVURISRCBIN_RESPECT_CONFIGURED_PROTOCOL=true` in `.env`.
Confirmed by 3-soak: 0 outages in 330s with TCP vs drops every ~300s with UDP.

**Stream 0 five-minute cycle ROOT CAUSE (found 2026-06-12):** The residual ~298.5s drop cycle
(persisting after the transport fix, ~167 outages/day) was caused by camera 0's default gateway
being set to **192.168.28.1, which does not exist** (ARP never resolves). The Hikvision firmware's
periodic gateway health check failed forever; its recovery action bounced the camera's
network/streaming stack every ~298.5s, silently killing all RTSP sessions (camera OS never
rebooted). Identical cameras 1&2 point at 192.168.27.1 (exists) — never dropped. Also explains
why the CP Plus → Hikvision camera replacement didn't help (same position, same dead gateway).
**Fix:** gateway changed to 192.168.28.8 (NVR-1, alive, answers ARP) via ISAPI + camera reboot.
The gateway is never routed through — it only needs to answer ARP. Do NOT set camera 0's gateway
back to 28.1. Full evidence: 2026-06-12 addendum in `ai_vision/docs/rtsp_stream0_investigation.md`.

**Segment buffer code** (legacy, disabled): Still in codebase at `pipeline/segment_buffer_helper.py`
and `gst_builder.py` segment buffer paths. Available if future cameras exhibit similar firmware drops.

**Investigation doc:** `ai_vision/docs/rtsp_stream0_investigation.md`

## Cloud Sync (AGNI API)
- **POST /api/v1/melting** — tapping, deslagging, spectro events
- **POST /api/v1/pouring** — pouring sessions with mould counts
- Auth: HMAC-SHA256 on request body → `X-HMAC-Signature` header
- Retry: up to 5× on failure, buffer in SQLite (`synced=0`)
- Cycle: 30-second background sync thread
- Cleanup: 7-day retention, auto-delete on INSERT

## Network & Camera Operations
- **Limit login attempts to 3 per device** before stopping — CP Plus and Hikvision NVR lockouts cause 30+ min delays
- Always validate one known-good stream first before bulk-testing credentials
- If auth fails twice, stop and ask the user — never brute-force RTSP URLs or credentials

## Code Quality
- Before creating skill files (`.claude/skills/*/SKILL.md`) or config YAML, read existing working examples first to confirm supported attributes
- Reference `.claude/skills/document-codebase/SKILL.md` for valid frontmatter schema
- Do NOT invent frontmatter keys — unsupported attributes silently break skills with no error message

## Analysis & Debugging
- When proposing threshold-based detection logic (pour clustering, brightness ratios, FPS watchdog), always state explicitly whether the threshold applies **per individual event** or **to an aggregate total**
- Walk through 2–3 concrete data examples before writing any detection code
- e.g. "trigger fires when individual mould pour duration > X" vs "trigger fires when total session pour duration > X" produce very different mould counts


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
