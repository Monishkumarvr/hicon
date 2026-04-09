# HiCon Pipeline - Configuration Trace Report
**Generated:** 2026-04-04  
**Status:** Analysis of current env/config loaded by systemd service

---

## Systemd Service Configuration
**File:** `hicon-vision.service`

### Environment Setup
```
User: hicon
WorkingDirectory: /home/hicon/hicon/ai_vision
LD_LIBRARY_PATH: /opt/nvidia/deepstream/deepstream-7.1/lib:/usr/local/cuda-12.6/lib64
GST_PLUGIN_PATH: /opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins
GST_DEBUG: 2
ExecStart: /usr/bin/python3 hicon_pipeline.py
```

✅ DeepStream 7.1 paths correctly configured  
✅ Python3 running from working directory  

---

## .env Configuration Loaded (ai_vision/.env)

### Stream 0 Analysis Pipeline Topology
```
HICON_STREAM_0_DECOUPLED_ANALYSIS_MODE           = true
HICON_STREAM_0_ANALYSIS_BRANCH_ENABLED           = true
HICON_STREAM_0_ANALYSIS_RGBA_ENABLED             = false
HICON_STREAM_0_ANALYSIS_CPP_PLUGIN_ENABLED       = true
HICON_STREAM_0_ANALYSIS_PROBE_ENABLED            = false  ⚠️ CRITICAL
```

### Stream Configuration
```
HICON_RTSP_STREAM_0                              = rtsp://admin:india%40789@192.168.28.119:554/Streaming/Channels/101
HICON_RTSP_STREAM_1                              = rtsp://admin:india%40789@192.168.27.253:554/Streaming/Channels/102
HICON_RTSP_STREAM_2                              = false (DISABLED)

HICON_STREAM_0_MUX_WIDTH                         = 1600
HICON_STREAM_0_MUX_HEIGHT                        = 900
HICON_STREAM_0_TRACKER_WIDTH                     = 800
HICON_STREAM_0_TRACKER_HEIGHT                    = 480
```

### Inference Recording
```
HICON_ENABLE_INFERENCE_VIDEO                     = true
HICON_INFERENCE_VIDEO_SCHEDULE                   = 11:00-14:00,19:00-24:00
HICON_INFERENCE_VIDEO_MAX_DURATION_S             = 3600
```

---

## Processor Initialization Decision Tree

### Stream 0: Pouring Processor
Binary check: `HICON_ENABLE_STREAM_0_POURING_PROCESSOR`
- **Default value:** `true` (not in .env, uses config.py default)
- **Status:** ✅ Initialized (line 1022 in hicon_pipeline.py)
- **Actually used in probe?** ❌ NO (see below)

### Stream 0: Brightness Processor
Binary check: `HICON_ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR`
- **Default value:** `true` (not in .env, uses config.py default)
- **Status:** ✅ Initialized (line 977 in hicon_pipeline.py)
- **Actually used in probe?** Conditional (see topology check below)

---

## Critical Topology Check (hicon_pipeline.py, line 910–911)

```python
safe_cuda_topology_ready = (
    config.STREAM_0_DECOUPLED_ANALYSIS_MODE        # true ✅
    and config.STREAM_0_ANALYSIS_BRANCH_ENABLED    # true ✅
    and config.STREAM_0_ANALYSIS_PROBE_ENABLED     # FALSE ❌
)
```

**Result:** `safe_cuda_topology_ready = FALSE`

This topology check gates:
1. C++ melting plugin loading (safe CUDA brightness)
2. MeltingAnalysisController initialization
3. **Probe callback execution** for Python pouring/brightness analysis

---

## What This Means: Which Paths Are Running?

### ✅ RUNNING (in systemd service) **— CORRECTED based on actual startup logs**
1. **Python Brightness Processor** — ✅ YES (CPU NumPy path)
   - Tracks: tapping, deslagging, spectro
   - Thresholds: Y>180 (tapping), Y>250 (deslagging/spectro)
   - Logging: Active frame-by-frame ratio monitoring
2. **Hybrid C++ Pouring Controller** — ✅ YES (C++ path for Stream 0 + 2)
   - Python pouring processor initialized but **bypassed** ("Python-only pouring processor bypassed")
   - Uses native C++ state plugin instead
   - Handles session management, pour detection, mould counting
3. **DeepStream nvinfer** — ✅ YES (GIE-1 & GIE-2 inference, tracking)
4. **Recording manager** — ✅ YES (post-OSD branch for inference video)
   - Schedule: 11:00-14:00, 19:00-24:00 (local time)
   - Max duration: 3600s per file
   - Retention: 3 days
5. **Cloud sync** — ✅ YES (every 600s to AGNI API)
   - URL: http://ai-bakend-v2.ap-south-1.elasticbeanstalk.com/api/v1
   - Customer: 1157
   - Batch size: 50 records

### ❌ NOT RUNNING (in systemd service) **— Also corrected**
1. **Python-Only Pouring Processor** — ❌ NO (bypassed in favor of C++ hybrid)
2. **C++ Melting Plugin (tapping inference)** — ❌ NO (CUDA path not available)
   - Safe topology check: `STREAM_0_ANALYSIS_PROBE_ENABLED=false` makes it unavailable
   - Fallback: Use Python CPU brightness instead ✅

---

## Why Python Path Is Disabled

The `.env` file has hardcoded:
```ini
HICON_STREAM_0_ANALYSIS_PROBE_ENABLED=false
```

This **intentionally disables** the Python probe callbacks. The pipeline:
1. ✅ Still decodes streams
2. ✅ Still runs inference (nvinfer)
3. ✅ Still records video (post-OSD)
4. ❌ Skips all Python probe analysis (pouring, brightness, tapping, deslagging, spectro)

---

## Recommendation

**Current state:** Pipeline is in "**Hybrid C++ Pouring + Python Brightness**" mode
- ✅ Python brightness processor is ACTIVE (CPU NumPy, monitoring tapping/deslagging/spectro)
- ✅ C++ hybrid pouring controller is ACTIVE (native state plugin for session/pour/mould analysis)
- ✅ Inference video is being recorded during scheduled times (11:00-14:00, 19:00-24:00)
- ✅ Cloud sync is sending structured events every 10 minutes
- ℹ️ Python-only pouring path is **initialized but intentionally bypassed** in favor of C++ hybrid

**Pouring analysis is happening via C++, not Python probe.** This means:
1. ✅ Pouring detection, mould counting, and session management ARE working
2. ℹ️ They're processed in the native C++ plugin, not in Python probe callbacks
3. ℹ️ Results are fed to Python as metadata through `PouringMetaReader`

**No errors detected.** Pipeline is running normally:
- Stream 0: 24.8–25.2 FPS ✅
- Stream 1: 24.6–25.2 FPS ✅
- Brightness processor logging regularly with no exceptions ✅

---

## Actual Startup Logs (April 2 Boot)

```
HiCon Pipeline Starting
Database initialized: /home/hicon/hicon/ai_vision/data/hicon.db
✓ HeatCycleManager initialized (ladle timeout: 300.0s)
C++ pouring meta readers initialized
BrightnessTracker[tapping] initialized: Y>180, start_ratio>=0.25 x20f, end_ratio<0.1 x25f
BrightnessTracker[deslagging] initialized: Y>250, start_ratio>=0.01 x10f, end_ratio<0.01 x15f
BrightnessTracker[spectro] initialized: Y>250, start_ratio>=0.03 x10f, end_ratio<0.03 x15f, max_ratio<0.2

Stream 0: CPU brightness processor initialized (NumPy)
Stream 0: Hybrid pouring controller initialized
Stream 0: Python-only pouring processor bypassed (hybrid C++ path active)

Stream 0 (CP Plus): decoupled analysis mode enabled
  (main path NV12 -> nvdsosd GPU, CPU analysis on leaky RGBA side branch)

RecordingManager initialized for stream 0
  target_fps=10.0, size=640x360
  schedule=11:00-14:00,19:00-24:00
  max_duration=3600s, retention=3d
```

---

## Actual Current State (April 4, 01:55 Running)

```
[FPS] Stream 0: 124-127 frames (24.8-25.2 fps)
[FPS] Stream 1: 123-126 frames (24.6-25.2 fps)

brightness_processor: [tapping] ratio=0.000 (need>=0.25) state=IDLE ✅ (actively monitoring)
brightness_processor logs every 1-2 seconds with no errors ✅

C++ pouring meta reader heartbeat: 
  meta_frames=250/250, session=False, mouth=False, probe=False
  trolley tracked, norm coordinates updated ✅
```

---

## To Verify Pouring Path (Python vs C++)

Check if Python pouring is bypassed:
```bash
sudo journalctl -u hicon-vision -b --no-pager | grep "bypassed\|hybrid"
# Expected: "Stream 0: Python-only pouring processor bypassed (hybrid C++ path active)"
```

Check if C++ metadata is being read:
```bash
sudo journalctl -u hicon-vision -f | grep "CPP-POURING\|meta_reader"
# Should see heartbeat messages with trolley tracking every 10s
```

Check if brightness is running:
```bash
sudo journalctl -u hicon-vision -f | grep "brightness_processor"
# Should see per-frame ratio updates (tapping/deslagging/spectro)
```

---

## Recommendation

**Current state:** Pipeline is in "**Hybrid C++ Pouring + Python Brightness**" mode
- ✅ Python brightness processor is ACTIVE (CPU NumPy, monitoring tapping/deslagging/spectro)
- ✅ C++ hybrid pouring controller is ACTIVE (native state plugin for session/pour/mould analysis)
- ✅ Inference video is being recorded during scheduled times (11:00-14:00, 19:00-24:00)
- ✅ Cloud sync is sending structured events every 10 minutes
- ℹ️ Python-only pouring path is **initialized but intentionally bypassed** in favor of C++ hybrid

**Pouring analysis is happening via C++, not Python probe.** This means:
1. ✅ Pouring detection, mould counting, and session management ARE working
2. ℹ️ They're processed in the native C++ plugin, not in Python probe callbacks
3. ℹ️ Results are fed to Python as metadata through `PouringMetaReader`

**No errors detected.** Pipeline is running normally:
- Stream 0: 24.8–25.2 FPS ✅
- Stream 1: 24.6–25.2 FPS ✅
- Brightness processor logging regularly with no exceptions ✅
