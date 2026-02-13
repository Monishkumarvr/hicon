# TRSL Edge AI Vision System

Industrial safety monitoring and process automation system for furnace operations using NVIDIA DeepStream 7.1 on Jetson Orin.

## Quick Start

```bash
# Install Python dependencies
cd /home/trsl/trsl/ai_vision
pip3 install -r requirements.txt

# Run production pipeline
python3 ympl_pipeline.py
```

## System Overview

**Hardware:** NVIDIA Jetson Orin (ARM64, Compute Capability 8.7)
**Framework:** DeepStream 7.1 with parallel inference architecture
**Deployment:** Edge processing with cloud sync via HMAC-authenticated API

### Stream Configuration

- **Stream 0** (PPE + Heat Start Time): Person → Helmet → Pan detection
- **Stream 1** (Furnace Operations): Ladle detection → Tapping + Deslagging monitoring

### Features

- ✅ PPE violation detection (helmet compliance)
- ✅ Heat start time (HST) tracking via pan zone entry
- ✅ Tapping detection with brightness + height validation
- ✅ Deslagging monitoring with brightness thresholds
- ✅ SQLite edge buffer with 7-day rolling retention
- ✅ Cloud sync with retry logic and idempotent operations

## Recent Updates

### Tapping Detection Fix (Dec 30, 2024)

**Issue:** Tapping events not detected despite ladles being in zone.

**Root Cause:** Zone detection using wrong point (bbox center instead of bottom_center where ladle physically touches ground).

**Files Modified:**
1. [ai_vision/state_machines.py](ai_vision/state_machines.py) - TappingTracker uses `bottom_center` for zone detection
2. [ai_vision/processors/tapping_processor.py](ai_vision/processors/tapping_processor.py) - Added `bottom_center` calculation
3. [ai_vision/archive/diagnostics/tapping_diagnostic.py](ai_vision/archive/diagnostics/tapping_diagnostic.py) - Diagnostic tool (archived)

**Validation Results:**
- ✅ 10 tapping events detected in FURNACE_1 (where tapping occurs)
- ✅ 0 false positives in FURNACE_2 and FURNACE_3 (no tapping)
- ✅ State machine transitions: IDLE → IN_ZONE → TAPPING working correctly
- ✅ Thresholds verified correct (no changes needed)

**Technical Details:**
- Ladles sit on GROUND at furnace base
- Zones drawn around BASE area where ladles rest
- Using bbox center (middle height) placed detection point OUTSIDE zone
- Using bottom_center (y2 coordinate) correctly represents physical position

**Before:** `center = (x_center, y_center)` ❌
**After:** `bottom_center = (x_center, y2)` ✅

## Installation

### Prerequisites (JetPack 6.2+)

These are pre-installed on Jetson Orin via JetPack SDK:
- NVIDIA DeepStream 7.1+
- GStreamer 1.0
- CUDA Toolkit
- TensorRT
- PyTorch 2.4+
- pyds (DeepStream Python bindings)
- pygobject (GStreamer Python bindings)

### Python Dependencies

```bash
cd /home/trsl/trsl/ai_vision
pip3 install -r requirements.txt
```

**Core packages:**
- opencv-python (computer vision)
- numpy (array operations)
- requests (HTTP API client)
- python-dotenv (environment management)

**Optional profiling tools:**
```bash
pip3 install psutil nvidia-ml-py
```

## Configuration

Create `.env` file in `ai_vision/` directory:

```bash
# Required
TRSL_CUSTOMER_ID=451
TRSL_RTSP_STREAM_0=rtsp://192.168.1.100:554/stream1
TRSL_RTSP_STREAM_1=rtsp://192.168.1.101:554/stream1
TRSL_HMAC_SECRET=your-secret-key

# Optional (disable cloud sync for testing)
TRSL_ENABLE_SYNC=false
```

See [ai_vision/ENVIRONMENT_VARIABLES.md](ai_vision/ENVIRONMENT_VARIABLES.md) for complete configuration reference.

## Zone Configuration

Zone polygons and thresholds are defined in JSON files:

- [ai_vision/configs/zones.json](ai_vision/configs/zones.json) - Pan zones, ladle zones, deslagging zones, PPE zone
- Coordinates are pixel values for specific camera angles
- Thresholds calibrated for production conditions

**Ladle Zone Thresholds (Tapping Detection):**
- FURNACE_1: brightness >200, height <190px
- FURNACE_2: brightness >210, height <210px
- FURNACE_3: brightness >210, height <190px

## Testing

### Pipeline Validation

```bash
cd /home/trsl/trsl/ai_vision

# Comprehensive test (dual-stream, 60 seconds)
python3 test_pipeline_validation.py --test TC3 --duration 60

# Metadata routing validation
python3 test_pipeline_validation.py --test TC4 --duration 60

# Performance benchmark
python3 test_pipeline_validation.py --benchmark
```

### Model Testing (Ultralytics)

```bash
# Test models without DeepStream overhead
python3 test_detections.py --stream 0 --frames 100  # PPE stream
python3 test_detections.py --stream 1 --frames 100  # Furnace stream
```

### Performance Profiling

```bash
python3 test_single_model.py
# See ai_vision/PROFILING.md for metrics interpretation
```

## Architecture

### Parallel Inference Pipeline (DeepStream 7.1)

```
Stream 0 + Stream 1 → main_mux → tee ┬→ bypass → metamux.sink_0
                                      └→ demux ┬→ Stream 0 → Person → Helmet → Pan → Tracker → metamux.sink_1
                                               └→ Stream 1 → Ladle → Tracker → metamux.sink_2
                                                                                    ↓
                                                                                metamux → OSD → PROBE
```

**Performance:**
- Stream 0 runs 3 models (Person, Helmet, Pan) - no wasted inference
- Stream 1 runs 1 model (Ladle) - no wasted inference
- 60% reduction in inference operations vs sequential architecture
- ~40% FPS improvement

### Database Schema

**Edge Buffer (SQLite):**
- `safety_violations` - PPE violations with screenshots
- `agni_heats` - Heat cycle records (HST to completion)
- `pan_zone_entries` - Pan detection events
- `tapping_events` - Tapping start/end times
- `deslagging_events` - Brightness-based deslagging detection

**Retention:** 7-day rolling window, auto-cleanup on each insert

### Cloud Sync

- Background thread with 30-second interval
- HMAC-SHA256 authentication on request body
- Idempotent sync_id generation (hash of event metadata)
- Retry logic with max 5 attempts per record
- Screenshots sent as Base64-encoded JPEG in POST body

## Project Structure

```
ympl/
├── ai_vision/                  # Main application directory
│   ├── ympl_pipeline.py        # Production pipeline (721 lines)
│   ├── processors/             # Business logic processors
│   │   ├── ppe_processor.py    # PPE violation detection
│   │   ├── heat_processor.py   # Heat cycle tracking
│   │   ├── tapping_processor.py # Tapping detection
│   │   └── deslagging_processor.py # Deslagging monitoring
│   ├── pipeline/               # GStreamer pipeline construction
│   │   └── gst_builder.py      # DeepStream pipeline builder
│   ├── configs/                # Model and zone configs
│   │   ├── zones.json          # Unified zone configuration
│   │   ├── config_metamux.txt  # Stream-to-model routing
│   │   └── *_yolov11n.txt      # Model-specific configs
│   ├── models/                 # PyTorch .pt files
│   ├── onnx/                   # ONNX exported models
│   ├── engines/                # TensorRT engine files
│   ├── archive/                # Archived code
│   │   └── diagnostics/        # Tapping diagnostic tools
│   ├── requirements.txt        # Python dependencies
│   └── test_*.py               # Test suites
├── data/                       # SQLite database
└── screenshots/                # Violation screenshots
```

## Deployment

### Production Deployment Checklist

1. ✅ Install JetPack 6.2+ on Jetson Orin
2. ✅ Install Python dependencies: `pip3 install -r requirements.txt`
3. ✅ Configure environment variables (`.env` file)
4. ✅ Verify RTSP streams accessible: `ffplay rtsp://...`
5. ✅ Run test suite: `python3 test_pipeline_validation.py --test TC3 --duration 30`
6. ✅ Start production pipeline: `python3 ympl_pipeline.py`

### Verification

```bash
# Check pipeline status
tail -f /home/trsl/trsl/ai_vision/logs/ympl_pipeline.log

# Query database
sqlite3 /home/trsl/trsl/data/trsl.db "SELECT * FROM tapping_events ORDER BY created_at DESC LIMIT 10;"

# Monitor GPU
tegrastats
```

## Troubleshooting

### Tapping Detection Issues

**Symptom:** 0 tapping events detected despite ladles in zone
**Solution:** Verify zone detection uses `bottom_center` (fixed in Dec 30, 2024 update)
**Validation:** Run diagnostic script (archived in `archive/diagnostics/`)

### RTSP Stream Issues

**Symptom:** Pipeline crashes after 1-2 seconds during decoder initialization
**Cause:** H.264 profile incompatibility (High profile vs Baseline/Main)
**Solution:** Configure RTSP source to send H.264 Baseline profile
**Fallback:** Pipeline auto-falls back to software decoder (avdec_h264) if hardware decoder fails

### Performance Issues

**Check:**
- GPU utilization: `tegrastats`
- Model batch sizes: main_mux=2, ppe_mux=1, furnace_mux=1
- Bypass branch linked first (timing reference requirement)
- Metadata routing: Stream 0 should have NO ladle detections, Stream 1 should have NO person/helmet/pan

## Documentation

- [ai_vision/ARCHITECTURE.md](ai_vision/ARCHITECTURE.md) - Complete architecture documentation
- [ai_vision/ENVIRONMENT_VARIABLES.md](ai_vision/ENVIRONMENT_VARIABLES.md) - Configuration reference
- [ai_vision/PROFILING.md](ai_vision/PROFILING.md) - Performance profiling guide
- [ai_vision/CLAUDE.md](ai_vision/CLAUDE.md) - Development guidelines for Claude Code

## License

Proprietary - TRSL Edge AI Vision System

## Support

For issues or questions:
- Check [ai_vision/ARCHITECTURE.md](ai_vision/ARCHITECTURE.md) for technical details
- Review test output logs in `ai_vision/test_output/`
- Consult diagnostic tools in `ai_vision/archive/diagnostics/`

---

**Last Updated:** December 30, 2024
**Version:** Production (DeepStream 7.1, Tapping Detection Fixed)
**Status:** ✅ Ready for Deployment
