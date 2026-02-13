# HiCon Edge AI Vision System

HiCon is a 2-camera DeepStream 7.1 pipeline for furnace monitoring on NVIDIA Jetson Orin Nano.

- Stream 0 (Process): pouring + tapping + deslagging + spectro
- Stream 1 (Pyrometer): rod insertion detection
- Local buffering: SQLite
- Cloud sync: AGNI API with HMAC authentication

## Quick Start

```bash
cd /home/hicon/hicon/ai_vision
pip3 install -r requirements.txt
python3 hicon_pipeline.py
```

## Required Environment Variables (`ai_vision/.env`)

```bash
HICON_CUSTOMER_ID=682
HICON_RTSP_STREAM_0=rtsp://<process-camera>
HICON_RTSP_STREAM_1=rtsp://<pyrometer-camera>
HICON_HMAC_SECRET=<secret>

# Optional
HICON_ENABLE_SYNC=false
HICON_ENABLE_INFERENCE_VIDEO=true
HICON_INFERENCE_VIDEO_FPS=10
```

## Main Components

- `ai_vision/hicon_pipeline.py`: pipeline bootstrap and probe wiring
- `ai_vision/processors/pouring_processor.py`: session/pour/mould logic
- `ai_vision/processors/brightness_processor.py`: tapping/deslagging/spectro ROI brightness logic
- `ai_vision/processors/pyrometer_processor.py`: pyrometer zone + temporal logic
- `ai_vision/state/heat_cycle_manager.py`: heat-cycle aggregation and lifecycle
- `ai_vision/pipeline/recording.py`: DS-native inference recording branch

## Validation

```bash
cd /home/hicon/hicon/ai_vision
python3 -m pytest tests -q
python3 hicon_pipeline.py
```

## Output Paths

- Screenshots: `ai_vision/output/screenshots/`
- Inference recordings: `ai_vision/output/videos/inference/`
- Database: `ai_vision/data/hicon.db`

## Notes

- `pyds.unmap_nvds_buf_surface()` is mandatory after frame extraction on Jetson.
- Inference video is recorded from a post-OSD tee branch.
