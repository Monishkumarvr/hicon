# DeepStream Pouring Detection — Usage Guide

## Overview

The DeepStream pipeline processes foundry CCTV footage to detect molten metal pouring events. It runs inside a Docker container using NVIDIA DeepStream 7.1 and outputs:

- **Annotated video** — bounding boxes, brightness probes, mould count overlay
- **JSON** — per-trolley summary with mould clusters, pour durations, and spatial positions
- **CSV** — one row per mould per trolley, for downstream analysis

Processing speed: ~75 FPS on RTX 4060 Laptop.

---

## Prerequisites

- Docker Desktop with NVIDIA GPU support
- Container `deepstream-hicon-pouring` running (see [Container Setup](#container-setup) below)
- App already built inside the container (`/workspace/apps/pouring/deepstream-pouring-app`)

---

## Running the Pipeline

Open **Command Prompt** (not PowerShell) and run:

```cmd
docker exec deepstream-hicon-pouring bash -c "/workspace/scripts/run_pouring.sh '/data/raw_videos/your_video.mp4' /workspace/output"
```

Replace the video path with the path as seen **inside the container** (the `Data/` folder is mounted at `/data/`).

### Examples

```cmd
docker exec deepstream-hicon-pouring bash -c "/workspace/scripts/run_pouring.sh '/data/raw_videos/test_clips/test_clip1_day2.mp4' /workspace/output"
```

```cmd
docker exec deepstream-hicon-pouring bash -c "/workspace/scripts/run_pouring.sh '/data/raw_videos/Melting_Track_Day1_clip2.mp4' /workspace/output"
```

Output files are written to `DeepStream_Pipeline/output/` on the Windows host (mapped from `/workspace/output/` in the container). Output filenames are derived from the input video name, e.g.:

| Input | Output |
|-------|--------|
| `test_clip1_day2.mp4` | `test_clip1_day2_annotated.mp4` |
| | `test_clip1_day2_annotated.json` |
| | `test_clip1_day2_annotated.csv` |

---

## Output Files

### CSV — `<name>_annotated.csv`

One row per detected mould pour, derived from clustering results:

```
trolley_id,mould_id,pouring_time_s
0,1,9.08
0,2,8.68
...
```

- `trolley_id` — YOLO tracker ID of the trolley
- `mould_id` — spatial cluster index (1-based, ordered by first detection)
- `pouring_time_s` — total pour duration for that mould cluster in seconds
- CSV is written at end-of-stream from the same final clustered summaries as JSON, so CSV and JSON per-mould times should match.

### JSON — `<name>_annotated.json`

Array of objects, one per trolley. Key fields:

```json
{
  "trolley_id": 0,
  "mould_count": 16,
  "session_start_frame": 11520,
  "session_end_frame": 12103,
  "mould_times": {
    "1": 9.08, "2": 8.68, ...
  },
  "per_mould_summary": [
    {
      "mould_cluster": 1,
      "num_pours": 2,
      "total_pour_s": 9.08,
      "centroid_rel": [0.67, 0.74]
    }, ...
  ],
  "pours": [
    {
      "mould_cluster": 1,
      "pour_idx": 1,
      "start_frame": 2991,
      "end_frame": 3062,
      "duration_s": 2.88,
      "rep_rel": [0.67, 0.71]
    }, ...
  ]
}
```

- `mould_times` and `per_mould_summary` always match — both come from clustering
- CSV is generated from the same clustered summary objects at EOS for consistency with JSON
- `centroid_rel` / `rep_rel` are normalised (0–1) coordinates within the trolley bounding box
- `pours` shows individual pour segments within each mould cluster (a cluster may have multiple segments if the ladle briefly lifted and returned)

### Overlay Semantics

- Overlay mould counts/times use the same split+cluster pipeline as final JSON/CSV (not the raw live counter)
- The currently active clustered mould is shown in green with `(pouring)`
- Cluster reassignment is recency-guarded: assignment can only go to the newest cluster or the immediately previous one

---

## Rebuilding the App

Only needed after changing the C++ source code. Run inside the container:

```cmd
docker exec deepstream-hicon-pouring bash -c "/workspace/scripts/build_pouring.sh"
```

Or clean rebuild:

```cmd
docker exec deepstream-hicon-pouring bash -c "cd /workspace/apps/pouring && make clean && make"
```

---

## Key Tuning Parameters

All parameters are hard-coded constants at the top of [apps/pouring/deepstream_pouring_app.cpp](apps/pouring/deepstream_pouring_app.cpp). Edit and rebuild to change them.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TH_ON` | 240 | HSV V-channel brightness threshold to detect pour start (NVDEC-compensated; notebook uses 230) |
| `TH_OFF` | 180 | Brightness threshold for pour end (60-unit hysteresis gap) |
| `MIN_POUR_DURATION_S` | 2.0s | Minimum pour length — shorter segments are discarded |
| `MOULD_SWITCH_HOLD_S` | 1.5s | Sustained ladle displacement required before counting a new mould |
| `D_SPLIT` | 0.15 | Normalised displacement threshold to trigger mould switch (~90px on a 600px trolley) |
| `SESSION_ENTER_S` | 1.0s | Trolley must be present this long before session opens |
| `SESSION_EXIT_S` | 1.5s | Trolley must be absent this long before session closes |
| `POUR_ON_S` | 0.25s | Frames above TH_ON needed to confirm pour start |
| `POUR_OFF_S` | 1.0s | Frames below TH_OFF needed to confirm pour end |
| `R_CLUSTER` | 0.08 | Spatial radius for clustering pour segments into the same mould |
| `CLUSTER_BACKTRACK_CID_GUARD` | 1 | Cluster recency guard (`cid >= latest_cid - 1`) to prevent jumps back to old mould IDs |
| `MIN_CLUSTER_POUR_S` | 1.5s | Minimum total duration of a mould cluster to be included in output |

---

## Container Setup

The container only needs to be created once. Start it with:

```cmd
docker start deepstream-hicon-pouring
```

If the container does not exist, create it:

```cmd
docker run -d --name deepstream-hicon-pouring --gpus all ^
  -v "D:\Projects\HI-CON\DeepStream_Pipeline:/workspace" ^
  -v "D:\Projects\HI-CON\Model_Outputs:/models" ^
  -v "D:\Projects\HI-CON\Data:/data" ^
  nvcr.io/nvidia/deepstream:7.1-samples-multiarch ^
  sleep infinity
```

After first creation, install ultralytics and build the app:

```cmd
docker exec deepstream-hicon-pouring pip install ultralytics onnx onnxslim
docker exec deepstream-hicon-pouring bash -c "/workspace/scripts/build_pouring.sh"
```

The TensorRT engine (`model_b1_gpu0_fp16.engine`) is built automatically on first run and cached at `/workspace/apps/pouring/`. Subsequent runs load the cached engine and start much faster.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `bad interpreter: /bin/bash^M` | CRLF line endings in script | Run: `docker exec deepstream-hicon-pouring sed -i 's/\r//' /workspace/scripts/run_pouring.sh` |
| `App not built` error | Binary missing | Run `build_pouring.sh` |
| 0 detections | Wrong ONNX format | ONNX must be exported with `export_yolo11.py` from the DeepStream-Yolo repo, not standard Ultralytics export |
| Engine build takes a long time on first run | TensorRT FP16 compilation | Normal — takes 2–5 min; subsequent runs use the cached `.engine` file |
| `NvBufSurfaceMap` error | dGPU memory type mismatch | Should not occur — pipeline uses `cudaMemcpy2D` for frame access |
