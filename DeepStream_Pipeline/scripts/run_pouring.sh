#!/bin/bash
# Run inference script for DeepStream Pouring Detection
set -e

APP_DIR="/workspace/apps/pouring"
APP="$APP_DIR/deepstream-pouring-app"

if [ ! -f "$APP" ]; then
    echo "ERROR: App not built. Run ./scripts/build_pouring.sh first"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: ./run_pouring.sh <video_path> [output_dir]"
    echo "Example: ./run_pouring.sh /data/Melting_Track_Day1_clip2.mp4"
    echo "         ./run_pouring.sh /data/video.mp4 /workspace/output"
    exit 1
fi

OUTPUT_DIR="${2:-/workspace/output}"

echo "========================================"
echo "DeepStream Pouring Detection"
echo "Input:  $1"
echo "Output: $OUTPUT_DIR"
echo "========================================"

cd "$APP_DIR"
./deepstream-pouring-app "$1" "$OUTPUT_DIR"
