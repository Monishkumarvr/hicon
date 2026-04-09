#!/bin/bash
# Build script for DeepStream Pouring Detection
set -e

APP_DIR="/workspace/apps/pouring"

cd "$APP_DIR"

if [ "$1" = "clean" ]; then
    make clean
    echo "Clean complete."
    exit 0
fi

echo "========================================"
echo "Building pouring detection app..."
echo "========================================"

make clean && make

echo "========================================"
echo "Build successful!"
echo "Run: cd $APP_DIR && ./deepstream-pouring-app /data/your_video.mp4"
echo "========================================"
