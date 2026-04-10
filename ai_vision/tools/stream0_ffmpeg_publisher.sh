#!/usr/bin/env bash
# Continuously reads from Hikvision camera and publishes to MediaMTX.
# Reconnects immediately on camera drop — much faster than nvurisrcbin's 10s interval.
set -a
source /home/hicon/hicon/ai_vision/.env
set +a

CAM_URL="${HICON_STREAM0_CAMERA_URL:-rtsp://admin:india%40789@192.168.28.119:554/Streaming/Channels/102}"
MTX_URL="rtsp://127.0.0.1:8554/stream0"

echo "[stream0-publisher] Starting — camera: $CAM_URL"

while true; do
    echo "[stream0-publisher] Connecting to camera"
    ffmpeg -hide_banner -loglevel warning \
        -rtsp_transport tcp \
        -i "$CAM_URL" \
        -c copy \
        -f rtsp \
        -rtsp_transport tcp \
        "$MTX_URL"
    echo "[stream0-publisher] Disconnected — reconnecting in 1s"
    sleep 1
done
