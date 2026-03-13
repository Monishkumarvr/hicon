#!/usr/bin/env bash
set -euo pipefail

STREAM0_URL="rtsp://127.0.0.1:8554/stream0"
check_stream() {
  local url="$1"
  /usr/bin/ffprobe \
    -v error \
    -rtsp_transport tcp \
    -select_streams v:0 \
    -show_entries stream=codec_type \
    -of default=noprint_wrappers=1:nokey=1 \
    "$url" 2>/dev/null | grep -qx "video"
}

# Wait for Stream 0 (required)
echo "Waiting for Stream 0 relay: $STREAM0_URL"
for _ in $(seq 1 45); do
  if check_stream "$STREAM0_URL"; then
    echo "Stream 0 relay is ready"
    break
  fi
  sleep 1
done

if ! check_stream "$STREAM0_URL"; then
  echo "Timed out waiting for Stream 0 relay: $STREAM0_URL" >&2
  exit 1
fi

exit 0
