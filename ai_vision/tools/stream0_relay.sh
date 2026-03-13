#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env
set -a
source /home/hicon/hicon/ai_vision/.env
set +a

if [[ -z "${HICON_CPPLUS_SOURCE_STREAM_0:-}" ]]; then
  echo "HICON_CPPLUS_SOURCE_STREAM_0 is not set" >&2
  exit 1
fi

RTSP_PORT="$1"
MTX_PATH="$2"

# Note: -reconnect flags only work with HTTP, not RTSP.
# For RTSP reconnect, MediaMTX runOnInitRestart handles process restart.
exec /usr/bin/ffmpeg \
  -nostdin \
  -loglevel warning \
  -rtsp_transport tcp \
  -fflags +genpts \
  -use_wallclock_as_timestamps 1 \
  -i "${HICON_CPPLUS_SOURCE_STREAM_0}" \
  -map 0:v:0 \
  -an \
  -c:v copy \
  -f rtsp \
  -rtsp_transport tcp \
  "rtsp://127.0.0.1:${RTSP_PORT}/${MTX_PATH}"
