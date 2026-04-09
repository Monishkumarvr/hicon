#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env
set -a
source /home/hicon/hicon/ai_vision/.env
set +a

if [[ -z "${HICON_CPPLUS_SOURCE_STREAM_0:-}" ]]; then
  echo "MediaMTX stream0 source proxy disabled: HICON_CPPLUS_SOURCE_STREAM_0 is not set" >&2
fi

# Override MediaMTX path behavior via environment variables
eval "$(/usr/bin/python3 /home/hicon/hicon/ai_vision/tools/mediamtx_env.py)"

if [[ -n "${HICON_CPPLUS_SOURCE_STREAM_0:-}" && "${HICON_USE_SEGMENT_BUFFER_0:-false}" == "true" ]]; then
  echo "MediaMTX stream0 relay set to on-demand because HICON_USE_SEGMENT_BUFFER_0=true" >&2
fi

if [[ -n "${HICON_STREAM0_REMOTE_RELAY_URL:-}" ]]; then
  echo "MediaMTX stream0_overlay remote relay enabled" >&2
else
  echo "MediaMTX stream0_overlay remote relay disabled: HICON_STREAM0_REMOTE_RELAY_URL is empty" >&2
fi

exec /usr/local/bin/mediamtx /home/hicon/hicon/ai_vision/configs/mediamtx.stream0.yml
