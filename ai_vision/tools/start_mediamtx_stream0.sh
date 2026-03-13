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

# Override MediaMTX path sources via environment variables
export MTX_PATHS_STREAM0_SOURCE="${HICON_CPPLUS_SOURCE_STREAM_0}"
if [[ "${HICON_USE_SEGMENT_BUFFER_0:-false}" == "true" ]]; then
  export MTX_PATHS_STREAM0_SOURCEONDEMAND="yes"
  echo "MediaMTX stream0 relay set to on-demand because HICON_USE_SEGMENT_BUFFER_0=true" >&2
fi

exec /usr/local/bin/mediamtx /home/hicon/hicon/ai_vision/configs/mediamtx.stream0.yml
