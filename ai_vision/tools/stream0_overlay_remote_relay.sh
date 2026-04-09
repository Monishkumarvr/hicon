#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env
set -a
source /home/hicon/hicon/ai_vision/.env
set +a

exec /usr/bin/python3 /home/hicon/hicon/ai_vision/tools/stream0_overlay_remote_relay.py "$@"
