#!/bin/bash
# RTSP control-puller probe (differential diagnosis for the Jul-15-onset outage storm).
#
# Pulls camera 1's low-bitrate sub-stream over TCP into a null sink, completely
# independent of the DeepStream pipeline, and logs every session start/end with
# duration. Comparing these session lifetimes against hicon-vision [RTSP-OUTAGE]
# timestamps discriminates:
#   - probe sessions die at the same instants  -> camera/network-side kill (external)
#   - probe survives while pipeline drops      -> pipeline-side problem
#
# This is the same method that isolated the June 2026 stream-0 gateway fault.
# Runs via hicon-rtsp-probe.service; log rotated by /etc/logrotate.d/hicon-rtsp-probe.

set -u

URL="${HICON_PROBE_URL:-rtsp://admin:india%40789@192.168.28.172:554/Streaming/Channels/102}"
LOG="${HICON_PROBE_LOG:-/var/log/hicon/rtsp_probe.log}"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"
}

log "probe starting (url host: $(echo "$URL" | sed -E 's#rtsp://[^@]*@##'))"

while true; do
    START=$(date +%s)
    log "session OPEN"
    # -stimeout: 10s socket timeout so a dead session exits instead of hanging.
    ffmpeg -nostdin -loglevel error -rtsp_transport tcp -stimeout 10000000 \
        -i "$URL" -f null - >> "$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    log "session CLOSED rc=$RC duration_s=$((END - START))"
    sleep 2
done
