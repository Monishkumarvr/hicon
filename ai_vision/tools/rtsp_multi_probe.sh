#!/bin/bash
# rtsp_multi_probe.sh — 6 concurrent control probes: {cam0,cam1,cam2} x {tcp,udp}.
#
# Purpose: side-by-side, pipeline-independent RTSP pullers. Two questions at once:
#   1) Does UDP survive past the ~300s TCP kill?  (udp session durations >> tcp -> UDP mitigation
#      viable; the pipeline can switch to select-rtp-protocol=0.)
#   2) Per-camera kill cadence with NO pipeline backpressure (isolates camera/network from us).
#
# No sudo required. Each probe logs "OPEN" / "CLOSED rc=.. duration_s=.." to its own file.
# Runs until Ctrl-C (or `kill` the process group). Analyze with: tools/outage_report.sh
#
# Env: HICON_CAM_USER/PASS override creds. HICON_PROBE_DIR overrides log dir.

set -u
U="${HICON_CAM_USER:-admin}"; P="${HICON_CAM_PASS:-india%40789}"   # NOTE: URL-encoded @ for RTSP URL
DIR="${HICON_PROBE_DIR:-/var/log/hicon}"
mkdir -p "$DIR" 2>/dev/null || DIR="$(dirname "$0")/../logs"; mkdir -p "$DIR"

# stream -> ip ; substream channel 102 (low bitrate) for all
declare -A CAM=( [0]=192.168.28.119 [1]=192.168.28.172 [2]=192.168.28.174 )

probe() {
  local sid=$1 ip=$2 transport=$3
  local log="$DIR/probe_cam${sid}_${transport}.log"
  local url="rtsp://${U}:${P}@${ip}:554/Streaming/Channels/102"
  echo "$(date '+%F %T') probe start cam$sid $transport $ip" >> "$log"
  while true; do
    local start; start=$(date +%s)
    echo "$(date '+%F %T') session OPEN" >> "$log"
    ffmpeg -nostdin -loglevel error -rtsp_transport "$transport" -stimeout 10000000 \
      -i "$url" -f null - >> "$log" 2>&1
    local rc=$? end; end=$(date +%s)
    echo "$(date '+%F %T') session CLOSED rc=$rc duration_s=$((end-start))" >> "$log"
    sleep 2
  done
}

echo "starting 6 probes -> $DIR/probe_cam{0,1,2}_{tcp,udp}.log  (Ctrl-C to stop)"
pids=()
for sid in 0 1 2; do
  probe "$sid" "${CAM[$sid]}" tcp & pids+=($!)
  probe "$sid" "${CAM[$sid]}" udp & pids+=($!)
done
trap 'echo; echo stopping; kill "${pids[@]}" 2>/dev/null; exit 0' INT TERM
wait
