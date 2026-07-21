#!/bin/bash
# rtsp_storm_capture.sh — promiscuous packet capture to find the COMMON ~300s trigger.
#
# REQUIRES sudo (raw capture). Run during a PEAK storm hour (business hours, ~08:00-17:00).
#
# The earlier "nothing talks to the camera" conclusion was based on a 12-packet, ARP-only
# capture (host-filtered) — it could not see broadcast/multicast. This captures the FULL wire
# (no host filter) so a periodic broadcast/multicast/scan or an injected RST becomes visible.
#
# What to look for afterwards (analyze section below prints it):
#   - a periodic (~300s) broadcast/multicast just before kills: SADP (udp 8610), IGMP query,
#     mDNS/SSDP, ARP sweep, or a Sophos/NVR poll -> the common external trigger.
#   - the RST/FIN at each kill: source MAC bc:29:78:* (camera-side) vs c8:4f:86:* (Sophos) vs
#     other; and IP TTL (odd TTL => injected by a middlebox).
#
# Usage:
#   sudo ./rtsp_storm_capture.sh [seconds]     # default 1200s (20 min ~= 4 kill cycles)
# Output: /var/log/hicon/storm_cap/  (rotating pcaps) + a summary printed at the end.

set -u
IFACE="${HICON_CAP_IFACE:-enP8p1s0}"
DUR="${1:-1200}"
OUT="${HICON_CAP_DIR:-/var/log/hicon/storm_cap}"
mkdir -p "$OUT"
CAMS="192.168.28.119 192.168.28.172 192.168.28.174"

echo "capturing $DUR s on $IFACE (NO host filter — includes broadcast/multicast) -> $OUT"
# Ring buffer: 50MB x 40 files. Exclude nothing. -s0 full packets.
timeout "$DUR" tcpdump -i "$IFACE" -s0 -w "$OUT/storm_%Y%m%d_%H%M%S.pcap" -C 50 -W 40 -Z root \
  -n not port 22 2>/dev/null &
CAPPID=$!
echo "tcpdump pid $CAPPID; also snapshotting management traffic (non-RTSP) to/from cameras..."
# Second lightweight capture: everything to/from the cameras that is NOT the RTSP media port.
FILT=""; for c in $CAMS; do FILT="$FILT or host $c"; done; FILT="${FILT# or }"
timeout "$DUR" tcpdump -i "$IFACE" -s0 -w "$OUT/cam_mgmt_%Y%m%d_%H%M%S.pcap" \
  "($FILT) and not port 554 and not port 22" 2>/dev/null &
wait

echo; echo "=== quick triage of the full capture ==="
LAST=$(ls -1t "$OUT"/storm_*.pcap 2>/dev/null | head -1)
[ -z "$LAST" ] && { echo "no pcap produced"; exit 1; }
echo "newest pcap: $LAST"
echo "-- broadcast/multicast senders (candidate common triggers) --"
tcpdump -nr "$LAST" 'broadcast or multicast' 2>/dev/null | awk '{print $3}' | sort | uniq -c | sort -rn | head
echo "-- TCP RST packets seen (who resets RTSP :554) --"
tcpdump -ner "$LAST" 'tcp port 554 and tcp[tcpflags] & tcp-rst != 0' 2>/dev/null | head
echo "-- SADP(8610)/IGMP/mDNS/SSDP presence --"
tcpdump -nr "$LAST" 'udp port 8610 or igmp or udp port 5353 or udp port 1900' 2>/dev/null | awk '{print $1,$3,$4}' | head
echo
echo "NEXT: cross-reference kill timestamps from /var/log/hicon/rtsp_probe.log (session CLOSED)"
echo "against the pcap: tcpdump -nr $LAST 'host <cam>' | grep -C5 <kill-time>"
