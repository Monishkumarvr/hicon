#!/bin/bash
# l2_blackout_autopsy.sh — settle WHAT the ~5-min all-camera blackout is at layer 2.
#
# Evidence so far (2026-07-21, no-sudo probes):
#   - All 3 cameras + the NVR (28.8, different vendor) lose ICMP at the SAME second, ~every few
#     min, ~30s each — a COMMON event, not a per-camera firmware timer.
#   - During each blackout the Jetson NIC rx_pps DOUBLES and multicast jumps ~10x (packets flood
#     IN, do not go silent) — the fingerprint of a spanning-tree topology change (MAC-flush ->
#     unknown-unicast flooding) with ~30s forward-delay reconvergence, i.e. a flapping switch port.
#
# This confirms/denies that and finds the source. REQUIRES sudo (promiscuous capture).
# Run during work hours (storm active). Default 7 min ≈ 1-2 blackout cycles.
#
#   sudo ./l2_blackout_autopsy.sh [seconds]
#
# Reads the wire for: (1) STP/RSTP/PVST BPDUs (decoded, to see the Topology-Change flag + which
# bridge/MAC sets it), (2) multicast/broadcast talkers (storm alternative), while (3) logging the
# exact blackout windows by pinging all cameras + NVR at 2 Hz. Then correlates them and prints a verdict.
set -u
IFACE="${HICON_CAP_IFACE:-enP8p1s0}"
DUR="${1:-420}"
OUT="${HICON_CAP_DIR:-/tmp/hicon_l2}"; mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S)
STP="$OUT/stp_$TS.log"; BLK="$OUT/blackout_$TS.log"; MC="$OUT/mcast_$TS.log"
CAMS="192.168.28.119 192.168.28.172 192.168.28.174 192.168.28.8"

if [ "$(id -u)" != 0 ]; then echo "must run with sudo (raw capture)"; exit 1; fi
command -v tcpdump >/dev/null || { echo "tcpdump not found"; exit 1; }

echo "[l2-autopsy] ${DUR}s on $IFACE   STP->$STP  blackouts->$BLK  mcast->$MC"

# 1) STP/RSTP + Cisco PVST+ BPDUs, decoded + timestamped (src MAC = field 3 with -e -tttt)
timeout "$DUR" tcpdump -ni "$IFACE" -e -tttt -vvv \
  'ether dst 01:80:c2:00:00:00 or ether dst 01:00:0c:cc:cc:cd' > "$STP" 2>/dev/null &
# 2) multicast/broadcast (non-ARP) talkers — for the storm-vs-flood composition
timeout "$DUR" tcpdump -ni "$IFACE" -e -tttt '(multicast or broadcast) and not arp' > "$MC" 2>/dev/null &
# 3) blackout windows: 2 Hz ping to every camera + the NVR
( END=$((SECONDS+DUR))
  while [ $SECONDS -lt $END ]; do
    line="$(date '+%H:%M:%S')"; down=0
    for c in $CAMS; do
      if ping -n -c1 -W1 "$c" >/dev/null 2>&1; then r=U; else r=D; down=1; fi
      line="$line .${c##*.}:$r"
    done
    [ "$down" = 1 ] && echo "$line" >> "$BLK"
    sleep 0.5
  done ) &
wait

echo; echo "=================== RESULTS ==================="
nb=$(grep -c ':D' "$BLK" 2>/dev/null || echo 0)
echo "--- blackout samples (any target Down); total=$nb ---"; head -40 "$BLK" 2>/dev/null
echo
tc=$(grep -icE 'topology.change|TC( bit)?\b' "$STP" 2>/dev/null || echo 0)
nbpdu=$(grep -c . "$STP" 2>/dev/null || echo 0)
echo "--- STP: $nbpdu BPDUs captured, $tc with Topology-Change flag ---"
if [ "$nbpdu" = 0 ]; then
  echo "   (0 BPDUs — switch is BPDU-filtering the Jetson's port; infer from flood pattern instead)"
else
  grep -iE 'topology' "$STP" | head -20
  echo "   -- BPDU source bridges (ether src) --"; awk 'NF>3{print $3}' "$STP" | sort | uniq -c | sort -rn | head
fi
echo
echo "--- multicast/broadcast top talkers (src IP/proto) — a periodic storm would dominate here ---"
awk 'NF>4{print $3,$5,$6}' "$MC" 2>/dev/null | sed 's/,$//' | sort | uniq -c | sort -rn | head -15
echo
echo "=== VERDICT HINTS ==="
echo " * Topology-Change BPDUs clustered at the blackout times  => STP reconvergence (find the"
echo "   flapping port on that source bridge; fix: RSTP + portfast/edge + BPDU-guard, or the bad device)."
echo " * A single multicast/broadcast src dominating during blackouts => storm (kill that talker)."
echo " * 0 BPDUs + rx flood already measured => STP with edge BPDU-filtering; get switch port-flap"
echo "   / topology-change counters directly from switch admin."
