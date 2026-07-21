#!/bin/bash
# camera_net_normalize.sh — safely change a Hikvision camera's DefaultGateway and/or DNS via ISAPI.
#
# Why: 2026-07-20 investigation found the RTSP "excess" bounces on cam1/cam2 track their network
# config vs the quiet cam0 (cam0: gateway 192.168.28.8 + no DNS). This tool normalizes a camera
# toward that known-good config, one variable at a time, and records a rollback file first.
#
# It ONLY rewrites the unique gateway/DNS values in the fetched XML (never the camera's own IP),
# shows a diff, and requires --apply to actually PUT. Reboot is separate (--reboot) because the
# firmware needs a reboot to apply network changes (~40-60s offline).
#
# Usage:
#   ./camera_net_normalize.sh <camIP> [--gateway <ip>] [--dns-clear] [--apply] [--reboot]
# Examples (the Track-1 split test):
#   cam1 gateway fix:  ./camera_net_normalize.sh 192.168.28.172 --gateway 192.168.28.8 --apply --reboot
#   cam2 DNS clear:    ./camera_net_normalize.sh 192.168.28.174 --dns-clear --apply --reboot
# Rollback (restore the saved original, then reboot):
#   ./camera_net_normalize.sh <camIP> --restore --apply --reboot
#
# Dry run (no --apply) just prints the diff. Credentials via env or defaults below.

set -u
USER_CRED="${HICON_CAM_USER:-admin}"
PASS_CRED="${HICON_CAM_PASS:-india@789}"
OUTDIR="${HICON_CAM_BACKUP_DIR:-$(dirname "$0")/../../.cam_net_backups}"
EP="System/Network/interfaces/1/ipAddress"

CAM="${1:-}"; shift || true
[ -z "$CAM" ] && { echo "usage: $0 <camIP> [--gateway ip] [--dns-clear] [--restore] [--apply] [--reboot]"; exit 2; }

NEW_GW=""; DNS_CLEAR=0; APPLY=0; REBOOT=0; RESTORE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gateway) NEW_GW="$2"; shift 2;;
    --dns-clear) DNS_CLEAR=1; shift;;
    --restore) RESTORE=1; shift;;
    --apply) APPLY=1; shift;;
    --reboot) REBOOT=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

mkdir -p "$OUTDIR"
CURL="curl -s --digest -u ${USER_CRED}:${PASS_CRED} --max-time 10"
STAMP=$(date +%Y%m%d_%H%M%S)
ORIG="$OUTDIR/${CAM}.ipAddress.${STAMP}.orig.xml"
NEWF="$OUTDIR/${CAM}.ipAddress.${STAMP}.new.xml"

echo "== fetching current config from $CAM =="
# The camera itself bounces its net stack ~every 300s (10-65s blackout), so a single GET can
# land in a dead window. Retry across it. Camera ISAPI also returns CRLF line endings; strip \r
# so awk-extracted values don't carry a trailing CR that breaks the sed match (and garbles the
# display). XML body is whitespace-insensitive, so an LF-only PUT is accepted.
fetch_ok=0
for attempt in $(seq 1 10); do
  $CURL "http://$CAM/ISAPI/$EP" > "$ORIG" 2>/dev/null
  sed -i 's/\r$//' "$ORIG"
  if grep -q '<ipAddress>' "$ORIG"; then fetch_ok=1; break; fi
  echo "  fetch attempt $attempt failed (camera may be mid-bounce) — retrying in 6s..."
  sleep 6
done
if [ "$fetch_ok" != 1 ]; then echo "FAILED to fetch after 10 tries (auth? reachability?)."; exit 1; fi
SELF_IP=$(grep -oE '<ipAddress>[0-9.]+</ipAddress>' "$ORIG" | head -1 | grep -oE '[0-9.]+')
echo "  camera self IP: $SELF_IP   (saved rollback: $ORIG)"

if [ "$RESTORE" = 1 ]; then
  LAST=$(ls -1t "$OUTDIR/${CAM}.ipAddress."*.orig.xml 2>/dev/null | grep -v "$STAMP" | head -1)
  [ -z "$LAST" ] && { echo "no prior backup to restore"; exit 1; }
  echo "  restoring from: $LAST"; cp "$LAST" "$NEWF"
else
  cp "$ORIG" "$NEWF"
  if [ -n "$NEW_GW" ]; then
    OLD_GW=$(awk '/<DefaultGateway>/{f=1} f&&/<ipAddress>/{gsub(/<[^>]*>/,""); gsub(/ /,""); print; exit}' "$ORIG")
    [ -z "$OLD_GW" ] && { echo "could not parse gateway"; exit 1; }
    echo "  gateway: $OLD_GW -> $NEW_GW"
    sed -i "s#<ipAddress>${OLD_GW}</ipAddress>#<ipAddress>${NEW_GW}</ipAddress>#" "$NEWF"
  fi
  if [ "$DNS_CLEAR" = 1 ]; then
    for d in $(awk '/<(Primary|Secondary)DNS>/{f=1} f&&/<ipAddress>/{gsub(/<[^>]*>/,""); gsub(/ /,""); print; f=0}' "$ORIG"); do
      [ "$d" = "0.0.0.0" ] && continue
      echo "  dns: $d -> 0.0.0.0"
      sed -i "s#<ipAddress>${d}</ipAddress>#<ipAddress>0.0.0.0</ipAddress>#" "$NEWF"
    done
  fi
fi

# Safety: camera's own IP must be unchanged
if [ "$(grep -c "<ipAddress>${SELF_IP}</ipAddress>" "$NEWF")" -lt 1 ]; then
  echo "ABORT: camera self IP $SELF_IP missing from new config — refusing to PUT"; exit 1
fi

echo "== proposed change =="
gw_of()  { awk '/<DefaultGateway>/{f=1} f&&/<ipAddress>/{gsub(/<[^>]*>|[ \t]/,""); print; exit}' "$1"; }
dns_of() { awk '/<(Primary|Secondary)DNS>/{f=1} f&&/<ipAddress>/{gsub(/<[^>]*>|[ \t]/,""); printf "%s ",$0; f=0} END{print ""}' "$1"; }
printf "  self IP : %s (UNCHANGED)\n" "$SELF_IP"
printf "  gateway : %s  ->  %s\n" "$(gw_of "$ORIG")" "$(gw_of "$NEWF")"
printf "  dns     : %s ->  %s\n" "$(dns_of "$ORIG")" "$(dns_of "$NEWF")"
if diff -q "$ORIG" "$NEWF" >/dev/null; then echo "  (no change — nothing to apply)"; exit 0; fi

if [ "$APPLY" != 1 ]; then echo "(dry run — re-run with --apply to PUT)"; exit 0; fi

echo "== PUT =="
put_ok=0
for attempt in $(seq 1 10); do
  RESP=$($CURL -X PUT -H "Content-Type: application/xml" --data-binary @"$NEWF" "http://$CAM/ISAPI/$EP" 2>/dev/null)
  if echo "$RESP" | grep -qE '<statusString>OK</statusString>|<statusCode>1</statusCode>'; then put_ok=1; break; fi
  echo "  PUT attempt $attempt failed (camera may be mid-bounce) — retrying in 6s... resp: $(echo "$RESP" | tr -d '\r\n' | head -c 120)"
  sleep 6
done
echo "$RESP" | grep -oE '<statusCode>[0-9]+</statusCode>|<statusString>[^<]*</statusString>|<subStatusCode>[^<]*'
if [ "$put_ok" != 1 ]; then echo "  WARNING: PUT did not confirm OK after retries — NOT rebooting. Re-run or check manually."; exit 1; fi

if [ "$REBOOT" = 1 ]; then
  echo "== reboot (applies network change; ~40-60s offline) =="
  $CURL -X PUT "http://$CAM/ISAPI/System/reboot" 2>/dev/null | grep -oE '<statusString>[^<]*' || true
  echo "  waiting for $CAM to return..."
  for i in $(seq 1 30); do
    sleep 5
    if $CURL "http://$CAM/ISAPI/$EP" 2>/dev/null | grep -q '<ipAddress>'; then
      echo "  back online after ~$((i*5))s. New config:"
      $CURL "http://$CAM/ISAPI/$EP" 2>/dev/null | grep -E 'DefaultGateway|PrimaryDNS|SecondaryDNS' -A1 | grep ipAddress
      exit 0
    fi
  done
  echo "  WARNING: $CAM not back after 150s — check manually"
fi
