#!/bin/bash
# outage_report.sh — per-stream RTSP outage counts + interval stats from pipeline.log.
# Use to measure BEFORE/AFTER each change (config normalize, UDP switch).
#
# Usage: ./outage_report.sh [YYYY-MM-DD]   (default: today)
set -u
DAY="${1:-$(date +%F)}"
LOGDIR="$(dirname "$0")/../logs"
SRC=$(ls "$LOGDIR"/pipeline.log "$LOGDIR"/pipeline.log.1.gz 2>/dev/null)

echo "=== RTSP outages for $DAY ==="
echo "-- recovered events per stream --"
zcat -f $SRC 2>/dev/null | grep 'RTSP-OUTAGE' | grep 'phase=recovered' | grep "$DAY" \
  | grep -oE 'stream=[0-9]' | sort | uniq -c

echo "-- outage duration_s per stream (mean/max) --"
for s in 0 1 2; do
  zcat -f $SRC 2>/dev/null | grep 'RTSP-OUTAGE' | grep 'phase=recovered' | grep "$DAY" | grep "stream=$s" \
    | grep -oE 'duration_s=[0-9.]+' | grep -oE '[0-9.]+' \
    | awk -v s="$s" '{n++;t+=$1; if($1>m)m=$1} END{if(n)printf "  stream %s: n=%d mean=%.1fs max=%.1fs\n",s,n,t/n,m; else print "  stream "s": none"}'
done

echo "-- interval (s) between outage starts per stream: min/median/max --"
for s in 0 1 2; do
  echo -n "  stream $s: "
  zcat -f $SRC 2>/dev/null | grep 'RTSP-OUTAGE' | grep 'phase=start' | grep "$DAY" | grep "stream=$s" \
    | grep -oE 'start=[0-9T:-]+' | sed 's/start=//' \
    | awk -F'T' '{split($2,a,":"); print a[1]*3600+a[2]*60+a[3]}' \
    | awk 'NR>1{d=$1-p; if(d>0&&d<3000)print d} {p=$1}' | sort -n \
    | awk '{v[NR]=$1} END{if(NR)printf "n=%d min=%d median=%d max=%d\n",NR,v[1],v[int(NR/2)+1],v[NR]; else print "none"}'
done

echo "-- hourly outage histogram (all streams) --"
zcat -f $SRC 2>/dev/null | grep 'RTSP-OUTAGE' | grep 'phase=recovered' | grep "$DAY" \
  | grep -oE "$DAY"'[T ][0-9]{2}' | grep -oE '[0-9]{2}$' | sort | uniq -c \
  | awk '{printf "  %02sh: %s\n",$2,$1}'
