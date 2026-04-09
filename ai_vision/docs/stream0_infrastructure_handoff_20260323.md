# Stream 0 Infrastructure Handoff - March 23, 2026

## Summary

Stream 0 instability is currently best explained as a camera or network-path fault on `192.168.28.119`, not an unresolved DeepStream keepalive bug and not overall 3-stream load.

The software baseline is intentionally unchanged for handoff reproducibility:
- Stream 0 uses regular `rtspsrc` (not `nvurisrcbin`)
- Stream 0 C++ pouring runs on the analysis branch, off the main display path
- Stream 2 remains enabled in the normal baseline

## Current Mapping

- Stream 0: `192.168.28.119`
- Stream 1: `192.168.27.253`
- Stream 2: `192.168.27.226`
- Stream 0 flags: `HICON_STREAM_0_DECOUPLED_ANALYSIS_MODE=true`, `HICON_USE_CPP_POURING_PLUGIN=true`, `HICON_USE_NVURISRCBIN_0=false`

## Verified Software Baseline

From the current service startup logs:
- Stream 0 starts on the regular RTSP path: `Stream 0: RTSP config protocol=tcp, latency=2000ms, drop-on-latency=True, buffer-mode=0(none), timeout=0us, tcp-timeout=60000000us, retry=20, do-retransmission=True`
- Stream 0 C++ plugin remains off the critical path: `Stream 0: C++ pouring plugin placed on analysis branch (off main path)`

## Evidence

### 1. Stream 0 still fails on regular `rtspsrc`

With the current baseline, Stream 0 still stalls and escalates to the 90-second watchdog restart.

Representative log window with Streams 0, 1, and 2 enabled:
- `2026-03-23 16:18:13 IST` - pipeline starts
- `2026-03-23 16:18:13 IST` - Stream 0 uses regular RTSP config
- `2026-03-23 16:18:13 IST` - Stream 2 also starts normally
- `2026-03-23 16:18:13 IST` - Stream 0 C++ pouring is placed on the analysis branch
- `2026-03-23 16:21:43 IST` - `Stream 0 at 0fps for 5s`
- `2026-03-23 16:23:08 IST` - `Stream 0 stale 90s - escalating to restart`

This shows the failure persists even after removing `nvurisrcbin` from Stream 0.

### 2. Disabling Stream 2 did not stop the failure

Representative log window with Stream 2 disabled:
- `2026-03-23 15:49:40 IST` - pipeline starts
- `2026-03-23 15:49:40 IST` - Stream 0 uses regular RTSP config
- no Stream 2 RTSP startup line appears in this run
- `2026-03-23 15:51:46 IST` - `Stream 0 at 0fps for 5s`
- `2026-03-23 15:53:11 IST` - `Stream 0 stale 90s - escalating to restart`

This weakens the theory that total 3-stream GPU/CPU load is the primary cause.

### 3. Stable cameras are on `192.168.27.x`; unstable camera is on `192.168.28.x`

Current subnet split:
- stable cameras observed in the normal pipeline mapping: `192.168.27.253` and `192.168.27.226`
- unstable Stream 0 camera: `192.168.28.119`

This makes the `192.168.28.x` path a strong suspect: camera, PoE, switch port, VLAN, cable, or subnet-specific policy.

### 4. Operator-run standalone `ffmpeg` test also failed on `192.168.28.119`

User-provided standalone evidence from March 23, 2026:
- a short standalone `ffmpeg` test to `192.168.28.119` held for 2 minutes and exited only because of the test timeout
- a longer standalone `ffmpeg` soak to `192.168.28.119` then degraded rapidly and exited early, before the requested 6-minute timeout

That result removes the DeepStream pipeline from the critical path and points back to the camera or network path itself.

## Requested Infrastructure Checks

Please check the following on the Stream 0 camera path (`192.168.28.119`):
- camera health and firmware stability
- RTSP session stability and session timeout behavior
- stream profile settings on the camera
- cable integrity and connector condition
- PoE stability or power budget issues
- switch port errors, flaps, or CRC counters
- VLAN or subnet policy differences between `192.168.28.x` and `192.168.27.x`
- any firewall, ACL, or session timeout rule specific to the `192.168.28.x` segment

## Post-Fix Validation

After infrastructure remediation:
1. Run a standalone `ffmpeg` soak against `192.168.28.119` for at least 10 to 15 minutes.
2. Pass criteria for standalone soak:
   - no throughput collapse
   - no disconnect before timeout
3. Restart `hicon-vision` without changing the current software baseline.
4. Run a 20-minute pipeline soak.
5. Pass criteria for the pipeline soak:
   - no `Pipeline warning from source0: No data from source since last 10 sec`
   - no `FPS-WATCHDOG` warnings or recoveries for Stream 0
   - no 90-second Stream 0 watchdog escalation
   - Stream 0 remains near 25 fps
   - Stream 0 C++ meta-reader heartbeats continue
   - Stream 2 remains unaffected

## Follow-Up Rule

If the standalone `ffmpeg` soak passes after infrastructure work but the pipeline still fails, reopen software investigation with the existing Stream 0 stage-isolation plan.
