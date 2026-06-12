# Stream 0 Investigation Summary (March 6–11, 2026)

## Scope
This file summarizes the Stream 0 testing, analysis, and conclusions from **March 6, 2026 through March 11, 2026**. It is the canonical incident log for the later investigation phase, after the initial March 5 fixes had already been applied.

March 5 prehistory is included only where it is required to explain the March 6+ experiments.

## Starting State On March 6
By the start of March 6, 2026, Stream 0 was already known to be unstable while Stream 1 remained a stable control.

Pre-March-6 context that materially affected the later work:
- `proto=Onvif` had already been removed from the CP Plus URL, improving Stream 0 lifetime from about 2 minutes to about 4.5 minutes.
- audio pads had already been linked to a discard `fakesink` to stop unconsumed-audio RTCP starvation.
- watchdog and systemd exit handling had already been hardened so fatal conditions produced restarts instead of silent dead streams.
- Stream 1 (Hikvision) was already established as the stable reference path.

## Chronological Test Log

### March 6, 2026: Stage 1 RTSP Config Cleanup
**Goal**
Remove known `rtspsrc` misconfiguration and make transport handling explicit before continuing deeper root-cause work.

**Change/Test**
- replaced the old `RTSP_TIMEOUT_SEC` wiring with protocol-aware settings:
  - `RTSP_UDP_TIMEOUT_US`
  - `RTSP_TCP_TIMEOUT_US`
  - `RTSP_PORT_RETRY`
  - per-stream `RTSP_PROTOCOL_N`
- corrected the bug where `rtspsrc.timeout` was being fed a seconds-style value even though the property is in microseconds.
- made Stream 0 explicitly `tcp` and Stream 1 `auto` during the validation run.

**Observed Result**
- startup logging showed the corrected RTSP config was actually being applied.
- Stream 0 still dropped to `0.0 fps` before later `source0` timeout warnings.
- the failure timing changed, but the failure mode did not disappear.

**Finding**
The timeout-unit bug was real and worth fixing, but it was not the full cause of the March 6+ Stream 0 failure.

**Decision**
Keep the RTSP config cleanup permanently, then continue with Stream 0-only topology isolation.

---

### March 6, 2026: Stream 0 Source Isolation Queues
**Goal**
Test whether downstream backpressure on the CP Plus branch was propagating all the way back to `rtspsrc`.

**Change/Test**
- added Stream 0-only `srcq0` between `rtspsrc` and depay.
- added Stream 0-only `premuxq0` between caps and `mux_0`.
- both queues were leaky downstream with small bounded buffers.

**Observed Result**
- Stream 0 lived longer than earlier failing runs.
- Stream 0 still degraded to low FPS and then `0.0 fps`.
- RTSP timeout warnings still came after the FPS collapse, not before it.

**Finding**
Source-path backpressure was part of the problem, but isolating the source leg alone did not eliminate the stall.

**Decision**
Continue isolating the CP Plus path further downstream.

---

### March 6, 2026: Post-Mux Isolation Queues
**Goal**
Determine whether the stall was being introduced after `nvstreammux` rather than on the raw ingest side.

**Change/Test**
- added Stream 0-only post-mux queues around the `pgie_pouring` / tracker / OSD side of the branch.
- kept Stream 1 unchanged.

**Observed Result**
- runtime improved again, but Stream 0 still eventually collapsed to `0.0 fps`.
- Stream 1 remained stable.

**Finding**
The failure boundary moved further downstream, but the added isolation still did not identify a single blocking element.

**Decision**
Run A/B bypass tests on the Stream 0-only inference path.

---

### March 6, 2026: Stream 0 A/B Bypass Tests (`tracker_0`, `pgie_pouring`)
**Goal**
Identify whether `tracker_0` or `pgie_pouring` was the first hard failure point.

**Change/Test**
- first run: bypassed `tracker_0` only.
- second run: bypassed `pgie_pouring`, which also removed the tracker from the active Stream 0 path.

**Observed Result**
- `tracker_0` bypass made the failure happen sooner.
- `pgie_pouring` bypass extended runtime materially, but Stream 0 still stalled and later timed out.

**Finding**
- `tracker_0` was not the root cause.
- `pgie_pouring` contributed pressure, but it was not a necessary condition for failure.

**Decision**
Reduce the path further and identify the smallest reproducible failing topology.

---

### March 6, 2026: Stream 0 Diagnostic Topology Modes
**Goal**
Find the minimal post-mux Stream 0 path that could still reproduce the stall.

**Change/Test**
Three diagnostic modes were tested in sequence:
- `post-mux-only`
- `pre-OSD-only`
- `post-convert-only`

**Observed Result**
- `post-mux-only` was stable.
- `pre-OSD-only` still failed after several minutes.
- `post-convert-only` also failed after several minutes.

**Finding**
- `nvdsosd` was not required to reproduce the failure.
- forced RGBA caps were not required to reproduce the failure.
- the smallest reproduced failing stage was the Stream 0 post-mux `nvvideoconvert` path.

**Decision**
Try direct `nvvideoconvert` tuning before redesigning the topology again.

---

### March 6, 2026: `nvvideoconvert` Tuning Experiment
**Goal**
Check whether Stream 0 post-mux conversion was stalling because of converter configuration rather than because the element belonged on the path at all.

**Change/Test**
Applied Stream 0-only converter tuning on the post-mux `nvvideoconvert`:
- `compute-hw=GPU`
- `copy-hw=GPU`
- `output-buffers=32`
- `disable-passthrough=true`

**Observed Result**
- first valid tuned samples still failed.
- runtime moved around, but the same basic pattern remained: healthy FPS, then collapse, then later RTSP fallout.
- the tuning was not accepted as a fix.

**Finding**
The converter configuration influenced timing, but it did not eliminate the failure mode.

**Decision**
Stop tuning the existing hot path and redesign the Stream 0 path so CPU work and RGBA conversion were no longer on the main branch.

---

### March 6, 2026: Decoupled-Analysis Architecture
**Goal**
Remove the proven-bad pre-OSD RGBA conversion and CPU extraction from the Stream 0 main path while keeping business processing alive on a side branch.

**Change/Test**
- kept the Stream 0 main path on NV12 into `nvdsosd` GPU mode.
- introduced a tee after the inference/tracker path.
- moved CPU frame extraction and CPU processors onto a leaky RGBA analysis branch.
- split liveness tracking into:
  - main-path heartbeat
  - analysis-branch heartbeat
- disabled CPU-generated live display metadata for Stream 0 in decoupled mode.

**Observed Result**
- Stream 0 still failed.
- on the clean failing run, `main_age` and `analysis_age` went stale together.

**Finding**
The removed pre-OSD RGBA hot path and the CPU side branch were not the primary failure source. The shared path before the tee still collapsed as a unit.

**Decision**
Instrument the shared boundaries directly instead of continuing to guess.

---

### March 6, 2026: Stage-Boundary Probes (`mux/postmuxq/pgie/tracker`)
**Goal**
Find the first dead boundary inside the shared Stream 0 branch after decoupling.

**Change/Test**
Added Stream 0 stage probes to:
- `mux_0.src`
- `postmuxq0.src`
- `pgie_pouring.sink`
- `pgie_pouring.src`
- `tracker_0.sink`
- `tracker_0.src`

**Observed Result**
On clean failing runs:
- Stream 0 stayed healthy for several minutes.
- as failure began, all stage ages increased together.
- no single stage was clearly stale ahead of the others.

**Finding**
The shared branch was collapsing as a unit. The probes did not isolate one first-dead element inside `mux -> postmuxq -> pgie -> tracker`.

**Decision**
Move the next instrumentation boundary upstream of `mux_0`.

---

### March 6, 2026: Upstream Probes and PTS Delta Checks (`decoder/nvvidconv/caps/premuxq`)
**Goal**
Test whether upstream timestamp or cadence corruption was appearing before the branch collapse.

**Change/Test**
Added Stream 0 probes at:
- `decoder0.src`
- `nvvidconv0.src`
- `caps0.src`
- `premuxq0.src`

Also logged PTS deltas at those points.

**Observed Result**
- during healthy periods, the PTS cadence was stable at `40.00ms` across all four boundaries.
- on a clean failing run, Stream 0 FPS began degrading while the upstream PTS cadence still looked normal.
- those boundaries also went stale together when the failure arrived.

**Finding**
The decode-to-premux path did not show gradual upstream PTS drift before failure. The earlier collapse was not explained by a simple pre-mux timestamp-cadence failure.

**Decision**
Stop expecting one more pad probe to reveal a hidden first-dead boundary and re-examine the ingest boundary instead.

---

### March 6, 2026: Decoder Observations
**Goal**
Determine whether the NVIDIA decoder itself was reporting real corruption before the stall.

**Change/Test**
Observed decoder-level output while running the isolation and probe experiments.

**Observed Result**
- some runs showed repeated `decoder0 decreasing timestamp` warnings while the stream was still healthy.
- a later clean failing run showed no such warning at all.
- separate decoder frame logs kept showing:
  - `ErrorType=0`
  - `Concealed MBs=0`
  - only `I/P` frames in the sampled output

**Finding**
Decoder corruption was not established. The `decreasing timestamp` warning was a clue early on, but it weakened because a clean failing run reproduced without it.

**Decision**
Treat the decoder warning as non-conclusive and avoid calling it the root cause.

---

### March 6–7, 2026: MediaMTX FFmpeg Copy-Publisher Relay
**Goal**
Move the failure boundary out of DeepStream by inserting a local RTSP relay on `127.0.0.1`.

**Change/Test**
- MediaMTX was deployed as a local relay.
- FFmpeg pulled the CP Plus stream and republished it into MediaMTX.
- the DeepStream pipeline read `rtsp://127.0.0.1:8554/stream0`.

**Observed Result**
- Stream 0 still dropped.
- recovery got worse because the relay had to reconnect upstream before the local reader could recover.
- the publisher side proved to be its own failure point.

**Finding**
The FFmpeg copy-publisher relay added latency and another unstable process without solving the root failure.

**Decision**
Reject the FFmpeg copy-publisher path as the primary solution.

---

### March 7, 2026: `nvurisrcbin` Direct To Camera
**Goal**
Reduce blind time by using NVIDIA's RTSP auto-reconnect wrapper instead of relying on full pipeline restarts.

**Change/Test**
- replaced the direct `rtspsrc` path for Stream 0 with `nvurisrcbin`.
- configured Stream 0 for unlimited reconnect attempts.
- used a warn-style zero-FPS policy so reconnect could happen in place.

**Observed Result**
- the CP Plus session still dropped.
- reconnect blind time improved to roughly 10–20 seconds in the better runs.
- `nvurisrcbin` did not expose the internal `rtspsrc` keepalive controls directly.

**Finding**
`nvurisrcbin` improved recovery behavior but did not eliminate the recurring upstream drop.

**Decision**
Keep `nvurisrcbin` as a recovery option, but do not treat it as a root-cause fix.

---

### March 9, 2026: Service And Runtime Hygiene
**Goal**
Make later test interpretation trustworthy by removing service-management artifacts and log spam.

**Change/Test**
- filtered decoder `ErrorType=` spam so journal output stayed readable.
- changed service handling to `KillMode=control-group`.

**Observed Result**
- earlier "faster drop" behavior was traced in part to zombie Python processes surviving restarts.
- those extra processes caused GPU contention and contaminated some timing conclusions.
- after the `KillMode=control-group` fix, later measurements were based on clean single-instance runs.

**Finding**
Some earlier runs were not valid comparisons because service restarts were leaving competing pipeline instances alive.

**Decision**
Treat post-`KillMode=control-group` runs as the authoritative timing data.

---

### March 9, 2026: NVR Ingest Tests
**Goal**
Determine whether pulling the CP Plus camera through the Hikvision NVR would avoid the CP Plus RTSP failure mode.

**Change/Test**
Two NVR-based variants were tested:
- `nvurisrcbin` reading the NVR-served RTSP stream
- plain `rtspsrc` with keepalive reading the NVR-served RTSP stream

**Observed Result**
- the NVR stream matched the camera feed characteristics.
- `nvurisrcbin + NVR` still dropped every few minutes, with reconnect blind time around tens of seconds.
- clean retest of `rtspsrc + NVR + keepalive` still dropped and then stayed dead until the watchdog restarted the pipeline.

**Finding**
The NVR did not act as a true shield against the CP Plus instability. It behaved like a proxy for an unstable upstream source rather than an independent stable origin.

**Decision**
Do not treat the NVR path as a proven universal fix based on the tested runs alone.

---

### March 9, 2026: Direct MediaMTX Source Proxy
**Goal**
Keep the local relay architecture but remove the rejected FFmpeg copy-publisher process, so the local relay could be judged on its own.

**Change/Test**
- switched MediaMTX to a direct RTSP source-proxy role.
- DeepStream still consumed `rtsp://127.0.0.1:8554/stream0`.
- startup readiness was handled through the existing wait script.

**Observed Result**
- startup could show a temporary `404 Not Found` before the source became available; this was treated as a readiness issue, not a steady-state fault.
- once running, Stream 0 could again hold normal FPS for a while.
- on the failing run, MediaMTX logged an upstream RTSP source **TCP timeout** first.
- only after that did DeepStream report that `source0` had been closed by the server.

**Finding**
This was the cleanest relay result because it moved the first failing boundary upstream of DeepStream. The remaining failure was between MediaMTX and the CP Plus source session, not between DeepStream and the local relay.

**Decision**
Keep the direct-proxy architecture as the cleaner relay model, but investigate source normalization and upstream transport next rather than doing more DeepStream-only forensics.

---

### March 9, 2026: CP Plus Camera Normalization And Relay-Side Conclusions
**Goal**
Reduce upstream CP Plus instability at the source after the direct proxy made the upstream boundary visible.

**Change/Test**
Camera-side and relay-side follow-ups were identified:
- move the CP Plus main stream to `H.264`
- disable audio so the source exposes one video track instead of video plus G.711
- reduce the I-frame interval from `50` to `25` at `25 FPS`
- prefer upstream UDP on the MediaMTX-to-camera leg because the observed direct-proxy failure was a MediaMTX **TCP timeout**

**Observed Result**
- by the end of the investigation, the camera UI had already been adjusted toward `H.264`, `25 FPS`, `CBR`, `4096 kbps`.
- audio-off and the I-frame interval change were still treated as required follow-up items.
- no completed clean soak was recorded yet for the full normalized-camera plus proxy-UDP combination.

**Finding**
Camera normalization plus a non-TCP upstream proxy leg became the next rational experiment because the relay work had isolated the remaining problem to the upstream CP Plus session.

**Decision**
Do not call camera normalization a completed fix. Treat it as the next targeted mitigation path that the March 6–9 analysis had converged on.

---

### March 10, 2026: NVR Admin Panel And ISAPI Investigation
**Goal**
Determine whether the Hikvision NVR (DS-7764NI-M4, 192.168.28.8) was hitting a session or connection limit that could explain the worsening drop frequency.

**Change/Test**
- inspected NVR web UI: Network Service settings, Integration Protocol, Platform Access (ISUP and Hik-Connect), Online Users.
- queried NVR ISAPI endpoints from the Jetson:
  - `ISAPI/System/deviceInfo` — firmware version, uptime
  - `ISAPI/System/status` — memory usage
  - `ISAPI/Streaming/channels` — stream configuration
  - `ISAPI/ContentMgmt/InputProxy/channels/status` — camera connection status
  - `ISAPI/System/Network/extension/sessionList` — active SDK sessions

**Observed Result**
- firmware: V5.04.050 build 240816, uptime 3.4 days.
- NVR memory: 87 MB free out of total (low but functional).
- HTTP API response time: 31 seconds per request — indicates NVR CPU overload.
- Hik-Connect: shown as disabled in web UI. ISAPI ISUP endpoint reported `enabled=true` but `registerStatus=offline` — no active cloud connection consuming sessions.
- online users: 2 Windows PCs (192.168.28.235 with SMB+VNC, 192.168.28.250 with SMB+RDP+VNC). Neither was actively streaming RTSP.
- ISAPI streaming session list returned 0 sessions — this endpoint only tracks SDK sessions, not RTSP connections.
- no visible "max connections" or "max remote sessions" setting in the NVR web UI.

**Finding**
The NVR was under CPU load (31s API responses, 87MB free RAM) but no session limit was proven. The session tracking APIs do not cover RTSP, so a connection limit could not be confirmed or denied through ISAPI alone.

**Decision**
Stop pursuing the NVR session-limit theory and test direct camera connections to isolate whether the NVR or the camera was the primary failure point.

---

### March 10, 2026: NVR Standalone Soak Tests
**Goal**
Re-measure NVR relay stability with standalone ffmpeg (no pipeline load) to establish a current baseline.

**Change/Test**
Ran standalone ffmpeg TCP soak tests against the NVR-relayed Stream 0 URL:
```
ffmpeg -rtsp_transport tcp -i "rtsp://admin:***@192.168.28.8:554/Streaming/Channels/501" \
  -c:v copy -an -f null /dev/null
```

**Observed Result**
- first run: dropped at 33 seconds.
- second run: connection timeout during setup.
- third run: dropped at 25 seconds.
- all failures were significantly worse than the camera's native ~3 minute cycle.

**Finding**
The NVR relay path was compounding the camera's inherent instability, adding its own failure mode (NVR CPU overload, slow API responses) on top of the upstream camera drops.

**Decision**
Test direct camera connections bypassing the NVR entirely.

---

### March 10, 2026: Direct Camera Soak Tests — Root Cause Confirmed
**Goal**
Determine whether the CP Plus cameras themselves had the ~3 minute session timeout, independent of the NVR.

**Change/Test**
Ran standalone ffmpeg TCP soak tests directly to each CP Plus camera, bypassing the NVR:
- camera 155 (Stream 0): `rtsp://admin:***@192.168.28.155:554/video/live?channel=1&subtype=0`
- camera 162 (Stream 2): `rtsp://admin:***@192.168.28.162:554/video/live?channel=1&subtype=0`
- also confirmed Stream 1 (Hikvision camera 192.168.27.253) as control.

**Observed Result**
- camera 155 (CP Plus, Stream 0): dropped at **2 minutes 46 seconds**.
- camera 162 (CP Plus, Stream 2): dropped at **3 minutes 20 seconds**.
- Stream 1 (Hikvision, 192.168.27.253): **never dropped** — ran indefinitely as expected.

**Finding**
**ROOT CAUSE CONFIRMED**: CP Plus cameras (Dahua OEM) have an approximately 3-minute RTSP TCP session timeout baked into their firmware. This is a camera-level behavior, not an NVR issue, not a GStreamer issue, and not a DeepStream issue. The Hikvision camera on Stream 1 does not have this limitation, which explains why Stream 1 was always the stable control throughout the entire March 6–11 investigation.

**Decision**
The NVR is not the primary cause. All further mitigation must target the CP Plus camera's firmware-level TCP session timeout.

---

### March 10, 2026: CP Plus RTSP URL Discovery
**Goal**
Determine all available RTSP URL paths on the CP Plus cameras (Dahua OEM) to ensure the correct endpoints were being used.

**Change/Test**
Tested multiple Dahua-standard RTSP paths against camera 155:
- `/video/live?channel=1&subtype=0` (main stream)
- `/video/live?channel=1&subtype=1` (sub stream)
- `/cam/realmonitor?channel=1&subtype=0` (Dahua standard)
- `/live` (generic)
- Various ONVIF media paths

**Observed Result**
- only `/video/live?channel=1&subtype=0` (main) and `subtype=1` (sub) responded with valid streams.
- Dahua-standard ONVIF and alternative paths did not respond.

**Finding**
The CP Plus firmware exposes a limited RTSP path set. No alternative URL or path avoids the session timeout.

**Decision**
URL selection is not a variable; focus on transport protocol changes.

---

### March 10, 2026: UDP Transport Discovery
**Goal**
Test whether switching from TCP interleaved to UDP transport would avoid the CP Plus ~3 minute session timeout.

**Change/Test**
Ran standalone ffmpeg with UDP transport directly to camera 155, outputting to `/dev/null`:
```
ffmpeg -rtsp_transport udp -i "rtsp://admin:***@192.168.28.155:554/video/live?channel=1&subtype=0" \
  -c:v copy -an -f null /dev/null
```

**Observed Result**
- standalone UDP survived **18+ minutes** with zero drops.
- no session timeout, no reconnects, continuous stable stream.

**Finding**
**UDP eliminates the 3-minute TCP session timeout.** The CP Plus firmware's session timeout applies only to TCP interleaved RTSP sessions. UDP transport uses a different session management path in the camera firmware and does not trigger the timeout.

**Decision**
Switch the ffmpeg bridge from TCP to UDP transport. Test under pipeline load.

---

### March 10–11, 2026: UDP Under Pipeline Load
**Goal**
Validate that UDP transport through the ffmpeg-to-fdsrc pipe bridge remains stable when the full DeepStream pipeline is consuming the stream.

**Change/Test**
- modified `gst_builder.py` to read per-stream protocol from config instead of hardcoding TCP.
- added UDP-specific ffmpeg transport options: `-buffer_size 4194304 -max_delay 500000 -reorder_queue_size 2000`.
- deployed with `HICON_RTSP_PROTOCOL_0=udp` in `.env`.

**Observed Result**
- ffmpeg showed continuous warnings every second: `max delay reached. need to consume packet` and `RTP: missed 13 packets`, `RTP: missed 30 packets`.
- stream died at approximately 2–3 minutes from accumulated packet loss corrupting the H.264 bitstream.
- root cause investigation revealed: kernel `net.core.rmem_max=212992` (208 KB) was silently capping ffmpeg's `-buffer_size` request regardless of the value passed on the command line.
- raised kernel limit: `sysctl -w net.core.rmem_max=8388608` (8 MB) — packet loss continued.
- additional mitigations attempted:
  - `buffer_size=4194304` (4 MB) — no effect after kernel cap was raised
  - `reorder_queue_size=2000` — no effect
  - `max_delay=500000` (500 ms) — no effect
  - `dd bs=4M iflag=fullblock` buffer pipe between ffmpeg stdout and fdsrc — `dd` also blocks on stdout when downstream is full, no async benefit
  - `thread_queue_size` — no effect

**Finding**
**Pipe backpressure stalls ffmpeg's event loop, causing UDP receive buffer overflow regardless of kernel buffer limits.** When the DeepStream pipeline applies backpressure through the fdsrc → parser → decoder chain, ffmpeg's single-threaded event loop cannot service the UDP socket fast enough. Incoming UDP packets are dropped by the kernel before ffmpeg can read them. This is a fundamental architecture limitation of piping UDP-received data through a synchronous stdout pipe to a GStreamer fdsrc element.

**Decision**
UDP through the pipe bridge is not viable under pipeline load. The pipe architecture fundamentally cannot decouple ffmpeg's UDP receive loop from downstream backpressure. Revert to TCP transport with auto-restart as the current best available approach.

---

### March 11, 2026: Direct Camera TCP + FFmpeg Auto-Restart Bridge
**Goal**
Establish the best achievable stability by combining direct camera connection (bypassing NVR) with TCP transport and the existing ffmpeg auto-restart wrapper.

**Change/Test**
- changed `.env` Stream 0 URL from NVR relay to direct camera 155: `HICON_RTSP_STREAM_0=rtsp://admin:***@192.168.28.155:554/video/live?channel=1&subtype=0`
- set `HICON_RTSP_PROTOCOL_0=tcp`.
- fixed URL quoting bug in `gst_builder.py`: the `&subtype=0` in the camera URL was being interpreted by bash as a background operator, causing ffmpeg to exit with code 127. Fixed by wrapping the URL in single quotes in the wrapper script.

**Observed Result**
- stable intervals of approximately 3.5 minutes between drops (matching the camera's firmware TCP session timeout).
- 10-minute soak test: 14 drops with cascading reconnects, each recovery taking 15–20 seconds via the bash auto-restart wrapper.
- improvement over NVR relay path: drops every ~3.5 min instead of every 25–33s.

**Finding**
Direct camera TCP with auto-restart is the current best achievable configuration. The ~3.5 minute drop cycle is a hard firmware limit that cannot be eliminated with TCP transport. Recovery is reliable through the bash wrapper (pipe stays open, fdsrc never sees EOF).

**Decision**
Deploy this configuration as the current operating mode. The remaining improvement path would require an async proxy (e.g., MediaMTX with UDP upstream to camera + TCP downstream to pipeline) to decouple camera transport from pipeline backpressure.

---

### March 11, 2026: Code Changes Summary (Pre-MediaMTX)
**Goal**
Document code modifications made during the March 10–11 direct-camera investigation.

**Changes**
- `ai_vision/pipeline/gst_builder.py`:
  - ffmpeg command now reads per-stream protocol from `config.get(f'rtsp_protocol_{stream_id}')` instead of hardcoding `tcp`.
  - added UDP-specific transport options (`-buffer_size`, `-max_delay`, `-reorder_queue_size`) when protocol is `udp`.
  - fixed URL quoting bug: wrapped `rtsp_url` in single quotes in the bash wrapper script to prevent shell interpretation of `&` in query parameters.
- `ai_vision/.env`:
  - Stream 0 URL changed from NVR relay to direct camera.
  - protocol set to TCP.
  - Stream 2 remains on NVR relay (pending direct camera switch).

---

### March 11, 2026: MediaMTX UDP Proxy — Standalone Soak Test
**Goal**
Test whether MediaMTX v1.16.3, configured as a local RTSP proxy with UDP upstream transport to the cameras and TCP downstream to the pipeline, could achieve zero-downtime by decoupling the camera's UDP session from pipeline backpressure.

**Change/Test**
- deployed MediaMTX as a systemd service (`hicon-mediamtx.service`) proxying both CP Plus camera streams.
- configured `rtspTransport: udp` for both paths in `mediamtx.stream0.yml`.
- MediaMTX listened on `127.0.0.1:8554`, serving `stream0` and `stream2`.
- `.env` updated: pipeline reads from `rtsp://127.0.0.1:8554/stream0` and `/stream2`.
- camera source URLs stored in `HICON_CPPLUS_SOURCE_STREAM_0/2` env vars (single-quoted to handle `&` in bash).
- ran standalone ffmpeg TCP soak against `rtsp://127.0.0.1:8554/stream0` for 10 minutes.

**Observed Result**
- MediaMTX started successfully: both streams online via UDP — `stream0` (1 track, H264) and `stream2` (2 tracks, H264 + G711).
- `sourceProtocol` was deprecated in v1.16.3; renamed to `rtspTransport`.
- **10-minute standalone soak: zero drops, zero warnings, clean timeout exit.**
- this was the first time Stream 0 went 10 minutes without a single drop.

**Finding**
MediaMTX UDP proxy eliminates drops in standalone testing. MediaMTX's goroutine-based architecture successfully decouples UDP receive from downstream TCP consumers — unlike the ffmpeg pipe bridge which suffered from backpressure stalling the event loop.

**Decision**
Proceed to full pipeline soak test.

---

### March 11, 2026: MediaMTX UDP Proxy — Full Pipeline Soak Test #1
**Goal**
Test MediaMTX UDP proxy under full DeepStream pipeline load.

**Change/Test**
- started `hicon-vision` service with `HICON_RTSP_STREAM_0=rtsp://127.0.0.1:8554/stream0`.
- ffmpeg bridge reads from local MediaMTX relay via TCP (no firmware timeout on localhost).
- MediaMTX `readTimeout: 30s` (default).
- monitored for 10 minutes.

**Observed Result**
- both streams immediately at 25.0 fps.
- **1 drop at ~9 minutes**: MediaMTX logged `ERR [path stream0] [RTSP source] UDP timeout`. Stream went offline briefly, ffmpeg bridge got `404 Not Found` on reconnect attempts until MediaMTX re-established the upstream source (~6s recovery).
- remaining 9+ minutes were clean.

**Finding**
Major improvement: 1 drop in 10 min vs 14 drops with direct TCP. However, MediaMTX still experiences occasional UDP timeouts from the camera.

**Decision**
Increase `readTimeout` from 30s to 60s to be more tolerant of brief UDP hiccups, then re-test.

---

### March 11, 2026: MediaMTX UDP Proxy — Full Pipeline Soak Test #2 (60s readTimeout)
**Goal**
Test whether increasing `readTimeout` to 60s eliminates the remaining drops.

**Change/Test**
- changed `readTimeout: 60s` in `mediamtx.stream0.yml`.
- restarted MediaMTX and monitored pipeline for 10 minutes.

**Observed Result**
- 2 drops in 10 minutes, at ~3 min and ~8 min marks.
- MediaMTX error: `ERR [path stream0] [RTSP source] read tcp 192.168.28.44:XXXXX->192.168.28.155:554: read: connection reset by peer`
- both cameras (155 and 162) showed the same error simultaneously each time.
- MediaMTX auto-reconnected in ~5-6 seconds; pipeline recovered in ~15-25 seconds total.

**Finding**
The drops were caused by `connection reset by peer` on the **RTSP control TCP connection**, not a UDP timeout. Even with `rtspTransport: udp`, the RTSP signaling channel always uses TCP. The CP Plus firmware kills this TCP control connection after ~5 minutes, identical to the session timeout observed with TCP interleaved transport. MediaMTX correctly detects the broken control connection and tears down the source to reconnect.

Network socket analysis confirmed: MediaMTX had both TCP connections (RTSP control, port 554) and UDP sockets (RTP media) open. The `rtspTransport: udp` setting was working correctly for media transport. The failure was on the control channel.

**Decision**
The MediaMTX UDP proxy does not eliminate the CP Plus firmware's RTSP session timeout because the timeout applies to the TCP control connection, which is used regardless of media transport protocol.

---

### March 11, 2026: Standalone ffmpeg UDP Re-Test
**Goal**
Verify whether standalone ffmpeg UDP still survives past 5 minutes (as it did in the earlier 18-minute test).

**Change/Test**
Ran standalone ffmpeg UDP soak for 6.5 minutes directly to camera 155:
```
ffmpeg -rtsp_transport udp -i 'rtsp://...@192.168.28.155:554/...' -c:v copy -an -f null /dev/null
```

**Observed Result**
- at 4 minutes 47 seconds: `max delay reached. need to consume packet`, `RTP: missed 15400 packets`.
- ffmpeg continued running after the packet loss burst (no backpressure with `/dev/null` sink).
- the stream was degraded but not dead.

**Finding**
The earlier 18-minute zero-drop test was not reproducible. Standalone ffmpeg UDP also experiences disruptions at ~5 minutes. The difference from MediaMTX: ffmpeg tolerates the disruption (keeps reading UDP after TCP control dies), while MediaMTX actively tears down and reconnects. Both are caused by the same CP Plus firmware TCP control session timeout — when the control connection dies, the camera may also briefly disrupt UDP media delivery.

**Decision**
Accept that the CP Plus firmware session timeout affects all transport modes (TCP interleaved, UDP+TCP control). The timeout cannot be avoided without camera firmware changes.

---

### March 11, 2026: MediaMTX Extended Soak — Stream 0 Only (No Stream 2)
**Goal**
Run an extended soak with only Stream 0 + Stream 1 (Stream 2 disabled) to get cleaner drop data and identify patterns.

**Change/Test**
- disabled Stream 2 in pipeline config.
- restarted pipeline at 14:38 with Stream 0 (MediaMTX UDP proxy) + Stream 1 (Hikvision direct).
- monitored for 36 minutes.

**Observed Result**
- 8 drops in 36 minutes, consistent ~5 minute interval:
  - 14:38:04 (startup), 14:39:31 (1m27s), 14:49:35 (10m04s), 14:54:10 (4m35s), 14:59:29 (5m19s), 15:04:18 (4m49s), 15:09:22 (5m04s), 15:14:25 (5m03s).
- the 10-minute gap between drops #2 and #3 was explained by cross-referencing MediaMTX logs:
  - 14:39:31 — ERR: `connection reset by peer` (hard drop, ffmpeg died)
  - 14:44:02 — WAR: `2913 RTP packets lost`, `49 processing errors: invalid FU-A packet` (soft drop, ffmpeg survived)
  - 14:49:34 — ERR: `connection reset by peer` (hard drop, ffmpeg died)
- also observed a second soft drop at 15:19:01: WAR `4244 RTP packets lost` (ffmpeg survived).

**Finding**
**Two distinct drop types identified:**
1. **Hard drop (ERR)**: Camera sends TCP RST on the RTSP control connection. MediaMTX tears down the source path and reconnects. ffmpeg's downstream connection also breaks — pipeline loses ~15-20s of frames.
2. **Soft drop (WAR)**: Camera briefly disrupts UDP media delivery without killing the TCP control session. MediaMTX logs RTP packet loss and FU-A parsing errors but keeps the source path alive. ffmpeg continues reading — pipeline is unaffected.

The camera firmware resets every ~5 minutes, but sometimes the reset only disrupts UDP media briefly (soft drop) rather than killing the TCP control session (hard drop). MediaMTX absorbs soft drops transparently. This explains why visible drop intervals occasionally appear as ~10 minutes — two 5-minute camera resets occurred, but the middle one was a soft drop that MediaMTX absorbed.

**Decision**
The ~5 minute firmware cycle is fixed. MediaMTX's value is absorbing the soft drops transparently. No further software mitigation available without camera firmware changes.

---

### March 11, 2026: FPS Watchdog Startup Issue
**Goal**
Document pipeline crash caused by Stream 1 startup delay.

**Observed Result**
- pipeline started at 21:25:20.
- Stream 0 reached 25 fps within 5 seconds.
- Stream 1 (Hikvision) stayed at 0 fps for 5 seconds.
- FPS watchdog triggered at 21:25:52: `[FPS-WATCHDOG] Stream 1 at 0fps for 5s — restarting`.
- pipeline shut down after only 32 seconds of operation.

**Finding**
The FPS watchdog does not have a startup grace period. Stream 1 (Hikvision direct RTSP) sometimes takes >5s to negotiate the RTSP session and deliver first frames. The watchdog interprets this as a stall and kills the pipeline.

**Decision**
Add a startup grace period to the FPS watchdog so it does not trigger during the first N seconds after pipeline start.

---

### March 11, 2026: Code Changes Summary (Final)
**Goal**
Document all code modifications made during the March 10–11 investigation.

**Changes**
- `ai_vision/pipeline/gst_builder.py`:
  - ffmpeg command reads per-stream protocol from config instead of hardcoding TCP.
  - added UDP-specific transport options when protocol is `udp`.
  - fixed URL quoting bug (single quotes around RTSP URL in bash wrapper).
- `ai_vision/configs/mediamtx.stream0.yml`:
  - added `stream2` path alongside `stream0`.
  - changed `sourceProtocol: tcp` → `rtspTransport: udp` (v1.16.3 renamed param).
  - increased `readTimeout` from 30s to 60s.
- `ai_vision/tools/start_mediamtx_stream0.sh`:
  - added `HICON_CPPLUS_SOURCE_STREAM_2` env var export for Stream 2.
- `ai_vision/tools/wait_for_stream0_relay.sh`:
  - added Stream 2 readiness check alongside Stream 0.
- `ai_vision/.env`:
  - added `HICON_CPPLUS_SOURCE_STREAM_0/2` (single-quoted for `&` safety).
  - pipeline URLs point to local MediaMTX relay (`127.0.0.1:8554`).
  - protocol remains TCP (relay-to-pipeline leg).
- `ai_vision/systemd/hicon-vision.service`:
  - added `After=hicon-mediamtx.service` and `Wants=hicon-mediamtx.service`.
- `ai_vision/systemd/hicon-mediamtx.service`:
  - installed and enabled as systemd service.

---

### March 12, 2026: Dahua-Standard URL Test (`/cam/realmonitor`)
**Goal**
Test whether the Dahua-standard RTSP URL path works on CP Plus cameras, as an alternative to `/video/live`.

**Change/Test**
Tested `/cam/realmonitor?channel=1&subtype=1` and `/cam/realmonitor?channel=1&subtype=0` against camera 155.

**Observed Result**
- both URLs returned **404 Not Found**.
- also tested Dahua CGI endpoint (`/cgi-bin/magicBox.cgi?action=getProductDefinition`) — also 404.

**Finding**
CP Plus firmware does not expose standard Dahua API or RTSP paths despite being a Dahua OEM. Only `/video/live?channel=1&subtype=0/1` works.

**Decision**
No alternative URLs available. Move on to sub-stream testing.

---

### March 12, 2026: Sub-Stream Standalone Soak Test (TCP, No Load)
**Goal**
Test whether the sub-stream (`subtype=1`, 720p) has the same ~3 min TCP session timeout as the main stream (`subtype=0`, 1080p).

**Change/Test**
Ran standalone ffmpeg TCP soak directly to camera 155 sub-stream:
```
ffmpeg -rtsp_transport tcp -i 'rtsp://admin:***@192.168.28.155:554/video/live?channel=1&subtype=1' \
  -c:v copy -an -f null /dev/null
```

**Observed Result**
- **27 minutes 28 seconds, ZERO drops.**
- stream characteristics: 1280x720, H.264 Main profile, 25 fps.
- for comparison, main stream standalone TCP drops at ~2m46s on the same camera.

**Finding**
The sub-stream does NOT have the main stream's ~3 min TCP session timeout in standalone (no backpressure) testing. This suggests the firmware treats main and sub streams differently for session management, or the lower bitrate of the sub-stream avoids triggering the timeout condition.

**Decision**
Test sub-stream under pipeline load (via MediaMTX and direct).

---

### March 12, 2026: Sub-Stream via MediaMTX UDP Proxy — Pipeline Soak
**Goal**
Test whether the sub-stream's standalone stability holds when consumed through the MediaMTX UDP proxy under full pipeline load.

**Change/Test**
- changed `HICON_CPPLUS_SOURCE_STREAM_0` from `subtype=0` to `subtype=1` in `.env`.
- restarted MediaMTX and pipeline.
- monitored for 10 minutes.

**Observed Result**
- 3 drops in 10 minutes (ERR: `connection reset by peer` at ~5 min intervals):
  - 15:30:40 (startup), 15:35:30 (4m50s), 15:40:49 (5m19s).
- same ~5 min firmware TCP control session timeout pattern as main stream through MediaMTX.
- soft drops also observed (WAR: `1 RTP packet lost` at 15:34:55, `2 RTP packets lost` at 15:36:22).

**Finding**
Sub-stream through MediaMTX drops at the same rate as main stream through MediaMTX. The MediaMTX RTSP control TCP connection to the camera still triggers the firmware's session timeout. The sub-stream's standalone stability advantage is lost when MediaMTX manages the RTSP session.

**Decision**
Test sub-stream directly without MediaMTX to isolate whether MediaMTX is the cause.

---

### March 12, 2026: Sub-Stream Direct (No MediaMTX) — Pipeline Soak
**Goal**
Test whether the sub-stream's standalone stability holds when consumed directly by the ffmpeg pipe bridge under pipeline load, bypassing MediaMTX.

**Change/Test**
- changed `HICON_RTSP_STREAM_0` to point directly at camera 155 sub-stream URL.
- stopped MediaMTX service.
- restarted pipeline.
- monitored for 10 minutes.

**Observed Result**
- 4 drops in 10 minutes:
  - 15:50:29 (2m29s after start), 15:55:28 (4m59s).
  - additional reconnect cascades at 15:55:40, 15:55:52, 15:56:20, 15:56:32, 15:56:44.
- pipeline recovered each time, current FPS stable at 25.0 after recovery.

**Finding**
**Sub-stream also drops under pipeline load**, even without MediaMTX. The standalone ffmpeg test (27 min zero drops) worked because ffmpeg had no downstream backpressure (`/dev/null` sink). Under real pipeline load, the ffmpeg pipe bridge creates backpressure that stalls TCP reads, triggering the camera's session timeout — same mechanism as the main stream.

The sub-stream's advantage is standalone-only: when ffmpeg can read TCP packets without delay, the firmware doesn't time out. But the pipe-to-fdsrc architecture reintroduces the backpressure that causes the timeout.

**Decision**
Sub-stream switching does not solve the problem under pipeline load. The root cause is pipe backpressure stalling TCP reads, not stream type. Next approach: UDP loopback to eliminate pipe backpressure entirely.

---

### March 12, 2026: UDP Loopback Standalone Soak
**Goal**
Test whether routing camera data through a localhost UDP hop eliminates pipe backpressure while preserving the standalone "no downstream load" success case.

**Architecture**
```
camera (TCP) → ffmpeg → MPEGTS → UDP 127.0.0.1:5000 → second ffmpeg → /dev/null
```

**Observed Result**
- **10 minutes 32 seconds, zero drops, zero resets.**
- no `ffmpeg exited`, no packet-loss bursts, no reconnects.

**Finding**
UDP loopback preserves the standalone "no blocking write" property. It proved that ffmpeg can hand off media to localhost UDP without the stdout-pipe backpressure seen in the `fdsrc` bridge.

**Decision**
Integrate UDP loopback into `gst_builder.py` and test under full pipeline load.

---

### March 12, 2026: UDP Loopback Integrated Into Pipeline
**Goal**
Replace the ffmpeg stdout pipe bridge with `ffmpeg → UDP localhost → udpsrc → tsdemux` for Stream 0.

**Code Change**
- added per-stream UDP loopback config and port wiring.
- added `udpsrc → tsdemux → parser → decoder` source path for Stream 0.
- updated `.env` toggles to allow Stream 0 UDP loopback.
- fixed a bootstrap bug where `hicon_pipeline.py` was not passing the new config keys through to the builder.

**Observed Result**
- pipeline started cleanly after the config handoff fix.
- Stream 0 reached **25 fps** after startup.
- `tsdemux` linked dynamically as expected.

**Finding**
The UDP loopback integration itself was correct. The remaining question was whether it actually reduced drop frequency under load.

**Decision**
Run 10-minute soaks on both main stream and sub-stream.

---

### March 12, 2026: UDP Loopback Under Pipeline Load
**Goal**
Verify whether UDP loopback reduces the recurring ~5 minute Stream 0 drops under full DeepStream load.

**Observed Result**
- main stream (`subtype=0`): still dropped at ~5 minute intervals under load.
- `nice -n -10` on ffmpeg made **no difference**; drops remained at the same ~5 minute cadence.
- sub-stream (`subtype=1`): also dropped under pipeline load, even though it had survived 27m28s standalone.
- recovery was often **worse** than the older pipe bridge because `tsdemux` had to re-lock after the MPEG-TS source resumed. Recovery windows of **10–30s** were observed.

**Finding**
UDP loopback removed stdout-pipe backpressure, but it did **not** eliminate the visible Stream 0 drops under full pipeline load. The camera/control-session problem still surfaced, and the `tsdemux` resynchronization penalty made recovery slower than expected.

**Decision**
Reject UDP loopback as the production direction for Stream 0 and move to a delayed local spool that hides reconnects instead of trying to prevent them.

---

### March 12, 2026: Stream 0 Segment Buffer Mode Implemented
**Goal**
Create a no-local-drop Stream 0 path that records the direct camera sub-stream into `/dev/shm`, then feeds DeepStream from a delayed FIFO-backed spool instead of the live RTSP session.

**Code Change**
- `HICON_RTSP_STREAM_0` switched to the direct camera **sub-stream** URL (`subtype=1`).
- new Stream 0 source mode added: `use_segment_buffer_0`.
- new helper process added:
  - ffmpeg writes 2-second MPEG-TS segments into `/dev/shm/hicon/stream0-buffer`.
  - helper feeds completed segments into a FIFO.
  - builder reads FIFO via `fdsrc -> tsparse -> identity(sync=true) -> tsdemux -> parser -> decoder`.
- Stream 0 MediaMTX relay was changed to **on-demand** when segment-buffer mode is enabled so it does not hold a second always-on RTSP session.
- builder/source precedence updated so segment-buffer mode overrides UDP loopback, ffmpeg pipe, and `nvurisrcbin`.

**Observed Result**
- code compiled and loaded.
- Stream 0 entered the new helper-driven ingest path.
- helper logs showed the intended 60-second fill target and `buffer primed` transition.

**Finding**
The architecture was successfully integrated into the codebase. The next step was to validate whether the delayed spool actually held backlog instead of draining immediately.

**Decision**
Run a live journal-based startup test and inspect backlog behavior.

---

### March 12, 2026: First Live Segment-Buffer Test Exposed Local Pacing Bug
**Goal**
Validate the first end-to-end segment-buffer implementation during cold start and initial playback.

**Observed Result**
- Stream 0 stayed at `0fps` during the initial 60-second fill, which was expected.
- however, the Stream 0 watchdog started warning at 30 seconds because bus-handler startup grace was still only 30 seconds.
- once the helper logged `buffer primed (30 pending segments, target=30)`, the backlog collapsed almost immediately:
  - within ~3 seconds the helper logged `buffer depth dropped to 14 segments, pausing to rebuffer to 60s`.
- this created a repeating loop:
  - **prime → brief playback burst → drain to 14 → rebuffer → reprime**
- Stream 0 showed repeated `0fps` stalls and recoveries even though the helper was successfully writing segments locally.

**Finding**
This was **not** the original upstream RTSP failure mode. It was a new local pacing/timing bug in the first segment-buffer implementation:
- the helper was writing buffered `.ts` files into the FIFO as fast as the kernel would accept them, instead of pacing playback in real time.
- ffmpeg was also resetting timestamps per segment epoch (`-reset_timestamps 1`), which made continuous delayed playback harder for `tsparse/identity/tsdemux`.
- the watchdog treated intentional buffering/rebuffering as a failure because it had no helper-state awareness.

**Decision**
Repair the helper and watchdog logic without changing the overall helper + FIFO + `fdsrc -> tsparse -> tsdemux` architecture.

---

### March 12, 2026: Segment Buffer Repair Implemented
**Goal**
Repair the first segment-buffer implementation so it behaves like a delayed playback buffer rather than a burst-drain spool.

**Code Change**
- removed ffmpeg `-reset_timestamps 1` from the helper so segments within an ffmpeg run keep continuous timing.
- added helper pacing: each completed segment is handed off on a segment-duration wall-clock schedule instead of being dumped into the FIFO immediately.
- added helper `state.json` in the buffer directory with:
  - `mode`: `buffering`, `playing`, `rebuffering`, `stopped`
  - `pending_segments`
  - `target_segments`
  - `updated_at`
  - `active_epoch`
- wired the bus handler to:
  - use a longer Stream 0 startup grace (`delay + 10s`)
  - suppress Stream 0 `0fps` watchdog warnings while helper state is `buffering` or `rebuffering`
  - keep normal Stream 0 watchdog behavior during `playing`
- fixed the misleading tsdemux log string so Stream 0 now logs `segment buffer chain fully linked via tsdemux pad-added` instead of `UDP loopback`.
- added/updated unit-test coverage for:
  - helper ffmpeg args
  - helper pacing
  - helper state publication
  - bus-handler buffering/rebuffering suppression
  - segment-buffer-specific tsdemux log wording

**Verification Performed**
- `python3 -m compileall` passed for the touched runtime files and tests.
- module import smoke test passed.
- `pytest` could **not** be run in this environment because `pytest` is not installed.

**Finding**
The repair is implemented in code, but the repaired segment-buffer path still needs a fresh live restart/soak to prove that the `prime -> drain -> reprime` loop is gone.

**Decision**
Current repo state at end of March 12 is the repaired Stream 0 segment-buffer experiment. Live validation remains pending.

---

## Consolidated Test Matrix

| Date | Test / Change | What Was Being Tested | Observed Runtime / Failure Pattern | Conclusion |
|---|---|---|---|---|
| March 6, 2026 | Stage 1 RTSP config cleanup | Whether the `rtspsrc.timeout` unit bug and protocol ambiguity were the main cause | Stream 0 still dropped after the config was corrected; `0.0 fps` still preceded later RTSP warnings | Real bug fixed, but not the full fix |
| March 6, 2026 | `srcq0` / `premuxq0` isolation | Whether source-path backpressure from the Stream 0 branch was the primary failure | Runtime improved, but Stream 0 still collapsed | Backpressure was part of the issue, not the whole issue |
| March 6, 2026 | Post-mux queue isolation | Whether the stall began after `nvstreammux` | Runtime improved again, but failure remained | The failure boundary was further downstream than raw ingest |
| March 6, 2026 | `tracker_0` bypass | Whether tracker was the first hard failure point | Failure happened sooner | `tracker_0` was not the root cause |
| March 6, 2026 | `pgie_pouring` bypass | Whether PGIE was the single blocking element | Runtime extended but Stream 0 still died | PGIE contributed pressure but was not necessary for failure |
| March 6, 2026 | `post-mux-only` / `pre-OSD-only` / `post-convert-only` | What the smallest failing post-mux topology was | `post-mux-only` stable; `pre-OSD-only` and `post-convert-only` still failed | `nvdsosd` and forced RGBA were not necessary; post-mux `nvvideoconvert` remained suspect |
| March 6, 2026 | `nvvideoconvert` tuning | Whether converter properties, not topology, were the real issue | Timing changed but clean tuned runs still failed | Tuning was not accepted as a fix |
| March 6, 2026 | Decoupled-analysis mode | Whether removing main-path RGBA conversion and CPU work would stabilize Stream 0 | Main and analysis liveness went stale together on failure | CPU side branch and removed hot path were not the primary cause |
| March 6, 2026 | Stage-boundary probes | Whether one stage inside `mux -> postmuxq -> pgie -> tracker` died first | All tracked stages aged together | Shared branch collapsed as a unit |
| March 6, 2026 | Upstream PTS probes | Whether decode-to-premux cadence drifted before failure | `40.00ms` PTS deltas stayed stable even near failure | No simple pre-mux timestamp drift was proven |
| March 6–7, 2026 | MediaMTX + FFmpeg copy-publisher relay | Whether a local relay could shield DeepStream from camera instability | Relay added its own reconnect delay and failed at publisher boundary | Reject this relay model |
| March 7, 2026 | `nvurisrcbin` direct camera | Whether built-in auto-reconnect was enough | Session still dropped, but blind time improved to roughly 10–20s in the better runs | Recovery improved; root cause remained |
| March 9, 2026 | Decoder log filtering + `KillMode=control-group` | Whether service hygiene was contaminating conclusions | Earlier zombie-process contamination was eliminated | Post-fix runs became the authoritative comparisons |
| March 9, 2026 | NVR ingest (`nvurisrcbin` and `rtspsrc`) | Whether the Hikvision NVR would shield Stream 0 from the CP Plus camera | NVR path still dropped; plain `rtspsrc` stayed dead after drop until watchdog restart | NVR was not a proven shield |
| March 9, 2026 | Direct MediaMTX source proxy | Whether the relay remained useful without the FFmpeg publisher layer | MediaMTX upstream RTSP source timed out first; DeepStream failed afterward | Cleanest proof that the remaining failure boundary was upstream of DeepStream |
| March 9, 2026 | Camera normalization direction | Whether source-side `H.264`, audio-off, GOP tightening, and proxy-leg UDP should be the next move | Camera was partly moved toward H.264/25fps/CBR; full normalized soak had not yet been completed | Strong next mitigation direction, not a completed fix |
| March 10, 2026 | NVR admin panel + ISAPI investigation | Whether the NVR was hitting a session/connection limit | 87MB free RAM, 31s API response, 0 tracked sessions, 2 online users, Hik-Connect disabled | No session limit proven; NVR under CPU load but not the primary cause |
| March 10, 2026 | NVR standalone soak tests | Current NVR relay stability baseline | Dropped at 33s, timeout, 25s — worse than camera native ~3 min | NVR compounds camera instability with its own failures |
| March 10, 2026 | Direct camera TCP soak tests | Whether CP Plus cameras drop independent of NVR | Camera 155: 2m46s, Camera 162: 3m20s, Hikvision: never drops | **ROOT CAUSE**: CP Plus firmware ~3 min TCP session timeout |
| March 10, 2026 | CP Plus RTSP URL discovery | Whether alternative URL paths avoid the timeout | Only `/video/live?channel=1&subtype=0/1` responded | No alternative URLs available |
| March 10, 2026 | UDP standalone soak test | Whether UDP avoids the TCP session timeout | 18+ minutes, zero drops | UDP eliminates the timeout — camera firmware issue is TCP-specific |
| March 10–11, 2026 | UDP under pipeline load | Whether UDP works through the ffmpeg pipe bridge | Packet loss every second, stream dies at 2–3 min; kernel buffer raised to 8MB — still fails | Pipe backpressure stalls ffmpeg UDP recv; architecture incompatible |
| March 11, 2026 | Direct camera TCP + auto-restart | Best achievable stability with current architecture | ~3.5 min stable, 14 drops in 10 min, 15–20s recovery each | Best TCP-only config; firmware limit cannot be eliminated with TCP |
| March 11, 2026 | MediaMTX UDP proxy — standalone soak | Whether async proxy with UDP upstream eliminates drops | 10 min, zero drops, zero warnings | First zero-drop 10-min test; proxy decouples UDP recv from backpressure |
| March 11, 2026 | MediaMTX UDP proxy — pipeline soak #1 (30s readTimeout) | Whether zero-drop behavior holds under pipeline load | 1 drop at ~9 min (UDP timeout), ~6s recovery | Major improvement: 1 drop vs 14 with direct TCP |
| March 11, 2026 | MediaMTX UDP proxy — pipeline soak #2 (60s readTimeout) | Whether increased timeout eliminates remaining drops | 2 drops at ~3 min and ~8 min (`connection reset by peer` on TCP control) | Camera firmware kills RTSP control TCP regardless of media transport |
| March 11, 2026 | Standalone ffmpeg UDP re-test | Whether earlier 18-min zero-drop result is reproducible | Packet loss burst at 4m47s (`missed 15400 packets`), stream continued degraded | Earlier 18-min test was not reproducible; camera disrupts at ~5 min mark |
| March 11, 2026 | MediaMTX extended soak — Stream 0 only (36 min) | Drop pattern and consistency with Stream 2 disabled | 8 drops in 36 min, consistent ~5 min interval; 1 transparent soft drop absorbed by MediaMTX | Two drop types identified: hard (TCP RST, cascades) and soft (UDP packet loss, absorbed) |
| March 12, 2026 | Dahua-standard URL test (`/cam/realmonitor`) | Whether CP Plus supports standard Dahua RTSP paths | 404 Not Found for both main and sub stream | CP Plus firmware only supports `/video/live` paths |
| March 12, 2026 | Sub-stream standalone TCP soak (no load) | Whether sub-stream has same ~3 min session timeout | **27m28s, zero drops** (vs main stream 2m46s) | Sub-stream does NOT timeout standalone — firmware treats streams differently |
| March 12, 2026 | Sub-stream via MediaMTX UDP proxy (pipeline load) | Whether sub-stream stability holds through MediaMTX | 3 drops in 10 min, ~5 min interval (same as main stream) | MediaMTX RTSP control TCP still triggers firmware timeout regardless of stream type |
| March 12, 2026 | Sub-stream direct TCP (pipeline load, no MediaMTX) | Whether sub-stream stability holds under pipeline backpressure | 4 drops in 10 min, first at 2m29s | Pipe backpressure stalls TCP reads — same failure mechanism as main stream |
| March 12, 2026 | UDP loopback standalone soak | Whether UDP localhost decouples ffmpeg from backpressure | **10m32s, zero drops, zero resets** | Loopback removes stdout-pipe backpressure in standalone |
| March 12, 2026 | UDP loopback integrated into pipeline | Whether `ffmpeg -> UDP localhost -> udpsrc -> tsdemux` works end-to-end | Stream 0 started and reached 25 fps after config handoff fix | Integration succeeded; stability still needed soak validation |
| March 12, 2026 | UDP loopback under pipeline load | Whether UDP loopback reduces Stream 0 drop frequency under DeepStream load | Still dropped at ~5 min intervals; recovery often 10–30s due `tsdemux` re-lock | Not better than earlier approaches; rejected |
| March 12, 2026 | Segment-buffer implementation | Whether delayed local spool can decouple Stream 0 from live source stalls | Helper/FIFO/tsdemux path integrated; Stream 0 enters delayed spool mode | Architecture added successfully |
| March 12, 2026 | First segment-buffer live test | Whether the first spool implementation preserves the 60s backlog | Backlog collapsed from 30 to 14 segments within ~3s; repeated reprime loop | Initial implementation had a local pacing/timeline bug |
| March 12, 2026 | Segment-buffer repair | Whether pacing/state/watchdog fixes are in code | Helper pacing, `state.json`, watchdog suppression, and log fixes implemented; live soak pending | Repaired architecture is current experimental state |
| March 12, 2026 | Segment-buffer: tsdemux deadlock | Why Stream 0 stayed at 0fps after `buffer primed` | Feeder blocked in `os.write` → FIFO full → fdsrc not reading → tsdemux/tsparse chain blocking without linked downstream pads | Deadlock: tsdemux needs data to fire pad-added, but data blocked because pad-added hasn't fired |
| March 12, 2026 | Segment-buffer: switch to raw H264 | Whether replacing MPEGTS+tsdemux with raw H264 eliminates the tsdemux dynamic-pad deadlock | Static chain `fdsrc → segbufq → h264parse → decoder` avoids tsdemux entirely | Deadlock eliminated by removing dynamic pads from source chain |
| March 12, 2026 | Segment-buffer: FIFO race condition | Why Stream 0 still at 0fps after switching to raw H264 | fd showed as `(deleted)` — gst_builder opened OLD FIFO from previous run before new helper's `shutil.rmtree` deleted it; fdsrc read from a deleted inode with no writer — blocked forever | Fix: delete old FIFO in gst_builder BEFORE spawning helper |
| **March 12, 2026** | **Segment-buffer fully working soak** | **Whether 60s delayed local spool absorbs all camera drops invisibly** | **4 camera drops, ZERO watchdog fires in 10 min; pipeline never saw 0fps despite 4 TCP resets** | **SOLUTION: Segment buffer with raw H264 + FIFO race fix completely hides CP Plus firmware drops** |

## What Was Ruled Out
- The `rtspsrc.timeout` unit bug as the sole cause of the March 6+ failures.
- `tracker_0` as the first or only blocking element.
- `pgie_pouring` as a necessary condition for failure.
- `nvdsosd` as a required element for the reproduced stall.
- Forced RGBA conversion before OSD as the sole root cause.
- The Stream 0 CPU probe or CPU analysis branch as the primary failure source once decoupled mode was tested.
- A simple decode-to-premux PTS drift explanation before the branch collapse.
- The FFmpeg copy-publisher MediaMTX relay as a production-worthy fix.
- The Hikvision NVR as a guaranteed shield against CP Plus source instability.
- NVR session/connection limits as the primary cause — ISAPI showed 0 tracked sessions; no max connection setting visible.
- NVR memory pressure as the primary cause — drops reproduced identically with direct camera connections bypassing the NVR.
- UDP transport through the ffmpeg pipe bridge — pipe backpressure stalls ffmpeg's UDP recv loop regardless of kernel buffer sizes.
- Alternative CP Plus RTSP URL paths — only `/video/live?channel=1&subtype=0/1` respond.
- MediaMTX UDP proxy as a zero-downtime solution — camera firmware kills the RTSP control TCP connection regardless of whether RTP media uses UDP. MediaMTX detects the TCP reset and briefly tears down the source (~5-6s reconnect).
- UDP transport as a complete bypass of the firmware timeout — re-testing showed standalone ffmpeg UDP also experiences disruptions at ~5 minutes. The earlier 18-minute zero-drop test was not reproducible.
- Sub-stream (`subtype=1`, 720p) as a fix under pipeline load — survived 27m28s standalone but drops at ~2-5 min intervals under pipeline backpressure (both via MediaMTX and direct). The standalone stability is lost when ffmpeg's TCP read loop is stalled by pipe backpressure.
- UDP loopback as the production Stream 0 fix — it integrated cleanly but still dropped at ~5 min intervals under pipeline load and often recovered more slowly because `tsdemux` had to re-lock.
- CPU priority (`nice -n -10`) as a fix for the ~5 min drop cycle — it did not change drop timing.
- The first-pass segment-buffer implementation as production-ready — its initial live test exposed a local pacing/timeline bug (`prime -> drain -> reprime`) unrelated to the original RTSP failure.

## What Remained True Across Tests
- Stream 0 was the only unstable branch; Stream 1 (Hikvision) remained the reliable control throughout all testing.
- In the direct-pipeline tests, Stream 0 FPS generally collapsed before RTSP timeout warnings appeared.
- Clean single-instance runs still reproduced the failure, so the issue was not only service-management noise.
- Decoder logs repeatedly showed `ErrorType=0` and `Concealed MBs=0` around healthy and failing periods.
- Recovery time could be improved by architecture changes (`nvurisrcbin`, relay boundaries, ffmpeg auto-restart wrapper, MediaMTX proxy), but repeated instability still came back because the CP Plus firmware RTSP session timeout is the root cause.
- Both CP Plus cameras (155 and 162) exhibited the same ~3–5 minute session timeout, confirming it is a firmware-level behavior common to the model.
- The CP Plus firmware timeout affects the RTSP control TCP connection, which is used by all transport modes (TCP interleaved and UDP+TCP control). No transport selection avoids the timeout.

## Final Findings
### Confirmed (Root Cause)
- **CP Plus cameras (Dahua OEM) have an approximately 3–5 minute RTSP session timeout baked into firmware.** This is the definitive root cause of all Stream 0 and Stream 2 drops observed since March 5.
- The timeout applies to the RTSP control TCP connection, which is always TCP regardless of media transport mode. This means both TCP interleaved and UDP media transport are affected.
- Direct camera TCP tests proved the timeout: camera 155 dropped at 2m46s, camera 162 dropped at 3m20s. The Hikvision camera on Stream 1 never dropped.
- The NVR compounded the camera instability — NVR relay drops were every 25–33s vs camera-native ~3–5 minutes — but the NVR was not the primary cause.

### Confirmed (Architecture)
- The Stage 1 RTSP timeout-unit cleanup fixed a real bug, but not the full Stream 0 problem.
- Stream 0-only queue isolation and later post-mux isolation improved timing but did not eliminate the stall.
- `tracker_0` was not the root cause, and `pgie_pouring` was not the only cause.
- The FFmpeg copy-publisher MediaMTX relay was rejected because it introduced another failure layer and worse recovery.
- `KillMode=control-group` was required; earlier zombie processes had contaminated some prior timing conclusions.
- The direct MediaMTX source proxy shifted the first observed failure boundary upstream of DeepStream.

### Confirmed (MediaMTX UDP Proxy)
- **MediaMTX UDP proxy is the best available configuration.** Under pipeline load: 1–2 drops per 10 minutes vs 14 drops with direct TCP. Recovery time ~15-25s.
- MediaMTX standalone soak test (no pipeline load): 10 minutes, zero drops — the only zero-drop 10-minute test ever recorded.
- The remaining drops under pipeline load are caused by the camera firmware resetting the RTSP control TCP connection (`connection reset by peer`). MediaMTX detects this and reconnects in ~5-6 seconds.
- `rtspTransport: udp` correctly uses UDP for RTP media but the RTSP signaling always uses TCP — this is inherent to the RTSP protocol, not a MediaMTX limitation.

### Confirmed (Segment Buffer — FINAL SOLUTION)
- A delayed local spool for Stream 0 was fully implemented and validated on March 12.
- **Final architecture (working)**:
  - ffmpeg reads camera sub-stream (`subtype=1`) directly via TCP
  - ffmpeg writes **raw H264 segments** (2s each) to `/dev/shm/hicon/stream0-buffer/segments/`
  - Python helper (`segment_buffer_helper.py`) feeds completed segments into a named FIFO at real-time pace (one segment per 2 real-world seconds)
  - GStreamer reads FIFO via `fdsrc → segbufq(leaky) → h264parse(config-interval=-1) → nvv4l2decoder → pipeline`
  - No tsdemux, no dynamic pads — fully static chain eliminates deadlock
- **Bugs found and fixed during implementation**:
  1. **tsdemux deadlock**: Original MPEGTS chain had a chicken-and-egg deadlock where tsdemux needed data to fire pad-added but data was blocked because pad-added hadn't fired. Fixed by switching to raw H264 segments and removing tsdemux entirely.
  2. **FIFO race condition**: gst_builder opened the OLD FIFO from a previous run before the new helper's `shutil.rmtree` deleted it. fdsrc ended up reading from a deleted inode with no writer — blocked forever. Fixed by deleting any leftover FIFO in gst_builder BEFORE spawning the helper.
- **Validated result**: 4 camera TCP resets in 10-minute soak → **ZERO watchdog fires, ZERO 0fps events** in the pipeline. CP Plus firmware drops completely invisible to the inference pipeline.
- **Trade-off**: 60-second intentional latency at startup and after extended outages. Within the 60s buffer window, camera reconnects are fully transparent. Acceptable for furnace monitoring (events last minutes).

### Confirmed (Drop Types)
- **Two distinct drop types occur on the ~5 min firmware cycle:**
  - **Hard drop (ERR)**: Camera sends TCP RST on RTSP control connection. MediaMTX tears down the source, ffmpeg dies, pipeline loses ~15-20s. These are the visible drops.
  - **Soft drop (WAR)**: Camera briefly disrupts UDP media without killing TCP control. MediaMTX logs RTP packet loss but keeps the path alive. ffmpeg and pipeline are unaffected. These are transparent.
- MediaMTX absorbs soft drops, explaining occasional ~10 min gaps between visible drops (two 5-min resets, middle one soft).
- Extended 36-min soak confirmed consistent ~5 min cycle with 8 hard drops and 2 soft drops absorbed.

### Not achievable without camera firmware changes
- Zero-downtime operation — the camera firmware session timeout cannot be bypassed by any transport or proxy configuration.
- The timeout applies to the RTSP control TCP channel, which is mandatory in all RTSP implementations.

### Comparison of all tested configurations (10-minute soak)

| Configuration | Drops in 10 min | Recovery time | Drop interval |
|---|---|---|---|
| rtspsrc direct TCP (March 6) | continuous stall | pipeline restart | ~2–5 min |
| nvurisrcbin direct TCP (March 7) | ~3–4 | 10–20s | ~3–5 min |
| ffmpeg bridge via NVR TCP (March 9) | many | 15–20s | 25–33s |
| ffmpeg bridge direct TCP (March 11) | 14 | 15–20s | ~3.5 min |
| MediaMTX UDP proxy (March 11) | 1–2 | 15–25s | ~5–9 min |
| **Segment buffer 60s spool (March 12)** | **0** | **0s (transparent)** | **∞** |

## Current Status At End Of Investigation (March 12, 2026)
The investigation concluded on March 12 with a fully validated solution:
- **Stream 0 segment buffer deployed**: ffmpeg writes raw H264 2s segments to `/dev/shm`, helper feeds FIFO at real-time pace, GStreamer reads via static `fdsrc → h264parse → decoder` chain.
- `HICON_USE_SEGMENT_BUFFER_0=true`, `HICON_SEGMENT_BUFFER_DELAY_SEC_0=60`, direct sub-stream URL (`subtype=1`).
- **Validated**: 4 CP Plus TCP resets in 10 min → 0 watchdog fires, 0 0fps events. Drops are fully transparent.
- MediaMTX service (`hicon-mediamtx.service`) remains deployed but Stream 0 no longer reads from it (the segment buffer reads directly from the camera).

Previous best configuration (MediaMTX UDP proxy, still in code but superseded):
- `hicon-mediamtx.service` proxying both CP Plus cameras via UDP upstream.
- Stream 0 and Stream 2 pipeline URLs pointed to `rtsp://127.0.0.1:8554/stream0` and `/stream2`.
- `HICON_USE_FFMPEG_SRC_0=true` — ffmpeg bridge reads from local MediaMTX relay via TCP.
- Stream 0 and Stream 2 pipeline URLs point to `rtsp://127.0.0.1:8554/stream0` and `/stream2`.
- Camera source URLs stored in `HICON_CPPLUS_SOURCE_STREAM_0/2` env vars (single-quoted for bash safety).
- `HICON_USE_FFMPEG_SRC_0=true` — ffmpeg bridge reads from local MediaMTX relay via TCP (no firmware timeout on localhost).
- `hicon-vision.service` depends on `hicon-mediamtx.service` (`After=` and `Wants=`).
- Stream 1 pointed at the direct Hikvision URL and remained the stable reference path.
- MediaMTX config: `rtspTransport: udp`, `readTimeout: 60s`, listener on `127.0.0.1:8554`.
- `gst_builder.py` supported per-stream protocol selection and UDP transport options.
- URL quoting bug fixed in both `.env` (single quotes) and bash wrapper script.
- The full chain for CP Plus streams: `camera (UDP) → MediaMTX → (TCP) → ffmpeg bridge → pipe → fdsrc → decoder → pipeline`.

### Current Status At End Of March 12, 2026
At the end of March 12, the repo had moved to an **experimental Stream 0 segment-buffer configuration**:
- Stream 0 points directly at the CP Plus **sub-stream** (`subtype=1`) rather than the local MediaMTX relay.
- `HICON_USE_SEGMENT_BUFFER_0=true`; Stream 0 segment-buffer mode takes priority over UDP loopback, ffmpeg pipe, and `nvurisrcbin`.
- Stream 0 helper writes 2-second MPEG-TS segments into `/dev/shm/hicon/stream0-buffer`.
- helper publishes `/dev/shm/hicon/stream0-buffer/state.json` for watchdog-aware buffering/rebuffering state.
- builder ingests Stream 0 as `fdsrc -> tsparse -> identity(sync=true) -> tsdemux -> parser -> decoder`.
- Stream 0 MediaMTX relay is forced to **on-demand** when segment-buffer mode is enabled so it does not hold a second always-on RTSP session.
- same-day repair for helper pacing, helper state publication, and bus-handler suppression is in code.
- **Live validation of the repaired segment-buffer path is still pending.**

---

## March 13, 2026: Segment Buffer Validation, Bug Fixes, and /dev/null Emulation Planning

### Validation of Segment Buffer (March 12–13 soak)

**MediaMTX hypothesis disproven.**
Hypothesis going into March 13 was that `hicon-mediamtx.service` was competing for the same camera
sub-stream URL, creating two RTSP sessions and halving the drop interval. `hicon-mediamtx` was
stopped and a soak test was run. Result: ffmpeg `code=0` drops continued at the same ~4-5 minute
rate. MediaMTX was not the cause.

**Segment buffer validated: drops are fully transparent.**
With the segment buffer enabled (`HICON_USE_SEGMENT_BUFFER_0=true`, `HICON_SEGMENT_BUFFER_DELAY_SEC_0=60`):
- Camera drops every ~4-5 min (ffmpeg exits code=0) → ffmpeg restarts, fills new epoch's segments.
- 60s of buffered segments cover the reconnect window → pipeline FIFO never drains.
- Result: **0 watchdog fires, 0 0fps events** across multi-hour soak. CP Plus drops completely
  invisible to the inference pipeline.

### Bug Fixes Applied (March 13)

**Bug 1: 14fps burst (burst-then-idle decode pattern)**
- **Root cause**: Default Linux FIFO pipe buffer = 64KB. A 2s raw H264 segment at the sub-stream
  bitrate is ~530KB. The entire segment was written to the FIFO in a single instant (once the pipe
  buffer was enlarged), causing the HW decoder to receive all ~50 frames at once, burst-decode
  them in <100ms, then sit idle for ~1.9s. tegrastats showed GPU alternating 33%↔1% in 2s cycles.
  Average pipeline throughput measured at ~14fps.
- **Fix**: Rate-limited `_write_segment()` in `segment_buffer_helper.py`. Writes the segment in
  64KB chunks paced at `file_size / segment_seconds` bytes/sec (matching the stream's natural
  bitrate). The decoder receives a steady chunk every ~12ms → GPU stays at ~4% continuously →
  25fps confirmed in logs (`125 frames (25.0 fps)` every 5s window).
- **Deadline base fix**: `_advance_feed_deadline()` now uses `write_start` (not write-end) as the
  base. The rate-limited write takes ~2s; using write-end would make the feeder wait 2s after
  finishing a 2s write, doubling the inter-segment gap and causing rebuffering stalls.

**Bug 2: FIFO pipe size silent failure**
- **Root cause**: `F_SETPIPE_SZ` was requested at 4MB but the system `pipe-max-size` on this
  Jetson is 1MB (`/proc/sys/fs/pipe-max-size`). The OSError was caught and passed silently with no
  fallback — the FIFO stayed at the default 64KB.
- **Fix**: Fallback loop trying 1MB → 512KB → 256KB, logging success at each level. System cap
  of 1MB is successfully set.

**Bug 3: DTS log flood making pipeline.log binary**
- **Root cause**: ffmpeg `-loglevel warning` caused a continuous stream of
  "Non-monotonous DTS in output stream" messages at every segment boundary. These warnings are
  cosmetic (the muxer auto-corrects DTS ordering), but they made the log file binary and
  unreadable with standard tools.
- **Fix**: Changed to `-loglevel error` in `_build_ffmpeg_cmd()`.

### Standalone /dev/null Test (March 13)

To understand the root cause of persistent ffmpeg `code=0` drops, the following standalone command
was run directly on the Jetson without any pipeline involvement:

```bash
ffmpeg -rtsp_transport tcp -i 'rtsp://admin:India%40789@192.168.28.155:554/video/live?channel=1&subtype=1' /dev/null
```

**Result: survived 27+ minutes with zero drops.**

Key differences from the segment buffer ffmpeg command:
1. **No `-stimeout`** — no socket timeout configured.
2. **No `-f segment`** — no disk writes; output is `/dev/null` (instant, zero I/O).
3. **No `-map`/`-c:v copy`/`-an`** — ffmpeg reads the stream but does no processing beyond
   the demuxer.

The segment buffer command uses `-f segment -segment_time 2 seg_%06d.h264`, which writes a new
2-second file to tmpfs (`/dev/shm`) every 2 seconds. Even on tmpfs, brief kernel I/O overhead
during segment file creation/flushing is sufficient to introduce stalls that trigger the CP Plus
firmware's TCP control-channel session timeout (graceful close, code=0).

### Root Cause Conclusion

The CP Plus firmware's ~4-5 minute TCP RTSP session timeout is triggered by brief stalls in the
ffmpeg process caused by `-f segment` disk I/O. When the RTSP TCP control channel goes quiet for
even a few seconds during a segment rotation, the firmware closes the session gracefully (code=0).

The `/dev/null` test emulates an ideal zero-overhead consumer: the camera session is never starved
because ffmpeg never blocks. The segment buffer absorbs the resulting drops, but the drops still
occur.

**Next step (planned, not yet implemented):** Dual-ffmpeg architecture to emulate `/dev/null`
conditions for the camera session while still producing segments for the pipeline:
1. **Reader**: `ffmpeg -rtsp_transport tcp -i {url} -f mpegts pipe:1` — reads RTSP, writes MPEGTS
   to stdout pipe. Zero disk I/O. No `-stimeout`. Camera session never sees a stall.
2. **Segmenter**: `ffmpeg -f mpegts -i pipe:0 -f segment -segment_time 2 seg_%06d.h264` — reads
   MPEGTS from stdin pipe, writes segments. Disk I/O fully decoupled from camera session.

A 1MB pipe between reader and segmenter absorbs momentary segmenter stalls. If the segmenter
stalls, only the pipe fills — the reader's RTSP session is unaffected.

---

## March 13, 2026 (continued): Dual-ffmpeg Implementation and Buffer Tuning

### Dual-ffmpeg Architecture Implemented

The dual-ffmpeg `/dev/null` emulation was implemented in `segment_buffer_helper.py`:

**Reader command** (`_build_ffmpeg_reader_cmd`):
```
ffmpeg -hide_banner -loglevel error -nostdin
       -rtsp_transport tcp -stimeout 10000000
       -i {rtsp_url}
       -map 0:v:0 -c:v copy -an -f h264 pipe:1
```

**Segmenter command** (`_build_ffmpeg_segmenter_cmd`):
```
ffmpeg -hide_banner -loglevel error
       -f h264 -r {fps} -i pipe:0
       -map 0:v:0 -c:v copy -an
       -f segment -segment_time 2 -reset_timestamps 1 seg_%06d.h264
```

Reader and segmenter are connected by a 1MB pipe (`F_SETPIPE_SZ`). Reader writes raw H264
to the pipe; segmenter reads from it and writes `.h264` segment files.

### Bug: MPEGTS Timestamp Discontinuity (Test 1)

**Symptom**: After a camera drop (code=0) and reconnect, the segmenter stopped writing after
exactly 262144 bytes (256KB = ~1s video). Segmenter process stayed alive but produced no more
data. Segment files accumulated at exactly 256KB.

**Root cause**: Initial implementation used MPEGTS (`-f mpegts`) as the pipe format. When the
camera had a TCP event, MPEGTS timestamps jumped discontinuously. The segment muxer compared
the new timestamps against its internal state from before the reconnect and got stuck on the
segment boundary, unable to determine when to cut.

**Fix**: Changed pipe format to raw H264 (`-f h264` on both reader and segmenter, with
`-r {fps}` added to segmenter to assign its own timestamps). Raw H264 carries no timing
metadata. The segmenter assigns timestamps based solely on the `-r fps` parameter, which
is monotonically increasing and unaffected by camera timestamp resets.

### Bug: TCP FIN-WAIT-1 Reader Stall Without -stimeout (Test 2)

**Symptom**: After a camera drop (code=0), the reader process stayed alive in `do_sys_poll`
kernel state. The pipe to the segmenter was empty. The segmenter was blocked in `pipe_read`.
`ss -tnp` showed the reader's TCP socket to the camera in state `FIN-WAIT-1` indefinitely.

**Diagnosis**: Without `-stimeout`, when the camera stops sending RTP (RTCP timeout), ffmpeg
sends a TCP FIN to the camera but the camera firmware does not ACK it promptly. The reader
gets stuck in FIN-WAIT-1 waiting for the ACK, during which it stops writing to the pipe.
The segmenter's stdin starves. Buffer drains.

**Fix**: Added `-stimeout 10000000` (10 seconds) back to the reader command. When no data
arrives for 10 seconds, the reader exits. The writer_loop detects the exit, terminates the
segmenter, and starts a new epoch.

### Result: First Transparent Drop (Test 3, same session)

With raw H264 pipe format and `-stimeout 10000000`:
- `14:07:25`: reader exited code=0 (camera drop)
- `14:07:26–14:07:31`: Stream 0 FPS = 124-125 frames (24.8–25.0 fps)

First confirmed transparent drop absorption in dual-ffmpeg mode.

### Remaining Issue: Buffer Drain During Camera Recovery

After each camera drop (code=0), the reconnect attempt immediately times out with
`Connection timed out` (code=1, after the 10s `-stimeout`). The CP Plus firmware needs
~25-40 seconds before it accepts a new TCP connection after closing the previous session.

With the 60s buffer (30 segments, `low_watermark = 15`):
- Buffer at steady state was empirically ~21 segments (not 30), due to slight drift between
  feeder rate (0.5/s) and writer rate (~0.47/s, from camera running at ~24fps actual vs 25
  assigned in `-r 25.0`).
- Camera recovery: 2-3 failed 10s attempts = 24-36s with no new segments.
- 21 segments, draining at 0.5/s for 30s = 6 remaining → below `low_watermark = 15`.
- Result: rebuffer triggered → Stream 0 0fps for ~30-60 seconds until buffer refilled.

### Fix 1: Lower Low Watermark (`target // 2` → `target // 4`)

`low_watermark_segments` changed from `target // 2` to `target // 4`. For 30 segments:
- Old: watermark = 15, safe window = (30-15)/0.5 = 30 seconds
- New: watermark = 7, safe window = (21-7)/0.5 = 28 seconds from empirical baseline

With empirical baseline of 21, this extended safe window from ~12s to ~28s before rebuffer
triggered. Still not enough to survive 30-40s camera recovery.

### Fix 2: Increase Buffer to 120 seconds

`HICON_SEGMENT_BUFFER_DELAY_SEC_0` increased from 60 → 120 (`.env`).
`HICON_SEGMENT_BUFFER_RETENTION_SEC_0` increased from 120 → 180 (`.env`).

With 120s buffer:
- `target = 60 segments`, `low_watermark = 15` (60 // 4)
- Steady-state empirical buffer at first drop: ~54 segments (drift ~6 over 3 minutes)
- Safe window: (54 - 15) / 0.5 = **78 seconds**
- Camera recovery ~30s: 54 - 30×0.5 = 39 > 15 → **no rebuffer triggered**

### Confirmed Result (March 13, 15:00–15:10 soak)

With dual-ffmpeg + raw H264 + `-stimeout` + `low_watermark = target // 4` + 120s buffer:

| Drop time | Exit sequence | FPS during recovery | Rebuffer? |
|-----------|--------------|---------------------|-----------|
| 14:56:59  | code=0 → code=1 (×1) | 24.4–25.6 fps continuously | None |
| 15:01:58  | code=0 → code=1 (×2) | 24.4–25.4 fps continuously | None |

Both drops fully transparent. No 0fps events, no rebuffer log entries.
Stream 0 FPS held at 24-25fps through the entire camera recovery period.

### Summary of Final Configuration

| Parameter | Value |
|-----------|-------|
| Reader format | `-f h264 pipe:1` |
| Segmenter format | `-f h264 -r 25.0 -i pipe:0` |
| Reader `-stimeout` | `10000000` (10s) |
| Inter-ffmpeg pipe size | 1MB (`F_SETPIPE_SZ`) |
| Segment duration | 2s |
| Buffer delay | 120s (60 segments) |
| Retention | 180s |
| Low watermark | `target // 4` = 15 segments |
| Safe window | ~78 seconds |


---

## March 13, 2026 (afternoon) — Stream 2 Segment Buffer Rollout

### Context

Stream 2 is the second CP Plus pouring camera (192.168.28.162), same firmware family as
Stream 0 (192.168.28.155). Having validated the segment buffer approach on Stream 0, the
same architecture was applied to Stream 2 to eliminate its TCP drop behavior.

### Bugs Found During Stream 2 Rollout

#### Bug 1: Missing pipeline_config keys (root cause of Stream 2 using rtspsrc)

`hicon_pipeline.py` builds a `pipeline_config` dict and passes it to `DeepStreamPipelineBuilder`.
The Stream 0 segment buffer keys (`use_segment_buffer_0`, `segment_buffer_dir_0`, etc.) were
present, but the five corresponding Stream 2 keys were never added to the dict.

Result: `config.get('use_segment_buffer_2', False)` returned the default `False` inside
`DeepStreamPipelineBuilder.__init__`, so the Stream 2 source fell through to `_create_decode_chain`
(rtspsrc) even when `HICON_USE_SEGMENT_BUFFER_2=true` was set in `.env`.

Symptom: "Stream 2: rtpbin configured" appearing in logs instead of
"Stream 2: segment buffer source created".

Fix: Added all five `segment_buffer_2` keys to the `pipeline_config` dict in `hicon_pipeline.py`.

#### Bug 2: Wrong codec — Stream 2 camera sends HEVC, not H.264

`.env` had `HICON_RTSP_CODEC_2=h264`. Direct ffprobe test revealed the camera at
192.168.28.162 actually sends **HEVC (H.265)**:

```
ffprobe /tmp/stream2_test.h264
→ Input #0, hevc, from '...': Video: hevc (Main), yuv420p(tv), 640x480, 25 fps
```

The segment buffer helper's ffmpeg commands used `-f h264` hardcoded (not `self.codec`),
so the reader output HEVC data labeled as H264. The segmenter received HEVC NAL units but
tried to parse them as H264:

```
ffmpeg-segmenter: dimensions not set
ffmpeg-segmenter: Could not write header for output file #0 (incorrect codec parameters ?): Invalid argument
```

Fix 1: Changed `HICON_RTSP_CODEC_2=h265` in `.env`.

Fix 2: Made segment_buffer_helper.py codec-aware:
- `_build_ffmpeg_reader_cmd()`: use `"hevc" if self.codec == "h265" else "h264"` for `-f` format
- `_build_ffmpeg_segmenter_cmd()`: same, plus codec-aware output extension (`.h265` / `.h264`)
- `parse_segment_ref()`: added `.h265` to accepted suffixes
- `list_complete_segments()`: changed glob from `seg_*.h264` to `seg_*.*`
- GStreamer chain in `_create_segment_buffer_chain()`: already used `self.config.get(f'rtsp_codec_{stream_id}')` to select `h265parse` vs `h264parse` — no change needed there.

#### Bug 3: FPS watchdog startup grace too short

The bus handler's `startup_grace` for Stream 0 was `delay_sec + 10 = 130s`. Under normal
conditions this is enough — 120s to fill the buffer, ~5s for first frames to reach the
pipeline. But if the camera is slow to accept the first TCP connection (e.g., needs 30-40s
recovery from a prior session), total startup can reach `50 + 120 = 170s`, exceeding the
130s grace.

Stream 2 had the default 30s grace (no override was wired). After the fixes above, the
segment buffer helper started correctly but the FPS watchdog fired before priming completed:

```
[FPS-WATCHDOG] Stream 2 at 0fps for 5s — restarting
```

Fix: Changed grace formula from `delay_sec + 10` to `delay_sec + 30` (→ 150s for 120s buffer).
Added `stream_startup_grace_overrides` dict to `BusHandler` to support per-stream overrides,
and wired Stream 2's grace from `hicon_pipeline.py`.

#### Bug 4: Stream 2 warn→restart escalation fired before recovery completed

After the grace fix, a camera TCP drop (both cameras simultaneously at ~16:04:34) caused
Stream 2 to go to 0fps. The `warn` policy logs warnings but escalates to a hard restart
after 90s of consecutive 0fps. Stream 2 was already at 0fps for 75s when the grace expired,
so only 15s remained before the 90s cap triggered a restart.

Root cause: Stream 0 has `_stream0_watchdog_suppressed()` which reads the helper's
`state.json`. While the helper is in `mode=rebuffering`, the watchdog counter resets
(suppressing the escalation). Stream 2 had no equivalent suppression.

Fix: Added `_segment_buffer_watchdog_suppressed(stream_id)` to `BusHandler` — a generic
version that reads any stream's `state.json`. Extended the watchdog check to call this
for any stream, not just Stream 0. Wired `stream_segment_buffer_state_paths[2]` from
`hicon_pipeline.py`.

The helper writes `state.json` with `mode` set to `"buffering"` (initial fill) or
`"rebuffering"` (post-drop recovery). While either mode is active, the watchdog suppresses
the 0fps counter for that stream, allowing arbitrarily long recovery without pipeline restart.

### Validated Result

After all fixes, both segment buffer streams prime simultaneously:
```
16:16:17 Stream 0: buffer primed (60 pending segments, target=60)
16:16:17 Stream 2: buffer primed (60 pending segments, target=60)
```

All three streams at ~25fps steady:
```
[FPS] Stream 0: 25.2 fps | Stream 1: 25.0 fps | Stream 2: 26.0 fps
```

Inference active immediately after priming (NEW HEAT CYCLE fired within 1s of first frames).

### Summary of Stream 2 Configuration

| Parameter | Value |
|-----------|-------|
| Camera | CP Plus 192.168.28.162, `subtype=1` |
| Codec | **HEVC (H.265)** — different from Stream 0 H.264 |
| Reader format | `-f hevc pipe:1` |
| Segmenter format | `-f hevc -r 25.0 -i pipe:0` |
| Segment files | `seg_%06d.h265` |
| GStreamer parser | `h265parse(config-interval=-1)` |
| Buffer delay | 120s (60 segments) |
| Retention | 180s |
| Low watermark | `target // 4` = 15 segments |
| Startup grace | 150s (`delay_sec + 30`) |
| Watchdog policy | `warn` + state.json suppression during rebuffering |

---

## March 16–19, 2026: nvurisrcbin Migration, NVR Investigation, and Backpressure Proof

### March 16: Migration from Segment Buffer to nvurisrcbin

**Goal**
Simplify the pipeline by removing the dual-ffmpeg segment buffer architecture and using DeepStream's
built-in `nvurisrcbin` element, which has native RTSP reconnection support.

**Change**
- All 3 streams switched to `nvurisrcbin` (built-in RTSP auto-reconnect, 2s reconnect interval).
- Segment buffer code retained but disabled (`HICON_USE_SEGMENT_BUFFER_0/2=false`).
- Stream 0: `reconnect-interval=10`, `protocol=auto`, `latency=4000`, `num-extra-surfaces=12`.
- Streams 1&2: `reconnect-interval=2`, `protocol=tcp`, `latency=2000`, `num-extra-surfaces=8`.
- `warn_safety_cap_sec=300` when nvurisrcbin active (was 90 for segment buffer).
- **CRITICAL**: `async-process=False` on mux must NEVER be used with nvurisrcbin — blocks reconnection permanently.

**Observed Result**
- Pipeline runs with all 3 streams active. nvurisrcbin auto-reconnects after each CP Plus drop.
- Recovery takes 10–65s depending on stream. Zero pipeline restarts needed.
- Drops still occur every ~5 minutes (CP Plus firmware unchanged).

**Finding**
nvurisrcbin provides acceptable "tolerate and recover" behavior without the complexity of segment buffers
or MediaMTX proxies. The trade-off is brief inference gaps (~10-65s) every 5 minutes vs the segment
buffer's fully transparent recovery. Acceptable for the current deployment given the reduction in
moving parts (no ffmpeg subprocesses, no /dev/shm spool, no helper process).

---

### March 19: Camera Hardware Identification via ONVIF

**Goal**
Identify exact camera models and firmware versions for all 3 CP Plus cameras and the NVR.

**Method**
ONVIF `GetDeviceInformation` SOAP request to each camera's HTTP port 80 (no authentication required
for this endpoint on CP Plus).

**Results**

| Device | IP | Model | Firmware | Serial |
|--------|-----|-------|----------|--------|
| Stream 0 (Process) | 192.168.28.155 | CP-UNC-TC41L5C-VMD-LQ | 2.860.00AT001.0.R | P49B5F2HBFVWK386 |
| Stream 1 (Pyrometer) | 192.168.28.152 | CP-UNC-TA41L3C-D-LQ | 2.860.00AT002.0.R | OKI36U73AQ00SAOS |
| Stream 2 | 192.168.28.162 | CP-UNC-TC41L5C-VMD-LQ | 2.860.00AT001.0.R | HHR46PNVQR1HTZ68 |
| NVR | 192.168.28.6 | *(HTTPS-only ONVIF, couldn't pull)* | — | — |

- Streams 0 & 2: same model (TC41L5C-VMD-LQ, 4MP bullet, 5mm lens, same firmware).
- Stream 1: different model (TA41L3C-D-LQ, 4MP turret/dome, 3.6mm lens, slightly newer firmware).
- All cameras confirmed as CP Plus (Dahua OEM), 4MP sensors.
- HTTP port 80: web UI accessible. No Dahua CGI endpoints (`/cgi-bin/magicBox.cgi` → 404).
- No RPC2 endpoints either. Only ONVIF SOAP and the proprietary web UI.

---

### March 19: NVR Channel Mapping Investigation

**Goal**
Access the CP Plus NVR (192.168.28.6) web UI to understand channel-to-camera mapping and evaluate
NVR-relay RTSP as a stability solution.

**Method**
SSH SOCKS proxy (`ssh -D 9090`) through the Jetson, then Chrome with `--proxy-server=socks5://localhost:9090`
to access `https://192.168.28.6` directly.

**Findings**
- NVR web UI: `CPPLUS NVR - Web View`, login `admin` / `NVR@321#`.
- NVR has 4 Hikvision NVRs registered on its "ADD IP CAM" page (not cameras):
  - Ch1: 192.168.28.7 (DS-8664NI-I8), Ch2: 192.168.27.3 (DS-7P32NI-K4),
    Ch3: 192.168.28.8 (DS-7764NI-M4), Ch4: 192.168.27.1 (DS-7P32NI-K4).
- Additionally has Ch8-10: other CP Plus cameras on 192.168.28.x (CP-UNC-TA41... and CP-UNC-DA41...).
- The 3 HiCon cameras are on a separate "Added Device" list:

| NVR Channel | IP | Port | Protocol | Status | Camera |
|-------------|-----|------|----------|--------|--------|
| 1 | 192.168.28.152 | 25001 | CPPLUS | Red (offline) | Pyrometer |
| 2 | 192.168.28.155 | 25001 | CPPLUS | Red (wrong password) | Process |
| 3 | 192.168.28.162 | 80 | ONVIF | Green (online) | Stream 2 |

- Ch1 and Ch2 use CP Plus proprietary protocol (port 25001), Ch3 uses ONVIF (port 80).
- Ch1/Ch2 showed "Wrong username or password" — credentials need to be fixed in NVR settings.
- NVR RTSP URL format: `rtsp://admin:NVR@321#@192.168.28.6:554/cam/realmonitor?channel={N}&subtype=1`

---

### March 19: NVR Live RTSP Soak Test Under DeepStream Load

**Goal**
Test whether routing streams through the NVR eliminates the CP Plus firmware drops. Theory: the NVR
records to HDD continuously without drops, so its RTSP re-stream might also be stable.

**Configuration**
- Streams 0 & 1 disabled. Only Stream 2 active, pointed at NVR Ch3:
  `rtsp://admin:NVR%40321%23@192.168.28.6:554/cam/realmonitor?channel=3&subtype=1`
- Pipeline running with nvurisrcbin, full inference active.

**Observed Result — 55 minutes of monitoring**
- **Drops every ~5 minutes, identical pattern to direct camera connection:**
  ```
  14:45:51 → 14:50:50 (+4:59)
  14:50:50 → 14:55:49 (+4:59)
  14:55:49 → 15:00:49 (+5:00)
  15:00:49 → 15:05:48 (+4:59)
  15:05:48 → 15:10:48 (+5:00)
  15:10:48 → 15:15:46 (+4:58)
  15:15:46 → 15:20:46 (+5:00)
  15:20:46 → 15:25:45 (+4:59)
  15:25:45 → 15:30:45 (+5:00)
  15:30:45 → 15:35:44 (+4:59)
  ```
- Each drop: 5–60s at 0fps, then recovery to 25fps via nvurisrcbin auto-reconnect.
- 204 total 0fps/reconnect log events in 55 minutes.

**Finding**
**NVR live RTSP does NOT solve the drop problem.** The NVR is just proxying the camera stream —
when the camera kills the TCP session to the NVR, the NVR also drops the downstream RTSP to the
Jetson. The NVR adds no buffering or stability to the live RTSP path.

---

### March 19: Simultaneous Soak — Proof That Drops Are Backpressure, Not Firmware

**Goal**
Run a standalone ffmpeg test (`/dev/null` sink) and the DeepStream pipeline **simultaneously**,
both consuming the same NVR RTSP stream, to determine whether the drops are caused by the camera
firmware or by pipeline backpressure.

**Test**
```
# Ran simultaneously for 6 minutes:
Standalone:  ffmpeg -rtsp_transport tcp -i rtsp://...NVR.../channel=3&subtype=1 -f null /dev/null
Pipeline:    hicon-vision.service (nvurisrcbin → nvinfer → tracker → osd → sink)
```

**Result**

| Client | Drops in 6 min | Notes |
|--------|---------------|-------|
| **Standalone ffmpeg → /dev/null** | **0** | Ran clean for full 6 minutes, terminated by timeout |
| **DeepStream pipeline** | **1** | Drop at 15:45:43, 0fps for ~40s, recovered at 15:46:29 |

Both clients were pulling the **same NVR RTSP stream** at the same time. The NVR did not kill the
session globally — only the pipeline's connection dropped.

**Finding**
**THIS IS THE SMOKING GUN.** The drops are caused by **pipeline backpressure**, not CP Plus firmware.

When the DeepStream pipeline stalls momentarily (GPU inference scheduling, GStreamer element blocking),
frames queue up and backpressure propagates to `nvurisrcbin`'s internal `rtspsrc`. The TCP socket
receive buffer fills up. The RTSP server (NVR or camera) sees the client isn't consuming data, the
TCP window shrinks to zero, and after a timeout the server closes that specific connection.

The standalone ffmpeg with `/dev/null` sink consumes frames instantly — zero backpressure. The TCP
socket never fills up. The server keeps the connection alive indefinitely.

**This reframes the entire investigation.** The CP Plus firmware timeout (~5 min) is real, but it
is **triggered by backpressure from the consumer**, not by an unconditional timer. A consumer that
reads fast enough (like ffmpeg → /dev/null) never triggers the timeout. The DeepStream pipeline's
occasional processing stalls are enough to trigger it.

**Implication:** The fix should focus on **decoupling the RTSP reader from the pipeline processing**
with sufficient buffering between them, rather than replacing cameras or using NVR proxies. The
segment buffer architecture (March 12) already solved this by fully decoupling the reader (ffmpeg)
from the consumer (GStreamer pipeline). Larger GStreamer queues between `nvurisrcbin` and `streammux`
might also work without the full segment buffer complexity.

---

### March 19: Camera 192.168.28.152 Firmware Upgrade Test

**Goal**
Test pyrometer camera (192.168.28.152) after firmware upgrade by site team.

**Observed Result**
- Camera reachable (ping OK, HTTP port 80 OK, ONVIF responds).
- ONVIF GetDeviceInformation: still reports firmware `2.860.00AT002.0.R` (same as before).
- **RTSP port 554 open but non-functional**: ffprobe returns `ECONNRESET` (connection reset) on
  OPTIONS request. Both `/cam/realmonitor` and `/video/live` paths fail with "Invalid data found".
- TCP and UDP transport both fail.
- ONVIF-initiated reboot (`SystemReboot`): camera rebooted successfully (60s downtime), but RTSP
  service remained non-functional after reboot.
- Camera then went fully offline (not pinging) — consistent with intermittent power/network issues
  seen throughout investigation.

**Finding**
The firmware version string is unchanged, so either the upgrade didn't take effect or it was a
configuration change rather than a firmware flash. The RTSP service appears broken — port 554 accepts
TCP connections but immediately resets them. The camera's HTTP web UI and ONVIF work fine.

**Next Step**
Camera needs physical inspection. Possible causes: RTSP service disabled in camera settings after
firmware upgrade, or firmware upgrade corrupted the RTSP module. Access the camera's web UI
(via SSH SOCKS proxy at `http://192.168.28.152`) to check if RTSP is enabled in the network settings.

---

### March 19: API 422 Fix — Pouring Sync

**Bug**
`pouring_end_time` set to empty string (`""`) for incomplete pouring sessions. API rejects:
`"Value error, Timestamp must be in YYYY-MM-DD HH:MM:SS format"`.

Same pattern as the March 18 melting/tapping fix where `tapping_start_time: null` was rejected.

**Fix**
- Added `pouring_skipped_sync_ids` set in `sync_manager.py`.
- Skip pouring records with missing `pouring_start_time` or `pouring_end_time` from API payload.
- Include skipped IDs in `pouring_complete` set so they don't block melting sync intersection.
- Pattern matches the existing `melting_skipped_sync_ids` fix.

---

### Updated Comparison Table (All Configurations Tested)

| Configuration | Drops in 10 min | Recovery time | Drop interval | Notes |
|---|---|---|---|---|
| rtspsrc direct TCP (March 6) | continuous stall | pipeline restart | ~2–5 min | |
| nvurisrcbin direct TCP (March 7) | ~3–4 | 10–20s | ~3–5 min | |
| ffmpeg bridge via NVR TCP (March 9) | many | 15–20s | 25–33s | NVR overloaded |
| ffmpeg bridge direct TCP (March 11) | 14 | 15–20s | ~3.5 min | |
| MediaMTX UDP proxy (March 11) | 1–2 | 15–25s | ~5–9 min | |
| **Segment buffer 60s spool (March 12)** | **0** | **0s (transparent)** | **∞** | **Best — fully decoupled** |
| nvurisrcbin direct TCP (March 16) | ~2 | 10–65s | ~5 min | Simpler, tolerates drops |
| **NVR relay via nvurisrcbin (March 19)** | **~2** | **5–60s** | **~5 min** | **NVR does NOT help** |
| Standalone ffmpeg → /dev/null (March 19) | **0** | — | **∞** | **Proves backpressure is root cause** |

### Updated Root Cause Analysis

The March 10 finding ("CP Plus firmware has ~3-5 min RTSP session timeout baked into firmware") is
now refined:

**The CP Plus firmware has a TCP receive-window timeout, not an unconditional session timer.** When a
client stops consuming RTSP data fast enough (TCP window shrinks toward zero due to downstream
backpressure), the camera firmware closes the connection after a timeout (~3-5 minutes of degraded
consumption, or shorter under heavy backpressure). A client that reads at line rate (like ffmpeg
to /dev/null) can maintain the connection indefinitely.

This means:
1. **Camera replacement is NOT required** — the firmware behavior is triggered by the consumer, not
   by an internal timer.
2. **The segment buffer remains the optimal solution** — it fully decouples the RTSP reader (ffmpeg,
   which reads at line rate) from the pipeline consumer (GStreamer, which has variable processing time).
3. **Larger GStreamer queues** between nvurisrcbin and processing elements might reduce drop frequency
   by absorbing brief stalls, but cannot eliminate drops entirely because prolonged inference stalls
   will still trigger the firmware timeout.
4. **The NVR adds no value** for drop mitigation — it just proxies the same stream with the same
   backpressure sensitivity.

---

## March 19, 2026 (continued): NVR Full Soak Test and Camera Replacement

### NVR 12-Minute Soak Test (All Streams via NVR)

**Goal**
Test whether routing Streams 1 and 2 through the NVR (instead of direct camera) provides stability.

**Configuration**
- Stream 1: NVR Ch1 → `rtsp://admin:NVR@321#@192.168.28.6:554/cam/realmonitor?channel=1&subtype=1`
- Stream 2: NVR Ch3 → `rtsp://admin:NVR@321#@192.168.28.6:554/cam/realmonitor?channel=3&subtype=1`
- Pipeline settings: `drop-on-latency=True`, `num-extra-surfaces=16`, `premuxq=64`

**Result**

| Minute | Stream 1 FPS | Stream 2 FPS | Drops |
|--------|-------------|-------------|-------|
| 1 | 0.0 | 25.0 | 5 |
| 2 | 0.0 | 0.0 | 14 |
| 3 | 0.0 | 25.0 | 13 |
| 4 | 0.0 | 25.2 | 13 |
| 5 | 0.0 | 25.0 | 13 |
| 6 | 0.0 | 24.8 | 16 |
| 7 | 0.0 | 0.0 | 27 |
| 8–12 | 0.0 | 0.0 | 13–27 |

**Finding**
**NVR route is definitively worse than direct camera.** Stream 1 never achieved a single frame in 12
minutes. Stream 2 died at minute 7 and never recovered. nvurisrcbin's auto-reconnect failed to
re-establish NVR RTSP sessions, while direct camera connections consistently recovered in 10-65s.

**Decision**
NVR relay route permanently abandoned. Direct camera + nvurisrcbin is the production configuration.

---

### Camera 192.168.28.152 Post-Firmware-Upgrade Test

**Context**
Site team upgraded firmware on pyrometer camera (192.168.28.152). Camera rebooted.

**Result**
- ONVIF `GetDeviceInformation` still reports firmware `2.860.00AT002.0.R` (unchanged string).
- **RTSP now works** via `/video/live?channel=1&subtype=1` path — was returning `ECONNRESET` before reboot.
- Stream: H.265 (HEVC), Main profile, 640×480, 25fps.
- `/cam/realmonitor` path still times out (consistent with other CP Plus cameras).

**Finding**
The firmware upgrade (or the reboot itself) restored RTSP functionality. The camera was previously
stuck in a state where port 554 accepted TCP connections but immediately reset them.

---

## March 20, 2026: New Camera Deployment

### New Camera Installation

Three new **Hikvision DS-2CD2043G2-LI2U** (4MP ColorVu bullet) cameras deployed to replace
the old CP Plus (Dahua OEM) cameras. Identified via ISAPI (`/ISAPI/System/deviceInfo`).

| Stream | Old Camera | New IP | New Camera | Firmware | Serial (suffix) |
|--------|-----------|--------|------------|----------|-----------------|
| 0 | 192.168.28.155 (CP-UNC-TC41L5C-VMD-LQ) | **192.168.27.226** | DS-2CD2043G2-LI2U | V5.7.18 | FR7128559 |
| 1 | 192.168.28.152 (CP-UNC-TA41L3C-D-LQ) | **192.168.27.253** | DS-2CD2043G2-LI2U | V5.7.18 | FR7129271 |
| 2 | 192.168.28.162 (CP-UNC-TC41L5C-VMD-LQ) | **192.168.28.119** | DS-2CD2043G2-LI2U | V5.7.19 | FW8319581 |

**Credentials:** `admin` / `india@789` (lowercase, URL-encoded: `india%40789`)
**RTSP URL:** `rtsp://admin:india%40789@{IP}:554/video/live?channel=1&subtype=1`

**Camera specs (all 3 identical model):**
- Model: Hikvision DS-2CD2043G2-LI2U (4MP ColorVu bullet with hybrid light)
- Codec: H.265 (HEVC), Main profile
- Resolution: **2688×1520** (4MP full resolution on both main and sub-stream)
- Frame rate: 25fps
- MAC OUI: `bc:29:78` (Hikvision)
- ONVIF: **disabled** in camera settings (404 on `/onvif/device_service`)
- ISAPI: available on HTTP :80 (streams 0, 1) / HTTPS :443 (stream 2)
- RTSP OPTIONS: returns 200 without authentication (different from CP Plus which required auth)

**Note:** Sub-stream (`subtype=1`) returns the same 2688×1520 as main stream — not configured to
lower resolution. Should be set to ~720p via camera web UI to reduce HW decoder load on Jetson.

**Initial pipeline test (2026-03-20 14:19):** All 3 streams running at 25fps. Stream 0 shows
occasional FPS dips (14→32fps burst pattern) likely due to full 4MP HEVC decode load. No RTSP
disconnections observed in first 35+ minutes — already better than old CP Plus cameras.

---

### Backpressure Buffer Improvements Applied

Based on the March 19 simultaneous soak proof that drops are caused by pipeline backpressure,
the following GStreamer buffer tuning was applied to `gst_builder.py`:

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `premuxq` max-size-buffers | 64 | **128** | ~5s at 25fps, absorbs longer inference stalls |
| `premuxq` max-size-time | 0 (unlimited) | **5,000,000,000** (5s) | Time-based safety net |
| `postmuxq0` max-size-buffers | 16 | **64** | Match upstream queue depth |
| `latency` (nvurisrcbin) | 2000ms | **4000ms** | Larger jitter buffer prevents early packet drops |
| `drop-on-latency` | True | True (unchanged) | Drops late packets instead of accumulating |
| `num-extra-surfaces` | 16 | 16 (unchanged) | Sufficient decoder output surfaces |

**Expected effect:** Larger buffers absorb momentary inference stalls for up to ~5 seconds.
The TCP socket continues draining during this buffer period, so the camera/NVR server never sees
a zero-window condition. If stalls exceed 5 seconds, the leaky queue drops old frames (losing
~5s of video) instead of blocking the TCP socket (which would lose ~60s of reconnection time).

---

### Updated Comparison Table (All Configurations Tested)

| Configuration | Drops in 10 min | Recovery time | Drop interval | Notes |
|---|---|---|---|---|
| rtspsrc direct TCP (March 6) | continuous stall | pipeline restart | ~2–5 min | |
| nvurisrcbin direct TCP (March 7) | ~3–4 | 10–20s | ~3–5 min | |
| ffmpeg bridge via NVR TCP (March 9) | many | 15–20s | 25–33s | NVR overloaded |
| ffmpeg bridge direct TCP (March 11) | 14 | 15–20s | ~3.5 min | |
| MediaMTX UDP proxy (March 11) | 1–2 | 15–25s | ~5–9 min | |
| **Segment buffer 60s spool (March 12)** | **0** | **0s (transparent)** | **∞** | **Best — fully decoupled** |
| nvurisrcbin direct TCP (March 16) | ~2 | 10–65s | ~5 min | Simpler, tolerates drops |
| NVR relay via nvurisrcbin (March 19) | ~2 | 5–60s | ~5 min | NVR does NOT help |
| **NVR relay all-stream soak (March 19)** | **never recovered** | **∞** | **immediate** | **NVR definitively worse** |
| Standalone ffmpeg → /dev/null (March 19) | **0** | — | **∞** | Proves backpressure is root cause |
| nvurisrcbin + buffer tuning (March 20) | **TBD** | TBD | TBD | premuxq=128, latency=4000ms |

---

## April 10, 2026: Stream 0 MediaMTX / ffmpeg Relay Closure

### Goal

Re-test whether a MediaMTX relay could eliminate the recurring Stream 0 dropouts by interposing a
local RTSP hop between DeepStream and the Hikvision camera currently used for Stream 0
(`192.168.28.119`, `Channels/101` main stream).

### Findings

1. **MediaMTX v1.16.3 does not support `sourceNotReadyPolicy`.**
   - Confirmed from the installed binary on April 10, 2026.
   - This version can run `runOnNotReady`, but it has no `sourceNotReadyPolicy: wait`-style setting
     to keep existing readers attached while the upstream source disappears.
   - That means MediaMTX cannot provide the exact "hold reader sessions through source loss"
     behavior this test needed.

2. **The ffmpeg publisher experiment did not solve Stream 0 continuity.**
   - Test architecture:
     - DeepStream Stream 0 reader: `rtsp://127.0.0.1:8554/stream0`
     - MediaMTX path `stream0`: `source: publisher`
     - External publisher: `ffmpeg` loop publishing camera RTSP into MediaMTX

3. **`Channels/102` failed first.**
   - `ffmpeg` emitted repeated timestamp warnings:
     - packets arrived without timestamps
     - non-monotonic DTS was rewritten
   - MediaMTX initially showed the path online, but Stream 0 dropped to `0 fps` around
     `2026-04-10 18:12`.
   - After publisher timeout, MediaMTX repeatedly returned:
     - `no stream is available on path 'stream0'`

4. **`Channels/101` also failed.**
   - After restart on the main stream, MediaMTX accepted a publisher at `2026-04-10 18:17:05`.
   - The published path came up as **2 tracks (`H265`, `G711`)**, not the earlier single-track H.265
     path.
   - The Stream 0 reader churned repeatedly, reconnecting to MediaMTX every ~10-12 seconds, then the
     publisher timed out again.
   - MediaMTX again fell back to repeated:
     - `no stream is available on path 'stream0'`

5. **This experiment was worse than the supported baseline.**
   - Baseline direct-camera mode still suffers the known Stream 0 drop behavior under this stack, but
     it remains the least-bad supported path.
   - The MediaMTX/ffmpeg relay introduced new failure modes:
     - publisher-side timestamp issues
     - reader churn against localhost MediaMTX
     - complete path disappearance when the publisher timed out

### Decision

**Close the MediaMTX/ffmpeg relay investigation as not a fix on the April 10, 2026 stack.**

The supported production baseline remains:
- Stream 0 input: direct Hikvision main stream (`rtsp://.../Streaming/Channels/101`)
- MediaMTX: retained only for `stream0_overlay` / local relay duties
- `hicon-stream0-publisher.service`: disabled and removed from runtime startup

Any future zero-drop investigation should be treated as a new branch of work and gated on one of:
- upgrading beyond MediaMTX `v1.16.3`, or
- camera firmware / hardware changes

### Post-Rollback Verification

After disabling the experimental `hicon-stream0-publisher.service`, removing its installed unit,
reloading systemd, and restarting `hicon-mediamtx.service` plus `hicon-vision.service`:

- MediaMTX came back in overlay-only mode:
  - `MediaMTX stream0 source proxy disabled: HICON_CPPLUS_SOURCE_STREAM_0 is not set`
- The restarted pipeline config again showed Stream 0 on the direct camera URL:
  - `rtsp://admin:india%40789@192.168.28.119:554/Streaming/Channels/101`
- A post-rollback soak from approximately `18:30` to `18:45` confirmed that the relay-specific
  failure mode was gone:
  - no `localhost:8554/stream0` input after restart
  - no MediaMTX publisher churn
  - no repeated `no stream is available on path 'stream0'` loop during the soak

The remaining behavior matched the known direct-camera limitation instead:
- Stream 0 stalled at `18:37:28` and recovered after **30s**
- Stream 0 stalled again at `18:42:28` and recovered after **20s**

This is the expected closure state for this issue:
- rollback successful
- relay experiment retired
- remaining drops attributed to the camera-side / direct RTSP baseline, not to MediaMTX or ffmpeg

### Ops Contract Update

With Stream 0 back on the direct camera URL, `hicon-mediamtx.service` is no longer a production
ingest dependency for `hicon-vision.service`.

- `hicon-vision.service` should be allowed to start even if MediaMTX is stopped.
- MediaMTX remains installed only as an auxiliary service for optional `stream0_overlay` /
  local-relay workflows.
- No conditional auto-start is added for those optional workflows.
- If operators later enable `HICON_ENABLE_STREAM0_LOCAL_RELAY=true` or set
  `HICON_STREAM0_REMOTE_RELAY_URL`, they must ensure `hicon-mediamtx.service` is started
  separately before expecting overlay relay publishing to work.

---

## 2026-06-12 Addendum: ACTUAL ROOT CAUSE FOUND — Dead Default Gateway

All prior conclusions attributing the ~5-minute Stream 0 drops to "irreducible camera-side
hardware behavior" were **wrong**. Live multi-layer probing on 2026-06-12 found the real cause.

### Root Cause
Camera 0 (192.168.28.119) had its default gateway configured as **192.168.28.1 — an IP that
does not exist on the network** (ARP `(incomplete)` permanently). The camera firmware's periodic
gateway health check failed every cycle, and its recovery action bounced the camera's
network/streaming stack on a free-running **~298.5s** timer:

- established RTSP sessions silently died (no FIN/RST at kill time; later packets answered with
  `RST win 0` — the daemon lost its TCP state)
- brief 0–9s ICMP/HTTP blackout, then service resumed within seconds
- camera OS never rebooted (`deviceUpTime` 5.6 days at diagnosis)

### Evidence (live probes, 16:18–16:30 IST)
| Probe | Result |
|---|---|
| Outage cadence | 298.5s free-running, drifting ~1s earlier per cycle |
| Independent parallel ffmpeg session | froze at the same cycle instant; conn stayed ESTAB |
| Fresh TCP connects to :554 every 5s | succeeded throughout (old "95s refusal" no longer applies) |
| ICMP/HTTP to camera | 0–9s gap per cycle |
| tcpdump | camera ARPs for 192.168.28.1 before the cycle; 28.1 never resolves |
| Controls | cameras 1&2 (identical model/firmware) use gateway 192.168.27.1 which EXISTS → zero drops ever |
| ARP audit | no rogue/conflicting claims for .119 or .44 |

This also explains why replacing the CP Plus camera with a Hikvision (2026-03-20) did not stop
the cycle: the camera position stayed on 28.x with the same dead gateway.

### Fix (2026-06-12 19:09 IST)
- ISAPI PUT `DefaultGateway` 192.168.28.1 → **192.168.28.8** (NVR-1, always-on, answers ARP)
- Camera reboot to apply (required by firmware), back online 19:11:03
- The gateway is never actually used for routing (camera only talks on-subnet to the Jetson and
  NVR-1) — it just needs to answer ARP so the firmware health check passes.

### Fallback ladder (if the cycle ever returns)
1. Disable NVR-1 channel 34 temporarily (NVR-perpetrator test; `enableTiming=true` on ch34)
2. Change camera RTSP port 554→8554 (external-prober test)
3. Firmware update beyond V5.7.23 build 260320
4. Swap camera units 0↔2 (unit vs position differential)
