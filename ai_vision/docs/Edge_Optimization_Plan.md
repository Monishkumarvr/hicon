# Edge Optimization Plan — hicon-vision on Jetson Orin Nano 8GB (FINAL, merged)

**Date:** 2026-07-16
**Supersedes:** the audit-derived draft (this file's previous version) and the standalone "Jetson Orin Nano Production Optimization Plan". This is the merged, executable plan.
**Trigger:** Mould-GIE interval 9 failed the saturation gate (99%-GPU samples 33.3% → 37.8%); interval 11 restored.
**Objective:** eliminate GPU saturation bursts and idle frame-work so mould inference cadence can rise (adaptively to interval 5 during active pours) without regressing accuracy, stability, or the 15W envelope.

---

## Evidence base (measured 2026-07-16, read-only audits of the live system)

1. **The saturation burst is the mould model.** Mould = YOLO11s, 9.51M params @1280² (3.6× the params of the other models; pouring is actually a 640² nano). Each mould inference is a 60–90 ms GPU monolith; pouring (every 40 ms) and pyro (every 120 ms) queue behind it → 100–150 ms saturated windows ≈ the 25–38% of samples at 99%.
2. **Part of the "99%" is DVFS lag, not capacity.** tegrastats caught GR3D 98% at 509 MHz — `nvhost_podgov` upclocks late. Clocks are not pinned.
3. **CUDA brightness silently falls back to CPU** ("safe NV12 analysis topology is unavailable"): full-frame RGBA copy + full-frame cvtColor per frame on Streams 0 and 2 ≈ 240 MB/s of unified-memory traffic + 3–5 ms/frame GIL-held CPU. Deployed reality made explicit: `HICON_USE_CUDA_BRIGHTNESS=false` is now set in `.env` (behavior-identical; removes the misleading startup warning). `HICON_STREAM_0_ANALYSIS_PROBE_ENABLED=false` is a permanent safety setting (shared-NVMM/IOMMU crash boundary), not an optimization candidate.
4. **`kernel.panic=0` + `panic_on_oops=1`**: an oops hangs the box forever at a remote site.
5. `sugov:0` (schedutil governor) burns a measured 8.2% of a core; cores dither 729–1510 MHz under load.
6. Stream-0 convert `nvvidconv_osd_0` is force-pinned to GPU (`compute-hw=1`, `disable-passthrough`) as a workaround for CP Plus cameras replaced 2026-03-20 — a free 3–5 GPU-pts sits behind a two-line revert.
7. Healthy/no-action (verified): no memory leak, all surface maps have matching unmaps, SQLite already WAL + async writer, MJPEG demand-gated (zero cost at zero clients), unit env already tuned (MALLOC_ARENA_MAX=2, BLAS caps, lazy CUDA), NVMe/journald/socket buffers fine, no GUI.
8. **This kernel has no PSI** (`CONFIG_PSI` unset) — memory-pressure telemetry must come from cgroup `memory.events`/`memory.stat`, not `/proc/pressure`.
9. The ~45s all-stream RTSP outages (430 start events on 2026-07-16) are upstream L2/switch, proven not this box — tracked as a separate ops item.

## Constraints (fixed safety decisions)

- Existing model weights and architectures only. FP16/INT8 engine rebuilds allowed; **no retraining, no architecture change, no resolution-altered model exports.**
- No kernel replacement, PREEMPT_RT, RPS/XPS, IRQ affinity, `isolcpus`, `nohz_full`, or boot-parameter changes.
- No MAXN, no overclocking, no realtime scheduling, no DLA/PVA (absent on Orin Nano), no CUDA MPS.
- **Amended from the draft plan (evidence-driven exceptions to the kernel freeze):**
  - `kernel.panic=10` sysctl IS allowed — availability fix, not performance tuning.
  - GPU `min_freq` pin to the 15W ceiling (624.75 MHz) IS allowed — envelope-neutral; removes measured DVFS lag. This is not `jetson_clocks` (no CPU/EMC pinning, no power-mode change).
  - CPU governor `performance` IS allowed — envelope-neutral (cores already ride near max); reclaims the measured sugov cost. Revert if VDD_IN or temps object.
  - The 25W SUPER profile is **not banned but gated** (Phase 6): physical PSU verification + explicit approval + 24 h brownout soak. It is +47% GPU / +50% EMC sitting unused on disk.
- Keep THP→madvise as the only memory-policy change; keep zram, swappiness, CMA, NVMe scheduler, TCP autotuning, decoder surface counts as-is.
- **Stream-0 mux stays 1600×900** (a downscale of the 2688×1520 CH101 main stream — defensible as-is; zones/probe geometry are site-calibrated to it). 1280×720 would cut post-mux surface traffic ~36% but risks brightness-ROI calibration and small-object mould recall; it may only be revisited via A/B on recorded pours (never directly in production), and its payoff largely disappears once Phases 2–3 remove idle-frame conversion/copy work.
- Unchanged external event, DB, and API schemas. No raw frames through Python multiprocessing. No hand-written assembly unless profiling proves a persistent compiler-unvectorized hotspot after all architectural fixes.

---

## Phase 0 — Quick wins — **DONE, verified 2026-07-16**

| # | Change | Where | Result | Risk |
|---|--------|-------|--------|------|
| 0.1 | `kernel.panic=10` (keep `panic_on_oops=1`) | `/etc/sysctl.d/90-hicon.conf` | Live (`sysctl kernel.panic` = 10) | None |
| 0.2 | Pin GPU min_freq = 624.75 MHz | `hicon-clocks.service` (oneshot, enabled) → `/sys/class/devfreq/17000000.gpu/min_freq` | Live; tegrastats confirms GR3D always reports `@[624]`, even at 0% and 99% load — DVFS lag eliminated | Negligible at 15W |
| 0.3 | CPU governor → performance | same unit | Live; all 6 cores hold 1510 MHz under load (no more 729–1510 MHz dithering) | Watched VDD_IN/temps — no change |
| 0.4 | Pouring nvinfer `interval=0 → 1` | `configs/config_pouring_pgie.txt:11` | Live; confirmed on disk + picked up at startup | Low — state machines filtered ≥0.2 s |
| 0.5 | Revert CP-Plus forced-GPU convert → VIC | `pipeline/gst_builder.py`: guarded `_tune_stream0_postmux_convert_for_cp_plus` behind `if not self.use_nvurisrcbin_0` at all 3 call sites (matches the existing mux-tuning guard pattern) | Live; journal confirms `"skipping CP Plus postmux-convert tuning for nvurisrcbin (VIC path)"` on the production branch | Old stall could resurface — watching |
| 0.6 | Rebuild pouring engine on-device | deleted `best_pouring_hicon_hikvision_v1_557.onnx_b1_gpu0_fp16.engine`; nvinfer rebuilt on restart | Done; rebuild took ~7 min, "serialize cuda engine to file... successfully" | None |
| 0.7 | Demote per-second INFO logs to DEBUG/10s cadence | `processors/brightness_processor.py:389` (tapping ratio), `processors/pyrometer_processor.py:168` (raw detections) | Done; both switched to `logger.debug` at 250-frame (~10s) cadence; event start/end still INFO | None |
| 0.8 | THP → madvise | same oneshot unit | Live (`cat .../enabled` shows `[madvise]`) | Low |

**Gate result (tegrastats @200ms, 60s samples, before/after the same restart window):**

| Metric | Before | After | Target |
|---|---|---|---|
| avg GR3D | 58.3% | **45.2%** | ~40–45% |
| %samples ≥99% | 30.8% | **16.5%** | materially down |

FPS held steady at ~25fps × 3 streams for 90s post-restart with zero outages/errors after the one expected ~5s reconnect during the 7-min engine-build window (stream 2 starved of frames while the build thread was blocking; self-recovered via nvurisrcbin). Pre-existing test failures (`test_nvurisrcbin_stream0_honors_configured_tcp`, 2× `test_segment_buffer_helper` tests) confirmed unrelated via `git stash` diff — not regressions from this phase. Tracked as `hicon-2db` (closed).

## Update 2026-07-17 — 12h telemetry findings, interval rebalance, canonical mould registry

**Telemetry correction:** tegrastats `GR3D_FREQ %` is a short-window point sample, NOT an
interval average — "saturated minutes" in earlier drafts actually meant "samples landing
inside a burst". 12h analysis (733 samples): avg GR3D 42%, burst residency 19%,
**uncorrelated with all activity** (P(sat|pour)=P(sat|idle)=0.19) — and **zero pours occurred
in the whole window**, so every mould-GIE inference (the 60–90 ms monolith, unconditional at
interval 11) was wasted on an empty bay. Phase 0 pins held all night (only 624 MHz seen,
CPU floor 1510); tj ≤61.9 °C, VDD_IN ≤10.1 W, no leak, no GPU-caused frame loss.

**Mould jitter root-caused and fixed (hicon-7kk):** 819 distinct NvDCF mould IDs in 13h vs
peak 25 visible; old diagnostics ran post-trolley-filter and reported `visible=0
id_switches=0` all day. Landed:
1. **Interval rebalance:** pyro 2→4 (`config_pyrometer_pgie.txt`, with interval-derived
   temporal scaling in `pyrometer_processor.py`: `effective_out=12`, `idle_grace=0.8s`) funded
   mould 11→**7** (`config_mould_pgie.txt` + `.env`). Gate result (5 min @200 ms):
   **avg GR3D 34.5%, burst residency 15.3%** — 1.7× mould cadence for −10.7 GPU points net.
   Stage 2 (mould →5) allowed once a pouring shift of jitter data confirms.
2. **Canonical mould registry** (`pouring_processor.py`): moulds latch after ≥3 matches over
   ≥1 s at a stable trolley-relative position, then hold a stable `canonical_id` + EMA
   position matched **by position, not tracker ID** — churn-immune ("freeze unless moved /
   new mould placed", ported from the C++ CanonicalMould concept). Pour assignment + shadow
   `tracker_mould_count` use canonical IDs; raw-ID count kept as `raw_tracker_mould_count`
   diagnostic. Raw churning rects suppressed on OSD; stable canonical boxes drawn on the
   MJPEG overlay (no trolley ⇒ no mould boxes by design; `HICON_MOULD_RAW_OVERLAY=true`
   shows raw ghosts). Confidence hysteresis: latch ≥0.35, refresh ≥0.20. Rollback:
   `HICON_MOULD_CANONICAL_ENABLED=false`.
3. **Max logging:** `[mould-raw]` (pre-filter count/conf/area ~5 s), trolley-independent
   lifecycle (births/deaths/lifespan p50/global ID-switches) in the extended
   `[mould-tracker]` line + `[METRICS]` counters/gauges, per-frame CSV
   (`HICON_MOULD_DIAG_CSV=true` → `output/csv/mould_diag_*.csv`, bounded async writer),
   canonical latch/expiry events at INFO. First live data: mould confidences 0.62–0.88
   (solid — flicker is tracker churn, not detector noise).
4. Tests: `tests/test_mould_canonical_registry.py` (10 cases); suite 113 pass / same 3
   pre-existing unrelated failures.

**Solution space evaluated (recorded for posterity):** canonical registry (chosen);
interval rebalance (chosen, staged); confidence hysteresis in registry (chosen);
NvDCF probation/termination tuning (pending CSV data — params shared with mouth/trolley);
INT8 mould engine (Phase 4, needs calibration set); adaptive per-state scheduling
(deferred until registry data in); overlay-only smoothing (subsumed by registry);
960² re-export (banned: no resolution-altered models); mux/letterbox review (only if
CSV shows small-area flicker).

**RTSP outage storm is a NEW fault, not the June gateway bug (hicon-b22):** June fix
verified intact (camera 0 gw 28.8 alive; cams 1&2 gw 28.200 alive, MAC c8:4f:86:a2:62:44;
ARP entries match camera MACs; keepalive applied every startup). New signature: onset
Jul 15, **diurnal** (~50/hr 08:00–17:00, ~3/hr overnight), ~300 s per-stream periodicity,
52 all-stream-simultaneous clusters/day, TCP connect-phase failures, local side proven
clean. `hicon-rtsp-probe.service` (ffmpeg TCP control puller on cam 1 sub-stream →
`/var/log/hicon/rtsp_probe.log`) now running to prove external-vs-internal. **Site
hand-off: identify 192.168.28.200, what changed on the camera VLAN ~Jul 15, switch port
counters during a storm hour.**

## Phase 1 — Baseline & telemetry — **landed 2026-07-17 (app-level + sidecar); soak/NVTX still open**

Implemented:
- `ai_vision/utils/metrics.py` — `MetricsRegistry` (bounded per-name latency deques, counters, zero-arg gauge callables) + `MetricsReporter` background thread. Emits one structured `[METRICS]` JSON line per `HICON_METRICS_INTERVAL_SEC` (default 60s) via a dedicated `metrics` logger, then resets the latency window — a per-interval snapshot, not a running average.
- `utils/perf.py: timed_section` now always records into the registry (previously only logged on threshold breach) — the 3 existing call sites (`probe.stream0.cpu_analysis`, `probe.stream1.pyrometer`, `probe.stream2.pouring`) get p50/p95/p99/max for free, no new hot-path instrumentation.
- Queue-depth gauges registered for `AsyncDBWriter` (`_queue.qsize`, `_queue_full_count`) and `AsyncScreenshotWriter` (`_queue.qsize`) — read at report time, not pushed per-item.
- Process RSS/threads, thermal zones, zram, and this process's own cgroup `memory.current`/`memory.stat`/`memory.events` (path resolved from `/proc/self/cgroup`, not hardcoded to a unit name) — all unprivileged-readable, verified on-device.
- **GPU/NVDEC/VIC/EMC/power are deliberately NOT in the app process** — confirmed no unprivileged sysfs path exists for GR3D load% on this kernel (checked `/sys/devices/17000000.gpu`, `/sys/class/devfreq/17000000.gpu`, `/sys/kernel/debug`). Added a separate root-owned `hicon-tegrastats.service` (`tegrastats --interval 60000 --logfile /var/log/hicon/tegrastats.log`, weekly-rotated via `/etc/logrotate.d/hicon-tegrastats`) — live and logging, confirmed one line/60s with full GR3D/NVDEC/VIC/EMC/VDD_IN/temp coverage. Does not touch `hicon-vision.service`.
- Tests: `tests/test_metrics.py` (8 cases — percentile correctness, reset vs. preserve, counters, gauges incl. exception handling, `timed_section` integration). Full suite: 103 passed, same 3 pre-existing unrelated failures, zero regressions.
- **Not yet activated in the live process** — the code ships in this commit but `MetricsReporter` starts on the *next* `hicon-vision` restart (bundled with Phase 2, to avoid a second same-night restart after Phase 0's).

Still open (deferred, not blocking Phase 2):
- DeepStream native component-latency measurement + NVTX ranges.
- The four 30-min labeled baselines (idle / 1-client / 3-client / active-event) — need real operating windows once the reporter is live.
- Nsight Systems traces (only during an approved outage, service stopped) — schedule separately.
- Passive `/proc/interrupts`/softnet/CMA observation — already covered by the earlier kernel/OS audit in this doc's evidence base; re-check only if Phase 2+ changes threading.

## Phase 2 — Remove continuous frame-processing waste

1. **Keep the analytics path in NV12.** Restructure the tail: `analytics → tee → NV12 fakesink` and `└→ leaky queue → subscriber valve → conversion → OSD → JPEG encoder`. Valve sits before conversion+OSD and closes after the existing 5 s subscriber grace — idle preview performs zero conversion/render/map/encode.
2. Preview encoding: `nvjpegenc` where available; **encode each source frame once and broadcast to all clients** — never per-client encode in generators. (Defer the separate `hicon-preview.service` + `/dev/shm` ring: well-designed but solves a measured non-problem today; revisit only if preview cost shows up in Phase 1 telemetry.)
3. Interim (until Phase 3): **ROI-first brightness** — slice zone bboxes from RGBA before `cvtColor` (or use the R channel per the documented pattern) instead of full-frame convert at `brightness_processor.py:263`; compute ROI stats on the mapped view and copy full frames only on tracker start/end events (screenshot path).
4. Pyrometer: refactor to metadata/state evaluation + transition-only screenshot requests; remove `_last_frame`; Stream 1 must not map surfaces continuously during an active event.
5. Drop raw screenshot frame references before enqueue when raw saving is disabled; remove redundant copies in color-convert/resize.

## Phase 3 — Native NV12 fused analysis hot path

- **The concurrent decoupled analysis branch stays permanently disabled** (`HICON_STREAM_0_ANALYSIS_PROBE_ENABLED=false` guards the known shared-NVMM/IOMMU crash boundary; it was the live failure boundary: "Unable to draw rectangles", pyro CUDA OOM). The sequential inline NV12 element below is the ONLY sanctioned path to GPU/native brightness; flipping the topology flags is never an acceptable shortcut.
- Build a **sequential in-line NV12 analysis element before inference/tracking** that finishes and unmaps before downstream — no concurrent NvBufSurface access.
- Fuse tapping + deslagging + spectro + pouring-probe brightness into one map/sync per frame; read only configured ROI rects; publish compact metadata. Python keeps temporal state, event decisions, API behavior.
- Implement `native_cpu` first: preallocated masks/work-queues/labels/result buffers; early rejection in NV12 Y/UV space; scanline or two-pass connected components (no `vector<bool>`/BFS); `-O3 -mcpu=cortex-a78 -flto`; check vectorization reports; NEON intrinsics only for loops the compiler misses. No assembly, no `-ffast-math`.
- `native_cuda` (fused ROI-only kernel, reusable device/pinned workspace, scalar results only) **only if** native_cpu misses its latency gate AND GPU 99%-residency is already <20%.
- Selection gate: Python and native run in shadow on recorded input (not live surfaces); require frame-level ratio tolerance + exact event-transition parity.

## Phase 4 — Smooth and reduce inference work

1. **Parser rewrite** (`NvDsInferParseYoloCuda`): thread-local reusable workspace instead of per-inference `thrust::device_vector`; GPU decode+threshold; CUB/bounded-atomic compaction; copy only count + topk=300 candidates; keep DeepStream NMS initially. Move the pyrometer's 33,600-row CPU parser to the CUDA parser only after bbox/class/confidence parity passes.
2. **INT8 engine builds — reordered to match the burst thesis: mould first** (the 60–90 ms monolith causing saturation), **then pyro** (~25–29% GPU duty), **then pouring** (640² nano, already halved by Phase 0.4). Calibration: 500–1,000 representative site frames (empty scenes, glare, occlusion, smoke, trolley movement, complete pours). Preserve input geometry; reject any engine missing accuracy gates. Service stopped during every export/build.
3. **Adaptive mould scheduling** (after the above pass, shadow-tested):
   - Idle/no trolley: interval 24; trolley or ladle-mouth armed: 11; confirmed active pour: 5; hold 5 for 5 s post-pour, then idle.
   - Set nvinfer properties via the GLib main context, never from a streaming probe.
   - **Joint constraint with Phase 5: tracker `earlyTerminationAge` and shadow age must exceed the max interval (24)** — a 12-frame termination age would kill every mould track at idle cadence (documented pitfall).
   - If adaptive fails parity: stay static and benchmark 11 → 9 → 7 → 5, promoting only gate-passing candidates.

## Phase 5 — Tracker, queues, state ownership, services

- **NvDCF A/B, one variable each:** featureImgSizeLevel 2→1; shadow age 2400→600; earlyTerminationAge 50→**28** (not 12 — must stay > max adaptive interval 24); tracker scaling `compute-hw=2` (VIC); target cap 64→32 only if measured peak occupancy <24. Retain NvDCF — simpler trackers won't bridge the mould detector's long gaps.
- Queues: instrument occupancy first, then bound post-mux/display queues to 3–4 buffers (~160 ms), downstream-leaky. Don't change decoder surface counts during mould rollout.
- **Single-owner event-coordinator thread:** probes enqueue immutable metadata events only; coordinator exclusively mutates `HeatCycleManager`; ordering = timestamp, source ID, source sequence; DB backfill and reads off-probe.
- DB writer: one persistent WAL connection owned by the writer; short batched transactions; lossless O(1) terminal-event ingress with high-water alerts; keyed latest-value storage for replaceable checkpoints; sync status through the same writer contract; indexes for synced/date, event-type/end-time, start-time.
- Logging: bounded `QueueHandler` → single `QueueListener`; routine diagnostics aggregated every 60 s; never per-frame INFO.
- Move cloud sync to `hicon-sync.service`, communicating via SQLite WAL. Keep GStreamer/pyds/TRT in one vision process.
- Benchmark `cv2.setNumThreads(1)` vs `(2)`; pick lower p99.
- Shutdown ordering: reject new work → quiesce probes → stop pipeline → drain coordinator → drain screenshots/DB/logging/sync.
- `hicon-vision.service`: **`MemoryHigh=4G` + `MemoryMax=5G`** (amended: memory.high throttling on a PSI-less kernel presents as silent probe stalls; a hard kill + `Restart` is diagnosable). `CPUWeight=200`. A/B default placement vs `CPUAffinity=1-5`, keep only if p99 improves. No realtime scheduling. Keep restart limiting (5/300 s).
- Disable Docker/containerd, printing, Bluetooth, ModemManager, Samba/Avahi, PackageKit, udisks only after dependency inspection confirms unused. Retain SSH, Tailscale, NetworkManager, nvfancontrol, nvzram, NVIDIA services.

## Phase 6 — Power headroom (optional, gated, requires explicit approval)

1. Physically verify PSU class (45W-class supply required for SUPER modes); review VDD_IN peaks from Phase 1 telemetry (~9.9 W avg today).
2. `ln -sf /etc/nvpmodel/nvpmodel_p3767_0003_super.conf /etc/nvpmodel.conf` → reboot → `nvpmodel -m 1` (25W: GPU 918 MHz +47%, EMC 3199 +50%; CPU max drops 1510→1344 MHz — if probes tighten, that's the signal to stop, not to jump to MAXN_SUPER).
3. Fan profile quiet → cool in `/etc/nvfancontrol.conf`.
4. 24 h brownout soak: any spontaneous reset with empty dmesg = revert immediately (sister-site learning).

---

## Runtime interfaces (add or normalize)

```
HICON_MELTING_BACKEND=python|native_cpu|native_cuda
HICON_MELTING_SHADOW_MODE=true|false
HICON_PREVIEW_BACKEND=gstreamer|opencv
HICON_OPENCV_THREADS=1|2
HICON_METRICS_INTERVAL_SEC=60
HICON_PERIODIC_LOG_INTERVAL_SEC=60
HICON_MOULD_GIE_SCHEDULE=static|adaptive
HICON_MOULD_GIE_INTERVAL_IDLE=24
HICON_MOULD_GIE_INTERVAL_ARMED=11
HICON_MOULD_GIE_INTERVAL_ACTIVE=5
HICON_MOULD_GIE_POST_POUR_GRACE_SEC=5
```

`HICON_MOULD_GIE_INTERVAL=11` remains the static fallback. (Dropped from the draft: `HICON_STREAM_0_MUX_WIDTH/HEIGHT` — mux resolution is out of scope.) External API payloads, DB records, and event meanings unchanged.

## Validation and rollout

- Track with `bd`: one issue per optimization group; configuration rollback retained for every phase.
- Pre-commit: compile, parser parity, processor replay, shutdown, DB overflow, multi-client preview, representative pipeline tests.
- Per candidate: 30-min comparison → 8 h shadow soak, one independent variable at a time.
- **Performance gates:**
  - Steady-state GR3D average ≤60%; 99%-samples <20% (stretch <10% after Phase 4).
  - CPU average <55%; ≥24 fps per healthy source; no resource-induced <20 fps period >5 s.
  - Metadata-only probe p99 <5 ms; mapped native analysis p99 <10 ms; no probe >40 ms.
  - Zero idle Python surface maps on all streams; preview encode cost independent of client count.
  - No probe-side queue blocking; zero terminal-event and warning/error drops.
  - ≥2 GB available RAM; RSS growth <1 MB/h; no swap in/out, OOM, or `memory.events` high/max counts (PSI unavailable on this kernel).
  - Temps <80 °C; within the active power mode's envelope; no RTSP-outage increase.
- **Accuracy gates:**
  - ≥200 annotated mould frames + ≥10 complete pours: mould recall ≥95%; exact final count on ≥95% of pours; max error ≤1.
  - No regression in pour, tapping, deslagging, spectro, pyrometer transitions.
  - Parser bbox/class/confidence parity within defined FP tolerance.
- Do not lower the global mould interval until gates pass. **Adaptive interval 5 during active pour is the preferred first cadence increase.**

## Separate operational items (parallel track, not gated on the above)

1. **Upstream network investigation** — 430 simultaneous all-stream outage starts on 2026-07-16, proven not local (TCP connect failures to all 3 camera IPs at once; zero link flaps/NIC errors/softnet drops). Candidate for `differential-layered-diagnosis` against the camera-VLAN switch / 28.x↔27.x path.
2. **GitHub push blocked by expired auth** — re-authenticate; session-close protocol currently unsatisfiable.
3. **Dev tooling hygiene** — close VS Code/agent sessions during production soaks and benchmarks.
