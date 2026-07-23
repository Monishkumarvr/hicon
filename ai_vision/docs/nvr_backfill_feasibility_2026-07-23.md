# NVR / Recording Backfill Feasibility Spike — Go/No-Go

**Date:** 2026-07-23
**Question:** Can the HiCon cameras' recurring 300s L2 blackouts be backfilled from recorded
footage (the proposed "gap-free delayed timeline")?
**Verdict:** **GO — feasible.** A reachable Hikvision NVR records all three cameras and its
footage is continuous, at full frame rate, across the exact windows the Jetson goes blind.

> Scope note: this spike answers *feasibility only*. Whether/how to build the timeline is a
> separate decision. Nothing in the running pipeline was changed.

## Correction to the pre-spike review

The pre-spike critique claimed the proposal named the wrong NVR (`192.168.28.6`) and a
"fabricated" channel map. **That critique was wrong** — it relied on stale repo/memory that still
described `.28.6` as the old CP Plus/Dahua box (decommissioned, no ISAPI). In reality `.28.6` has
been **replaced by a new Hikvision NVR** (same `admin:NVR@321#` credentials), and the proposal's
NVR + channel/track mapping were **correct and current**.

## Verified facts (all read-only ISAPI; ≤ a few successful calls per device, no failed logins)

**Recorder:** `192.168.28.6` — Hikvision **DS-7716NXI-K4** (16ch), fw **V4.76.015**, clock **IST
(+05:30)**. It is the melting/furnace-area recorder (channel names: "100 KG", "500 KG", "Ladle
Preheater", "Melting Entrance", …).

**Channel → track map (verified against live channel list — matches the proposal exactly):**

| AI stream | Camera | NVR channel (name) | Track |
|---|---|---|---|
| Stream 0 / Process | `192.168.28.119` | ch12 ("POURING") | **1201** |
| Stream 1 / Pyrometer | `192.168.28.172` | ch9 ("FURNACE & PANEL") | **901** |
| Stream 2 / Pouring2 | `192.168.28.174` | ch13 ("Camera 01") | **1301** |

**Known Jetson blackout windows (today, from FPS-watchdog in `pipeline/bus_handler.py`):** the
L2 break starts at exactly `:XX:59` every 300s — e.g. 17:26:59, 17:31:59, 17:36:59 (110s, the
longest), 17:41:59, 17:46:59, 17:51:59, 17:56:59, 18:01:59 IST. All three streams drop together.

**Segment-level continuity (ISAPI `ContentMgmt/search`):** every track returns one/few
**contiguous** segments with **no gap > 2s**, spanning all 8 blackout instants above
(e.g. track 1201 = one unbroken segment 17:23:46 → 18:19:44 IST).

**Frame-level continuity (RTSP playback + ffprobe):**
- **Track 1201 (Process/.119)** across the **110s** outage (17:36:30→17:39:20 window): 3746
  frames, **25.0 fps**, max inter-frame gap **0.040s**, **0 gaps > 0.2s → CONTINUOUS**.
- **Track 901 (Pyro/.172)** across the 17:46:59 lockstep outage (full 63.7s window): 1594
  frames, **25.0 fps**, max gap **0.040s**, **0 gaps > 0.2s → CONTINUOUS**.
- **Track 1301 (Pouring2/.174)** through the 17:46:59 onset (20s capture): 500 frames, **25.1
  fps**, max gap **0.040s**, **0 gaps > 0.2s → CONTINUOUS**.

**All three cameras pass at both segment and frame level.**

## Why it works (topology)

The 2026-07-21 investigation noted `.28.6` **drops in lockstep with the cameras** from the
Jetson's viewpoint (whereas `.28.8` is "spared"). That lockstep is the *good* sign: `.28.6` sits
on the **same downstream segment as the cameras**, so it is isolated *together with them* from the
Jetson during the L2 break — but camera→NVR traffic stays local, so the NVR keeps recording at
full frame rate throughout. Hence the footage is gap-free exactly where the Jetson is blind.

## Practical notes for a future build (learned during the spike)

- **Retrieval method:** on this firmware (V4.76) the ISAPI `ContentMgmt/download` endpoint rejects
  the body format used by `tools/nvr_download_heats.py` (`statusCode 6 / badXmlContent`). **RTSP
  playback works cleanly:** `rtsp://admin:…@192.168.28.6/Streaming/tracks/{TRACK}/?starttime=YYYYMMDDThhmmssZ&endtime=…`.
- **Timezone quirk:** the NVR labels times with a `Z` suffix but the digits are **local IST**
  (not UTC). Search results and playback URIs both use this convention; pass IST wall-clock digits
  with a `Z`.
- **Audio:** playback tracks carry `pcm_mulaw` audio → won't mux into MP4; use MKV or `-an`.
- **`ContentMgmt/search`** works with ISO8601 + explicit `+05:30` offset and returns segment
  timeSpans (usable for gap detection without downloading).
- **Retention** (max look-back → sets the deliverable delay bound / buffer size) was **not yet
  measured** — confirm before sizing any timeline buffers.

## Unrelated but confirmed stale — fix regardless

- `tools/nvr_download_heats.py` hardcodes `NVR_IP=192.168.28.8`, `TRACK_ID=3401` labeled
  "Cam-Process". On `.28.8` (a *different*, 64ch facility NVR) track 3401 is now **"EP Area"
  (`.151`)** — the HiCon cameras are **not on `.28.8` at all**. This tool currently downloads the
  wrong camera; fix its NVR/track or disable it.
- Camera/NVR credentials are hardcoded in `tools/*.py` and `.env` (hygiene).

## Sources checked and ruled out for HiCon footage

- **NVR-1 `.28.8`** (DS-7764NI-M4, 64ch): 64 *other* facility cameras; none of `.119/.172/.174`.
- **NVR-3 `.27.6`** (DS-8664NI-I8, 62ch): building surveillance; none of ours.
- **NVR `.27.7`** (documented pyrometer recorder): **down/unreachable**; stale `.27.253` mapping.
- **Camera SD (edge)** on `.119/.172/.174`: **no SD card** installed (empty `hddList`).
- → The **only** recorder holding the HiCon cameras is **`.28.6`**.
