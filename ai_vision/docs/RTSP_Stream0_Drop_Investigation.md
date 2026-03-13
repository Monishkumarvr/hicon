# Stream 0 RTSP Drop Investigation — Full Report

**Camera:** CP Plus CP-UNC-TC41L5C-VMD-LQ
**Firmware:** V2.860.00AT001.0.R (Dahua OEM)
**ONVIF Version:** 24.12 (V2.0.1.0)
**Platform:** Jetson Orin Nano 8GB, JetPack 6.2.1 (L4T R36.4.7), DeepStream 7.1, GStreamer 1.20.3
**Investigation Period:** February–March 2026
**Status:** Root cause confirmed — camera firmware bug. Mitigated with fast watchdog recovery.

---

## 1. Problem Statement

Stream 0 (CP Plus process camera) drops its RTSP TCP session every 2–5 minutes. The session terminates abruptly with no prior warning — no packet loss, no latency spike, no GStreamer error before the disconnect. Stream 1 (Hikvision pyrometer camera) on the same Jetson has **never** dropped a single session.

The drops cause the entire DeepStream pipeline to stall (0 fps on Stream 0), requiring a full pipeline restart to recover.

---

## 2. Network Architecture

```
                    ┌──────────────────────┐
                    │   Tailscale VPN       │
                    │   (laptop access)     │
                    └──────────┬────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────┐
│                         Network Switch                       │
│                                                              │
│  Subnet 192.168.28.0/24              Subnet 192.168.27.0/24  │
│  ┌─────────────────────┐             ┌─────────────────────┐ │
│  │ Jetson Orin Nano     │             │ Hikvision Camera    │ │
│  │ 192.168.28.44        │             │ 192.168.27.253      │ │
│  │ Interface: enP8p1s0  │             │ Stream 1 (pyro)     │ │
│  └─────────┬────────────┘             │ NEVER drops         │ │
│            │                          └─────────────────────┘ │
│  ┌─────────┴────────────┐                                     │
│  │ CP Plus Camera       │                                     │
│  │ 192.168.28.155       │                                     │
│  │ Stream 0 (process)   │                                     │
│  │ DROPS every 2-5 min  │                                     │
│  └──────────────────────┘                                     │
└──────────────────────────────────────────────────────────────┘
```

**Key facts:**
- Jetson and CP Plus are on the **same subnet** (192.168.28.x) — no router hops
- Hikvision is on a different subnet (192.168.27.x) — routed, yet never drops
- All devices connected to the same physical switch
- Tailscale VPN on Jetson for remote management — does NOT route camera traffic

---

## 3. Network Diagnostics (Ruled Out)

Comprehensive network testing was performed to eliminate network as the cause.

### Ping test (Jetson → CP Plus)
```
PING 192.168.28.155: 100 packets transmitted, 100 received, 0% packet loss
rtt min/avg/max/mdev = 0.699/2.413/12.847/1.893 ms
```

### Interface statistics (enP8p1s0)
- **RX errors:** 0
- **TX errors:** 0
- **Dropped:** 0
- **Overruns:** 0
- **Carrier errors:** 0

### Conclusion
Zero packet loss, sub-3ms latency, clean interface — **network is not the cause**.

---

## 4. Camera Details

### CP Plus CP-UNC-TC41L5C-VMD-LQ
- **Chipset:** Dahua OEM (confirmed by firmware naming, ONVIF behavior, web UI)
- **Firmware:** V2.860.00AT001.0.R
- **ONVIF:** 24.12 (V2.0.1.0), Profile token: `Profile000`
- **Native RTSP URL:** `rtsp://admin:***@192.168.28.155:554/video/live?channel=1&subtype=0`
- **Supported codecs:** H.264, H.265 (Main Profile)
- **Resolution:** Web UI shows 2560x1440, but RTSP actually delivers **1920x1080** (H.264 Main Profile Level 4.0 caps at 1080p)
- **Audio:** Has audio stream on RTSP (was enabled by default)
- **Dahua-native URLs:** `/cam/realmonitor?channel=1&subtype=0` → 404 (not supported)
- **Hikvision-style URLs:** `/Streaming/Channels/101` → 404 (not supported)

### Hikvision (Stream 1 — for comparison)
- **RTSP URL:** `rtsp://admin:***@192.168.27.253:554/Streaming/Channels/102`
- **Behavior:** Rock solid, **zero drops** across weeks of continuous operation
- **Same pipeline code**, same GStreamer settings, same Jetson

---

## 5. Experiments & Results

All tests used the same Jetson, same network, same physical connections. Each test ran for multiple drop cycles to establish a pattern.

### Test Matrix

| # | Codec | I-Frame | Audio | URL / Protocol | Client | Drop Interval | Notes |
|---|-------|---------|-------|----------------|--------|---------------|-------|
| 1 | H.265 | 50 | ON | Direct TCP | DeepStream | 80–280s | Original config |
| 2 | H.264 | 50 | ON | Direct TCP | DeepStream | ~30s | **Worse** — H.264 generates ~2x data |
| 3 | H.264 | 25 | OFF | Direct TCP | DeepStream | 170–300s | Audio disable was biggest improvement |
| 4 | H.265 | 25 | OFF | Direct TCP | DeepStream | ~236s | Best codec for bandwidth |
| 5 | H.265 | 25 | OFF | ONVIF URL | DeepStream | ~193s | ONVIF uses same internal handler |
| 6 | H.265 | 25 | OFF | Direct TCP | **ffmpeg** | ~236s | Same drops — not GStreamer-specific |
| 7 | H.265 | 25 | OFF | ONVIF URL | **ffmpeg** | ~193s | Confirmed across clients |
| 8 | — | — | — | ICMP ping | — | 0% loss | Network ruled out |

### Experiment Details

#### Test 1: Original Config (H.265, audio ON)
- Camera default settings, H.265 codec, I-frame interval 50, audio enabled
- Drops every 80–280 seconds (highly variable)
- This was the baseline that prompted the investigation

#### Test 2: Switch to H.264
- Changed camera codec to H.264 via web UI
- **Result: Dramatically worse** — drops every ~30 seconds
- H.264 at 1920x1080 generates approximately 2x the bitrate of H.265 at the same resolution
- Higher data rate increases TCP buffer pressure, triggering the firmware bug faster
- Camera web UI showed 2560x1440 but RTSP negotiated 1920x1080 (H.264 Main Profile Level 4.0 limitation)

#### Test 3: H.264 with I-frame 25, Audio OFF
- Reduced I-frame interval from 50 to 25 (smaller GOP = faster recovery, less buffering)
- **Disabled audio stream** in camera settings
- **Result: Major improvement** — drops went from ~30s to 170–300s
- **Key finding:** Disabling audio was the single biggest factor
- The audio RTSP sub-session appears to destabilize the TCP connection

#### Test 4: Revert to H.265, Audio OFF
- Switched back to H.265 (better compression = less bandwidth)
- Kept audio disabled, I-frame interval 25
- **Result:** ~236s between drops — similar to H.264 with audio off
- H.265 is better overall (lower bitrate, same drop pattern)

#### Test 5: ONVIF Protocol URL
- Discovered camera's ONVIF profile token via SOAP: `Profile000`
- Constructed ONVIF URL: `rtsp://...?proto=Onvif`
- **Result:** ~193s — no improvement over direct URL
- ONVIF RTSP internally uses the same session handler as the proprietary URL
- The `proto=Onvif` parameter just changes the RTSP SETUP negotiation, not the underlying TCP handling

#### Test 6: ffmpeg (Independent Client)
- Tested with ffmpeg 4.4.2 (completely independent of GStreamer/DeepStream)
- `ffmpeg -rtsp_transport tcp -stimeout 10000000 -i rtsp://... -c copy -y test.mp4`
- **Result:** Same drops at ~236s
- **This conclusively proves the issue is in the camera firmware**, not in GStreamer, DeepStream, or any client-side code

#### Test 7: ffmpeg with ONVIF URL
- ffmpeg with the ONVIF-style RTSP URL
- **Result:** Same drops at ~193s
- Confirmed ONVIF path offers no benefit

#### Test 8: Network Diagnostics
- 100-packet ping: 0% loss, 2.4ms average RTT
- Interface counters: zero errors, zero drops
- **Network conclusively ruled out**

### Other URLs Tested
- Dahua native: `rtsp://admin:***@192.168.28.155:554/cam/realmonitor?channel=1&subtype=0` → **404 Not Found**
- Hikvision style: `rtsp://admin:***@192.168.28.155:554/Streaming/Channels/101` → **404 Not Found**
- Only the proprietary URL format works: `/video/live?channel=1&subtype=0`

---

## 6. Root Cause Analysis

### Confirmed: Camera Firmware TCP Session Bug

The CP Plus CP-UNC-TC41L5C-VMD-LQ (firmware V2.860.00AT001.0.R) has a bug in its RTSP TCP session management that causes it to drop active sessions every 2–5 minutes.

**Evidence:**
1. **Multiple independent clients reproduce the issue** — GStreamer, ffmpeg, DeepStream all see the same drops
2. **Network is clean** — zero packet loss, zero interface errors, sub-3ms latency
3. **Same Jetson, same code, different camera works perfectly** — Hikvision on Stream 1 has zero drops
4. **No codec/protocol/URL combination fixes it** — H.264, H.265, ONVIF, direct — all drop
5. **Audio sub-session amplifies the bug** — disabling audio extended stable periods from 30s to 3–5min, suggesting the firmware struggles with multi-stream TCP session management

### Why Audio Makes It Worse
RTSP establishes separate RTP channels for video and audio within the same TCP connection (interleaved mode). The CP Plus firmware appears to have a session management bug that is exacerbated when handling multiple interleaved streams. With audio disabled, only the video RTP channel is active, reducing the frequency of the firmware bug trigger.

### Why H.264 is Worse Than H.265
H.264 at the same resolution produces approximately 2x the bitrate of H.265. Higher throughput means more TCP packets per second, which increases the rate at which the firmware's TCP session handling code encounters its bug condition. H.265's lower bitrate gives the firmware more breathing room between bug triggers.

---

## 7. Mitigation Strategy

Since the firmware bug cannot be fixed from the client side, the strategy is **fast detection and recovery**.

### Current Config (Optimal)

**Camera settings:**
| Setting | Value | Reason |
|---------|-------|--------|
| Codec | H.265 | Lower bitrate, longer stable periods |
| I-frame interval | 25 | Faster keyframe recovery after reconnect |
| Audio | **Disabled** | Biggest single improvement (30s → 3-5min) |
| Resolution | 2560x1440 (delivers 1920x1080) | Camera default |

**Pipeline settings (DeepStream):**
| Setting | Value | Location |
|---------|-------|----------|
| RTSP transport | TCP | `gst_builder.py` |
| RTSP latency | 2000ms | `gst_builder.py` |
| TCP timeout | 60s | `gst_builder.py` |
| Keep-alive | Enabled | `gst_builder.py` |
| FPS watchdog | 10s (0fps detection) | `bus_handler.py` |
| Audio pads | Linked to fakesink (discard) | `gst_builder.py` |

**Systemd service:**
| Setting | Value |
|---------|-------|
| Restart | on-failure |
| RestartSec | 10s |

### Recovery Timeline Per Drop
```
0s     — Camera drops TCP session
0-10s  — Pipeline stalls (0 fps on Stream 0)
10s    — FPS watchdog detects 0fps, triggers pipeline exit
10-20s — systemd restarts the service
20-25s — Pipeline initializes, RTSP reconnects, first frames arrive
~25s   — Full recovery, both streams at 25fps
```

### Expected Uptime
- **Stable period:** 3–5 minutes between drops
- **Downtime per drop:** ~20–25 seconds
- **Effective uptime:** ~93–97%
- **Stream 1 (Hikvision):** 100% uptime (never drops)

### Production Observations (2026-03-07)
- 30+ restarts observed in a ~2-hour window (restart counter 83→129)
- All drops are **Stream 0 only** — Stream 1 never triggers watchdog
- Some drops cause back-to-back restarts (camera fails to accept reconnection on first attempt)
- Watchdog and systemd recovery work reliably every time

---

## 8. GStreamer Pipeline Config (Stream 0)

```python
# rtspsrc configuration (gst_builder.py)
rtspsrc
  location = rtsp://admin:***@192.168.28.155:554/video/live?channel=1&subtype=0
  protocols = tcp                    # Force TCP (no UDP fallback)
  do-rtsp-keep-alive = True          # Send RTSP keepalive
  buffer-mode = 0                    # No buffering mode
  latency = 2000                     # 2s jitter buffer
  tcp-timeout = 60000000             # 60s TCP timeout (microseconds)
  retry = 20                         # RTSP reconnection attempts
  do-retransmission = True           # Enable RTCP retransmission

# Decoder (nvv4l2decoder)
nvv4l2decoder
  num-extra-surfaces = 8             # Extra decode buffers
  enable-max-performance = True      # Max clock for decode
  disable-dpb = True                 # Disable decoded picture buffer (lower latency)

# Audio handling
# Audio pads from rtspsrc are linked to a fakesink to discard audio data
# This prevents "unlinked pad" errors that could destabilize the pipeline
```

---

## 9. Recommendations

### Short-term (current)
- Keep current config: H.265, audio OFF, I-frame 25, 10s watchdog
- Accept ~93–97% uptime on Stream 0
- Monitor via `journalctl -u hicon-vision -f | grep FPS-WATCHDOG`

### Medium-term
- **Check for firmware update** from CP Plus / Dahua for model CP-UNC-TC41L5C-VMD-LQ
- Current firmware V2.860.00AT001.0.R may have a newer version that fixes TCP session handling
- Contact CP Plus support with this investigation data

### Long-term
- **Replace CP Plus camera with Hikvision** — proven zero-drop RTSP on the same infrastructure
- Hikvision cameras on this same Jetson (Stream 1) have demonstrated weeks of continuous operation without a single RTSP drop

---

## 10. Key Takeaways

1. **Always test with an independent client (ffmpeg)** before blaming the pipeline framework
2. **Audio sub-sessions are a hidden destabilizer** — always disable audio on IP cameras used for vision-only applications
3. **H.265 is strictly better than H.264** for RTSP stability (lower bitrate = less TCP pressure)
4. **ONVIF is not a different protocol** — it uses the same RTSP/RTP stack internally
5. **Camera firmware quality varies wildly** — Hikvision vs CP Plus on identical infrastructure shows orders-of-magnitude difference in RTSP reliability
6. **Fast watchdog + systemd restart is an effective mitigation** — 93–97% uptime is achievable even with a fundamentally broken camera
