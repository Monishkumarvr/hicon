# Live Streaming Quick Start Guide

## ✅ Implementation Complete!

All code changes have been applied. The pipeline now supports:
1. **2-line technical status** (non-overlapping)
2. **Clean per-mould timing panel** (notebook-style)
3. **Large visible probe dots** (12px circles)
4. **MJPEG live streaming** (optional, disabled by default)

---

## 🚀 Quick Test (Live Streaming Enabled)

### 1. Enable Live Streaming

```bash
cd /home/hicon/hicon/ai_vision
nano .env
```

**Change this line:**
```bash
HICON_ENABLE_LIVE_STREAM=false
```

**To:**
```bash
HICON_ENABLE_LIVE_STREAM=true
```

Save and exit (Ctrl+X, Y, Enter).

---

### 2. Run Pipeline

```bash
python3 hicon_pipeline.py
```

**Expected output:**
```
[INFO] ✓ Live streaming enabled: http://0.0.0.0:8080/
[INFO]   Index: http://0.0.0.0:8080/
[INFO]   Stream 0: http://0.0.0.0:8080/stream0
[INFO]   Stream 1: http://0.0.0.0:8080/stream1
[INFO] Pipeline PLAYING — waiting for streams...
```

---

### 3. Open Browser

**On Jetson:**
```
http://localhost:8080/
```

**From another device on same network:**
```
http://<jetson-ip>:8080/
```

Replace `<jetson-ip>` with your Jetson's IP (find with `hostname -I`).

---

## 📺 What You'll See

### Top Section (2-line Technical Status)
```
POURING INFERENCE | 2026-02-17 14:22:15 | SESSION:ON POUR:ON
MOULDS:3 CLUSTERS:2 B:245 TARGET_T:5 LOCK_T:5 CYCLE_AGE:45.2s ABSENCE:0.1s
```

### Middle Section (Clean Per-Mould Panel)
```
Trolley #5 [LOCKED]
  Total Moulds: 3
  Mould #1: 12.3s ✓
  Mould #2:  8.7s ✓
  Mould #3: 15.2s ● ← GREEN (actively pouring)

Session: 45s | Cycle: 12m
```

### Visual Elements
- **Large green circle** (12px) at probe position when pouring
- **Brightness value** "B:245" next to probe dot
- **Color-coded moulds:** Green for active, gray for completed
- **Expanded trolley bbox** (semi-transparent green outline)

---

## ⚙️ Configuration Options

### Live Streaming Settings (in .env)

```bash
# Enable/disable streaming
HICON_ENABLE_LIVE_STREAM=true

# Bind address (0.0.0.0 = all interfaces)
HICON_LIVE_STREAM_HOST=0.0.0.0

# HTTP port
HICON_LIVE_STREAM_PORT=8080

# JPEG quality (0-100, higher = better quality, more bandwidth)
HICON_LIVE_STREAM_QUALITY=85

# Max FPS for stream (reduce if CPU too high)
HICON_LIVE_STREAM_FPS=15
```

---

## 🔧 Troubleshooting

### Issue: "Address already in use" port 8080
**Solution:** Change port in .env:
```bash
HICON_LIVE_STREAM_PORT=8081
```

---

### Issue: Can't access from browser on another device
**Check firewall:**
```bash
sudo ufw allow 8080/tcp
```

**Check Jetson IP:**
```bash
hostname -I
```

---

### Issue: High CPU usage
**Reduce streaming FPS:**
```bash
HICON_LIVE_STREAM_FPS=10
```

**Or reduce quality:**
```bash
HICON_LIVE_STREAM_QUALITY=70
```

---

### Issue: Overlays not appearing
**Verify inference video is enabled:**
```bash
HICON_ENABLE_INFERENCE_VIDEO=true
```

Overlays are rendered by DeepStream nvosd and sent to both:
1. Recording branch (MKV files)
2. Live streaming (MJPEG HTTP)

---

## 📊 Performance Impact

**Baseline (no streaming):**
- CPU: ~60-80%
- GPU: ~50-70%
- RAM: ~4-5 GB

**With MJPEG streaming (15 FPS, 85% quality):**
- CPU: +5-10% (JPEG encoding on CPU)
- GPU: no change
- RAM: +100-200 MB
- Network: ~1.5-2.5 Mbps per stream (3-5 Mbps total)

---

## 🎯 What Changed

### 1. Per-Mould Timing Tracking
- Added `mould_completed_times` dict to `PouringProcessor`
- Updated on each pour completion
- Displayed in real-time overlay

### 2. Improved Overlay Layout
- Replaced single cluttered line with 2-line technical status + clean panel
- Per-mould times: "Mould #2: 8.7s ✓"
- Active mould highlighted in green: "Mould #3: 15.2s ●"
- Larger probe dots (12px circles instead of small rects)

### 3. MJPEG Server
- New `streaming/mjpeg_server.py` module
- Extracts annotated frames from OSD probe
- Serves via Flask HTTP (no plugins required)
- Auto index page with both streams

### 4. Config Updates
- Added `ENABLE_LIVE_STREAM` flag (default: false)
- Added live streaming config: host, port, quality, FPS
- Updated `.env` with streaming section

---

## 🔐 Optional: Add Authentication

**Edit `streaming/mjpeg_server.py` and uncomment authentication code:**
```python
from flask import request, Response
from functools import wraps

def check_auth(username, password):
    return username == 'admin' and password == 'your-password'

def authenticate():
    return Response('Authentication required', 401,
                    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Add to routes:
@app.route('/stream<int:stream_id>')
@requires_auth
def stream(stream_id):
    ...
```

---

## 📝 Next Steps

1. **Test on live video**: Verify per-mould timing updates in real time
2. **Check probe dot visibility**: Should be large green circle when pouring
3. **Monitor performance**: Use `tegrastats --interval 1000`
4. **Adjust FPS/quality**: If CPU too high, reduce LIVE_STREAM_FPS
5. **Record inference video**: Verify same overlays appear in MKV files

---

## 🎉 Success Criteria

✅ Pipeline starts without errors
✅ Browser shows live streams at http://localhost:8080/
✅ Per-mould timing visible in real time
✅ Probe dot large and clearly visible
✅ Active mould highlighted in green
✅ 2-line technical status not overlapping with panel
✅ CPU usage < 90% with streaming enabled
✅ Latency < 300ms

---

**To disable streaming:**
Set `HICON_ENABLE_LIVE_STREAM=false` in `.env` and restart pipeline.
