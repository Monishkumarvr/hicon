#!/usr/bin/env python3
"""
Visualize all zones from zones.json on live streams from both cameras.
Saves annotated screenshots with zone overlays to ai_vision/output/zones_mapped/
"""
import json
import subprocess
import cv2
import numpy as np
from pathlib import Path

# Load zones config
zones_file = Path("/home/hicon/hicon/ai_vision/configs/zones.json")
with open(zones_file) as f:
    zones_config = json.load(f)

# RTSP streams
stream_0_url = "rtsp://admin:india%40789@192.168.28.119:554/Streaming/Channels/101"  # main stream 2688x1520
stream_1_url = "rtsp://admin:india%40789@192.168.27.253:554/Streaming/Channels/102"   # sub stream 1280x720

def capture_frame(rtsp_url):
    """Capture one frame from RTSP stream using ffmpeg"""
    cmd = [
        'ffmpeg', '-rtsp_transport', 'tcp', '-i', rtsp_url,
        '-frames:v', '1', '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return None
        
        nparr = np.frombuffer(result.stdout, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return None

def draw_zone(frame, zone_name, roi_points, ref_w, ref_h, color=(0, 255, 0)):
    """Draw a single zone on frame with auto-scaling"""
    if not roi_points or len(roi_points) < 3:
        return frame
    
    h, w = frame.shape[:2]
    rel_w = w / float(ref_w) if ref_w > 0 else 1.0
    rel_h = h / float(ref_h) if ref_h > 0 else 1.0
    
    # Scale points
    scaled = [[int(x * rel_w), int(y * rel_h)] for x, y in roi_points]
    pts = np.array(scaled, dtype=np.int32)
    
    # Draw filled polygon with alpha
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw boundary
    cv2.polylines(frame, [pts], True, color, 2)
    
    # Add label
    if len(scaled) > 0:
        text_x, text_y = scaled[0][0] + 5, scaled[0][1] + 20
        cv2.putText(frame, zone_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def map_zones_on_frame(frame, zones_section, ref_w, ref_h, color):
    """Draw all zones from a section"""
    if frame is not None and "zones" in zones_section:
        for zone_name, zone_data in zones_section["zones"].items():
            roi_pts = zone_data.get("roi_points", [])
            frame = draw_zone(frame, zone_name, roi_pts, ref_w, ref_h, color)
    return frame

# Capture frames
print("Capturing Stream 0 (Process camera, main stream 2688x1520)...")
frame_0 = capture_frame(stream_0_url)

print("Capturing Stream 1 (Pyrometer camera, sub-stream 1280x720)...")
frame_1 = capture_frame(stream_1_url)

if frame_0 is None or frame_1 is None:
    print("Failed to capture frames")
    exit(1)

ref_w = zones_config['metadata']['ref_width']
ref_h = zones_config['metadata']['ref_height']

print(f"\nZones reference resolution: {ref_w}x{ref_h}")
print(f"Stream 0 frame size: {frame_0.shape[1]}x{frame_0.shape[0]}")
print(f"Stream 1 frame size: {frame_1.shape[1]}x{frame_1.shape[0]}")

# Draw zones on Stream 0
print("\nMapping zones on Stream 0...")
frame_0 = map_zones_on_frame(frame_0, zones_config.get('tapping', {}), ref_w, ref_h, (255, 0, 0))  # Blue
frame_0 = map_zones_on_frame(frame_0, zones_config.get('deslagging', {}), ref_w, ref_h, (0, 255, 0))  # Green
frame_0 = map_zones_on_frame(frame_0, zones_config.get('spectro', {}), ref_w, ref_h, (0, 165, 255))  # Orange

# Draw zones on Stream 1
print("Mapping zones on Stream 1...")
frame_1 = map_zones_on_frame(frame_1, zones_config.get('pyrometer', {}), ref_w, ref_h, (255, 0, 255))  # Magenta

# Add title bars
cv2.putText(frame_0, "Stream 0: Process Camera (Tapping=Blue, Deslagging=Green, Spectro=Orange)", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(frame_1, "Stream 1: Pyrometer Camera (Furnaces=Magenta)", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Save
out_dir = Path("/home/hicon/hicon/ai_vision/output/zones_mapped")
out_dir.mkdir(parents=True, exist_ok=True)

file_0 = out_dir / "stream0_zones_mapped.jpg"
file_1 = out_dir / "stream1_zones_mapped.jpg"

cv2.imwrite(str(file_0), frame_0)
cv2.imwrite(str(file_1), frame_1)

print(f"\n✓ Saved: {file_0}")
print(f"✓ Saved: {file_1}")
