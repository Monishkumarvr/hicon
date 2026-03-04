"""
Zone Snapshot Tool — capture a frame from the live RTSP stream and overlay
the current zones.json ROI regions so you can verify or recalibrate them.

Usage:
    cd /home/hicon/hicon/ai_vision
    python3 tools/zone_snapshot.py [--stream 0|1] [--out snapshot.jpg]
        [--dx-deslagging PX] [--dy-deslagging PX]
        [--dx-spectro PX]    [--dy-spectro PX]
        [--apply]

The tool connects directly to the camera RTSP URL (from .env / config) and
grabs one frame, then draws all ROI zones from configs/zones.json with their
vertex coordinates printed on the image and in the terminal.

Use --dx-* / --dy-* to test offset adjustments visually.
Add --apply to write the shifted coordinates back into zones.json automatically.
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.zone_loader import load_zones_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger('zone_snapshot')

# ---------------------------------------------------------------------------
# Zone definitions: which zones to draw per stream
# ---------------------------------------------------------------------------
_STREAM0_ZONES = {
    'tapping':    {'key': 'roi_points', 'color': (0, 165, 255),  'label': 'TAPPING'},    # orange
    'deslagging': {'key': 'roi_points', 'color': (0,   0, 255),  'label': 'DESLAGGING'}, # red
    'spectro':    {'key': 'roi_points', 'color': (255, 255,  0), 'label': 'SPECTRO'},     # cyan
}
_STREAM1_ZONES = {
    'pyrometer':  {'key': 'zone_polygon', 'color': (0, 255, 0), 'label': 'PYRO ZONE'},   # green
}


def _grab_frame(rtsp_url: str, mux_w: int, mux_h: int) -> np.ndarray:
    """Open RTSP stream, skip a few frames for auto-exposure to settle, return one frame.

    The frame is resized to mux_w x mux_h so the snapshot matches exactly what
    nvstreammux delivers to brightness_processor.py in the pipeline.
    """
    log.info(f"Connecting to: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {rtsp_url}")

    # Skip first few frames (auto-exposure / IDR refresh)
    for _ in range(30):
        cap.read()

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from stream")

    native_w, native_h = frame.shape[1], frame.shape[0]
    log.info(f"Captured frame: {native_w}x{native_h} BGR (native)")

    if native_w != mux_w or native_h != mux_h:
        frame = cv2.resize(frame, (mux_w, mux_h), interpolation=cv2.INTER_LINEAR)
        log.info(f"Resized to mux resolution: {mux_w}x{mux_h} (matches pipeline nvstreammux output)")

    return frame


def _scale_pts(pts_raw: list, sx: float, sy: float) -> list:
    """Scale zone coordinates from calibration resolution to frame resolution."""
    return [[int(round(x * sx)), int(round(y * sy))] for x, y in pts_raw]


def _draw_zones(frame: np.ndarray, zones_config: dict, zone_defs: dict,
                zone_offsets: dict) -> np.ndarray:
    """Draw ROI polygons + vertex labels on frame, auto-scaling from zones.json ref resolution.

    zone_offsets: {zone_name: (dx, dy)} — per-zone pixel offset after scaling.
    Vertex labels show the final (offset-applied) coordinates — these are the values
    to put into zones.json when --apply is used.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    meta = zones_config.get('metadata', {})
    ref_w = int(meta.get('ref_width', w))
    ref_h = int(meta.get('ref_height', h))
    sx = w / ref_w if ref_w > 0 else 1.0
    sy = h / ref_h if ref_h > 0 else 1.0

    if sx != 1.0 or sy != 1.0:
        log.info(f"Auto-scaling zones: {ref_w}x{ref_h} (calibration) → {w}x{h} (frame) "
                 f"[sx={sx:.3f}, sy={sy:.3f}]")
    else:
        log.info(f"No scaling needed: frame matches calibration resolution {w}x{h}")

    for zone_name, cfg in zone_defs.items():
        zone_data = zones_config.get(zone_name, {})
        pts_raw = zone_data.get(cfg['key'], [])
        if not pts_raw:
            log.warning(f"No points for zone '{zone_name}' (key='{cfg['key']}')")
            continue

        color = cfg['color']
        label = cfg['label']
        dx, dy = zone_offsets.get(zone_name, (0, 0))

        # Scale to frame resolution, then apply per-zone offset
        pts_final = _scale_pts(pts_raw, sx, sy)
        if dx or dy:
            pts_final = [[x + dx, y + dy] for x, y in pts_final]
        pts = np.array(pts_final, dtype=np.int32)

        # Semi-transparent fill
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)

        # Solid outline
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        # Zone label at centroid
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        cv2.putText(out, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

        # Vertex dots + coordinate text
        # Show final (offset-applied) coords — these go directly into zones.json
        for i, (fx, fy) in enumerate(pts_final):
            cv2.circle(out, (fx, fy), 5, color, -1)
            cv2.circle(out, (fx, fy), 6, (255, 255, 255), 1)
            coord_txt = f"P{i}({fx},{fy})"
            tx = min(fx + 8, w - 120)
            ty = max(fy - 8, 18)
            cv2.putText(out, coord_txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, coord_txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Terminal log
        log.info(f"  [{label}] final coords: {pts_final}"
                 + (f"  (offset {dx:+d},{dy:+d} from original {pts_raw})" if dx or dy else ""))

    # Frame info bar
    active = [f"{n}({dx:+d},{dy:+d})" for n, (dx, dy) in zone_offsets.items() if dx or dy]
    info = (f"Frame: {w}x{h}  |  ref: {ref_w}x{ref_h}"
            + (f"  |  scale: {sx:.3f}x{sy:.3f}" if sx != 1.0 or sy != 1.0 else "  |  1:1")
            + (f"  |  offsets: {', '.join(active)}" if active else ""))
    cv2.rectangle(out, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(out, info, (8, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def _apply_offsets_to_zones_json(zones_path: str, zones_config: dict,
                                  zone_offsets: dict, frame_w: int, frame_h: int):
    """Write offset-shifted coordinates back into zones.json.

    ALL roi_points zones are rescaled to frame resolution so the ref stays
    consistent. Only zones listed in zone_offsets with non-zero values get
    an additional pixel offset applied. zone_polygon zones (e.g. pyrometer)
    are left untouched — they are stream-native and don't use ref scaling.
    """
    meta = zones_config.get('metadata', {})
    ref_w = int(meta.get('ref_width', frame_w))
    ref_h = int(meta.get('ref_height', frame_h))
    sx = frame_w / ref_w if ref_w > 0 else 1.0
    sy = frame_h / ref_h if ref_h > 0 else 1.0

    with open(zones_path, 'r') as f:
        raw = json.load(f)

    for zone_name, zone_data in zones_config.items():
        if zone_name == 'metadata':
            continue
        # Only update roi_points zones (Stream 0). Leave zone_polygon (pyrometer) as-is.
        if 'roi_points' not in zone_data:
            continue

        pts_raw = zone_data['roi_points']
        pts_final = _scale_pts(pts_raw, sx, sy)

        dx, dy = zone_offsets.get(zone_name, (0, 0))
        if dx or dy:
            pts_final = [[x + dx, y + dy] for x, y in pts_final]
            log.info(f"  {zone_name}.roi_points → {pts_final}  (offset {dx:+d},{dy:+d})")
        else:
            log.info(f"  {zone_name}.roi_points → {pts_final}  (rescaled only)")

        raw[zone_name]['roi_points'] = pts_final

    # Update ref to match frame resolution — all roi_points now in frame space
    raw['metadata']['ref_width'] = frame_w
    raw['metadata']['ref_height'] = frame_h

    with open(zones_path, 'w') as f:
        json.dump(raw, f, indent=2)

    offset_zones = [n for n, (dx, dy) in zone_offsets.items() if dx or dy]
    log.info(f"zones.json saved at ref {frame_w}x{frame_h}. "
             f"Offsets applied to: {offset_zones or 'none (rescale only)'}")


def main():
    ap = argparse.ArgumentParser(description="Capture one frame and overlay zone ROIs")
    ap.add_argument('--stream', type=int, default=0, choices=[0, 1],
                    help='Camera stream to capture (0=process, 1=pyrometer)')
    ap.add_argument('--out', type=str, default='',
                    help='Output JPEG path (default: zone_snapshot_stream<N>.jpg)')
    ap.add_argument('--dx-deslagging', type=int, default=0, metavar='PX',
                    help='Shift deslagging zone on X (negative = left)')
    ap.add_argument('--dy-deslagging', type=int, default=0, metavar='PX',
                    help='Shift deslagging zone on Y (positive = down)')
    ap.add_argument('--dx-spectro', type=int, default=0, metavar='PX',
                    help='Shift spectro zone on X (negative = left)')
    ap.add_argument('--dy-spectro', type=int, default=0, metavar='PX',
                    help='Shift spectro zone on Y (positive = down)')
    ap.add_argument('--apply', action='store_true',
                    help='Write shifted coordinates directly into zones.json')
    args = ap.parse_args()

    zones_path = str(config.CONFIG_DIR / 'zones.json')
    zones_config = load_zones_config(zones_path)

    if args.stream == 0:
        rtsp_url = config.RTSP_STREAM_0
        zone_defs = _STREAM0_ZONES
        out_default = 'zone_snapshot_stream0.jpg'
    else:
        rtsp_url = config.RTSP_STREAM_1
        zone_defs = _STREAM1_ZONES
        out_default = 'zone_snapshot_stream1.jpg'

    if not rtsp_url:
        log.error(f"Stream {args.stream} RTSP URL is empty — check .env")
        sys.exit(1)

    # Mux resolution — must match nvstreammux width/height in gst_builder.py
    # Stream 0 and 1 use the default 1920x1080; Stream 2 uses 704x576
    mux_w, mux_h = (704, 576) if args.stream == 2 else (1920, 1080)

    frame = _grab_frame(rtsp_url, mux_w, mux_h)
    h, w = frame.shape[:2]

    zone_offsets = {
        'deslagging': (args.dx_deslagging, args.dy_deslagging),
        'spectro':    (args.dx_spectro,    args.dy_spectro),
    }

    log.info("Zone coordinates (final = after scale + offset):")
    annotated = _draw_zones(frame, zones_config, zone_defs, zone_offsets)

    out_path = args.out or out_default
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info(f"Saved: {out_path}")

    if args.apply:
        log.info("--apply: writing shifted coordinates to zones.json ...")
        _apply_offsets_to_zones_json(zones_path, zones_config, zone_offsets, w, h)
    elif any(dx or dy for dx, dy in zone_offsets.values()):
        log.info("Preview only — add --apply to write these coordinates into zones.json")


if __name__ == '__main__':
    main()
