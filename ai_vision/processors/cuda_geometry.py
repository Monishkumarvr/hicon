"""
CUDA Geometry Helpers
=====================
Shared geometry utilities for CUDA-based detection probes.
Ported from deepstream_yolo_hicon/inference_probes.py.
"""

import pyds
from typing import Tuple, List

# Type alias: flat tuple of pixel coords (x1,y1, x2,y2, ..., xn,yn).
Polygon = Tuple[float, ...]


def polygon_bbox(polygon: Polygon) -> Tuple[float, float, float, float]:
    """
    Return the axis-aligned bounding box of a polygon.

    Args:
        polygon: Flat sequence (x1,y1, x2,y2, ..., xn,yn).

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    xs = polygon[0::2]
    ys = polygon[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_polygon(pt: Tuple[float, float], polygon: Polygon) -> bool:
    """
    Ray-casting point-in-polygon test.

    Args:
        pt:      (x, y) point to test.
        polygon: Flat sequence of vertex coords (x1,y1, x2,y2, ..., xn,yn).

    Returns:
        True if pt is inside (or on the boundary of) the polygon.
    """
    px, py = pt
    verts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon) - 1, 2)]
    n = len(verts)
    inside = False
    xi, yi = verts[-1]
    for xj, yj in verts:
        if ((yj > py) != (yi > py)) and (px < (xi - xj) * (py - yj) / (yi - yj) + xj):
            inside = not inside
        xi, yi = xj, yj
    return inside


def draw_zone_polygon(
    batch_meta,
    frame_meta,
    polygon: Polygon,
    color: Tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0),
    line_width: int = 2,
):
    """
    Draw a polygon outline on the OSD using NvOSD line segments.

    Handles polygons with more than 16 edges by batching into multiple
    display_meta acquisitions (DeepStream limit: 16 lines per object).

    Args:
        batch_meta:  NvDsBatchMeta, needed to acquire display metadata.
        frame_meta:  NvDsFrameMeta, where the display metadata is attached.
        polygon:     Flat vertex sequence (x1,y1, x2,y2, ..., xn,yn).
        color:       RGBA tuple, each component in [0.0, 1.0].
        line_width:  Pixel width of drawn lines.
    """
    verts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon) - 1, 2)]
    n = len(verts)
    segments = [(verts[i], verts[(i + 1) % n]) for i in range(n)]

    MAX_LINES = 16
    for batch_start in range(0, len(segments), MAX_LINES):
        batch_segs = segments[batch_start : batch_start + MAX_LINES]
        dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        dm.num_lines = len(batch_segs)
        for i, ((x1, y1), (x2, y2)) in enumerate(batch_segs):
            lp = dm.line_params[i]
            lp.x1 = int(x1)
            lp.y1 = int(y1)
            lp.x2 = int(x2)
            lp.y2 = int(y2)
            lp.line_width = line_width
            lp.line_color.red   = color[0]
            lp.line_color.green = color[1]
            lp.line_color.blue  = color[2]
            lp.line_color.alpha = color[3]
        pyds.nvds_add_display_meta_to_frame(frame_meta, dm)


def draw_blob_rect(batch_meta, frame_meta, blob, label: str = ""):
    """Draw a blue bounding box around a molten-metal blob, with an optional text label."""
    dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    dm.num_rects = 1
    rp = dm.rect_params[0]
    rp.left         = blob.x1
    rp.top          = blob.y1
    rp.width        = max(1, blob.x2 - blob.x1)
    rp.height       = max(1, blob.y2 - blob.y1)
    rp.border_width = 2
    rp.border_color.red   = 0.0
    rp.border_color.green = 0.0
    rp.border_color.blue  = 0.8
    rp.border_color.alpha = 1.0
    rp.has_bg_color = 0
    if label:
        dm.num_labels = 1
        tp = dm.text_params[0]
        tp.display_text = label
        tp.x_offset = int(blob.x1)
        tp.y_offset = max(0, int(blob.y1) - 18)
        tp.font_params.font_name = "Serif"
        tp.font_params.font_size = 10
        tp.font_params.font_color.red   = 0.0
        tp.font_params.font_color.green = 0.0
        tp.font_params.font_color.blue  = 0.8
        tp.font_params.font_color.alpha = 1.0
        tp.set_bg_clr = 1
        tp.text_bg_clr.red   = 0.0
        tp.text_bg_clr.green = 0.0
        tp.text_bg_clr.blue  = 0.0
        tp.text_bg_clr.alpha = 0.6
    pyds.nvds_add_display_meta_to_frame(frame_meta, dm)


def draw_text_label(batch_meta, frame_meta, text: str, x: int, y: int,
                    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                    font_size: int = 12):
    """Draw a single text label at the given position on the OSD."""
    dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    dm.num_labels = 1
    tp = dm.text_params[0]
    tp.display_text = text
    tp.x_offset = int(x)
    tp.y_offset = int(y)
    tp.font_params.font_name = "Serif"
    tp.font_params.font_size = font_size
    tp.font_params.font_color.red   = color[0]
    tp.font_params.font_color.green = color[1]
    tp.font_params.font_color.blue  = color[2]
    tp.font_params.font_color.alpha = color[3]
    tp.set_bg_clr = 1
    tp.text_bg_clr.red   = 0.0
    tp.text_bg_clr.green = 0.0
    tp.text_bg_clr.blue  = 0.0
    tp.text_bg_clr.alpha = 0.6
    pyds.nvds_add_display_meta_to_frame(frame_meta, dm)


def scale_zones(zones_dict, annotation_size, mux_width=1280, mux_height=720):
    """
    Scale zone polygon coordinates from annotation space to pipeline frame space.

    Args:
        zones_dict: {zone_name: list-of-[x,y]-pairs}
        annotation_size: (width, height) of annotation space, or None for no scaling.
        mux_width: Pipeline mux output width.
        mux_height: Pipeline mux output height.

    Returns:
        {zone_name: flat-tuple-polygon-in-pipeline-space}
    """
    if annotation_size is None:
        # Convert list-of-pairs to flat tuples
        scaled = {}
        for name, pts in zones_dict.items():
            flat = []
            for x, y in pts:
                flat.extend([float(x), float(y)])
            scaled[name] = tuple(flat)
        return scaled

    aw, ah = annotation_size
    sx = mux_width / aw
    sy = mux_height / ah

    scaled = {}
    for name, pts in zones_dict.items():
        flat = []
        for x, y in pts:
            flat.extend([float(x) * sx, float(y) * sy])
        scaled[name] = tuple(flat)
    return scaled
