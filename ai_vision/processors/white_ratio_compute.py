"""
NvBufSurface White-Ratio Extraction
===================================
Wrapper for GPU white-ratio extraction from DeepStream NvBufSurface.
Uses libds_white_ratio.so CUDA kernel to count white pixels in a rectangular ROI
directly on GPU memory — no CPU frame copy needed.
"""

import os
import ctypes
from pathlib import Path
from typing import Tuple

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import pyds
    HAS_PYDS = True
except ImportError:
    HAS_PYDS = False
    print("WARNING: pyds not available - white ratio extraction disabled")


def _load_white_ratio_lib():
    lib_path = Path(__file__).resolve().parent.parent / "libs" / "libds_white_ratio.so"
    if not lib_path.exists():
        print(f"WARNING: libds_white_ratio.so not found at {lib_path}")
        return None
    try:
        library = ctypes.CDLL(str(lib_path))
        library.ds_extract_white_ratio_rgba.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        library.ds_extract_white_ratio_rgba.restype = ctypes.c_int
        return library
    except Exception as e:
        print(f"WARNING: failed to load white-ratio CUDA library: {e}")
        return None


WHITE_RATIO_LIB = _load_white_ratio_lib()


def extract_white_ratio(
    gst_buffer,
    surface_index: int,
    roi_xyxy: Tuple[float, float, float, float],
    absolute_brightness_threshold: int,
) -> float:
    """
    Extract white-pixel ratio from user-provided ROI on NvBufSurface.

    Args:
        gst_buffer: GStreamer buffer from pad probe
        surface_index: Frame index within batch
        roi_xyxy: ROI in pixel coordinates as (x1, y1, x2, y2)
        absolute_brightness_threshold: White threshold (0-255)

    Returns:
        White ratio in [0, 1], or -1 on extraction failure.
    """
    if not HAS_PYDS or WHITE_RATIO_LIB is None:
        return -1.0

    try:
        x1, y1, x2, y2 = roi_xyxy
        out_ratio = ctypes.c_float(0.0)

        rc = WHITE_RATIO_LIB.ds_extract_white_ratio_rgba(
            ctypes.c_uint64(hash(gst_buffer)),
            ctypes.c_int(surface_index),
            ctypes.c_int(int(x1)),
            ctypes.c_int(int(y1)),
            ctypes.c_int(int(x2)),
            ctypes.c_int(int(y2)),
            ctypes.c_int(int(absolute_brightness_threshold)),
            ctypes.byref(out_ratio),
        )

        if rc != 0:
            return -1.0

        return float(out_ratio.value)

    except Exception as e:
        print("White-ratio extraction error:", e)
        return -1.0
