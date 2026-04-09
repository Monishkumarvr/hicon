"""
Molten Metal Detection
======================
ctypes wrapper for libds_molten_detect.so — model-free detection of glowing
molten metal inside a rectangular ROI on an RGBA NvBufSurface.

Three-layer detection stack (implemented in the CUDA kernel):
  Layer 1 — intensity gate   : brightness > threshold
  Layer 2 — color signature  : hot-color OR white-hot pixel
  Layer 3 — area filter      : connected-component area >= min_blob_area
"""

import ctypes
from ctypes import c_int, c_uint64
from pathlib import Path
from typing import List, Tuple

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import pyds
    HAS_PYDS = True
except ImportError:
    HAS_PYDS = False
    print("WARNING: pyds not available — molten detection disabled")


class DsMoltenBlob(ctypes.Structure):
    """Blob descriptor returned by ds_detect_molten_blobs_rgba."""
    _fields_ = [
        ("x1",             c_int),
        ("y1",             c_int),
        ("x2",             c_int),
        ("y2",             c_int),
        ("area",           c_int),
        ("avg_brightness", c_int),
    ]


DS_MAX_MOLTEN_BLOBS = 64


def _load_molten_detect_lib():
    lib_path = Path(__file__).resolve().parent.parent / "libs" / "libds_molten_detect.so"
    if not lib_path.exists():
        print(f"WARNING: libds_molten_detect.so not found at {lib_path}")
        return None
    try:
        lib = ctypes.CDLL(str(lib_path))
        lib.ds_detect_molten_blobs_rgba.argtypes = [
            c_uint64,                                    # gst_buffer_addr
            c_int,                                       # batch_id
            c_int, c_int, c_int, c_int,                 # roi_x1, roi_y1, roi_x2, roi_y2
            ctypes.POINTER(DsMoltenBlob),               # blobs_out
            c_int,                                       # max_blobs
            ctypes.POINTER(c_int),                      # num_blobs_out
            c_int,                                       # min_blob_area
            c_int,                                       # brightness_thresh
        ]
        lib.ds_detect_molten_blobs_rgba.restype = c_int
        return lib
    except Exception as e:
        print(f"WARNING: failed to load libds_molten_detect.so: {e}")
        return None


MOLTEN_DETECT_LIB = _load_molten_detect_lib()


def detect_molten_blobs(
    gst_buffer,
    surface_index: int,
    roi_xyxy: Tuple[float, float, float, float],
    min_blob_area: int = 2000,
    brightness_thresh: int = 200,
) -> List[DsMoltenBlob]:
    """
    Detect glowing molten-metal blobs inside a rectangular ROI.

    Args:
        gst_buffer:    GStreamer buffer from a pad probe (passed as-is).
        surface_index: Frame index within the NvBufSurface batch.
        roi_xyxy:      ROI in pipeline pixel coordinates: (x1, y1, x2, y2).
                       Typically the axis-aligned bounding box of a zone polygon.
        min_blob_area: Minimum connected-component area in pixels.
        brightness_thresh: Intensity gate threshold (0-255).

    Returns:
        List of DsMoltenBlob objects for blobs that passed the area filter.
        Empty list on failure or no detection.
    """
    if not HAS_PYDS or MOLTEN_DETECT_LIB is None:
        return []

    try:
        x1, y1, x2, y2 = roi_xyxy
        BlobArray = DsMoltenBlob * DS_MAX_MOLTEN_BLOBS
        blobs_arr = BlobArray()
        num_blobs = c_int(0)

        rc = MOLTEN_DETECT_LIB.ds_detect_molten_blobs_rgba(
            c_uint64(hash(gst_buffer)),
            c_int(int(surface_index)),
            c_int(int(x1)),
            c_int(int(y1)),
            c_int(int(x2)),
            c_int(int(y2)),
            blobs_arr,
            c_int(DS_MAX_MOLTEN_BLOBS),
            ctypes.byref(num_blobs),
            c_int(int(min_blob_area)),
            c_int(int(brightness_thresh)),
        )

        if rc != 0:
            return []

        return list(blobs_arr[:num_blobs.value])

    except Exception as e:
        print(f"Molten detection error: {e}")
        return []
