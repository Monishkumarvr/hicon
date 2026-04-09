#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
    int area;
    int avg_brightness;  // average (r+g+b)/3 across all pixels in the blob
} DsMoltenBlob;

/**
 * Detect glowing molten-metal blobs in an RGBA NvBufSurface ROI.
 *
 * Three-layer model-free detection:
 *   Layer 1 — intensity gate   : brightness > 220
 *   Layer 2 — color signature  : hotColor (r>180,g>80,b<200) OR whiteHot (r,g,b>240)
 *   Layer 3 — area filter      : connected-component area >= 5000 pixels
 *
 * Upper 40% of frame height is ignored (furnace/background exclusion).
 *
 * @param gst_buffer_addr  Hash/address of the GstBuffer (from Python hash(buf)).
 * @param batch_id         Surface index within the NvBufSurface batch.
 * @param roi_x1           Left edge of the zone AABB (pipeline pixels).
 * @param roi_y1           Top edge  of the zone AABB.
 * @param roi_x2           Right edge of the zone AABB.
 * @param roi_y2           Bottom edge of the zone AABB.
 * @param blobs_out        Caller-allocated array to receive blob descriptors.
 * @param max_blobs        Capacity of blobs_out (use DS_MAX_MOLTEN_BLOBS).
 * @param num_blobs_out    Set to the number of valid blobs written.
 * @param min_blob_area    Minimum connected-component area in pixels to keep.
 * @param brightness_thresh Minimum average channel brightness (r+g+b)/3 to pass Layer 1.
 * @return 0 on success, negative error code on failure.
 */
int ds_detect_molten_blobs_rgba(
    uint64_t gst_buffer_addr,
    int      batch_id,
    int      roi_x1,
    int      roi_y1,
    int      roi_x2,
    int      roi_y2,
    DsMoltenBlob* blobs_out,
    int      max_blobs,
    int*     num_blobs_out,
    int      min_blob_area,
    int      brightness_thresh
);

#define DS_MAX_MOLTEN_BLOBS 64

int ds_detect_molten_blobs_rgba_device(
    const void* rgba_device_ptr,
    int      width,
    int      height,
    int      pitch_bytes,
    int      roi_x1,
    int      roi_y1,
    int      roi_x2,
    int      roi_y2,
    DsMoltenBlob* blobs_out,
    int      max_blobs,
    int*     num_blobs_out,
    int      min_blob_area,
    int      brightness_thresh
);

int ds_detect_molten_blobs_nv12_device(
    const void* y_plane_device_ptr,
    const void* uv_plane_device_ptr,
    int      width,
    int      height,
    int      pitch_y,
    int      pitch_uv,
    int      roi_x1,
    int      roi_y1,
    int      roi_x2,
    int      roi_y2,
    DsMoltenBlob* blobs_out,
    int      max_blobs,
    int*     num_blobs_out,
    int      min_blob_area,
    int      brightness_thresh
);

#ifdef __cplusplus
}
#endif
