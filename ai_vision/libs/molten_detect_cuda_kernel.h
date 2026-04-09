#pragma once

/**
 * Classify pixels in an RGBA ROI as "molten" or "not molten" using the
 * three-layer detection stack. Writes a 1-byte-per-pixel binary mask into
 * a caller-supplied host buffer.
 *
 * @param rgba_device_ptr  Device pointer to the RGBA surface.
 * @param width            Frame width in pixels.
 * @param height           Frame height in pixels.
 * @param pitch_bytes      Row stride in bytes.
 * @param roi_x1           ROI left (clamped to frame).
 * @param roi_y1           ROI top.
 * @param roi_x2           ROI right.
 * @param roi_y2           ROI bottom.
 * @param mask_host        Host buffer to receive binary mask.
 *                         Must be at least (roi_x2-roi_x1+1)*(roi_y2-roi_y1+1) bytes.
 * @param brightness_thresh Minimum average channel brightness (r+g+b)/3 to pass Layer 1.
 * @return 0 on success, negative on CUDA error.
 */
int ds_classify_molten_pixels_device(
    const void* rgba_device_ptr,
    int width,
    int height,
    int pitch_bytes,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    unsigned char* mask_host,
    int brightness_thresh
);

int ds_classify_molten_pixels_device_nv12(
    const void* y_plane_device_ptr,
    const void* uv_plane_device_ptr,
    int width,
    int height,
    int pitch_y,
    int pitch_uv,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    unsigned char* mask_host,
    int brightness_thresh
);
