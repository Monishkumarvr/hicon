#pragma once

int ds_compute_white_ratio_from_device_rgba(
    const void* rgba_device_ptr,
    int width,
    int height,
    int pitch_bytes,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int absolute_brightness_threshold,
    float* out_white_ratio
);

int ds_compute_white_ratio_from_device_nv12(
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
    int absolute_brightness_threshold,
    float* out_white_ratio
);
