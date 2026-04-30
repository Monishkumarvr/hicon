/*
 * Molten-pixel classification — CPU implementation (no CUDA).
 * Same function signatures as the former CUDA version so callers are unchanged.
 */

#include "molten_detect_cuda_kernel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#define MOLTEN_MIN_R         180
#define MOLTEN_MIN_G          80
#define MOLTEN_MAX_B         200
#define MOLTEN_WHITE_HOT_MIN 240

static inline int clamp_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

static inline void nv12_to_rgb(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    int pitch_y,
    int pitch_uv,
    int x,
    int y,
    int* out_r,
    int* out_g,
    int* out_b
) {
    int y_val  = static_cast<int>(y_plane[y * pitch_y + x]);
    int uv_idx = (y / 2) * pitch_uv + (x / 2) * 2;
    int u = static_cast<int>(uv_plane[uv_idx])     - 128;
    int v = static_cast<int>(uv_plane[uv_idx + 1]) - 128;
    int c = y_val - 16;
    if (c < 0) c = 0;
    *out_r = clamp_u8((298 * c + 409 * v + 128) >> 8);
    *out_g = clamp_u8((298 * c - 100 * u - 208 * v + 128) >> 8);
    *out_b = clamp_u8((298 * c + 516 * u + 128) >> 8);
}

int ds_classify_molten_pixels_device(
    const void*    rgba_ptr,
    int            width,
    int            height,
    int            pitch_bytes,
    int            roi_x1,
    int            roi_y1,
    int            roi_x2,
    int            roi_y2,
    unsigned char* mask_host,
    int            brightness_thresh
) {
    if (!rgba_ptr || !mask_host || width <= 0 || height <= 0)
        return -1;

    roi_x1 = std::max(0, std::min(width  - 1, roi_x1));
    roi_x2 = std::max(0, std::min(width  - 1, roi_x2));
    roi_y1 = std::max(0, std::min(height - 1, roi_y1));
    roi_y2 = std::max(0, std::min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1)
        return 0;

    int roi_w = roi_x2 - roi_x1 + 1;
    int roi_h = roi_y2 - roi_y1 + 1;
    const unsigned char* base = reinterpret_cast<const unsigned char*>(rgba_ptr);

    for (int ly = 0; ly < roi_h; ++ly) {
        int gy = roi_y1 + ly;
        const unsigned char* row = base + gy * pitch_bytes;
        for (int lx = 0; lx < roi_w; ++lx) {
            int gx = roi_x1 + lx;
            int r  = row[gx * 4 + 0];
            int g  = row[gx * 4 + 1];
            int b  = row[gx * 4 + 2];
            int brightness = (r + g + b) / 3;
            bool veryBright = brightness > brightness_thresh;
            if (!veryBright) {
                mask_host[ly * roi_w + lx] = 0;
                continue;
            }
            bool hotColor = (r > MOLTEN_MIN_R && g > MOLTEN_MIN_G && b < MOLTEN_MAX_B);
            bool whiteHot = (r > MOLTEN_WHITE_HOT_MIN && g > MOLTEN_WHITE_HOT_MIN &&
                             b > MOLTEN_WHITE_HOT_MIN);
            mask_host[ly * roi_w + lx] = (hotColor || whiteHot)
                ? static_cast<unsigned char>(brightness)
                : 0;
        }
    }
    return 0;
}

int ds_classify_molten_pixels_device_nv12(
    const void*    y_ptr,
    const void*    uv_ptr,
    int            width,
    int            height,
    int            pitch_y,
    int            pitch_uv,
    int            roi_x1,
    int            roi_y1,
    int            roi_x2,
    int            roi_y2,
    unsigned char* mask_host,
    int            brightness_thresh
) {
    if (!y_ptr || !uv_ptr || !mask_host ||
        width <= 0 || height <= 0 || pitch_y <= 0 || pitch_uv <= 0)
        return -1;

    roi_x1 = std::max(0, std::min(width  - 1, roi_x1));
    roi_x2 = std::max(0, std::min(width  - 1, roi_x2));
    roi_y1 = std::max(0, std::min(height - 1, roi_y1));
    roi_y2 = std::max(0, std::min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1)
        return 0;

    int roi_w = roi_x2 - roi_x1 + 1;
    int roi_h = roi_y2 - roi_y1 + 1;
    const unsigned char* y_plane  = reinterpret_cast<const unsigned char*>(y_ptr);
    const unsigned char* uv_plane = reinterpret_cast<const unsigned char*>(uv_ptr);

    for (int ly = 0; ly < roi_h; ++ly) {
        int gy = roi_y1 + ly;
        for (int lx = 0; lx < roi_w; ++lx) {
            int gx = roi_x1 + lx;
            int r, g, b;
            nv12_to_rgb(y_plane, uv_plane, pitch_y, pitch_uv, gx, gy, &r, &g, &b);
            int brightness = (r + g + b) / 3;
            bool veryBright = brightness > brightness_thresh;
            if (!veryBright) {
                mask_host[ly * roi_w + lx] = 0;
                continue;
            }
            bool hotColor = (r > MOLTEN_MIN_R && g > MOLTEN_MIN_G && b < MOLTEN_MAX_B);
            bool whiteHot = (r > MOLTEN_WHITE_HOT_MIN && g > MOLTEN_WHITE_HOT_MIN &&
                             b > MOLTEN_WHITE_HOT_MIN);
            mask_host[ly * roi_w + lx] = (hotColor || whiteHot)
                ? static_cast<unsigned char>(brightness)
                : 0;
        }
    }
    return 0;
}
