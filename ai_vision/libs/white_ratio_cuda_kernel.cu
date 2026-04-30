/*
 * White-ratio computation — CPU implementation (no CUDA).
 * Same function signatures as the former CUDA version so callers are unchanged.
 */

#include "white_ratio_cuda_kernel.h"

#include <algorithm>
#include <cstdint>

static inline int clamp_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

int ds_compute_white_ratio_from_device_rgba(
    const void* rgba_ptr,
    int width,
    int height,
    int pitch_bytes,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int threshold,
    float* out_white_ratio
) {
    if (!rgba_ptr || !out_white_ratio || width <= 0 || height <= 0 || threshold < 0)
        return -1;

    roi_x1 = std::max(0, std::min(width  - 1, roi_x1));
    roi_x2 = std::max(0, std::min(width  - 1, roi_x2));
    roi_y1 = std::max(0, std::min(height - 1, roi_y1));
    roi_y2 = std::max(0, std::min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        *out_white_ratio = 0.0f;
        return 0;
    }

    const unsigned char* base = reinterpret_cast<const unsigned char*>(rgba_ptr);
    unsigned int white_count = 0;
    unsigned int total_count = 0;

    for (int y = roi_y1; y <= roi_y2; ++y) {
        const unsigned char* row = base + y * pitch_bytes;
        for (int x = roi_x1; x <= roi_x2; ++x) {
            /* RGBA layout: R=0, G=1, B=2, A=3 */
            int r = row[x * 4 + 0];
            int g = row[x * 4 + 1];
            int b = row[x * 4 + 2];
            int value = r > g ? (r > b ? r : b) : (g > b ? g : b); /* max(R,G,B) */
            ++total_count;
            if (value > threshold)
                ++white_count;
        }
    }

    *out_white_ratio = (total_count == 0)
        ? 0.0f
        : static_cast<float>(white_count) / static_cast<float>(total_count);
    return 0;
}

int ds_compute_white_ratio_from_device_nv12(
    const void* y_ptr,
    const void* uv_ptr,
    int width,
    int height,
    int pitch_y,
    int pitch_uv,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int threshold,
    float* out_white_ratio
) {
    if (!y_ptr || !uv_ptr || !out_white_ratio ||
        width <= 0 || height <= 0 || pitch_y <= 0 || pitch_uv <= 0 || threshold < 0)
        return -1;

    roi_x1 = std::max(0, std::min(width  - 1, roi_x1));
    roi_x2 = std::max(0, std::min(width  - 1, roi_x2));
    roi_y1 = std::max(0, std::min(height - 1, roi_y1));
    roi_y2 = std::max(0, std::min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        *out_white_ratio = 0.0f;
        return 0;
    }

    const unsigned char* y_plane  = reinterpret_cast<const unsigned char*>(y_ptr);
    unsigned int white_count = 0;
    unsigned int total_count = 0;

    for (int y = roi_y1; y <= roi_y2; ++y) {
        for (int x = roi_x1; x <= roi_x2; ++x) {
            /* Use luma (Y) as brightness proxy — equivalent to max(R,G,B) for
             * bright molten-metal pixels and avoids the full NV12→RGB conversion. */
            int luma = y_plane[y * pitch_y + x];
            ++total_count;
            if (luma > threshold)
                ++white_count;
        }
    }

    *out_white_ratio = (total_count == 0)
        ? 0.0f
        : static_cast<float>(white_count) / static_cast<float>(total_count);
    return 0;
}
