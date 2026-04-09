#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int ds_extract_white_ratio_rgba(
    uint64_t gst_buffer_addr,
    int batch_id,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int absolute_brightness_threshold,
    float* out_white_ratio
);

#ifdef __cplusplus
}
#endif
