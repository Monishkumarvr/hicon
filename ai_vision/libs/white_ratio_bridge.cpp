#include "white_ratio_cuda.h"
#include "white_ratio_cuda_kernel.h"

#include <gst/gst.h>
#include "nvbufsurface.h"

extern "C" int ds_extract_white_ratio_rgba(
    uint64_t gst_buffer_addr,
    int batch_id,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int absolute_brightness_threshold,
    float* out_white_ratio
) {
    if (gst_buffer_addr == 0 || out_white_ratio == nullptr || absolute_brightness_threshold < 0) {
        return -1;
    }

    GstBuffer* gst_buffer = reinterpret_cast<GstBuffer*>(gst_buffer_addr);

    GstMapInfo map_info = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(gst_buffer, &map_info, GST_MAP_READ)) {
        return -2;
    }

    NvBufSurface* surface = reinterpret_cast<NvBufSurface*>(map_info.data);
    if (surface == nullptr || batch_id < 0 || batch_id >= static_cast<int>(surface->batchSize)) {
        gst_buffer_unmap(gst_buffer, &map_info);
        return -3;
    }

    NvBufSurfaceParams& params = surface->surfaceList[batch_id];
    if (params.colorFormat != NVBUF_COLOR_FORMAT_RGBA || params.dataPtr == nullptr) {
        gst_buffer_unmap(gst_buffer, &map_info);
        return -4;
    }

    int rc = ds_compute_white_ratio_from_device_rgba(
        params.dataPtr,
        static_cast<int>(params.width),
        static_cast<int>(params.height),
        static_cast<int>(params.pitch),
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        absolute_brightness_threshold,
        out_white_ratio
    );

    gst_buffer_unmap(gst_buffer, &map_info);
    return rc;
}
