#include "molten_detect.h"
#include "molten_detect_cuda_kernel.h"

#include <gst/gst.h>
#include "nvbufsurface.h"

#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────

/**
 * BFS connected-components on a binary mask (1 = molten, 0 = background).
 * Operates entirely on the host after the mask has been copied from device.
 *
 * @param mask       Row-major binary mask, roi_w * roi_h bytes.
 * @param roi_w      Mask width  (= roi_x2 - roi_x1 + 1).
 * @param roi_h      Mask height (= roi_y2 - roi_y1 + 1).
 * @param roi_x1     Left edge of the ROI in pipeline pixel space (for abs coords).
 * @param roi_y1     Top  edge of the ROI in pipeline pixel space.
 * @param blobs_out  Caller-supplied output array.
 * @param max_blobs  Capacity of blobs_out.
 * @param num_blobs  Set to the number of valid blobs written (area >= MIN).
 */
static void bfs_connected_components(
    const unsigned char* mask,
    int                  roi_w,
    int                  roi_h,
    int                  roi_x1,
    int                  roi_y1,
    DsMoltenBlob*        blobs_out,
    int                  max_blobs,
    int*                 num_blobs,
    int                  min_blob_area
) {
    *num_blobs = 0;

    // Visited array — use a flat vector<bool> for speed
    std::vector<bool> visited(static_cast<size_t>(roi_w) * roi_h, false);

    // 4-connected neighbours
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    std::queue<int> q;  // stores flat index = ly * roi_w + lx

    for (int ly = 0; ly < roi_h; ++ly) {
        for (int lx = 0; lx < roi_w; ++lx) {
            int idx = ly * roi_w + lx;
            if (mask[idx] == 0 || visited[idx]) {
                continue;
            }

            // Start a new BFS blob
            int area = 0;
            int brightness_sum = 0;
            int bx1 = lx, by1 = ly, bx2 = lx, by2 = ly;

            q.push(idx);
            visited[idx] = true;

            while (!q.empty()) {
                int cur = q.front();
                q.pop();

                int cx = cur % roi_w;
                int cy = cur / roi_w;
                ++area;
                brightness_sum += static_cast<int>(mask[cur]);

                if (cx < bx1) bx1 = cx;
                if (cy < by1) by1 = cy;
                if (cx > bx2) bx2 = cx;
                if (cy > by2) by2 = cy;

                for (int d = 0; d < 4; ++d) {
                    int nx = cx + dx[d];
                    int ny = cy + dy[d];
                    if (nx < 0 || nx >= roi_w || ny < 0 || ny >= roi_h) {
                        continue;
                    }
                    int nidx = ny * roi_w + nx;
                    if (!visited[nidx] && mask[nidx] != 0) {
                        visited[nidx] = true;
                        q.push(nidx);
                    }
                }
            }

            // Area filter
            if (area < min_blob_area) {
                continue;
            }

            if (*num_blobs >= max_blobs) {
                continue;  // output buffer full — skip remaining large blobs
            }

            DsMoltenBlob& blob = blobs_out[(*num_blobs)++];
            blob.x1            = roi_x1 + bx1;
            blob.y1            = roi_y1 + by1;
            blob.x2            = roi_x1 + bx2;
            blob.y2            = roi_y1 + by2;
            blob.area          = area;
            blob.avg_brightness = (area > 0) ? (brightness_sum / area) : 0;
        }
    }
}


// ─── Public bridge function ───────────────────────────────────────────────────

extern "C" int ds_detect_molten_blobs_rgba(
    uint64_t      gst_buffer_addr,
    int           batch_id,
    int           roi_x1,
    int           roi_y1,
    int           roi_x2,
    int           roi_y2,
    DsMoltenBlob* blobs_out,
    int           max_blobs,
    int*          num_blobs_out,
    int           min_blob_area,
    int           brightness_thresh
) {
    if (gst_buffer_addr == 0 || blobs_out == nullptr ||
        max_blobs <= 0 || num_blobs_out == nullptr) {
        return -1;
    }

    *num_blobs_out = 0;

    GstBuffer* gst_buffer = reinterpret_cast<GstBuffer*>(gst_buffer_addr);

    GstMapInfo map_info = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(gst_buffer, &map_info, GST_MAP_READ)) {
        return -2;
    }

    NvBufSurface* surface = reinterpret_cast<NvBufSurface*>(map_info.data);
    if (surface == nullptr ||
        batch_id < 0 || batch_id >= static_cast<int>(surface->batchSize)) {
        gst_buffer_unmap(gst_buffer, &map_info);
        return -3;
    }

    NvBufSurfaceParams& params = surface->surfaceList[batch_id];
    if (params.colorFormat != NVBUF_COLOR_FORMAT_RGBA || params.dataPtr == nullptr) {
        gst_buffer_unmap(gst_buffer, &map_info);
        return -4;
    }

    int frame_w = static_cast<int>(params.width);
    int frame_h = static_cast<int>(params.height);
    int pitch   = static_cast<int>(params.pitch);

    // Clamp ROI to frame
    roi_x1 = std::max(0, std::min(frame_w - 1, roi_x1));
    roi_x2 = std::max(0, std::min(frame_w - 1, roi_x2));
    roi_y1 = std::max(0, std::min(frame_h - 1, roi_y1));
    roi_y2 = std::max(0, std::min(frame_h - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        gst_buffer_unmap(gst_buffer, &map_info);
        return 0;
    }

    int roi_w = roi_x2 - roi_x1 + 1;
    int roi_h = roi_y2 - roi_y1 + 1;

    // Allocate host mask
    std::vector<unsigned char> mask(static_cast<size_t>(roi_w) * roi_h, 0);

    // Run CUDA pixel classifier
    int rc = ds_classify_molten_pixels_device(
        params.dataPtr,
        frame_w,
        frame_h,
        pitch,
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        mask.data(),
        brightness_thresh
    );

    gst_buffer_unmap(gst_buffer, &map_info);

    if (rc != 0) {
        return rc;
    }

    // CPU BFS connected components + area filter
    bfs_connected_components(
        mask.data(),
        roi_w,
        roi_h,
        roi_x1,
        roi_y1,
        blobs_out,
        max_blobs,
        num_blobs_out,
        min_blob_area
    );

    return 0;
}

extern "C" int ds_detect_molten_blobs_rgba_device(
    const void*      rgba_device_ptr,
    int              width,
    int              height,
    int              pitch_bytes,
    int              roi_x1,
    int              roi_y1,
    int              roi_x2,
    int              roi_y2,
    DsMoltenBlob*    blobs_out,
    int              max_blobs,
    int*             num_blobs_out,
    int              min_blob_area,
    int              brightness_thresh
) {
    if (rgba_device_ptr == nullptr || blobs_out == nullptr ||
        max_blobs <= 0 || num_blobs_out == nullptr) {
        return -1;
    }

    *num_blobs_out = 0;

    roi_x1 = std::max(0, std::min(width - 1, roi_x1));
    roi_x2 = std::max(0, std::min(width - 1, roi_x2));
    roi_y1 = std::max(0, std::min(height - 1, roi_y1));
    roi_y2 = std::max(0, std::min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        return 0;
    }

    int roi_w = roi_x2 - roi_x1 + 1;
    int roi_h = roi_y2 - roi_y1 + 1;

    std::vector<unsigned char> mask(static_cast<size_t>(roi_w) * roi_h, 0);

    int rc = ds_classify_molten_pixels_device(
        rgba_device_ptr,
        width,
        height,
        pitch_bytes,
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        mask.data(),
        brightness_thresh
    );
    if (rc != 0) {
        return rc;
    }

    bfs_connected_components(
        mask.data(),
        roi_w,
        roi_h,
        roi_x1,
        roi_y1,
        blobs_out,
        max_blobs,
        num_blobs_out,
        min_blob_area
    );

    return 0;
}

extern "C" int ds_detect_molten_blobs_nv12_device(
    const void*      y_plane_device_ptr,
    const void*      uv_plane_device_ptr,
    int              width,
    int              height,
    int              pitch_y,
    int              pitch_uv,
    int              roi_x1,
    int              roi_y1,
    int              roi_x2,
    int              roi_y2,
    DsMoltenBlob*    blobs_out,
    int              max_blobs,
    int*             num_blobs_out,
    int              min_blob_area,
    int              brightness_thresh
) {
    if (y_plane_device_ptr == nullptr || uv_plane_device_ptr == nullptr || blobs_out == nullptr ||
        max_blobs <= 0 || num_blobs_out == nullptr) {
        return -1;
    }

    *num_blobs_out = 0;

    roi_x1 = std::max(0, std::min(width - 1, roi_x1));
    roi_x2 = std::max(0, std::min(width - 1, roi_x2));
    roi_y1 = std::max(0, std::min(height - 1, roi_y1));
    roi_y2 = std::max(0, std::min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        return 0;
    }

    int roi_w = roi_x2 - roi_x1 + 1;
    int roi_h = roi_y2 - roi_y1 + 1;

    std::vector<unsigned char> mask(static_cast<size_t>(roi_w) * roi_h, 0);

    int rc = ds_classify_molten_pixels_device_nv12(
        y_plane_device_ptr,
        uv_plane_device_ptr,
        width,
        height,
        pitch_y,
        pitch_uv,
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        mask.data(),
        brightness_thresh
    );
    if (rc != 0) {
        return rc;
    }

    bfs_connected_components(
        mask.data(),
        roi_w,
        roi_h,
        roi_x1,
        roi_y1,
        blobs_out,
        max_blobs,
        num_blobs_out,
        min_blob_area
    );

    return 0;
}
