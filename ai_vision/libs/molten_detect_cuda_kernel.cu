#include "molten_detect_cuda_kernel.h"

#include <cuda_runtime.h>

// ─── Hardcoded detection thresholds ──────────────────────────────────────────
// Do NOT add function arguments for these — they are intentionally baked in.

#define MOLTEN_MIN_R              180   // hot-color gate: red floor
#define MOLTEN_MIN_G               80   // hot-color gate: green floor (not too dark)
#define MOLTEN_MAX_B              200   // hot-color gate: blue ceiling (avoid pure white)
#define MOLTEN_WHITE_HOT_MIN      240   // white-hot threshold for all three channels

// ─────────────────────────────────────────────────────────────────────────────

namespace {

__device__ inline int clamp_u8(int value) {
    return max(0, min(255, value));
}

__device__ inline void nv12_to_rgb(
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
    int y_val = static_cast<int>(y_plane[y * pitch_y + x]);
    int uv_index = (y / 2) * pitch_uv + (x / 2) * 2;
    int u = static_cast<int>(uv_plane[uv_index]) - 128;
    int v = static_cast<int>(uv_plane[uv_index + 1]) - 128;
    int c = max(0, y_val - 16);
    int r = (298 * c + 409 * v + 128) >> 8;
    int g = (298 * c - 100 * u - 208 * v + 128) >> 8;
    int b = (298 * c + 516 * u + 128) >> 8;
    *out_r = clamp_u8(r);
    *out_g = clamp_u8(g);
    *out_b = clamp_u8(b);
}

__global__ void molten_classify_rgba_kernel(
    const uchar4*  rgba,
    int            frame_width,
    int            frame_height,
    int            pitch_bytes,
    int            roi_x1,
    int            roi_y1,
    int            roi_w,           // roi_x2 - roi_x1 + 1
    int            roi_h,           // roi_y2 - roi_y1 + 1
    unsigned char* mask,            // roi_w * roi_h bytes, row-major
    int            brightness_thresh
) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x;  // local x within ROI
    int ly = blockIdx.y * blockDim.y + threadIdx.y;  // local y within ROI

    if (lx >= roi_w || ly >= roi_h) {
        return;
    }

    int gx = roi_x1 + lx;   // global frame x
    int gy = roi_y1 + ly;   // global frame y

    // Read pixel (RGBA stored as uchar4: x=R, y=G, z=B, w=A)
    const uchar4* row = reinterpret_cast<const uchar4*>(
        reinterpret_cast<const unsigned char*>(rgba) + gy * pitch_bytes
    );
    uchar4 px = row[gx];

    int r = static_cast<int>(px.x);
    int g = static_cast<int>(px.y);
    int b = static_cast<int>(px.z);

    // Layer 1: intensity gate
    int brightness = (r + g + b) / 3;
    bool veryBright = (brightness > brightness_thresh);

    if (!veryBright) {
        mask[ly * roi_w + lx] = 0;
        return;
    }

    // Layer 2: color signature
    // hot-color: orange / pink / white-orange range
    bool hotColor = (r > MOLTEN_MIN_R && g > MOLTEN_MIN_G && b < MOLTEN_MAX_B);
    // white-hot: all channels saturated
    bool whiteHot = (r > MOLTEN_WHITE_HOT_MIN && g > MOLTEN_WHITE_HOT_MIN && b > MOLTEN_WHITE_HOT_MIN);

    // Store brightness value for classified pixels (always >= brightness_thresh+1 > 0).
    // Unclassified pixels remain 0, so BFS can distinguish them.
    mask[ly * roi_w + lx] = (hotColor || whiteHot) ? (unsigned char)brightness : 0;
}

__global__ void molten_classify_nv12_kernel(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    int frame_width,
    int frame_height,
    int pitch_y,
    int pitch_uv,
    int roi_x1,
    int roi_y1,
    int roi_w,
    int roi_h,
    unsigned char* mask,
    int brightness_thresh
) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;

    if (lx >= roi_w || ly >= roi_h) {
        return;
    }

    int gx = roi_x1 + lx;
    int gy = roi_y1 + ly;
    if (gx < 0 || gy < 0 || gx >= frame_width || gy >= frame_height) {
        mask[ly * roi_w + lx] = 0;
        return;
    }

    int r = 0;
    int g = 0;
    int b = 0;
    nv12_to_rgb(y_plane, uv_plane, pitch_y, pitch_uv, gx, gy, &r, &g, &b);

    int brightness = (r + g + b) / 3;
    bool veryBright = (brightness > brightness_thresh);
    if (!veryBright) {
        mask[ly * roi_w + lx] = 0;
        return;
    }

    bool hotColor = (r > MOLTEN_MIN_R && g > MOLTEN_MIN_G && b < MOLTEN_MAX_B);
    bool whiteHot = (r > MOLTEN_WHITE_HOT_MIN &&
                     g > MOLTEN_WHITE_HOT_MIN &&
                     b > MOLTEN_WHITE_HOT_MIN);

    mask[ly * roi_w + lx] = (hotColor || whiteHot) ? (unsigned char)brightness : 0;
}

} // anonymous namespace


int ds_classify_molten_pixels_device(
    const void*    rgba_device_ptr,
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
    if (rgba_device_ptr == nullptr || mask_host == nullptr ||
        width <= 0 || height <= 0) {
        return -1;
    }

    // Clamp ROI to frame bounds
    roi_x1 = max(0, min(width  - 1, roi_x1));
    roi_x2 = max(0, min(width  - 1, roi_x2));
    roi_y1 = max(0, min(height - 1, roi_y1));
    roi_y2 = max(0, min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        return 0;  // zero-area ROI: nothing to do
    }

    int roi_w = roi_x2 - roi_x1 + 1;
    int roi_h = roi_y2 - roi_y1 + 1;
    size_t mask_bytes = static_cast<size_t>(roi_w) * roi_h;

    // Allocate device mask
    unsigned char* mask_device = nullptr;
    cudaError_t err = cudaMalloc(&mask_device, mask_bytes);
    if (err != cudaSuccess) {
        return -2;
    }

    err = cudaMemset(mask_device, 0, mask_bytes);
    if (err != cudaSuccess) {
        cudaFree(mask_device);
        return -3;
    }

    // Launch kernel
    dim3 threads(16, 16);
    dim3 blocks(
        (roi_w + threads.x - 1) / threads.x,
        (roi_h + threads.y - 1) / threads.y
    );

    molten_classify_rgba_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uchar4*>(rgba_device_ptr),
        width,
        height,
        pitch_bytes,
        roi_x1,
        roi_y1,
        roi_w,
        roi_h,
        mask_device,
        brightness_thresh
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mask_device);
        return -4;
    }

    // Copy mask back to host
    err = cudaMemcpy(mask_host, mask_device, mask_bytes, cudaMemcpyDeviceToHost);
    cudaFree(mask_device);
    if (err != cudaSuccess) {
        return -5;
    }

    return 0;
}

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
) {
    if (y_plane_device_ptr == nullptr || uv_plane_device_ptr == nullptr || mask_host == nullptr ||
        width <= 0 || height <= 0 || pitch_y <= 0 || pitch_uv <= 0) {
        return -1;
    }

    roi_x1 = max(0, min(width  - 1, roi_x1));
    roi_x2 = max(0, min(width  - 1, roi_x2));
    roi_y1 = max(0, min(height - 1, roi_y1));
    roi_y2 = max(0, min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        return 0;
    }

    int roi_w = roi_x2 - roi_x1 + 1;
    int roi_h = roi_y2 - roi_y1 + 1;
    size_t mask_bytes = static_cast<size_t>(roi_w) * roi_h;

    unsigned char* mask_device = nullptr;
    cudaError_t err = cudaMalloc(&mask_device, mask_bytes);
    if (err != cudaSuccess) {
        return -2;
    }

    err = cudaMemset(mask_device, 0, mask_bytes);
    if (err != cudaSuccess) {
        cudaFree(mask_device);
        return -3;
    }

    const unsigned char* y_plane =
        reinterpret_cast<const unsigned char*>(y_plane_device_ptr);
    const unsigned char* uv_plane =
        reinterpret_cast<const unsigned char*>(uv_plane_device_ptr);

    dim3 threads(16, 16);
    dim3 blocks(
        (roi_w + threads.x - 1) / threads.x,
        (roi_h + threads.y - 1) / threads.y
    );

    molten_classify_nv12_kernel<<<blocks, threads>>>(
        y_plane,
        uv_plane,
        width,
        height,
        pitch_y,
        pitch_uv,
        roi_x1,
        roi_y1,
        roi_w,
        roi_h,
        mask_device,
        brightness_thresh
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mask_device);
        return -4;
    }

    err = cudaMemcpy(mask_host, mask_device, mask_bytes, cudaMemcpyDeviceToHost);
    cudaFree(mask_device);
    if (err != cudaSuccess) {
        return -5;
    }

    return 0;
}
