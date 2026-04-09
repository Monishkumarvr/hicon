#include "white_ratio_cuda_kernel.h"

#include <cuda_runtime.h>

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

__global__ void white_ratio_rgba_kernel(
    const uchar4* rgba,
    int width,
    int height,
    int pitch_bytes,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int threshold,
    unsigned int* white_count,
    unsigned int* total_count
) {
    int x = roi_x1 + blockIdx.x * blockDim.x + threadIdx.x;
    int y = roi_y1 + blockIdx.y * blockDim.y + threadIdx.y;

    if (x > roi_x2 || y > roi_y2) {
        return;
    }

    const uchar4* row = reinterpret_cast<const uchar4*>(
        reinterpret_cast<const unsigned char*>(rgba) + y * pitch_bytes
    );

    uchar4 pixel = row[x];
    int value = max(static_cast<int>(pixel.x),
                    max(static_cast<int>(pixel.y), static_cast<int>(pixel.z)));

    atomicAdd(total_count, 1U);
    if (value > threshold) {
        atomicAdd(white_count, 1U);
    }
}

__global__ void white_ratio_nv12_kernel(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    int width,
    int height,
    int pitch_y,
    int pitch_uv,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int threshold,
    unsigned int* white_count,
    unsigned int* total_count
) {
    int x = roi_x1 + blockIdx.x * blockDim.x + threadIdx.x;
    int y = roi_y1 + blockIdx.y * blockDim.y + threadIdx.y;

    if (x > roi_x2 || y > roi_y2 || x < 0 || y < 0 || x >= width || y >= height) {
        return;
    }

    int r = 0;
    int g = 0;
    int b = 0;
    nv12_to_rgb(y_plane, uv_plane, pitch_y, pitch_uv, x, y, &r, &g, &b);
    int value = max(r, max(g, b));

    atomicAdd(total_count, 1U);
    if (value > threshold) {
        atomicAdd(white_count, 1U);
    }
}

} // namespace

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
) {
    if (
        rgba_device_ptr == nullptr ||
        out_white_ratio == nullptr ||
        width <= 0 ||
        height <= 0 ||
        absolute_brightness_threshold < 0
    ) {
        return -1;
    }

    roi_x1 = max(0, min(width - 1, roi_x1));
    roi_x2 = max(0, min(width - 1, roi_x2));
    roi_y1 = max(0, min(height - 1, roi_y1));
    roi_y2 = max(0, min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        *out_white_ratio = 0.0f;
        return 0;
    }

    unsigned int* white_count_device = nullptr;
    unsigned int* total_count_device = nullptr;

    cudaError_t cuda_status = cudaMalloc(&white_count_device, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        return -2;
    }

    cuda_status = cudaMalloc(&total_count_device, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        return -3;
    }

    cuda_status = cudaMemset(white_count_device, 0, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -4;
    }

    cuda_status = cudaMemset(total_count_device, 0, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -5;
    }

    dim3 threads(16, 16);
    dim3 blocks(
        (roi_x2 - roi_x1 + 1 + threads.x - 1) / threads.x,
        (roi_y2 - roi_y1 + 1 + threads.y - 1) / threads.y
    );

    white_ratio_rgba_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uchar4*>(rgba_device_ptr),
        width,
        height,
        pitch_bytes,
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        absolute_brightness_threshold,
        white_count_device,
        total_count_device
    );

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -6;
    }

    unsigned int white_count_host = 0;
    unsigned int total_count_host = 0;

    cuda_status = cudaMemcpy(
        &white_count_host,
        white_count_device,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -7;
    }

    cuda_status = cudaMemcpy(
        &total_count_host,
        total_count_device,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -8;
    }

    if (total_count_host == 0U) {
        *out_white_ratio = 0.0f;
    } else {
        *out_white_ratio = static_cast<float>(white_count_host) /
                           static_cast<float>(total_count_host);
    }

    cudaFree(white_count_device);
    cudaFree(total_count_device);
    return 0;
}

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
) {
    if (
        y_plane_device_ptr == nullptr ||
        uv_plane_device_ptr == nullptr ||
        out_white_ratio == nullptr ||
        width <= 0 ||
        height <= 0 ||
        pitch_y <= 0 ||
        pitch_uv <= 0 ||
        absolute_brightness_threshold < 0
    ) {
        return -1;
    }

    roi_x1 = max(0, min(width - 1, roi_x1));
    roi_x2 = max(0, min(width - 1, roi_x2));
    roi_y1 = max(0, min(height - 1, roi_y1));
    roi_y2 = max(0, min(height - 1, roi_y2));

    if (roi_x2 < roi_x1 || roi_y2 < roi_y1) {
        *out_white_ratio = 0.0f;
        return 0;
    }

    const unsigned char* y_plane =
        reinterpret_cast<const unsigned char*>(y_plane_device_ptr);
    const unsigned char* uv_plane =
        reinterpret_cast<const unsigned char*>(uv_plane_device_ptr);

    unsigned int* white_count_device = nullptr;
    unsigned int* total_count_device = nullptr;

    cudaError_t cuda_status = cudaMalloc(&white_count_device, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        return -2;
    }

    cuda_status = cudaMalloc(&total_count_device, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        return -3;
    }

    cuda_status = cudaMemset(white_count_device, 0, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -4;
    }

    cuda_status = cudaMemset(total_count_device, 0, sizeof(unsigned int));
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -5;
    }

    dim3 threads(16, 16);
    dim3 blocks(
        (roi_x2 - roi_x1 + 1 + threads.x - 1) / threads.x,
        (roi_y2 - roi_y1 + 1 + threads.y - 1) / threads.y
    );

    white_ratio_nv12_kernel<<<blocks, threads>>>(
        y_plane,
        uv_plane,
        width,
        height,
        pitch_y,
        pitch_uv,
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
        absolute_brightness_threshold,
        white_count_device,
        total_count_device
    );

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -6;
    }

    unsigned int white_count_host = 0;
    unsigned int total_count_host = 0;

    cuda_status = cudaMemcpy(
        &white_count_host,
        white_count_device,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -7;
    }

    cuda_status = cudaMemcpy(
        &total_count_host,
        total_count_device,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    if (cuda_status != cudaSuccess) {
        cudaFree(white_count_device);
        cudaFree(total_count_device);
        return -8;
    }

    if (total_count_host == 0U) {
        *out_white_ratio = 0.0f;
    } else {
        *out_white_ratio = static_cast<float>(white_count_host) /
                           static_cast<float>(total_count_host);
    }

    cudaFree(white_count_device);
    cudaFree(total_count_device);
    return 0;
}
