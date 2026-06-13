/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <sstream>

#include "nvdsinfer_custom_impl.h"

enum class YoloTensorLayout
{
  kRowMajor,
  kTransposed
};

struct YoloTensorSpec
{
  bool valid {false};
  YoloTensorLayout layout {YoloTensorLayout::kRowMajor};
  uint outputSize {0};
  uint channelCount {0};
  std::string dimsStr;
};

extern "C" bool
NvDsInferParseYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

static std::string
inferDimsToString(const NvDsInferDims& inferDims)
{
  std::stringstream s;
  s << "[";
  for (int i = 0; i < inferDims.numDims; ++i) {
    if (i > 0) {
      s << ", ";
    }
    s << inferDims.d[i];
  }
  s << "]";
  return s.str();
}

static YoloTensorSpec
getYoloTensorSpec(const NvDsInferDims& inferDims)
{
  YoloTensorSpec spec;
  spec.dimsStr = inferDimsToString(inferDims);

  std::vector<int> dims;
  dims.reserve(inferDims.numDims);
  for (int i = 0; i < inferDims.numDims; ++i) {
    if (inferDims.d[i] <= 0) {
      return spec;
    }
    dims.push_back(inferDims.d[i]);
  }

  if (dims.size() >= 3 && dims[0] == 1) {
    dims.erase(dims.begin());
  }

  if (dims.size() != 2) {
    return spec;
  }

  if (dims[1] == 6) {
    spec.valid = true;
    spec.layout = YoloTensorLayout::kRowMajor;
    spec.outputSize = dims[0];
    spec.channelCount = dims[1];
  }
  else if (dims[0] == 6) {
    spec.valid = true;
    spec.layout = YoloTensorLayout::kTransposed;
    spec.outputSize = dims[1];
    spec.channelCount = dims[0];
  }

  return spec;
}

__global__ void decodeTensorYoloCuda(NvDsInferParseObjectInfo *binfo, const float* output, const uint outputSize,
    const uint netW, const uint netH, const float* preclusterThreshold, const int numClasses, const bool transposed,
    const bool rawClassScores)
{
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize) {
    return;
  }

  float maxProb = 0.0;
  int maxIndex = -1;
  float bx1 = 0.0;
  float by1 = 0.0;
  float bx2 = 0.0;
  float by2 = 0.0;

  if (transposed) {
    float box0 = output[0 * outputSize + x_id];
    float box1 = output[1 * outputSize + x_id];
    float box2 = output[2 * outputSize + x_id];
    float box3 = output[3 * outputSize + x_id];

    if (rawClassScores) {
      /* Ultralytics opset-17 export emits transposed [4 + C, N] with boxes in
       * cx, cy, w, h order. Convert to x1, y1, x2, y2 before clamping. */
      float halfW = box2 * 0.5f;
      float halfH = box3 * 0.5f;
      bx1 = box0 - halfW;
      by1 = box1 - halfH;
      bx2 = box0 + halfW;
      by2 = box1 + halfH;
    }
    else {
      bx1 = box0;
      by1 = box1;
      bx2 = box2;
      by2 = box3;
    }

    if (rawClassScores) {
      for (int classIndex = 0; classIndex < numClasses; ++classIndex) {
        float classProb = output[(4 + classIndex) * outputSize + x_id];
        if (classProb > maxProb) {
          maxProb = classProb;
          maxIndex = classIndex;
        }
      }
    }
    else {
      maxProb = output[4 * outputSize + x_id];
      maxIndex = (int) output[5 * outputSize + x_id];
    }
  }
  else {
    maxProb = output[x_id * 6 + 4];
    maxIndex = (int) output[x_id * 6 + 5];
    bx1 = output[x_id * 6 + 0];
    by1 = output[x_id * 6 + 1];
    bx2 = output[x_id * 6 + 2];
    by2 = output[x_id * 6 + 3];
  }

  if (maxIndex < 0 || maxIndex >= numClasses || maxProb < preclusterThreshold[maxIndex]) {
    binfo[x_id].detectionConfidence = 0.0;
    return;
  }

  bx1 = fminf(float(netW), fmaxf(float(0.0), bx1));
  by1 = fminf(float(netH), fmaxf(float(0.0), by1));
  bx2 = fminf(float(netW), fmaxf(float(0.0), bx2));
  by2 = fminf(float(netH), fmaxf(float(0.0), by2));

  binfo[x_id].left = bx1;
  binfo[x_id].top = by1;
  binfo[x_id].width = fminf(float(netW), fmaxf(float(0.0), bx2 - bx1));
  binfo[x_id].height = fminf(float(netH), fmaxf(float(0.0), by2 - by1));
  binfo[x_id].detectionConfidence = maxProb;
  binfo[x_id].classId = maxIndex;
}

static bool NvDsInferParseCustomYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const YoloTensorSpec tensorSpec = getYoloTensorSpec(output.inferDims);

  if (!tensorSpec.valid) {
    static std::string lastUnsupportedShape;
    if (lastUnsupportedShape != tensorSpec.dimsStr) {
      std::cerr << "ERROR: Unsupported YOLO CUDA output shape " << tensorSpec.dimsStr
                << ". Expected [N, 6] or [6, N] after optional batch dimension." << std::endl;
      lastUnsupportedShape = tensorSpec.dimsStr;
    }
    objectList.clear();
    return true;
  }

  const bool rawClassScores = tensorSpec.layout == YoloTensorLayout::kTransposed &&
      tensorSpec.channelCount == 4 + detectionParams.perClassPreclusterThreshold.size();

  thrust::device_vector<float> perClassPreclusterThreshold = detectionParams.perClassPreclusterThreshold;

  thrust::device_vector<NvDsInferParseObjectInfo> objects(tensorSpec.outputSize);

  int threads_per_block = 1024;
  int number_of_blocks = ((tensorSpec.outputSize) / threads_per_block) + 1;

  decodeTensorYoloCuda<<<number_of_blocks, threads_per_block>>>(
      thrust::raw_pointer_cast(objects.data()), (float*) (output.buffer), tensorSpec.outputSize, networkInfo.width,
          networkInfo.height, thrust::raw_pointer_cast(perClassPreclusterThreshold.data()),
          detectionParams.perClassPreclusterThreshold.size(),
          tensorSpec.layout == YoloTensorLayout::kTransposed, rawClassScores);

  objectList.resize(tensorSpec.outputSize);
  thrust::copy(objects.begin(), objects.end(), objectList.begin());

  return true;
}

extern "C" bool
NvDsInferParseYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloCuda);
