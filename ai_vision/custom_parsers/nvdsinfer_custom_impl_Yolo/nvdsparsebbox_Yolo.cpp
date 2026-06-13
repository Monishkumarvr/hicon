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

#include "nvdsinfer_custom_impl.h"

#include "utils.h"

#include <sstream>

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
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

static std::string
inferDimsToString(const NvDsInferDims& inferDims)
{
  std::stringstream s;
  s << "[";
  for (unsigned int i = 0; i < inferDims.numDims; ++i) {
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
  for (unsigned int i = 0; i < inferDims.numDims; ++i) {
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

static NvDsInferParseObjectInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferParseObjectInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

  if (bbi.width < 1 || bbi.height < 1) {
    return;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
  binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(const float* output, const uint& outputSize, const uint& netW, const uint& netH,
    const std::vector<float>& preclusterThreshold, const YoloTensorLayout layout, const bool rawClassScores)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = 0.0;
    int maxIndex = -1;
    float bx1 = 0.0;
    float by1 = 0.0;
    float bx2 = 0.0;
    float by2 = 0.0;

    if (layout == YoloTensorLayout::kRowMajor) {
      maxProb = output[b * 6 + 4];
      maxIndex = (int) output[b * 6 + 5];
      bx1 = output[b * 6 + 0];
      by1 = output[b * 6 + 1];
      bx2 = output[b * 6 + 2];
      by2 = output[b * 6 + 3];
    }
    else {
      float box0 = output[0 * outputSize + b];
      float box1 = output[1 * outputSize + b];
      float box2 = output[2 * outputSize + b];
      float box3 = output[3 * outputSize + b];

      if (rawClassScores) {
        /* Ultralytics opset-17 export emits transposed [4 + C, N] with boxes
         * in cx, cy, w, h order, not x1, y1, x2, y2 like the legacy export. */
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
        for (int classIndex = 0; classIndex < (int) preclusterThreshold.size(); ++classIndex) {
          float classProb = output[(4 + classIndex) * outputSize + b];
          if (classProb > maxProb) {
            maxProb = classProb;
            maxIndex = classIndex;
          }
        }
      }
      else {
        maxProb = output[4 * outputSize + b];
        maxIndex = (int) output[5 * outputSize + b];
      }
    }

    if (maxIndex < 0 || maxIndex >= (int) preclusterThreshold.size()) {
      continue;
    }

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const YoloTensorSpec tensorSpec = getYoloTensorSpec(output.inferDims);

  if (!tensorSpec.valid) {
    static std::string lastUnsupportedShape;
    if (lastUnsupportedShape != tensorSpec.dimsStr) {
      std::cerr << "ERROR: Unsupported YOLO output shape " << tensorSpec.dimsStr
                << ". Expected [N, 6] or [6, N] after optional batch dimension." << std::endl;
      lastUnsupportedShape = tensorSpec.dimsStr;
    }
    objectList.clear();
    return true;
  }

  const bool rawClassScores = tensorSpec.layout == YoloTensorLayout::kTransposed &&
      tensorSpec.channelCount == 4 + detectionParams.perClassPreclusterThreshold.size();

  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo((const float*) (output.buffer),
      tensorSpec.outputSize, networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold,
      tensorSpec.layout, rawClassScores);

  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
