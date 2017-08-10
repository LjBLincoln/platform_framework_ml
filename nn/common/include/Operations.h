/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_COMMON_OPERATIONS_H
#define ANDROID_ML_NN_COMMON_OPERATIONS_H

#include <cstdint>
#include <stddef.h>
#include <vector>

namespace android {
namespace nn {

struct Shape;

enum PaddingScheme {
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
};
enum ActivationFn {
    kActivationNone = 0,
    kActivationRelu = 1,
    kActivationRelu1 = 2,
    kActivationRelu6 = 3,
};

bool addPrepare(const Shape& in1, const Shape& in2, Shape* out1);
bool addFloat32(const float* in1, const float* in2,
                int32_t activation,
                float* out, const Shape& shape);

bool mulPrepare(const Shape& in1, const Shape& in2, Shape* out1);
bool mulFloat32(const float* in1, const float* in2,
                int32_t activation,
                float* out, const Shape& shape);

bool floorPrepare(const Shape& input, Shape* output);
bool floorFloat32(const float* inputData,
                  float* outputData,
                  const Shape& shape);

bool dequantizePrepare(const Shape& input, Shape* output);
bool dequantizeQuant8ToFloat32(const uint8_t* inputData,
                               float* outputData,
                               const Shape& shape);

bool depthwiseConvPrepare(const Shape& input,
                                 const Shape& filter,
                                 const Shape& bias,
                                 int32_t padding,
                                 int32_t stride_width, int32_t stride_height,
                                 Shape* output);
bool depthwiseConvFloat32(const float* inputData, const Shape& inputShape,
                          const float* filterData, const Shape& filterShape,
                          const float* biasData, const Shape& biasShape,
                          int32_t padding, int32_t stride_width, int32_t stride_height,
                          int32_t depth_multiplier, int32_t activation,
                          float* outputData, const Shape& outputShape);
bool depthwiseConvQuant8(const uint8_t* inputData, const Shape& inputShape,
                         const uint8_t* filterData, const Shape& filterShape,
                         const int32_t* biasData, const Shape& biasShape,
                         int32_t padding, int32_t stride_width, int32_t stride_height,
                         int32_t depth_multiplier, int32_t activation,
                         uint8_t* outputData, const Shape& outputShape);

bool convPrepare(const Shape& input,
                 const Shape& filter,
                 const Shape& bias,
                 int32_t padding,
                 int32_t stride_width, int32_t stride_height,
                 Shape* output);
bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding, int32_t stride_width, int32_t stride_height,
                 int32_t activation,
                 float* outputData, const Shape& outputShape);
bool convQuant8(const uint8_t* inputData, const Shape& inputShape,
                const uint8_t* filterData, const Shape& filterShape,
                const int32_t* biasData, const Shape& biasShape,
                int32_t padding, int32_t stride_width, int32_t stride_height,
                int32_t activation,
                uint8_t* outputData, const Shape& outputShape);

bool genericPoolingPrepare(const Shape& input,
                           int32_t padding,
                           int32_t stride_width, int32_t stride_height,
                           int32_t filter_width, int32_t filter_height,
                           Shape* output);
bool averagePoolFloat32(const float* inputData, const Shape& inputShape,
                        int32_t padding, int32_t stride_width, int32_t stride_height,
                        int32_t filter_width, int32_t filter_height, int32_t activation,
                        float* outputData, const Shape& outputShape);
bool averagePoolQuant8(const uint8_t* inputData, const Shape& inputShape,
                       int32_t padding, int32_t stride_width, int32_t stride_height,
                       int32_t filter_width, int32_t filter_height, int32_t activation,
                       uint8_t* outputData, const Shape& outputShape);
bool l2PoolFloat32(const float* inputData, const Shape& inputShape,
                   int32_t padding, int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
                   float* outputData, const Shape& outputShape);
bool maxPoolFloat32(const float* inputData, const Shape& inputShape,
                    int32_t padding, int32_t stride_width, int32_t stride_height,
                    int32_t filter_width, int32_t filter_height, int32_t activation,
                    float* outputData, const Shape& outputShape);
bool maxPoolQuant8(const uint8_t* inputData, const Shape& inputShape,
                   int32_t padding, int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
                   uint8_t* outputData, const Shape& outputShape);

bool genericActivationPrepare(const Shape& input, Shape* output);
bool reluFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape);
bool relu6Float32(const float* inputData, const Shape& inputShape,
                  float* outputData, const Shape& outputShape);
bool tanhFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape);
bool logisticFloat32(const float* inputData, const Shape& inputShape,
                     float* outputData, const Shape& outputShape);
bool logisticQuant8(const uint8_t* inputData, const Shape& inputShape,
                    uint8_t* outputData, const Shape& outputShape);
bool softmaxFloat32(const float* inputData, const Shape& inputShape,
                    const float beta,
                    float* outputData, const Shape& outputShape);

bool fullyConnectedPrepare(const Shape& input,
                           const Shape& weights,
                           const Shape& bias,
                           Shape* output);
bool fullyConnectedFloat32(const float* inputData, const Shape& inputShape,
                           const float* weights, const Shape& weightsShape,
                           const float* biasData, const Shape& biasShape,
                           int32_t activation,
                           float* outputData, const Shape& outputShape);
bool fullyConnectedQuant8(const uint8_t* inputData, const Shape& inputShape,
                          const uint8_t* weights, const Shape& weightsShape,
                          const int32_t* biasData, const Shape& biasShape,
                          int32_t activation,
                          uint8_t* outputData, const Shape& outputShape);

bool concatenationPrepare(const std::vector<Shape>& inputShapes,
                          int32_t axis,
                          Shape* output);
bool concatenationFloat32(const std::vector<const float*>& inputDataPtrs,
                          const std::vector<Shape>& inputShapes,
                          int32_t axis, int32_t activation,
                          float* outputData, const Shape& outputShape);
bool concatenationQuant8(const std::vector<const uint8_t*>& inputDataPtrs,
                         const std::vector<Shape>& inputShapes,
                         int32_t axis, int32_t activation,
                         uint8_t* outputData, const Shape& outputShape);

bool genericNormalizationPrepare(const Shape& input, Shape* output);
bool l2normFloat32(const float* inputData, const Shape& inputShape,
                   float* outputData, const Shape& outputShape);
bool l2normQuant8(const uint8_t* inputData, const Shape& inputShape,
                  uint8_t* outputData, const Shape& outputShape);
bool localResponseNormFloat32(const float* inputData, const Shape& inputShape,
                              int32_t radius, float bias, float alpha, float beta,
                              float* outputData, const Shape& outputShape);
} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_OPERATIONS_H
