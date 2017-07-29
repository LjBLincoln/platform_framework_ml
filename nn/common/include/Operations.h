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
    kActivationRelu6 = 3,
};

bool addTensorsFloat32Prepare(const Shape& in1, const Shape& in2, Shape* out1);
bool addTensorsFloat32(const float* in1, const float* in2, float* out, const Shape& shape);

bool depthwiseConvFloat32Prepare(const Shape& input,
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

bool convFloat32Prepare(const Shape& input,
                        const Shape& filter,
                        const Shape& bias,
                        int32_t padding,
                        int32_t stride_width, int32_t stride_height,
                        Shape* output);
bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding, int32_t stride_width, int32_t stride_height, int32_t activation,
                 float* outputData, const Shape& outputShape);

bool genericPoolingFloat32Prepare(const Shape& input,
                                  int32_t padding,
                                  int32_t stride_width, int32_t stride_height,
                                  int32_t filter_width, int32_t filter_height,
                                  Shape* output);
bool averagePoolFloat32(const float* inputData, const Shape& inputShape,
                        int32_t padding, int32_t stride_width, int32_t stride_height,
                        int32_t filter_width, int32_t filter_height, int32_t activation,
                        float* outputData, const Shape& outputShape);
bool l2PoolFloat32(const float* inputData, const Shape& inputShape,
                   int32_t padding, int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
                   float* outputData, const Shape& outputShape);
bool maxPoolFloat32(const float* inputData, const Shape& inputShape,
                    int32_t padding, int32_t stride_width, int32_t stride_height,
                    int32_t filter_width, int32_t filter_height, int32_t activation,
                    float* outputData, const Shape& outputShape);

bool genericActivationFloat32Prepare(const Shape& input, Shape* output);
bool reluFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape);
bool relu6Float32(const float* inputData, const Shape& inputShape,
                  float* outputData, const Shape& outputShape);
bool tanhFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape);
bool logisticFloat32(const float* inputData, const Shape& inputShape,
                     float* outputData, const Shape& outputShape);

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_OPERATIONS_H
