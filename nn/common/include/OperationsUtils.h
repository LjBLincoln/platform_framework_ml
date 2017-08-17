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

#ifndef ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H
#define ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H

#include "Utils.h"

#include <cstdint>
#include <vector>

namespace android {
namespace nn {

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

// Verifies that the two shapes are the same.
bool SameShape(const Shape& in1, const Shape& in2);

// Sets out to the same shape as in.
bool SetShape(const Shape& in, Shape* out);

// Return the total number of elements, i.e. all the dimensions multiplied
// together. For a scalar, returns one.
uint32_t getNumberOfElements(const Shape& shape);

uint32_t getNumberOfDimensions(const Shape& shape);

uint32_t getSizeOfDimension(const Shape& shape, uint32_t dimensionIdx);

inline uint32_t ComputePadding(uint32_t stride, uint32_t in_size, uint32_t filter_size,
                               uint32_t out_size) {
    uint32_t tmp = (out_size - 1) * stride + filter_size;
    if (tmp > in_size) {
        return (tmp - in_size) / 2;
    } else {
        return 0;
    }
}

void QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int32_t* right_shift);

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

void GetQuantizedConvolutionMultipler(const Shape& inputShape,
                                      const Shape& filterShape,
                                      const Shape& biasShape,
                                      const Shape& outputShape,
                                      float* multiplier);

void CalculateActivationRangeUint8(int32_t activation,
                                   const Shape& outputShape,
                                   int32_t* act_min,
                                   int32_t* act_max);

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift);

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H
