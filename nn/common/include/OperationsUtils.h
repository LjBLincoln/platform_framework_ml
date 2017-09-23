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

enum PaddingScheme {
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
};

inline PaddingScheme getPaddingScheme(uint32_t filterWidth, uint32_t filterHeight,
                                      uint32_t paddingLeft, uint32_t paddingRight,
                                      uint32_t paddingTop, uint32_t paddingBottom) {
    if (paddingLeft > paddingRight || paddingTop > paddingBottom) {
        return kPaddingUnknown;
    }

    uint32_t totolPaddingWidth = paddingLeft + paddingRight;
    uint32_t totolPaddingHeight = paddingTop + paddingBottom;
    if (totolPaddingWidth == filterWidth - 1 &&
        totolPaddingHeight == filterHeight -1) {
        return kPaddingSame;
    } else if (totolPaddingWidth == 0 &&
               totolPaddingHeight == 0) {
        return kPaddingValid;
    } else {
        return kPaddingUnknown;
    }
}

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

inline uint32_t computeOutSize(uint32_t imageSize, uint32_t filterSize, uint32_t stride,
                               uint32_t paddingHead, uint32_t paddingTail) {
    return (imageSize - filterSize + stride + paddingHead + paddingTail) / stride;
}

__wur
bool QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int32_t* right_shift);

__wur
bool QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

__wur
bool GetQuantizedConvolutionMultipler(const Shape& inputShape,
                                      const Shape& filterShape,
                                      const Shape& biasShape,
                                      const Shape& outputShape,
                                      float* multiplier);

void CalculateActivationRangeUint8(int32_t activation,
                                   const Shape& outputShape,
                                   int32_t* act_min,
                                   int32_t* act_max);

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift);

inline void calculateExplicitPadding(int32_t in_size, int32_t stride,
                                     int32_t filter_size, int32_t padding_implicit,
                                     int32_t* padding_head, int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == kPaddingSame) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

// Preparation functions for the corresponding ops
bool addMulPrepare(const Shape& in1, const Shape& in2, Shape* out1);

bool floorPrepare(const Shape& input, Shape* output);

bool dequantizePrepare(const Shape& input, Shape* output);

bool depthwiseConvPrepare(const Shape& input,
                          const Shape& filter,
                          const Shape& bias,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          Shape* output);

bool convPrepare(const Shape& input,
                 const Shape& filter,
                 const Shape& bias,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 Shape* output);

bool genericPoolingPrepare(const Shape& input,
                           int32_t padding_left, int32_t padding_right,
                           int32_t padding_top, int32_t padding_bottom,
                           int32_t stride_width, int32_t stride_height,
                           int32_t filter_width, int32_t filter_height,
                           Shape* output);

bool genericActivationPrepare(const Shape& input, Shape* output);

bool fullyConnectedPrepare(const Shape& input,
                           const Shape& weights,
                           const Shape& bias,
                           Shape* output);

bool concatenationPrepare(const std::vector<Shape>& inputShapes,
                          int32_t axis,
                          Shape* output);

bool genericNormalizationPrepare(const Shape& input, Shape* output);

bool reshapePrepare(const Shape& input,
                    const int32_t* targetDims,
                    const int32_t targetDimsSize,
                    Shape* output);

bool resizeBilinearPrepare(const Shape& input,
                           int32_t height,
                           int32_t width,
                           Shape* output);

bool depthToSpacePrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output);

bool spaceToDepthPrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output);

#define ANDROID_NN_MACRO_DISPATCH(macro)                                    \
    switch (activation) {                                                   \
        case (int32_t) FusedActivationFunc::NONE:                           \
            macro(kNone);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU:                           \
            macro(kRelu);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU1:                          \
            macro(kRelu1);                                                  \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU6:                          \
            macro(kRelu6);                                                  \
            break;                                                          \
        default:                                                            \
            LOG(ERROR) << "Unsupported fused activation function type";     \
            return false;                                                   \
    }

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H
