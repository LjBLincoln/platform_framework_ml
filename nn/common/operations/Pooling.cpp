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

#include "Operations.h"
#include "OperationsUtils.h"

#include "internal/optimized/optimized_ops.h"
#include "internal/reference/reference_ops.h"

namespace android {
namespace nn {

bool genericPoolingFloat32Prepare(const Shape& input,
                                  int32_t padding,
                                  int32_t stride_width, int32_t stride_height,
                                  int32_t filter_width, int32_t filter_height,
                                  Shape* output) {
    DCHECK_EQ(getNumberOfDimensions(input), 4);
    DCHECK_EQ(stride_width, stride_height);

    uint32_t batches      = getSizeOfDimension(input, 0);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t channels_out = getSizeOfDimension(input, 3);

    // Matching GetWindowedOutputSize in TensorFlow.
    auto computeOutSize = [padding](uint32_t imageSize, uint32_t filterSize,
                                    uint32_t stride) -> int {
        return padding == kPaddingSame
                   ? (imageSize + stride - 1) / stride
                   : padding == kPaddingValid
                         ? (imageSize - filterSize + stride) / stride
                         : 0;
    };

    uint32_t outWidth = computeOutSize(width, filter_width, stride_width);
    uint32_t outHeight = computeOutSize(height, filter_height, stride_height);

    output->type = input.type;
    output->dimensions = {batches, outHeight, outWidth, channels_out};
    return true;
}

bool averagePoolFloat32(const float* inputData, const Shape& inputShape,
                        int32_t padding, int32_t stride_width, int32_t stride_height,
                        int32_t filter_width, int32_t filter_height, int32_t activation,
                        float* outputData, const Shape& outputShape) {
    uint32_t height       = getSizeOfDimension(inputShape, 1);
    uint32_t width        = getSizeOfDimension(inputShape, 2);
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);

    uint32_t paddingHeight =
            ComputePadding(stride_height, height, filter_height, outHeight);
    uint32_t paddingWidth =
            ComputePadding(stride_width, width, filter_width, outWidth);

    #define ANDROID_NN_AVERAGE_POOL(activation)                                \
        optimized_ops::AveragePool<FusedActivationFunctionType::activation>(   \
            inputData, convertShapeToDims(inputShape),                         \
            stride_width, paddingWidth, paddingHeight,                         \
            filter_width, filter_height,                                       \
            outputData, convertShapeToDims(outputShape))

    if (activation == kActivationNone) {
        ANDROID_NN_AVERAGE_POOL(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_AVERAGE_POOL(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_AVERAGE_POOL(kRelu6);
    }

    #undef ANDROID_NN_AVERAGE_POOL

    return true;
}

bool l2PoolFloat32(const float* inputData, const Shape& inputShape,
                   int32_t padding, int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
                   float* outputData, const Shape& outputShape) {
    uint32_t height       = getSizeOfDimension(inputShape, 1);
    uint32_t width        = getSizeOfDimension(inputShape, 2);
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);

    uint32_t paddingHeight =
            ComputePadding(stride_height, height, filter_height, outHeight);
    uint32_t paddingWidth =
            ComputePadding(stride_width, width, filter_width, outWidth);

    #define ANDROID_NN_L2_POOL(activation)                                     \
        optimized_ops::L2Pool<FusedActivationFunctionType::activation>(        \
            inputData, convertShapeToDims(inputShape),                         \
            stride_width, paddingWidth, paddingHeight,                         \
            filter_width, filter_height,                                       \
            outputData, convertShapeToDims(outputShape))

    if (activation == kActivationNone) {
        ANDROID_NN_L2_POOL(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_L2_POOL(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_L2_POOL(kRelu6);
    }

    #undef ANDROID_NN_L2_POOL

    return true;
}

bool maxPoolFloat32(const float* inputData, const Shape& inputShape,
                    int32_t padding, int32_t stride_width, int32_t stride_height,
                    int32_t filter_width, int32_t filter_height, int32_t activation,
                    float* outputData, const Shape& outputShape) {
    uint32_t height       = getSizeOfDimension(inputShape, 1);
    uint32_t width        = getSizeOfDimension(inputShape, 2);
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);

    uint32_t paddingHeight =
            ComputePadding(stride_height, height, filter_height, outHeight);
    uint32_t paddingWidth =
            ComputePadding(stride_width, width, filter_width, outWidth);

    #define ANDROID_NN_MAX_POOL(activation)                                    \
        optimized_ops::MaxPool<FusedActivationFunctionType::activation>(       \
            inputData, convertShapeToDims(inputShape),                         \
            stride_width, paddingWidth, paddingHeight,                         \
            filter_width, filter_height,                                       \
            outputData, convertShapeToDims(outputShape))

    if (activation == kActivationNone) {
        ANDROID_NN_MAX_POOL(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_MAX_POOL(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_MAX_POOL(kRelu6);
    }

    #undef ANDROID_NN_MAX_POOL

    return true;
}



}  // namespace nn
}  // namespace android
