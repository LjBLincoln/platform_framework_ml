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

#include "internal/optimized/depthwiseconv_float.h"
#include "internal/optimized/depthwiseconv_uint8.h"

namespace android {
namespace nn {

bool depthwiseConvPrepare(const Shape& input,
                          const Shape& filter,
                          const Shape& bias,
                          int32_t padding,
                          int32_t stride_width, int32_t stride_height,
                          Shape* output) {
    DCHECK_EQ(getNumberOfDimensions(input), 4);
    DCHECK_EQ(getNumberOfDimensions(filter), 4);
    DCHECK_EQ(getNumberOfDimensions(bias), 1);

    DCHECK_EQ(getSizeOfDimension(filter, 3), getSizeOfDimension(bias, 0));
    DCHECK_EQ(stride_width, stride_height);

    uint32_t channels_out = getSizeOfDimension(filter, 3);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t filterWidth  = getSizeOfDimension(filter, 2);
    uint32_t filterHeight = getSizeOfDimension(filter, 1);
    uint32_t batches      = getSizeOfDimension(input, 0);

    // Matching GetWindowedOutputSize in TensorFlow.
    auto computeOutSize = [padding](uint32_t imageSize, uint32_t filterSize,
                                    uint32_t stride) -> int {
        return padding == kPaddingSame
                   ? (imageSize + stride - 1) / stride
                   : padding == kPaddingValid
                         ? (imageSize - filterSize + stride) / stride
                         : 0;
    };

    uint32_t outWidth = computeOutSize(width, filterWidth, stride_width);
    uint32_t outHeight = computeOutSize(height, filterHeight, stride_height);

    output->type = input.type;
    output->dimensions = {batches, outHeight, outWidth, channels_out};
    return true;
}


#define ANDROID_NN_DEPTHWISE_CONV_PARAMETERS                                    \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);                 \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);                 \
                                                                                \
    uint32_t paddingHeight =                                                    \
            ComputePadding(stride_height, height, filterHeight, outHeight);     \
    uint32_t paddingWidth =                                                     \
            ComputePadding(stride_width, width, filterWidth, outWidth);

bool depthwiseConvFloat32(const float* inputData, const Shape& inputShape,
                          const float* filterData, const Shape& filterShape,
                          const float* biasData, const Shape& biasShape,
                          int32_t padding, int32_t stride_width, int32_t stride_height,
                          int32_t depth_multiplier, int32_t activation,
                          float* outputData, const Shape& outputShape) {

    ANDROID_NN_DEPTHWISE_CONV_PARAMETERS

    #define ANDROID_NN_DEPTHWISE_CONV(activation)                              \
        optimized_ops::DepthwiseConv<FusedActivationFunctionType::activation>( \
            inputData, convertShapeToDims(inputShape),                         \
            filterData, convertShapeToDims(filterShape),                       \
            biasData, convertShapeToDims(biasShape),                           \
            stride_width, paddingWidth, paddingHeight, depth_multiplier,       \
            outputData, convertShapeToDims(outputShape))

    if (activation == kActivationNone) {
        ANDROID_NN_DEPTHWISE_CONV(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_DEPTHWISE_CONV(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_DEPTHWISE_CONV(kRelu6);
    }

    #undef ANDROID_NN_DEPTHWISE_CONV

    return true;
}


bool depthwiseConvQuant8(const uint8_t* inputData, const Shape& inputShape,
                         const uint8_t* filterData, const Shape& filterShape,
                         const int32_t* biasData, const Shape& biasShape,
                         int32_t padding, int32_t stride_width, int32_t stride_height,
                         int32_t depth_multiplier, int32_t activation,
                         uint8_t* outputData, const Shape& outputShape) {

    ANDROID_NN_DEPTHWISE_CONV_PARAMETERS

    float real_multiplier = 0.0;
    int32_t output_multiplier = 0;
    int32_t output_shift = 0;
    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;

    GetQuantizedConvolutionMultipler(inputShape, filterShape, biasShape,
                                     outputShape, &real_multiplier);
    QuantizeMultiplierSmallerThanOne(real_multiplier, &output_multiplier,
                                     &output_shift);
    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

    uint32_t inputOffset = -inputShape.offset;
    uint32_t filterOffset = -filterShape.offset;
    uint32_t outputOffset = outputShape.offset;
    #define ANDROID_NN_DEPTHWISE_CONV(activation)                              \
        optimized_ops::DepthwiseConv<FusedActivationFunctionType::activation>( \
            inputData, convertShapeToDims(inputShape), inputOffset,            \
            filterData, convertShapeToDims(filterShape), filterOffset,         \
            biasData, convertShapeToDims(biasShape),                           \
            stride_width, paddingWidth, paddingHeight, depth_multiplier,       \
            outputOffset, output_multiplier, output_shift,                     \
            output_activation_min, output_activation_max,                      \
            outputData, convertShapeToDims(outputShape))

    if (activation == kActivationNone) {
        ANDROID_NN_DEPTHWISE_CONV(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_DEPTHWISE_CONV(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_DEPTHWISE_CONV(kRelu6);
    }

    #undef ANDROID_NN_DEPTHWISE_CONV

    return true;
}

#undef ANDROID_NN_DEPTHWISE_CONV_PARAMETERS
}  // namespace nn
}  // namespace android
