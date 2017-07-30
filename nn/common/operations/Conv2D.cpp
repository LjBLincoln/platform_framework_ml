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

// If possible we will use this static buffer for the tensor.
static constexpr int kStaticBufferSize = 1605632;
static char static_scratch_buffer[kStaticBufferSize];

bool convFloat32Prepare(const Shape& input,
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

    uint32_t channels_out = getSizeOfDimension(filter, 0);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t filterWidth  = getSizeOfDimension(filter, 2);
    uint32_t filterHeight = getSizeOfDimension(filter, 1);
    uint32_t batches      = getSizeOfDimension(input, 0);

    // Matching GetWindowedOutputSize in TensorFlow.
    // TODO: changing this to explicit padding.
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

bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding, int32_t stride_width, int32_t stride_height, int32_t activation,
                 float* outputData, const Shape& outputShape) {
    uint32_t height       = getSizeOfDimension(inputShape, 1);
    uint32_t width        = getSizeOfDimension(inputShape, 2);
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);
    uint32_t inDepth      = getSizeOfDimension(inputShape, 3);

    uint32_t paddingHeight =
            ComputePadding(stride_height, height, filterHeight, outHeight);
    uint32_t paddingWidth =
            ComputePadding(stride_width, width, filterWidth, outWidth);

    Dims<4> im2colDim;
    im2colDim.sizes[3] = (int)getSizeOfDimension(outputShape, 0);
    im2colDim.sizes[2] = (int)getSizeOfDimension(outputShape, 1);
    im2colDim.sizes[1] = (int)getSizeOfDimension(outputShape, 2);
    im2colDim.sizes[0] = (int)inDepth * filterHeight * filterWidth;

    im2colDim.strides[0] = 1;
    for (int i=1; i<4; i++) {
        im2colDim.strides[i] = im2colDim.strides[i-1] * im2colDim.sizes[i-1];
    }

    float* im2colData = nullptr;
    int im2colByteSize = sizeof(float);
    for (int i=0; i<4; i++) {
        im2colByteSize *= im2colDim.sizes[i];
    }
    if (im2colByteSize <= kStaticBufferSize) {
        im2colData = reinterpret_cast<float *>(static_scratch_buffer);
    } else {
        im2colData = new (std::nothrow) float[im2colByteSize / sizeof(float)];
    }

    #define ANDROID_NN_CONV(activation)                                        \
        optimized_ops::Conv<FusedActivationFunctionType::activation>(          \
            inputData, convertShapeToDims(inputShape),                         \
            filterData, convertShapeToDims(filterShape),                       \
            biasData, convertShapeToDims(biasShape),                           \
            stride_width, paddingWidth, paddingHeight,                         \
            outputData, convertShapeToDims(outputShape),                       \
            im2colData, im2colDim)

    if (activation == kActivationNone) {
        ANDROID_NN_CONV(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_CONV(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_CONV(kRelu6);
    }

    #undef ANDROID_NN_CONV

    if (im2colByteSize > kStaticBufferSize) {
        delete[] im2colData;
    }
    return true;
}

}  // namespace nn
}  // namespace android
