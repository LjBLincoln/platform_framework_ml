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

#define LOG_TAG "OperationsUtils"

#include "OperationsUtils.h"
#include "Operations.h"
#include "Utils.h"

#include <cmath>

namespace android {
namespace nn {

bool SameShape(const Shape& in1, const Shape& in2) {
    if (in1.type != in2.type || in1.dimensions.size() != in2.dimensions.size()) {
        return false;
    }
    for (size_t i = 0; i < in1.dimensions.size(); i++) {
        if (in1.dimensions[i] != in2.dimensions[i]) {
            return false;
        }
    }
    return true;
}

bool SetShape(const Shape& in, Shape* out) {
    if (in.type != out->type || in.dimensions.size() != out->dimensions.size()) {
        return false;
    }
    out->dimensions = in.dimensions;
    return true;
}

uint32_t getNumberOfElements(const Shape& shape) {
    uint32_t count = 1;
    for (size_t i = 0; i < shape.dimensions.size(); i++) {
        count *= shape.dimensions[i];
    }
    return count;
}

uint32_t getNumberOfDimensions(const Shape& shape) {
    return shape.dimensions.size();
}

uint32_t getSizeOfDimension(const Shape& shape, uint32_t dimensionIdx) {
    if (dimensionIdx >= shape.dimensions.size()) {
        // TODO, log the error
        return 0;
    }
    return shape.dimensions[dimensionIdx];
}


void QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int32_t* right_shift) {
    CHECK(double_multiplier >= 0.);
    CHECK(double_multiplier < 1.);
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *right_shift = 0;
        return;
    }
    CHECK(double_multiplier > 0.);
    const double q = std::frexp(double_multiplier, right_shift);
    *right_shift *= -1;
    int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
    CHECK(q_fixed <= (1ll << 31));
    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        --*right_shift;
    }
    CHECK_GE(*right_shift, 0);
    CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
    *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
    CHECK(double_multiplier > 1.);
    const double q = std::frexp(double_multiplier, left_shift);
    int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
    CHECK(q_fixed <= (1ll << 31));
    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++*left_shift;
    }
    CHECK_GE(*left_shift, 0);
    CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
    *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void GetQuantizedConvolutionMultipler(const Shape& inputShape,
                                      const Shape& filterShape,
                                      const Shape& biasShape,
                                      const Shape& outputShape,
                                      float* multiplier) {
    const float input_product_scale = inputShape.scale * filterShape.scale;
    const float bias_scale = biasShape.scale;
    const float output_scale = outputShape.scale;

    // The following conditions must be guaranteed by the training pipeline.
    CHECK(std::abs(input_product_scale - bias_scale) <=
              1e-6 * std::min(input_product_scale, bias_scale));
    CHECK(input_product_scale >= 0);
    CHECK(input_product_scale < output_scale);
    *multiplier = input_product_scale / output_scale;
}

void CalculateActivationRangeUint8(int32_t activation,
                                   const Shape& outputShape,
                                   int32_t* act_min,
                                   int32_t* act_max) {
    const int32_t qmin = std::numeric_limits<uint8_t>::min();
    const int32_t qmax = std::numeric_limits<uint8_t>::max();

    const auto scale = outputShape.scale;
    const auto zero_point = outputShape.offset;

    auto quantize = [scale, zero_point](float f) {
        return zero_point + static_cast<int32_t>(std::round(f / scale));
    };

    if (activation == kActivationRelu) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = qmax;
    } else if (activation == kActivationRelu6) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = std::min(qmax, quantize(6.0));
    } else if (activation == kActivationRelu1) {
        *act_min = std::max(qmin, quantize(-1.0));
        *act_max = std::min(qmax, quantize(1.0));
    } else {
        *act_min = qmin;
        *act_max = qmax;
    }
}

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift) {
    const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                      (1ll << (31 - input_integer_bits)) /
                                      (1ll << input_left_shift);
    // Tighten bound using floor.  Suppose that we could use the exact value.
    // After scaling the difference, the result would be at the maximum.  Thus we
    // must ensure that our value has lower magnitude.
    return static_cast<int32_t>(std::floor(max_input_rescaled));
}


// Macro to check if the input parameters for operation are valid or not.
#define nnOpsCheck(v)                                                                      \
    if (!(v)) {                                                                            \
        LOG(ERROR) << "nnOpsCheck failed: "  << #v << "'\n";                               \
        return false;                                                                      \
    }

bool addMulPrepare(const Shape& in1, const Shape& in2, Shape* out) {
    nnOpsCheck(getNumberOfDimensions(in1) <= 4 && getNumberOfDimensions(in2) <= 4);
    if (SameShape(in1, in2)) {
        return SetShape(in1, out);
    } else {
        // BroadcastAdd needed
        uint32_t numberOfDims1 = getNumberOfDimensions(in1);
        uint32_t numberOfDims2 = getNumberOfDimensions(in2);
        uint32_t maxDims = std::max(numberOfDims1, numberOfDims2);
        out->dimensions = std::vector<uint32_t>(maxDims);
        for (uint32_t i = 1; i <= maxDims; i++) {
            uint32_t dim1 = 1;
            if (i <= numberOfDims1) {
                dim1 = getSizeOfDimension(in1, numberOfDims1 - i);
            }
            uint32_t dim2 = 1;
            if (i <= numberOfDims2) {
                dim2 = getSizeOfDimension(in2, numberOfDims2 - i);
            }
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                LOG(ERROR) << "Dimensions mismatch for BroadcastAdd";
                return false;
            }
            out->dimensions[maxDims - i] = std::max(dim1, dim2);
        }
    }
    return true;
}

bool floorPrepare(const Shape& input, Shape* output) {
    return SetShape(input, output);
}

bool dequantizePrepare(const Shape& input, Shape* output) {
    if (input.type != OperandType::TENSOR_QUANT8_ASYMM ||
            output->type != OperandType::TENSOR_FLOAT32) {
        LOG(ERROR) << "bad input / output operand type.";
        return false;
    }
    if (input.dimensions.size() != output->dimensions.size()) {
        LOG(ERROR) << "input and output tensors don't have the same rank.";
        return false;
    }
    output->dimensions = input.dimensions;
    return true;
}

bool convPrepare(const Shape& input,
                 const Shape& filter,
                 const Shape& bias,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) == 4);
    nnOpsCheck(getNumberOfDimensions(filter) == 4);
    nnOpsCheck(getNumberOfDimensions(bias) == 1);

    nnOpsCheck(getSizeOfDimension(filter, 0) == getSizeOfDimension(bias, 0));
    nnOpsCheck(getSizeOfDimension(filter, 3) == getSizeOfDimension(input, 3));

    nnOpsCheck(stride_width == stride_height);

    uint32_t channels_out = getSizeOfDimension(filter, 0);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t filterWidth  = getSizeOfDimension(filter, 2);
    uint32_t filterHeight = getSizeOfDimension(filter, 1);
    uint32_t batches      = getSizeOfDimension(input, 0);

    uint32_t outWidth = computeOutSize(width, filterWidth, stride_width,
                                       padding_left, padding_right);
    uint32_t outHeight = computeOutSize(height, filterHeight, stride_height,
                                        padding_top, padding_bottom);

    output->type = input.type;
    output->dimensions = {batches, outHeight, outWidth, channels_out};
    return true;
}

bool depthwiseConvPrepare(const Shape& input,
                          const Shape& filter,
                          const Shape& bias,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) == 4);
    nnOpsCheck(getNumberOfDimensions(filter) == 4);
    nnOpsCheck(getNumberOfDimensions(bias) == 1);

    nnOpsCheck(getSizeOfDimension(filter, 3) == getSizeOfDimension(bias, 0));

    nnOpsCheck(stride_width == stride_height);

    uint32_t channels_out = getSizeOfDimension(filter, 3);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t filterWidth  = getSizeOfDimension(filter, 2);
    uint32_t filterHeight = getSizeOfDimension(filter, 1);
    uint32_t batches      = getSizeOfDimension(input, 0);

    uint32_t outWidth = computeOutSize(width, filterWidth, stride_width,
                                       padding_left, padding_right);
    uint32_t outHeight = computeOutSize(height, filterHeight, stride_height,
                                        padding_top, padding_bottom);

    output->type = input.type;
    output->dimensions = {batches, outHeight, outWidth, channels_out};
    return true;
}


bool genericPoolingPrepare(const Shape& input,
                           int32_t padding_left, int32_t padding_right,
                           int32_t padding_top, int32_t padding_bottom,
                           int32_t stride_width, int32_t stride_height,
                           int32_t filter_width, int32_t filter_height,
                           Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) == 4);
    nnOpsCheck(stride_width == stride_height);

    uint32_t batches      = getSizeOfDimension(input, 0);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t channels_out = getSizeOfDimension(input, 3);

    uint32_t outWidth = computeOutSize(width, filter_width, stride_width,
                                       padding_left, padding_right);
    uint32_t outHeight = computeOutSize(height, filter_height, stride_height,
                                        padding_top, padding_bottom);

    output->type = input.type;
    output->dimensions = {batches, outHeight, outWidth, channels_out};
    return true;
}


bool genericActivationPrepare(const Shape& input,
                              Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) <= 4);
    return SetShape(input, output);
}

bool fullyConnectedPrepare(const Shape& input,
                           const Shape& weights,
                           const Shape& bias,
                           Shape* output) {
    // Check all the parameters of tensor match within themselves and match the
    // input configuration.
    uint32_t input_size = getNumberOfElements(input);
    uint32_t num_units  = getSizeOfDimension(weights, 0);
    uint32_t batch_size = input_size / getSizeOfDimension(weights, 1);

    nnOpsCheck(getSizeOfDimension(bias, 0) == num_units);
    nnOpsCheck(getSizeOfDimension(weights, 1) * batch_size == input_size);
    nnOpsCheck(getNumberOfDimensions(weights) == 2);

    output->type = input.type;
    output->dimensions = {batch_size, num_units};

    return true;
}

bool concatenationPrepare(const std::vector<Shape>& inputShapes,
                          int32_t axis,
                          Shape* output) {

    int num_inputs = inputShapes.size();
    OperandType input_type = inputShapes[0].type;
    uint32_t num_dimensions = getNumberOfDimensions(inputShapes[0]);

    nnOpsCheck(axis >= 0);
    nnOpsCheck(axis < (int32_t)num_dimensions);

    int sum_axis = getSizeOfDimension(inputShapes[0], axis);
    for (int i = 1; i < num_inputs; ++i) {
        nnOpsCheck(getNumberOfDimensions(inputShapes[i]) == num_dimensions);
        nnOpsCheck(inputShapes[i].type == inputShapes[0].type);
        if (input_type == OperandType::TENSOR_QUANT8_ASYMM) {
            nnOpsCheck(inputShapes[0].offset == inputShapes[i].offset);
            nnOpsCheck(inputShapes[0].scale == inputShapes[i].scale);
        }
        for (int d = 0; d < (int32_t)num_dimensions; ++d) {
            if (d == axis) {
                sum_axis += getSizeOfDimension(inputShapes[i], axis);
            } else {
                nnOpsCheck(getSizeOfDimension(inputShapes[0], d) ==
                           getSizeOfDimension(inputShapes[i], d));
            }
        }
    }

    output->type = input_type;
    output->dimensions = inputShapes[0].dimensions;
    output->dimensions[axis] = sum_axis;

    if (input_type == OperandType::TENSOR_QUANT8_ASYMM) {
        nnOpsCheck(inputShapes[0].offset == output->offset);
        nnOpsCheck(inputShapes[0].scale == output->scale);
    }

    return true;
}


bool genericNormalizationPrepare(const Shape& input, Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) == 4);
    return SetShape(input, output);
}

bool reshapePrepare(const Shape& input,
                    const int32_t* targetDims,
                    const int32_t targetDimsSize,
                    Shape* output) {
    // Reshape allows one of the targetDims components to have the
    // special -1 value, meaning it will be calculated automatically based on the
    // input. Here we calculate what that dimension should be so that the number
    // of output elements in the same as the number of input elements.
    int32_t numInputElements = (int32_t) getNumberOfElements(input);

    std::vector<uint32_t> outDims(targetDimsSize);
    int32_t numOutputElements = 1;
    int32_t strechDim = -1;
    for (int32_t i = 0; i < targetDimsSize; ++i) {
        int32_t value = targetDims[i];
        if (value == -1) {
            nnOpsCheck(strechDim == -1);
            strechDim = i;
        } else {
            numOutputElements *= value;
            outDims[i] = (uint32_t)value;
        }
    }
    if (strechDim != -1) {
        int32_t strechValue = numInputElements / numOutputElements;
        outDims[strechDim] = (uint32_t) strechValue;
        numOutputElements *= strechValue;
    }

    nnOpsCheck(numInputElements == numOutputElements);

    output->type = input.type;
    output->dimensions = outDims;
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool resizeBilinearPrepare(const Shape& input,
                           int32_t width,
                           int32_t height,
                           Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) == 4);
    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t channels = getSizeOfDimension(input, 3);

    output->type = input.type;
    output->dimensions = {batches, (uint32_t)height, (uint32_t)width, channels};

    return true;
}

bool depthToSpacePrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) == 4);
    nnOpsCheck(blockSize > 0);

    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t height   = getSizeOfDimension(input, 1);
    uint32_t width    = getSizeOfDimension(input, 2);
    uint32_t channels = getSizeOfDimension(input, 3);

    nnOpsCheck(channels % (blockSize * blockSize) == 0);
    output->type = input.type;
    output->dimensions = {batches,
                          height * blockSize,
                          width * blockSize,
                          channels / (blockSize * blockSize)};
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool spaceToDepthPrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output) {
    nnOpsCheck(getNumberOfDimensions(input) == 4);
    nnOpsCheck(blockSize > 0);

    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t height   = getSizeOfDimension(input, 1);
    uint32_t width    = getSizeOfDimension(input, 2);
    uint32_t channels = getSizeOfDimension(input, 3);

    nnOpsCheck(height % blockSize == 0);
    nnOpsCheck(width % blockSize == 0);

    output->type = input.type;
    output->dimensions = {batches,
                          height / blockSize,
                          width / blockSize,
                          channels * (blockSize * blockSize)};
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

} // namespace nn
} // namespace android
