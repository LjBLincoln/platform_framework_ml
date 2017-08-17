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

} // namespace nn
} // namespace android
