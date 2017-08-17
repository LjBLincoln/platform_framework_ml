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

namespace android {
namespace nn {

bool genericActivationPrepare(const Shape& input,
                              Shape* output) {
    DCHECK_EQ(getNumberOfDimensions(input), 4);
    return SetShape(input, output);
}

bool reluFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::max(0.f, *inputData);
    }
    return true;
}

bool relu6Float32(const float* inputData, const Shape& inputShape,
                  float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::min(std::max(0.f, *inputData), 6.f);
    }
    return true;
}

bool tanhFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::tanh(*inputData);
    }
    return true;
}

bool logisticFloat32(const float* inputData, const Shape& inputShape,
                     float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = 1.f / (1.f + std::exp(-*inputData));
    }
    return true;
}

bool logisticQuant8(const uint8_t* inputData, const Shape& inputShape,
                    uint8_t* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    static constexpr int kInputIntegerBits = 4;

    const double input_real_multiplier =
            inputShape.scale *
            static_cast<double>(1 << (31 - kInputIntegerBits));

    int32_t input_multiplier = 0;
    int32_t input_left_shift = 0;
    QuantizeMultiplierGreaterThanOne(input_real_multiplier,
                                     &input_multiplier,
                                     &input_left_shift);
    int32_t input_range_radius =
            CalculateInputRadius(kInputIntegerBits, input_left_shift);

    optimized_ops::Logistic(
            inputData, convertShapeToDims(inputShape),
            inputShape.offset, input_range_radius,
            input_multiplier, input_left_shift,
            outputData, convertShapeToDims(outputShape));

    return true;
}

bool softmaxFloat32(const float* inputData, const Shape& inputShape,
                    const float beta,
                    float* outputData, const Shape& outputShape) {
    int batch_size = (int)getSizeOfDimension(inputShape, 0);
    int input_size = getNumberOfElements(inputShape) / batch_size;

    // For each batch
    for (int b=0; b<batch_size; b++) {
        // Find the max coeff.
        float max_coeff = inputData[0];
        for (int i=1; i<input_size; i++) {
            max_coeff = std::max(inputData[i], max_coeff);
        }
        // Compute the normalized sum of exps.
        float exp_sum = 0.0f;
        for (int i=0; i<input_size; i++) {
            outputData[i] = std::exp((inputData[i] - max_coeff) * beta);
            exp_sum += outputData[i];
        }
        // Divide by the sum of exps.
        float reciprocal_sum_exp = 1.f / exp_sum;
        for (int i=0; i<input_size; i++) {
          outputData[i] *= reciprocal_sum_exp;
        }
        // Advance in and out pointers for the next batch.
        inputData += input_size;
        outputData += input_size;
    }
    return true;
}

}  // namespace nn
}  // namespace android
