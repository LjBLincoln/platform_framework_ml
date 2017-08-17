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

bool concatenationPrepare(const std::vector<Shape>& inputShapes,
                          int32_t axis,
                          Shape* output) {

    int num_inputs = inputShapes.size();
    OperandType input_type = inputShapes[0].type;
    uint32_t num_dimensions = getNumberOfDimensions(inputShapes[0]);

    DCHECK_GE(axis, 0);
    DCHECK_LT(axis, (int32_t)num_dimensions);

    int sum_axis = getSizeOfDimension(inputShapes[0], axis);
    for (int i = 1; i < num_inputs; ++i) {
        DCHECK_EQ(getNumberOfDimensions(inputShapes[i]), num_dimensions);
        DCHECK_EQ((uint32_t)inputShapes[i].type, (uint32_t)inputShapes[0].type);
        if (input_type == OperandType::TENSOR_QUANT8_ASYMM) {
            DCHECK_EQ(inputShapes[0].offset, inputShapes[i].offset);
            DCHECK_EQ(inputShapes[0].scale, inputShapes[i].scale);
        }
        for (int d = 0; d < (int32_t)num_dimensions; ++d) {
            if (d == axis) {
                sum_axis += getSizeOfDimension(inputShapes[i], axis);
            } else {
                DCHECK_EQ(getSizeOfDimension(inputShapes[0], d),
                          getSizeOfDimension(inputShapes[i], d));
            }
        }
    }

    output->type = input_type;
    output->dimensions = inputShapes[0].dimensions;
    output->dimensions[axis] = sum_axis;

    if (input_type == OperandType::TENSOR_QUANT8_ASYMM) {
        DCHECK_EQ(inputShapes[0].offset, output->offset);
        DCHECK_EQ(inputShapes[0].scale, output->scale);
    }

    return true;
}

bool concatenationFloat32(const std::vector<const float*>& inputDataPtrs,
                          const std::vector<Shape>& inputShapes,
                          int32_t axis, int32_t activation,
                          float* outputData, const Shape& outputShape) {
    int num_inputs = inputShapes.size();
    std::vector<Dims<4>*> inputDimsPtr(num_inputs);
    std::vector<Dims<4> > inputDims(num_inputs);
    for (int i=0; i<num_inputs; i++) {
        inputDims[i] = convertShapeToDims(inputShapes[i]);
        inputDimsPtr[i] = &inputDims[i];
    }

    #define ANDROID_NN_CONCATENATION(activation)                                      \
        optimized_ops::Concatenation<FusedActivationFunctionType::activation, float>( \
            getNumberOfDimensions(outputShape) - axis - 1,                            \
            inputDataPtrs.data(), inputDimsPtr.data(), num_inputs,                    \
            outputData, convertShapeToDims(outputShape))

    if (activation == kActivationNone) {
        ANDROID_NN_CONCATENATION(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_CONCATENATION(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_CONCATENATION(kRelu6);
    }

    #undef ANDROID_NN_CONCATENATION

    return true;
}

bool concatenationQuant8(const std::vector<const uint8_t*>& inputDataPtrs,
                         const std::vector<Shape>& inputShapes,
                         int32_t axis, int32_t activation,
                         uint8_t* outputData, const Shape& outputShape) {
    int num_inputs = inputShapes.size();
    std::vector<Dims<4>*> inputDimsPtr(num_inputs);
    std::vector<Dims<4> > inputDims(num_inputs);
    for (int i=0; i<num_inputs; i++) {
        inputDims[i] = convertShapeToDims(inputShapes[i]);
        inputDimsPtr[i] = &inputDims[i];
    }

    #define ANDROID_NN_CONCATENATION(activation)                                        \
        optimized_ops::Concatenation<FusedActivationFunctionType::activation, uint8_t>( \
            getNumberOfDimensions(outputShape) - axis - 1,                              \
            inputDataPtrs.data(), inputDimsPtr.data(), num_inputs,                      \
            outputData, convertShapeToDims(outputShape))

    if (activation == kActivationNone) {
        ANDROID_NN_CONCATENATION(kNone);
    }
    if (activation == kActivationRelu) {
        ANDROID_NN_CONCATENATION(kRelu);
    }
    if (activation == kActivationRelu6) {
        ANDROID_NN_CONCATENATION(kRelu6);
    }

    #undef ANDROID_NN_CONCATENATION
    return true;
}

}  // namespace nn
}  // namespace android
