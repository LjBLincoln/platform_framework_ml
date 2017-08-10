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

// Contains the implementation of the operations.

#define LOG_TAG "Operations"

#include "Operations.h"
#include "OperationsUtils.h"

#include "internal/optimized/optimized_ops.h"

namespace android {
namespace nn {

bool addPrepare(const Shape& in1, const Shape& in2, Shape* out) {
    return SameShape(in1, in2) && SetShape(in1, out);
}

bool addFloat32(const float* in1, const float* in2,
                int32_t activation,
                float* out, const Shape& shape) {
    Dims<4> dim = convertShapeToDims(shape);
    #define ANDROID_NN_ADD(activation)                               \
        optimized_ops::Add<FusedActivationFunctionType::activation>( \
                in1, dim, in2, dim, out, dim)

    if (activation == kActivationNone) {
        ANDROID_NN_ADD(kNone);
    } else if (activation == kActivationRelu) {
        ANDROID_NN_ADD(kRelu);
    } else if (activation == kActivationRelu1) {
        ANDROID_NN_ADD(kRelu1);
    } else if (activation == kActivationRelu6) {
        ANDROID_NN_ADD(kRelu6);
    } else {
        return false;
    }

    #undef ANDROID_NN_ADD
    return true;
}

bool mulPrepare(const Shape& in1, const Shape& in2, Shape* out) {
    return SameShape(in1, in2) && SetShape(in1, out);
}

bool mulFloat32(const float* in1, const float* in2,
                int32_t activation,
                float* out, const Shape& shape) {
    Dims<4> dim = convertShapeToDims(shape);
    #define ANDROID_NN_MUL(activation)                               \
        optimized_ops::Mul<FusedActivationFunctionType::activation>( \
                in1, dim, in2, dim, out, dim)

    if (activation == kActivationNone) {
        ANDROID_NN_MUL(kNone);
    } else if (activation == kActivationRelu) {
        ANDROID_NN_MUL(kRelu);
    } else if (activation == kActivationRelu1) {
        ANDROID_NN_MUL(kRelu1);
    } else if (activation == kActivationRelu6) {
        ANDROID_NN_MUL(kRelu6);
    } else {
        return false;
    }

    #undef ANDROID_NN_MUL
    return true;
}

bool floorPrepare(const Shape& input, Shape* output) {
    return SetShape(input, output);
}

bool floorFloat32(const float* inputData,
                  float* outputData,
                  const Shape& shape) {
    Dims<4> dim = convertShapeToDims(shape);
    optimized_ops::Floor(inputData, dim, outputData, dim);
    return true;
}

bool dequantizePrepare(const Shape& input, Shape* output) {
    if (input.type != OperandType::TENSOR_QUANT8_ASYMM ||
            output->type != OperandType::TENSOR_FLOAT32) {
        LOG(ERROR) << "bad input / output operand type.";
        return false;
    }
    return SetShape(input, output);
}

bool dequantizeQuant8ToFloat32(const uint8_t* inputData,
                               float* outputData,
                               const Shape& shape) {
    Dims<4> dim = convertShapeToDims(shape);
    optimized_ops::Dequantize(inputData, dim,
                              shape.offset, shape.scale,
                              outputData, dim);
    return true;
}

} // namespace nn
} // namespace android
