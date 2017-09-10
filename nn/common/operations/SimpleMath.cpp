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

bool addMulPrepare(const Shape& in1, const Shape& in2, Shape* out) {
    if (getNumberOfDimensions(in1) > 4 || getNumberOfDimensions(in2) > 4) {
        LOG(ERROR) << "Only supports upto 4D tensors.";
        return false;
    }
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

bool addFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    bool needBroadcast = !SameShape(shape1, shape2);

    #define ANDROID_NN_NORMAL_ADD(activation)                        \
        optimized_ops::Add<FusedActivationFunctionType::activation>( \
                in1, convertShapeToDims(shape1),                     \
                in2, convertShapeToDims(shape2),                     \
                out, convertShapeToDims(shapeOut))

    #define ANDROID_NN_BROADCAST_ADD(activation)                              \
        optimized_ops::BroadcastAdd<FusedActivationFunctionType::activation>( \
                in1, convertShapeToDims(shape1),                              \
                in2, convertShapeToDims(shape2),                              \
                out, convertShapeToDims(shapeOut))

    #define ANDROID_NN_ADD_DISPATCH

    if (needBroadcast) {
        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_BROADCAST_ADD)
    } else {
        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_NORMAL_ADD)
    }

    #undef ANDROID_NN_ADD
    #undef ANDROID_NN_BROADCAST_ADD
    return true;
}

bool mulFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    bool needBroadcast = !SameShape(shape1, shape2);

    #define ANDROID_NN_NORMAL_MUL(activation)                        \
        optimized_ops::Mul<FusedActivationFunctionType::activation>( \
                in1, convertShapeToDims(shape1),                     \
                in2, convertShapeToDims(shape2),                     \
                out, convertShapeToDims(shapeOut))

    #define ANDROID_NN_BROADCAST_MUL(activation)                              \
        optimized_ops::BroadcastMul<FusedActivationFunctionType::activation>( \
                in1, convertShapeToDims(shape1),                              \
                in2, convertShapeToDims(shape2),                              \
                out, convertShapeToDims(shapeOut))

    if (needBroadcast) {
        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_BROADCAST_MUL)
    } else {
        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_NORMAL_MUL)
    }

    #undef ANDROID_NN_MUL
    #undef ANDROID_NN_BROADCAST_MUL
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
