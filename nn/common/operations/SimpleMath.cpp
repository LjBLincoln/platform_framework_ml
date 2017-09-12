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

bool floorFloat32(const float* inputData,
                  float* outputData,
                  const Shape& shape) {
    Dims<4> dim = convertShapeToDims(shape);
    optimized_ops::Floor(inputData, dim, outputData, dim);
    return true;
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
