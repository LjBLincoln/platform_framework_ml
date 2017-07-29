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

namespace android {
namespace nn {

bool addTensorsFloat32Prepare(const Shape& in1, const Shape& in2, Shape* out) {
    return SameShape(in1, in2) && SetShape(in1, out);
}

bool addTensorsFloat32(const float* in1, const float* in2, float* out, const Shape& shape) {
    uint32_t count = getNumberOfElements(shape);
    for (size_t i = 0; i < count; i++) {
        *(out++) = *(in1++) + *(in2++);
    }
    return true;
}

} // namespace nn
} // namespace android
