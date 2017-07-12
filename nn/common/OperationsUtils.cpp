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
#include "Utils.h"

namespace android {
namespace nn {

bool SameShape(const Shape& in1, const Shape& in2) {
    if (in1.type != in2.type || in1.numberOfDimensions != in2.numberOfDimensions) {
        return false;
    }
    for (uint32_t i = 0; i < in1.numberOfDimensions; i++) {
        if (in1.dimensions[i] != in2.dimensions[i]) {
            return false;
        }
    }
    return true;
}

bool SetShape(const Shape& in, const Shape* out) {
    if (in.type != out->type || in.numberOfDimensions != out->numberOfDimensions) {
        return false;
    }
    for (uint32_t i = 0; i < in.numberOfDimensions; i++) {
        out->dimensions[i] = in.dimensions[i];
    }
    return true;
}

uint32_t getNumberOfElements(const Shape& shape) {
    uint32_t count = 1;
    for (uint32_t i = 0; i < shape.numberOfDimensions; i++) {
        count *= shape.dimensions[i];
    }
    return count;
}

}  // namespace nn
}  // namespace android
