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

// Interface used by the CpuExecutor to communicate with the two model
// implementations.

#ifndef ANDROID_ML_NN_COMMON_MODEL_BUILDER_H
#define ANDROID_ML_NN_COMMON_MODEL_BUILDER_H

#include "Utils.h"

namespace android {
namespace nn {

class IModel {
public:
    virtual ~IModel() {}
    virtual Range<OperationEntry> getOperations() const = 0;
    virtual Range<OperandEntry> getOperands() const = 0;
    virtual Range<uint32_t> getOperandIndexes(const ArrayInfo& info) const = 0;
    virtual void copyDimensionStorage(std::vector<uint32_t>* dimensions) const = 0;
    virtual uint32_t getInputOperandIndex(uint32_t listIndex) const = 0;
    virtual uint32_t getOutputOperandIndex(uint32_t listIndex) const = 0;
    virtual const void* getDataPointer(uint32_t offset) const = 0;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_COMMON_MODEL_BUILDER_H
