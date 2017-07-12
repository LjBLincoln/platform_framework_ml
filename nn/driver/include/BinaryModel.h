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

// Class used by drivers to parse and navigate the serialized model.

#ifndef ANDROID_ML_NN_DRIVER_BINARY_MODEL_H
#define ANDROID_ML_NN_DRIVER_BINARY_MODEL_H

#include "Model.h"
#include "Utils.h"

#include <stddef.h>

namespace android {
namespace nn {

class BinaryModel : public IModel {
public:
    /* Verifies that the structure of the model is safe to use.  It does not
     * verify that:
     * - all the operands are used.
     * - all the operations can be reached from the inputs.
     * etc.
     * Assumes Model is not null.
     * TODO should make sure that we can't write into the control structures.
     */
    virtual ~BinaryModel() {}
    virtual Range<OperationEntry> getOperations() const { return mOperations; }
    virtual Range<OperandEntry> getOperands() const { return mOperands; }
    virtual Range<uint32_t> getOperandIndexes(const ArrayInfo& info) const {
        return Range<uint32_t>(mOperandIndexes, info);
    }
    virtual void copyDimensionStorage(std::vector<uint32_t>* dimensions) const {
        dimensions->resize(mDimensions.count());
        dimensions->insert(dimensions->begin(), mDimensions.begin(), mDimensions.end());
    }
    virtual uint32_t getInputOperandIndex(uint32_t listIndex) const {
        return getOperandIndex(mInfo->modelInputs, listIndex);
    }
    virtual uint32_t getOutputOperandIndex(uint32_t listIndex) const {
        return getOperandIndex(mInfo->modelOutputs, listIndex);
    }
    virtual const void* getDataPointer(uint32_t offset) const {
        return mOperandValues.begin() + offset;
    }

    static BinaryModel* Create(const uint8_t* buffer, size_t length);

protected:
    BinaryModel(){};
    // We don't take ownership of buffer.  It must outlive the lifetime of the
    // BinaryModel.
    bool initialize(const uint8_t* buffer, size_t length);
    bool validOperations() const;
    bool validOperands() const;
    bool validOperandIndexes() const;
    bool validModelInputsOutputs(const ArrayInfo& info) const;

    template <typename T>
    bool validTableInfo(const ArrayInfo& array) const;

    int32_t getOperandIndex(const ArrayInfo& info, uint32_t listIndex) const {
        nnAssert(listIndex < info.count);
        return mOperandIndexes[info.offset + listIndex];
    }

    const uint8_t* mBuffer = nullptr;  // We don't own.
    size_t mLength = 0;
    const ModelHeader* mInfo = nullptr;  // This is a pointer into mBuffer.
    Range<OperationEntry> mOperations;
    Range<OperandEntry> mOperands;
    Range<uint32_t> mDimensions;
    Range<uint32_t> mOperandIndexes;
    Range<uint32_t> mOperandValues;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_DRIVER_BINARY_MODEL_H
