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

#define LOG_TAG "BinaryModel"

#include "BinaryModel.h"
#include "HalAbstraction.h"

namespace android {
namespace nn {

static bool validArrayInfo(const ArrayInfo& array) {
    if (array.count == 0xFFFFFFFF) {
        // We don't allow that number to prevent infinite loops elsewhere.
        ALOGE("Number of entries in array greater than 0xFFFFFFFE");
        return false;
    }
    return true;
}

template <typename T>
static bool validSubList(const ArrayInfo& info, const Range<T>& range) {
    if (!validArrayInfo(info)) {
        return false;
    }
    if (info.offset + info.count > range.count()) {
        ALOGE("Sublist out of range.  Starts at %u, length %u, max %u", info.offset, info.count,
              range.count());
        return false;
    }
    return true;
}

BinaryModel* BinaryModel::Create(const uint8_t* buffer, size_t length) {
    BinaryModel* model = new BinaryModel();
    if (model->initialize(buffer, length)) {
        return model;
    }
    delete model;
    return nullptr;
}

bool BinaryModel::initialize(const uint8_t* buffer, size_t length) {
    mBuffer = buffer;
    mLength = length;
    if (mLength < sizeof(ModelHeader)) {
        ALOGE("Model buffer too small %zu", mLength);
        return false;
    }
    mInfo = reinterpret_cast<const ModelHeader*>(mBuffer);

    if (!validTableInfo<OperationEntry>(mInfo->operations) ||
        !validTableInfo<OperandEntry>(mInfo->operands) ||
        !validTableInfo<uint32_t>(mInfo->dimensions) ||
        !validTableInfo<uint32_t>(mInfo->operandIndexes) ||
        !validTableInfo<uint8_t>(mInfo->operandValues)) {
        return false;
    }
    mOperations.setFromBuffer(mInfo->operations, mBuffer);
    mOperands.setFromBuffer(mInfo->operands, mBuffer);
    mDimensions.setFromBuffer(mInfo->dimensions, mBuffer);
    mOperandIndexes.setFromBuffer(mInfo->operandIndexes, mBuffer);
    mOperandValues.setFromBuffer(mInfo->operandValues, mBuffer);

    return (validOperations() && validOperands() &&
            // Nothing to validate for mDimensions
            validOperandIndexes() &&
            // Nothing to validate for mOperandValues
            validSubList(mInfo->modelInputs, mOperandIndexes) &&
            validSubList(mInfo->modelOutputs, mOperandIndexes));
}

bool BinaryModel::validOperations() const {
    for (auto& op : mOperations) {
        if (static_cast<OperatorType>(op.opCode) >= OperatorType::NUM_OPERATOR_TYPES) {
            ALOGE("Invalid operation code %u", op.opCode);
            return false;
        }
        if (!validSubList(op.inputs, mOperandIndexes) ||
            !validSubList(op.outputs, mOperandIndexes)) {
            return false;
        }
    }
    return true;
}

bool BinaryModel::validOperands() const {
    for (auto& operand : mOperands) {
        if (static_cast<DataType>(operand.type) >= DataType::NUM_DATA_TYPES) {
            ALOGE("Invalid operand code %u", operand.type);
            return false;
        }
        if (!validSubList(operand.dimensions, mDimensions)) {
            return false;
        }
        if (operand.location.pool == LOCATION_SAME_BLOCK) {
            if (operand.location.offset + operand.length > mOperandValues.count()) {
                ALOGE("OperandValue location out of range.  Starts at %u, length %u, max "
                      "%u",
                      operand.location.offset, operand.length, mOperandValues.count());
                return false;
            }
        } else if (operand.location.pool == LOCATION_AT_RUN_TIME) {
            if (operand.location.offset != 0 || operand.length != 0) {
                ALOGE("Unexpected offset %u or length %u for runtime location.",
                      operand.location.offset, operand.length);
                return false;
            }
        } else {
            // TODO: Revisit when we add support for multiple pools.
            ALOGE("Unexpected pool %u", operand.location.pool);
            return false;
        }
    }
    return true;
}

bool BinaryModel::validOperandIndexes() const {
    const uint32_t operandCount = mOperands.count();
    for (uint32_t i : mOperandIndexes) {
        if (i >= operandCount) {
            ALOGE("Reference to operand %u of %u.", i, operandCount);
            return false;
        }
    }
    return true;
}

template <typename T>
bool BinaryModel::validTableInfo(const ArrayInfo& array) const {
    constexpr size_t entrySize = sizeof(T); /* assumes no padding? */
    if (!validArrayInfo(array)) {
        return false;
    }
    size_t spaceNeeded = array.count * entrySize;
    if (array.offset + spaceNeeded > mLength) {
        ALOGE("Array of %u entries of length %zu starting at %u exceeds buffer size "
              "of %zu",
              array.count, entrySize, array.offset, mLength);
        return false;
    }
    return true;
}

}  // namespace nn
}  // namespace android
