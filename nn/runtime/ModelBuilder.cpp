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

#define LOG_TAG "ModelBuilder"

#include "ModelBuilder.h"

#include "Request.h"
#include "Utils.h"

#include <map>
#include <utility>

namespace android {
namespace nn {

static uint32_t alignBytesNeeded(uint32_t index, size_t length) {
    uint32_t pattern;
    if (length < 2) {
        pattern = 0;  // No alignment necessary
    } else if (length < 4) {
        pattern = 1;  // Align on 2-byte boundary
    } else {
        pattern = 3;  // Align on 4-byte boundary
    }
    uint32_t extra = (~(index - 1)) & pattern;
    return extra;
}

static void storeIntList(const ANeuralNetworksIntList& from, std::vector<uint32_t>* into,
                         ArrayInfo* info) {
    info->count = from.count;
    if (from.count == 0) {
        info->offset = 0;
    } else {
        size_t size = into->size();
        info->offset = static_cast<uint32_t>(size);  // TODO not the same as in file
        into->reserve(size + from.count);
        into->insert(into->end(), from.data, from.data + from.count);
    }
}

int ModelBuilder::addOperand(const ANeuralNetworksOperandType& type) {
    if (mCompletedModel) {
        ALOGE("ANeuralNetworksModel_addOperand can't modify after request creation");
        return ANEURALNETWORKS_BAD_DATA;
    }
    size_t idx = operandCount();
    if (idx >= MAX_NUMBER_OF_OPERANDS) {
        ALOGE("ANeuralNetworksModel_addOperand exceed max operands");
        return ANEURALNETWORKS_BAD_DATA;
    }
    mOperands.resize(idx + 1);
    auto& entry = mOperands[idx];
    entry.type = type.type;
    entry.numberOfConsumers = 0;
    storeIntList(type.dimensions, &mDimensions, &entry.dimensions);
    entry.location = {.pool = LOCATION_AT_RUN_TIME, .offset = 0};
    entry.length = 0;

    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setOperandValue(uint32_t index, const void* buffer, size_t length) {
    if (index >= operandCount()) {
        ALOGE("ANeuralNetworksModel_setOperandValue setting operand %u of %u", index,
              operandCount());
        return ANEURALNETWORKS_BAD_DATA;
    }
    OperandEntry& operand = mOperands[index];
    uint32_t neededLength =
                sizeOfData(operand.type, Range<uint32_t>(mDimensions, operand.dimensions));
    if (neededLength != length) {
        ALOGE("ANeuralNetworksModel_setOperandValue setting %zu bytes when needing "
              "%u",
              length, neededLength);
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t existingSize = static_cast<uint32_t>(mOperandValues.size());
    uint32_t extraBytes = alignBytesNeeded(existingSize, length);
    mOperandValues.resize(existingSize + extraBytes + length);
    operand.location.offset = existingSize + extraBytes;
    operand.location.pool = LOCATION_SAME_BLOCK;
    memcpy(&mOperandValues[operand.location.offset], buffer, length);
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::addOperation(ANeuralNetworksOperationType type,
                               const ANeuralNetworksIntList* inputs,
                               const ANeuralNetworksIntList* outputs) {
    if (mCompletedModel) {
        ALOGE("ANeuralNetworksModel_addOperation can't modify after request "
              "creation");
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t operationIndex = operationCount();
    if (operationIndex >= MAX_NUMBER_OF_OPERATIONS) {
        ALOGE("ANeuralNetworksModel_addOperation exceed max operations");
        return ANEURALNETWORKS_BAD_DATA;
    }
    mOperations.resize(operationIndex + 1);
    auto& entry = mOperations[operationIndex];
    entry.opCode = type;

    storeIntList(*inputs, &mOperandIndexes, &entry.inputs);
    storeIntList(*outputs, &mOperandIndexes, &entry.outputs);
    for (uint32_t i = 0; i < inputs->count; i++) {
        mOperands[inputs->data[i]].numberOfConsumers++;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setInputsAndOutputs(const ANeuralNetworksIntList* inputs,
                                      const ANeuralNetworksIntList* outputs) {
    if (mCompletedModel) {
        ALOGE("ANeuralNetworksModel_setInputsAndOutputs can't modify after request "
              "creation");
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO Validate all inputs
    storeIntList(*inputs, &mOperandIndexes, &mModelInputs);
    storeIntList(*outputs, &mOperandIndexes, &mModelOutputs);
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::loadBaseLineModel(uint32_t modelId) {
    if (mCompletedModel) {
        ALOGE("ANeuralNetworksModel_loadBaseLineModel can't modify after request "
              "creation");
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO implement
    switch (modelId) {
        case ANEURALNETWORKS_INCEPTION_SMALL_20_20:
        case ANEURALNETWORKS_INCEPTION_LARGE_20_20:
        case ANEURALNETWORKS_MOBILE_NETS_100_100:
            break;
    }
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
}

Request* ModelBuilder::createRequest() {
    finishTheModel();
    return new Request(this);
}

void ModelBuilder::finishTheModel() {
    if (!mCompletedModel) {
        // We sort the operations so that they will be in the appropriate
        // order for a single-threaded, op at a time execution.
        sortIntoRunOrder();
        mCompletedModel = true;
    }
}

void ModelBuilder::sortIntoRunOrder() {
    // Tracks the operations that can be executed.
    std::vector<uint32_t> opsReadyToRun;
    std::vector<OperationEntry> runOrder;

    // Mark the inputs
    for (auto i : getOperandIndexes(mModelInputs)) {
        mOperands[i].location.pool = 0;  // We'll reset it to unknown aftewards
    }

    // Tracks how many inputs are needed for each operation to be ready to run.
    std::multimap<uint32_t, uint32_t> operandToOperations;
    std::vector<uint32_t> unknownInputCount(operationCount());
    for (uint32_t operationIndex = 0; operationIndex < operationCount(); operationIndex++) {
        uint32_t& count = unknownInputCount[operationIndex];
        count = 0;
        for (uint32_t operandIndex : getOperandIndexes(mOperations[operationIndex].inputs)) {
            if (mOperands[operandIndex].location.pool == LOCATION_AT_RUN_TIME) {
                count++;
                operandToOperations.insert(
                            std::pair<uint32_t, uint32_t>(operandIndex, operationIndex));
            }
        }
        if (count == 0) {
            opsReadyToRun.push_back(operationIndex);
        }
    }
    // TODO verify that a modelInput can't be set as output or vice-versa
    // TODO test what happens when a model output is also used as input to an
    // op!!!
    for (auto i : getOperandIndexes(mModelInputs)) {
        mOperands[i].location.pool = LOCATION_AT_RUN_TIME;
    }

    while (opsReadyToRun.size() > 0) {
        // Execute the next op
        int opIndex = opsReadyToRun.back();
        opsReadyToRun.pop_back();
        const OperationEntry& operation = mOperations[opIndex];

        runOrder.push_back(mOperations[opIndex]);

        // Mark all its output as known.
        for (uint32_t operandIndex : getOperandIndexes(operation.outputs)) {
            // const OperandEntry& output = mOperands[operandIndex];
            auto range = operandToOperations.equal_range(operandIndex);
            for (auto i = range.first; i != range.second; i++) {
                uint32_t& count = unknownInputCount[i->second];
                if (--count == 0) {
                    opsReadyToRun.push_back(i->second);
                }
            }
        }
    }
    mOperations = runOrder;
}

void ModelBuilder::serialize(std::vector<uint8_t>* buffer) const {
    auto roundUp = [](size_t x) { return (x + 0xF) & ~0xF; };

    ModelHeader header;
    header.modelInputs = mModelInputs;
    header.modelOutputs = mModelOutputs;

    header.operations.count = static_cast<uint32_t>(mOperations.size());
    header.operands.count = static_cast<uint32_t>(mOperands.size());
    header.dimensions.count = static_cast<uint32_t>(mDimensions.size());
    header.operandIndexes.count = static_cast<uint32_t>(mOperandIndexes.size());
    header.operandValues.count = static_cast<uint32_t>(mOperandValues.size());

    size_t sizeOfHeader = sizeof(ModelHeader);
    size_t sizeOfOperations = sizeof(OperationEntry) * header.operations.count;
    size_t sizeOfOperands = sizeof(OperandEntry) * header.operands.count;
    size_t sizeOfDimensions = sizeof(uint32_t) * header.dimensions.count;
    size_t sizeOfOperandIndexes = sizeof(uint32_t) * header.operandIndexes.count;
    size_t sizeOfOperandValues = sizeof(uint8_t) * header.operandValues.count;

    size_t totalSize = 0;
    auto addUp = [&totalSize, &roundUp](size_t length, ArrayInfo* info) {
        info->offset = static_cast<uint32_t>(totalSize);
        totalSize += roundUp(length);
    };
    ArrayInfo headerInfo;
    addUp(sizeOfHeader, &headerInfo);
    addUp(sizeOfOperations, &header.operations);
    addUp(sizeOfOperands, &header.operands);
    addUp(sizeOfDimensions, &header.dimensions);
    addUp(sizeOfOperandIndexes, &header.operandIndexes);
    addUp(sizeOfOperandValues, &header.operandValues);

    buffer->resize(totalSize);
    uint8_t* start = buffer->data();
    auto copy = [start](size_t length, const void* from, const ArrayInfo& info) {
        memcpy(start + info.offset, from, length);
    };
    copy(sizeOfHeader, &header, headerInfo);
    copy(sizeOfOperations, mOperations.data(), header.operations);
    copy(sizeOfOperands, mOperands.data(), header.operands);
    copy(sizeOfDimensions, mDimensions.data(), header.dimensions);
    copy(sizeOfOperandIndexes, mOperandIndexes.data(), header.operandIndexes);
    copy(sizeOfOperandValues, mOperandValues.data(), header.operandValues);
}

}  // namespace nn
}  // namespace android
