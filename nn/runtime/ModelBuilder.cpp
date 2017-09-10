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

#include "CompilationBuilder.h"
#include "Utils.h"

#include <map>
#include <utility>

namespace android {
namespace nn {

// The maximum number of operands and operations that a model may have.
const uint32_t MAX_NUMBER_OF_OPERANDS = 0xFFFFFFFE;
const uint32_t MAX_NUMBER_OF_OPERATIONS = 0xFFFFFFFE;

int ModelBuilder::addOperand(const ANeuralNetworksOperandType& type) {
    if (mCompletedModel) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperand can't modify after request creation";
        return ANEURALNETWORKS_BAD_DATA;
    }
    size_t idx = mOperands.size();
    if (idx >= MAX_NUMBER_OF_OPERANDS) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperand exceed max operands";
        return ANEURALNETWORKS_BAD_DATA;
    }
    mOperands.resize(idx + 1);
    auto& entry = mOperands[idx];
    entry.type = static_cast<OperandType>(type.type);
    entry.scale = type.scale;
    entry.zeroPoint = type.offset;

    // TODO  entry.numberOfConsumers = 0;
    setFromIntList(&entry.dimensions, type.dimensions);
    entry.location = {.poolIndex = RUN_TIME, .offset = 0, .length = 0};
    entry.numberOfConsumers = 0;

    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setOperandValue(uint32_t index, const void* buffer, size_t length) {
    //    auto roundUp = [](size_t x) { return (x + 0xF) & ~0xF; };

    if (index >= operandCount()) {
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValue setting operand " << index << " of "
                   << operandCount();
        return ANEURALNETWORKS_BAD_DATA;
    }
    Operand& operand = mOperands[index];
    uint32_t neededLength = sizeOfData(operand.type, operand.dimensions);
    if (neededLength != length) {
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValue setting " << length
                   << " bytes when needing " << neededLength;
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t existingSize = static_cast<uint32_t>(mOperandValues.size());
    uint32_t extraBytes = alignBytesNeeded(existingSize, length);
    mOperandValues.resize(existingSize + extraBytes + length);
    operand.location = {.poolIndex = SAME_BLOCK,
                        .offset = existingSize + extraBytes,
                        .length = neededLength};
    memcpy(&mOperandValues[operand.location.offset], buffer, length);
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setOperandValueFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                                            size_t length) {
    if (index >= operandCount()) {
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValueFromMemory setting operand " << index
                   << " of " << operandCount();
        return ANEURALNETWORKS_BAD_DATA;
    }
    Operand& operand = mOperands[index];
    uint32_t neededLength = sizeOfData(operand.type, operand.dimensions);
    if (neededLength != length) {
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValueFromMemory setting " << length
                   << " bytes when needing " << neededLength;
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO validate does not exceed length of memory
    operand.location = {.poolIndex = mMemories.add(memory),
                        .offset = offset,
                        .length = neededLength};
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::addOperation(ANeuralNetworksOperationType type,
                               const ANeuralNetworksIntList* inputs,
                               const ANeuralNetworksIntList* outputs) {
    if (mCompletedModel) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperation can't modify after request creation";
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t operationIndex = operationCount();
    if (operationIndex >= MAX_NUMBER_OF_OPERATIONS) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperation exceed max operations";
        return ANEURALNETWORKS_BAD_DATA;
    }
    mOperations.resize(operationIndex + 1);
    auto& entry = mOperations[operationIndex];
    entry.opTuple = {static_cast<OperationType>(type),
                     static_cast<OperandType>(mOperands[inputs->data[0]].type)};

    setFromIntList(&entry.inputs, *inputs);
    setFromIntList(&entry.outputs, *outputs);
    for (uint32_t i : entry.inputs) {
        mOperands[i].numberOfConsumers++;
        // TODO mOperands[i].consumers.push_back(operationIndex);
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setInputsAndOutputs(const ANeuralNetworksIntList* inputs,
                                      const ANeuralNetworksIntList* outputs) {
    if (mCompletedModel) {
        LOG(ERROR)
                << "ANeuralNetworksModel_setInputsAndOutputs can't modify after request creation";
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO Validate all inputs
    setFromIntList(&mInputIndexes, *inputs);
    setFromIntList(&mOutputIndexes, *outputs);
    return ANEURALNETWORKS_NO_ERROR;
}

CompilationBuilder* ModelBuilder::createCompilation() {
    finishTheModel();
    return new CompilationBuilder(this);
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
    std::vector<Operation> runOrder;

    // Mark the inputs
    for (auto i : mInputIndexes) {
        mOperands[i].location.poolIndex = 0; // We'll reset it to unknown aftewards
    }

    // Tracks how many inputs are needed for each operation to be ready to run.
    std::multimap<uint32_t, uint32_t> operandToOperations;
    std::vector<uint32_t> unknownInputCount(operationCount());
    for (uint32_t operationIndex = 0; operationIndex < operationCount(); operationIndex++) {
        uint32_t& count = unknownInputCount[operationIndex];
        count = 0;
        for (uint32_t operandIndex : mOperations[operationIndex].inputs) {
            if (mOperands[operandIndex].location.poolIndex == RUN_TIME) {
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
    for (auto i : mInputIndexes) {
        mOperands[i].location.poolIndex = RUN_TIME;
    }

    while (opsReadyToRun.size() > 0) {
        // Execute the next op
        int opIndex = opsReadyToRun.back();
        opsReadyToRun.pop_back();
        const Operation& operation = mOperations[opIndex];

        runOrder.push_back(mOperations[opIndex]);

        // Mark all its output as known.
        for (uint32_t operandIndex : operation.outputs) {
            // const Operand& output = mOperands[operandIndex];
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

void ModelBuilder::setHidlModel(Model* model) const {
    model->operands = mOperands;
    model->operations = mOperations;
    model->inputIndexes = mInputIndexes;
    model->outputIndexes = mOutputIndexes;
    model->operandValues = mOperandValues;

    uint32_t count = mMemories.size();
    model->pools.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        model->pools[i] = mMemories[i]->getHidlMemory();
    }
}

} // namespace nn
} // namespace android
