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

#define LOG_TAG "CpuExecutor"

#include "CpuExecutor.h"

#include "Model.h"
#include "NeuralNetworks.h"
#include "Operations.h"

namespace android {
namespace nn {

// If we don't have a buffer, allocate it.
static bool allocateIfNeeded(RunTimeOperandInfo* info) {
    if (info->buffer == nullptr) {
        uint32_t length =
                sizeOfData(info->shape.type,
                           Range<uint32_t>(info->shape.numberOfDimensions, info->shape.dimensions));
        info->buffer = malloc(length);
    }
    return true;
}

CpuExecutor::CpuExecutor(const IModel* model, const std::vector<InputOutputInfo>& modelInputs,
                         const std::vector<InputOutputInfo>& modelOutputs)
      : mModel(model) {
    mModel->copyDimensionStorage(&mDimensions);

    const Range<OperandEntry> modelOperands = model->getOperands();
    const size_t count = modelOperands.count();
    mOperands.resize(count);
    for (size_t i = 0; i < count; i++) {
        const OperandEntry& from = modelOperands[i];
        RunTimeOperandInfo& to = mOperands[i];
        to.shape.type = from.type;
        to.shape.numberOfDimensions = from.dimensions.count;
        // It's safe to take the address. The size of mDimensions won't change.
        to.shape.dimensions = &mDimensions[from.dimensions.offset];
        if (from.location.pool == LOCATION_AT_RUN_TIME) {
            to.buffer = nullptr;
            to.numberOfUsesLeft = from.numberOfConsumers;
        } else if (from.location.pool == LOCATION_SAME_BLOCK) {
            to.buffer = const_cast<void*>(mModel->getDataPointer(from.location.offset));
            to.numberOfUsesLeft = 0;
        } else {
            // TODO: Revisit when we add support for multiple pools.
            nnAssert(false);
        }
        to.length = from.length;
    }

    for (uint32_t i = 0; i < modelInputs.size(); i++) {
        overrideOperand(mModel->getInputOperandIndex(i), modelInputs[i]);
    }
    for (uint32_t i = 0; i < modelOutputs.size(); i++) {
        overrideOperand(mModel->getOutputOperandIndex(i), modelOutputs[i]);
    }
}

int CpuExecutor::run() {
    // The model has serialized the operation in execution order.
    for (const auto& operation : mModel->getOperations()) {
        int n = executeOperation(operation);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

void CpuExecutor::overrideOperand(uint32_t operandIndex, const InputOutputInfo& from) {
    RunTimeOperandInfo& to = mOperands[operandIndex];
    if (from.dimensionChanged) {
        nnAssert(to.shape.numberOfDimensions == from.dimensions.size());
        for (uint32_t i = 0; i < to.shape.numberOfDimensions; i++) {
            to.shape.dimensions[i] = from.dimensions[i];
        }
    }
    nnAssert(to.buffer == nullptr);
    to.buffer = from.buffer;
    to.length = from.length;
    to.numberOfUsesLeft = 0;
}

void CpuExecutor::freeNoLongerUsedOperands(const Range<uint32_t>& inputs) {
    for (uint32_t i : inputs) {
        auto& info = mOperands[i];
        // Check if it's a static or model input/output.
        if (info.numberOfUsesLeft == 0) {
            continue;
        }
        nnAssert(mModel->getOperands()[i].location.pool == LOCATION_AT_RUN_TIME);
        info.numberOfUsesLeft--;
        if (info.numberOfUsesLeft == 0) {
            auto* buffer = mOperands[i].buffer;
            nnAssert(buffer != nullptr);
            free(buffer);
            buffer = nullptr;
        }
    }
}

int CpuExecutor::executeOperation(const OperationEntry& operation) {
    ALOGI("Executing %s", getOperationName(operation.opCode));
    const Range<uint32_t> ins = mModel->getOperandIndexes(operation.inputs);
    const Range<uint32_t> outs = mModel->getOperandIndexes(operation.outputs);
    bool success = false;

    // Function to verify that the number of input and output parameters
    // matches what is expected.
    auto parameterCountIs = [&ins, &outs, &operation](uint32_t expectedIns,
                                                      uint32_t expectedOuts) -> bool {
        if (ins.count() != expectedIns || outs.count() != expectedOuts) {
            ALOGE("%s: Invalid number of ins %u/%u and outs %u/%u",
                  getOperationName(operation.opCode), ins.count(), expectedIns, outs.count(),
                  expectedOuts);
            return false;
        }
        return true;
    };

    switch (static_cast<OperatorType>(operation.opCode)) {
        case OperatorType::ADD_FLOAT32: {
            if (!parameterCountIs(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            RunTimeOperandInfo& out = mOperands[outs[0]];

            success = addTensorsFloat32Prepare(in1.shape, in2.shape, &out.shape) &&
                    allocateIfNeeded(&out) &&
                    addTensorsFloat32(reinterpret_cast<const float*>(in1.buffer),
                                      reinterpret_cast<const float*>(in2.buffer),
                                      reinterpret_cast<float*>(out.buffer), in1.shape);
        } break;
        default:
            nnAssert(false);
            break;
    }
    if (!success) {
        ALOGE("%s failed.", getOperationName(operation.opCode));
        return ANEURALNETWORKS_OP_FAILED;
    }

    freeNoLongerUsedOperands(ins);
    return ANEURALNETWORKS_NO_ERROR;
}

} // namespace nn
} // namespace android
