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

#define LOG_TAG "ExecutionPlan"

#include "ExecutionPlan.h"

#include "CompilationBuilder.h"
#include "Manager.h"
#include "ModelBuilder.h"
#include "Utils.h"

#include <forward_list>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace android {
namespace nn {

typedef std::function<void(uint32_t)> OperationReadyCallback;

// This class tracks whether we know the value of an operand as operations
// are processed.
class OperandTracker {
public:
    // Creates the tracker for this model. Figure out which operations can be
    // executed right away and cb for each one of them.
    OperandTracker(const ModelBuilder* model, OperationReadyCallback cb);
    // Mark the specified operation as having been processed. The output
    // of the operation now being known, this may make new operations to be
    // able to run.  Call cb for each one of them.
    void markProcessed(uint32_t operationIndex, OperationReadyCallback cb);

private:
    const ModelBuilder* mModel;
    std::multimap<uint32_t, uint32_t> mOperandToOperations;
    std::vector<uint32_t> mUnknownInputCount;  // For each operation
};

OperandTracker::OperandTracker(const ModelBuilder* model, OperationReadyCallback cb) :
        mModel(model) {
    const auto& operations = mModel->getOperations();
    mUnknownInputCount.resize(operations.size());
    for (uint32_t operationIndex = 0; operationIndex < operations.size(); operationIndex++) {
        const Operation& operation = operations[operationIndex];
        uint32_t count = 0;
        for (uint32_t operandIndex : operation.inputs) {
            auto lifetime = mModel->getOperand(operandIndex).lifetime;
            if (lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
                lifetime == OperandLifeTime::MODEL_OUTPUT) {
                count++;
                mOperandToOperations.insert(
                        std::pair<uint32_t, uint32_t>(operandIndex, operationIndex));
            }
        }
        if (count == 0) {
            cb(operationIndex);
        }
        mUnknownInputCount[operationIndex] = count;
    }
}

void OperandTracker::markProcessed(uint32_t operationIndex, OperationReadyCallback cb) {
    // Mark all its outputs as known.
    const Operation& operation = mModel->getOperations()[operationIndex];
    for (uint32_t operandIndex : operation.outputs) {
        auto range = mOperandToOperations.equal_range(operandIndex);
        for (auto i = range.first; i != range.second; i++) {
            uint32_t& count = mUnknownInputCount[i->second];
            if (--count == 0) {
                cb(i->second);
            }
        }
    }
}

ExecutionStep::ExecutionStep(std::shared_ptr<ModelBuilder> model, std::shared_ptr<Device> device)
      : mSubModel(model), mDevice(device) {}

// Adds an operand if it has not been added already.
// Sets the index in the submodel for the corresponding operand.
int ExecutionStep::addOperand(uint32_t fromOperandIndex, uint32_t* toOperandIndex,
                              const ModelBuilder& fromModel, OperandKind kind) {
    // Have we added this operand already?
    auto i = mOperandMap.find(fromOperandIndex);
    if (i != mOperandMap.end()) {
        nnAssert(kind == INPUT);
        *toOperandIndex = i->second;
        return ANEURALNETWORKS_NO_ERROR;
    }

    // First time we add this operand.
    *toOperandIndex = mSubModel->operandCount();
    mOperandMap.insert(std::pair<uint32_t, uint32_t>(fromOperandIndex, *toOperandIndex));

    // Add the operand to the submodel.
    const Operand& operand = fromModel.getOperand(fromOperandIndex);
    ANeuralNetworksOperandType type = {.type = static_cast<int32_t>(operand.type),
                                       .dimensionCount =
                                               static_cast<uint32_t>(operand.dimensions.size()),
                                       .dimensions = operand.dimensions.data(),
                                       .scale = operand.scale,
                                       .zeroPoint = operand.zeroPoint};
    int n = mSubModel->addOperand(type);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        LOG(ERROR) << "Previous error occurred when partitioning the graph";
        return n;
    }

    // Sets its value.
    switch (operand.lifetime) {
        case OperandLifeTime::CONSTANT_COPY: {
            const uint8_t* data = fromModel.getPointerToOperandValue(operand.location.offset);
            n = mSubModel->setOperandValue(*toOperandIndex, data, operand.location.length);
            if (n != ANEURALNETWORKS_NO_ERROR) {
                LOG(ERROR) << "Previous error occurred when partitioning the graph";
                return n;
            }
        } break;
        case OperandLifeTime::CONSTANT_REFERENCE: {
            const Memory* memory = fromModel.getMemories()[operand.location.poolIndex];
            n = mSubModel->setOperandValueFromMemory(*toOperandIndex, memory,
                                                     operand.location.offset,
                                                     operand.location.length);
            if (n != ANEURALNETWORKS_NO_ERROR) {
                LOG(ERROR) << "Previous error occurred when partitioning the graph";
                return n;
            }
        } break;
        case OperandLifeTime::TEMPORARY_VARIABLE:
            if (kind == INPUT) {
                // The first time we've seen this operand is as an
                // input.  That means it must be defined by a
                // different partition, and is an input to this one.
                mSubModelInputs.push_back(std::make_pair(fromOperandIndex, *toOperandIndex));
            }
            break;
        case OperandLifeTime::MODEL_INPUT:
            mModelInputs.push_back(std::make_pair(fromOperandIndex, *toOperandIndex));
            break;
        case OperandLifeTime::MODEL_OUTPUT:
            mModelOutputs.push_back(std::make_pair(fromOperandIndex, *toOperandIndex));
            break;
        default:
            nnAssert(false);
            break;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int ExecutionStep::addOperation(int operationIndex, const ModelBuilder& fromModel) {
    const Operation& operation = fromModel.getOperation(operationIndex);

    // Convert the input and output operand indexes.
    //
    // We expect operations to be added in topological order.  Therefore:
    //
    // - We may not have seen an input if it is a model input, a
    //   constant, or an operand written by a different partition.
    //
    // - We should not have seen any outputs.
    const uint32_t inputCount = static_cast<uint32_t>(operation.inputs.size());
    const uint32_t outputCount = static_cast<uint32_t>(operation.outputs.size());
    std::vector<uint32_t> inputs(inputCount);
    std::vector<uint32_t> outputs(outputCount);

    auto addOperands = [this, &fromModel](const hidl_vec<uint32_t>& globalOperands,
                                          std::vector<uint32_t>& localOperands,
                                          OperandKind kind) -> int {
        const uint32_t operandCount = static_cast<uint32_t>(globalOperands.size());
        for (uint32_t i = 0; i < operandCount; i++) {
            uint32_t localOperand = ~0U;
            int n = addOperand(globalOperands[i], &localOperand, fromModel, kind);
            if (n != ANEURALNETWORKS_NO_ERROR)
                return n;
            localOperands[i] = localOperand;
        }
        return ANEURALNETWORKS_NO_ERROR;
    };

    int n;
    if ((n = addOperands(operation.inputs, inputs, INPUT)) != ANEURALNETWORKS_NO_ERROR ||
        (n = addOperands(operation.outputs, outputs, OUTPUT)) != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    return mSubModel->addOperation(static_cast<uint32_t>(operation.type), inputCount, inputs.data(),
                                   outputCount, outputs.data());
}

int ModelBuilder::partitionTheWork(uint32_t preference, ExecutionPlan* plan) const {
    // This function uses a heuristic approach to partitioning the graph.
    // It should be good enough for the first release.

    // Get the list of HAL devices.
    const std::vector<std::shared_ptr<Device>>& devices = DeviceManager::get()->getDrivers();

    // The device count is the number of HAL devices + 1. The +1 is for the CPU.
    const size_t deviceCount = devices.size() + 1;
    const size_t operationCount = mOperations.size();

    LOG(DEBUG) << "ModelBuilder::partitionTheWork: deviceCount = " << deviceCount
               << ", operationCount = " << operationCount;

    // If we only have the CPU, or if the graph has no operations, no
    // need to try to partition.
    if (deviceCount == 1 || operationCount == 0) {
        // TODO plan->addStep(new ExecutionStep(this));
        return ANEURALNETWORKS_NO_ERROR;
    }

    // Figure out where each operation will best execute.
    // The value of the vector is the index in the devices vector, with devices.size()
    // representing the CPU.
    std::vector<int> bestDeviceForOperation(operationCount);
    findBestDeviceForEachOperation(preference, devices, operationCount, deviceCount,
                                   &bestDeviceForOperation);

    // If one device will run all the operations, we don't need to split the work.
    if (std::adjacent_find(bestDeviceForOperation.begin(), bestDeviceForOperation.end(),
                           std::not_equal_to<int>()) == bestDeviceForOperation.end()) {
        if (WOULD_LOG(DEBUG)) {
            const int bestDeviceIndex = bestDeviceForOperation[0];
            const bool cpu = (size_t(bestDeviceIndex) == deviceCount - 1);
            LOG(DEBUG) << "ModelBuilder::partitionTheWork: only one best device: "
                       << bestDeviceIndex << " = "
                       << (cpu ? "CPU" : devices[bestDeviceIndex]->getName());
        }
        // TODO int index = bestDeviceForOperation[0];
        // TODO plan->addStep(new ExecutionStep(this, devices[index]));
        return ANEURALNETWORKS_NO_ERROR;
    }

    // No easy solution, we need to split the work.

    // We keep track of the operations that are ready to run for each device.
    std::vector<std::forward_list<uint32_t>> perDeviceQueue(deviceCount);

    // This helper function enqueues the operation on the appropriate queue.
    auto enqueueOnAppropriateDevice = [&](uint32_t operationIndex) {
        int deviceIndex = bestDeviceForOperation[operationIndex];
        perDeviceQueue[deviceIndex].push_front(operationIndex);
    };

    // This helper function finds a device that has operations ready to process.
    // We start by looking at the CPU. We do this to try to maximize the
    // size of the graph we'll send to non-CPU devices. If the CPU runs first,
    // it will have the chance to prepare more of the inputs required by the
    // other devices. This function returns -1 if all queues are empty.
    auto findNextDeviceToProcess = [&]() -> int {
        for (int i = deviceCount - 1; i-- > 0;) {
            if (!perDeviceQueue[i].empty()) {
                return i;
            }
        }
        return -1;
    };

    OperandTracker tracker(this, enqueueOnAppropriateDevice);
    // For each iteration of this loop, we'll create an execution step.
    while (true) {
        // Find the device we'll do this step for.
        int deviceIndex = findNextDeviceToProcess();
        if (deviceIndex < 0) {
            break;
        }
        // nullptr represents the CPU.
        std::shared_ptr<Device> device =
                static_cast<size_t>(deviceIndex) < deviceCount ? devices[deviceIndex] : nullptr;

        // Assign as much as possible to this device.
        std::shared_ptr<ExecutionStep> step(
            new ExecutionStep(std::shared_ptr<ModelBuilder>(new ModelBuilder()), device));
        plan->addStep(step);
        auto& queue = perDeviceQueue[deviceIndex];
        while (!queue.empty()) {
            uint32_t operationIndex = queue.front();
            queue.pop_front();
            step->addOperation(operationIndex, *this);
            tracker.markProcessed(operationIndex, enqueueOnAppropriateDevice);
        }
    }

    // TODO: Visit every step's mSubModelInputs to discover operands
    // in other steps that must become submodel outputs.

    return ANEURALNETWORKS_NO_ERROR;
}

PerformanceInfo ModelBuilder::getPerformanceInfo(const std::shared_ptr<Device> device,
                                                 uint32_t operationIndex) const {
    const Operation& operation = getOperation(operationIndex);
    // TODO This assumes that the type is dictated by the first operand. This is
    // currently the case but is not a safe assumption to make in the long term.
    const uint32_t operandIndex = operation.inputs[0];
    const OperandType operandType = mOperands[operandIndex].type;
    switch(operandType) {
        case OperandType::FLOAT32:
        case OperandType::TENSOR_FLOAT32:
            return device->getFloat32Performance();
        case OperandType::INT32:
        case OperandType::UINT32:
        case OperandType::TENSOR_QUANT8_ASYMM:
            // For OEM, the real selection will be made from who can run the operand.
        case OperandType::OEM:
        case OperandType::TENSOR_OEM_BYTE:
            return device->getQuantized8Performance();
        default:
            nnAssert(false);
            return device->getQuantized8Performance();
    }
}

namespace {
// This class determines whether a given device can execute a given operation
class CanDo {
public:
    CanDo() {}

    void initialize(const ModelBuilder* model, std::shared_ptr<Device> device) {
        Model hidlModel;
        model->setHidlModel(&hidlModel);
        device->getSupportedOperations(hidlModel, &mSupportsOperationByIndex);
    }

    bool check(size_t operationIndex) const { return mSupportsOperationByIndex[operationIndex]; }

private:
    hidl_vec<bool> mSupportsOperationByIndex;
};
};  // anonymous namespace

int ModelBuilder::findBestDeviceForEachOperation(
        uint32_t preference,
        const std::vector<std::shared_ptr<Device>>& devices,
        const size_t operationCount, [[maybe_unused]] const size_t deviceCount,
        std::vector<int>* bestDeviceForOperation) const {

    // Note that deviceCount includes CPU, which has no entry in devices[]
    const size_t nonCpuDeviceCount = deviceCount - 1;

    std::vector<CanDo> canDo(nonCpuDeviceCount);
    for (size_t deviceIndex = 0; deviceIndex < nonCpuDeviceCount; deviceIndex++) {
        canDo[deviceIndex].initialize(this, devices[deviceIndex]);
    }

    // Figure out the best driver for each operation.
    //
    // TODO: If the best driver is inferior (higher-power or
    // longer-running, depending on preference) than the CPU, then we
    // should use the CPU.  We could do this by setting bestChoice
    // initially to the number representing the CPU
    // (nonCpuDeviceCount) and bestPerfVal to the CPU value.  Problem
    // is, we have no such number now, so that will have to be for
    // release P or later.  One option is that the float performance
    // is a ratio of device/cpu rather than a number in joules or
    // microseconds.
    for (size_t operationIndex = 0; operationIndex < operationCount; operationIndex++) {
        int bestChoice = -1;
        float bestPerfVal = 0.0;  // do not check bestPerfVal unless we have bestChoice >= 0
        for (size_t deviceIndex = 0; deviceIndex < nonCpuDeviceCount; deviceIndex++) {
            if (canDo[deviceIndex].check(operationIndex)) {
                const auto& device = devices[deviceIndex];
                const PerformanceInfo perf = getPerformanceInfo(device, operationIndex);
                const float perfVal =
                            (preference == ANEURALNETWORKS_PREFER_LOW_POWER ? perf.powerUsage
                                                                            : perf.execTime);
                if ((bestChoice >= 0) && (bestPerfVal <= perfVal)) {
                    continue;
                }
                bestChoice = deviceIndex;
                bestPerfVal = perfVal;
            }
        }
        // No drivers are available for this operation, so choose the CPU.
        // TODO What if it is an OEM op?
        (*bestDeviceForOperation)[operationIndex] =
                bestChoice >= 0 ? bestChoice : static_cast<int>(nonCpuDeviceCount);
        LOG(VERBOSE) << "ModelBuilder::findBestDeviceForEachOperation("
                     << toString(getOperation(operationIndex).type)
                     << ") = "
                     << (*bestDeviceForOperation)[operationIndex];
    }
    return ANEURALNETWORKS_NO_ERROR;
}

} // namespace nn
} // namespace android
