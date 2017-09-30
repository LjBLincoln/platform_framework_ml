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

#define LOG_TAG "ExecutionBuilder"

#include "ExecutionBuilder.h"

#include "CompilationBuilder.h"
#include "CpuExecutor.h"
#include "HalInterfaces.h"
#include "Manager.h"
#include "ModelBuilder.h"

#include <mutex>
#include <thread>
#include <vector>

namespace android {
namespace nn {

int ModelArgumentInfo::setFromPointer(const Operand& operand,
                                      const ANeuralNetworksOperandType* type, void* data,
                                      uint32_t length) {
    int n = updateDimensionInfo(operand, type);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    state = ModelArgumentInfo::POINTER;
    locationAndDimension.location = {.poolIndex = 0, .offset = 0, .length = length};
    buffer = data;
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelArgumentInfo::setFromMemory(const Operand& operand, const ANeuralNetworksOperandType* type,
                                     uint32_t poolIndex, uint32_t offset, uint32_t length) {
    int n = updateDimensionInfo(operand, type);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    state = ModelArgumentInfo::MEMORY;
    locationAndDimension.location = {.poolIndex = poolIndex, .offset = offset, .length = length};
    buffer = nullptr;
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelArgumentInfo::updateDimensionInfo(const Operand& operand,
                                           const ANeuralNetworksOperandType* newType) {
    if (newType == nullptr) {
        locationAndDimension.dimensions = hidl_vec<uint32_t>();
    } else {
        uint32_t count = newType->dimensionCount;
        if (static_cast<OperandType>(newType->type) != operand.type ||
            count != operand.dimensions.size()) {
            LOG(ERROR) << "ANeuralNetworksExecution_setInput/Output incompatible types";
            return ANEURALNETWORKS_BAD_DATA;
        }
        for (uint32_t i = 0; i < count; i++) {
            locationAndDimension.dimensions[i] = newType->dimensions[i];
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

ExecutionBuilder::ExecutionBuilder(const CompilationBuilder* compilation) :
        mModel(compilation->mModel),
        mPlan(&compilation->mPlan),
        mInputs(mModel->inputCount()),
        mOutputs(mModel->outputCount()),
        mMemories(mModel->getMemories()) {
    LOG(DEBUG) << "ExecutionBuilder::ExecutionBuilder";
}

int ExecutionBuilder::setInput(uint32_t index, const ANeuralNetworksOperandType* type,
                               const void* buffer, size_t length) {
    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksExecution_setInput bad index " << index << " " << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (type != nullptr) {
        int n = validateOperandType(*type, "ANeuralNetworksExecution_setInput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
        LOG(ERROR) << "ANeuralNetworksExecution_setInput input exceeds max length " << length;
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t l = static_cast<uint32_t>(length);
    return mInputs[index].setFromPointer(mModel->getInputOperand(index), type,
                                         const_cast<void*>(buffer), l);
}

int ExecutionBuilder::setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                         const Memory* memory, size_t offset, size_t length) {
    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksExecution_setInputFromMemory bad index " << index << " "
                   << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (!memory->validateSize(offset, length)) {
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO validate the rest
    uint32_t poolIndex = mMemories.add(memory);
    return mInputs[index].setFromMemory(mModel->getInputOperand(index), type, poolIndex, offset,
                                        length);
}

int ExecutionBuilder::setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                                size_t length) {
    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksExecution_setOutput bad index " << index << " " << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (type != nullptr) {
        int n = validateOperandType(*type, "ANeuralNetworksExecution_setOutput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
        LOG(ERROR) << "ANeuralNetworksExecution_setOutput input exceeds max length " << length;
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t l = static_cast<uint32_t>(length);
    return mOutputs[index].setFromPointer(mModel->getOutputOperand(index), type, buffer, l);
}

int ExecutionBuilder::setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                          const Memory* memory, size_t offset, size_t length) {
    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksExecution_setOutputFromMemory bad index " << index << " "
                   << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (!memory->validateSize(offset, length)) {
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO validate the rest
    uint32_t poolIndex = mMemories.add(memory);
    return mOutputs[index].setFromMemory(mModel->getOutputOperand(index), type, poolIndex, offset,
                                         length);
}

int ExecutionBuilder::startCompute(sp<Event>* event) {
    *event = nullptr;

    // TODO validate that we have full types for all inputs and outputs,
    // that the graph is not cyclic,
    /*
       TODO: For non-optional inputs, also verify that buffers are not null.

    for (auto& p : mInputs) {
        if (p.state == ModelArgumentInfo::UNSPECIFIED) {
            LOG(ERROR) << "ANeuralNetworksExecution_startCompute not all inputs specified";
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    */
    for (auto& p : mOutputs) {
        if (p.state == ModelArgumentInfo::UNSPECIFIED) {
            LOG(ERROR) << "ANeuralNetworksExecution_startCompute not all outputs specified";
            return ANEURALNETWORKS_BAD_DATA;
        }
    }

    // TODO: Remove the non-plan-based path once we've fully integrated ExecutionPlan
    // with the compilation and execution phases of the NN API.
#if NN_DEBUGGABLE
    if (DeviceManager::get()->getPartitioning() > 1) {
        std::shared_ptr<Device> device;
        sp<IPreparedModel> preparedModel;
        int n = mPlan->getSimplePlan(&device, &preparedModel);
        if (n == ANEURALNETWORKS_NO_ERROR) {
            LOG(DEBUG) << "ExecutionBuilder::startCompute (from plan) on "
                       << (device == nullptr ? "CPU" : device->getName());

            return device == nullptr ? startComputeOnCpu(event)
                                     : startComputeOnDevice(event, device->getInterface(), preparedModel);
        }
    }
#endif  // NN_DEBUGGABLE

    // Find a driver that can handle all the operations.
    Model hidlModel;
    mModel->setHidlModel(&hidlModel);
    const std::vector<std::shared_ptr<Device>>& devices = DeviceManager::get()->getDrivers();
    for (const auto& device : devices) {
        hidl_vec<bool> supports;
        LOG(DEBUG) << "Checking " << device->getName();
        device->getSupportedOperations(hidlModel, &supports);
        if (std::find(supports.begin(), supports.end(), false) == supports.end()) {
            LOG(DEBUG) << "ExecutionBuilder::startCompute (without plan) on " << device->getName();
            return startComputeOnDevice(event, device->getInterface());
        }
    }
    // If none can, run on the CPU.
    LOG(DEBUG) << "ExecutionBuilder::startCompute (without plan) on CPU";
    return startComputeOnCpu(event);
}

// Figures out how to place each of the input or outputs in a buffer. This just does the layout,
// it does not copy data.  Aligns each input a bit.
int ExecutionBuilder::allocatePointerArgumentsToPool(std::vector<ModelArgumentInfo>* args,
                                                     Memory* memory) {
    uint32_t nextPoolIndex = mMemories.size();
    int64_t total = 0;
    for (auto& info : *args) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndDimension.location;
            // TODO Good enough alignment?
            total += alignBytesNeeded(static_cast<uint32_t>(total), loc.length);
            loc.poolIndex = nextPoolIndex;
            loc.offset = static_cast<uint32_t>(total);
            total += loc.length;
        }
    };
    if (total > 0xFFFFFFFF) {
        LOG(ERROR) << "ANeuralNetworksExecution_startCompute Size of all inputs or outputs exceeds "
                      "2^32.";
        return ANEURALNETWORKS_BAD_DATA;
    }
    hidl_memory hidlMemory;
    if (total > 0) {
        memory->create(total);  // TODO check error
        mMemories.add(memory);
    }
    return ANEURALNETWORKS_NO_ERROR;
}

static void copyLocationAndDimension(const std::vector<ModelArgumentInfo>& argumentInfos,
                                     hidl_vec<RequestArgument>* ioInfos) {
    size_t count = argumentInfos.size();
    ioInfos->resize(count);
    for (size_t i = 0; i < count; i++) {
        (*ioInfos)[i] = argumentInfos[i].locationAndDimension;
    }
}

int ExecutionBuilder::startComputeOnDevice(sp<Event>* event, sp<IDevice> driver,
                                           sp<IPreparedModel> preparedModel) {
    *event = nullptr;

    // TODO: Remove the preparedModel == nullptr case once we've fully integrated
    // ExecutionPlan with the compilation and execution phases of the NN API
    if (preparedModel == nullptr) {
        Model model;
        mModel->setHidlModel(&model);

        // TODO Dangerous!  In async, the model will outlive it here. Safe for now
        sp<Event> preparationEvent = new Event();
        ErrorStatus prepareStatus = ErrorStatus::GENERAL_FAILURE;

        driver->prepareModel(model, preparationEvent,
                             [&](ErrorStatus status, const sp<IPreparedModel>& prepared) {
                                 prepareStatus = status;
                                 preparedModel = prepared;
                             });

        // Immediately synchronize with event for now
        // TODO: change to asynchronous later
        Event::Status eventStatus = preparationEvent->wait();

        if (prepareStatus != ErrorStatus::NONE || preparedModel == nullptr ||
            eventStatus != Event::Status::SUCCESS) {
            return ANEURALNETWORKS_OP_FAILED;
        }
    }

    // We separate the input & output pools so that we reduce the copying done if we
    // do an eventual remoting (hidl_memory->update()).  We could also use it to set
    // protection on read only memory but that's not currently done.
    Memory inputPointerArguments;
    Memory outputPointerArguments;

    // Layout the input and output data
    int n = allocatePointerArgumentsToPool(&mInputs, &inputPointerArguments);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = allocatePointerArgumentsToPool(&mOutputs, &outputPointerArguments);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    // Copy the input data that was specified via a pointer.
    // inputPointerArguments.update();
    for (auto& info : mInputs) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndDimension.location;
            uint8_t* data = nullptr;
            int n = inputPointerArguments.getPointer(&data);
            if (n != ANEURALNETWORKS_NO_ERROR) {
                return n;
            }
            memcpy(data + loc.offset, info.buffer, loc.length);
        }
    }
    // TODO: Add inputPointerArguments.commit() and .update() at all the right places

    Request request;
    copyLocationAndDimension(mInputs, &request.inputs);
    copyLocationAndDimension(mOutputs, &request.outputs);
    uint32_t count = mMemories.size();
    request.pools.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        request.pools[i] = mMemories[i]->getHidlMemory();
    }

    // Prepare the event for asynchronous execution. The sp<Event> object is
    // returned when the execution has been successfully launched, otherwise a
    // nullptr is returned. The sp is used for ref-counting purposes. Without
    // it, the HIDL service could attempt to communicate with a dead event
    // object.
    //
    // TODO: Explain the "dead event" problem further, either here or
    // in the design document.
    sp<Event> eventSp = new Event();

    LOG(DEBUG) << "Before preparedModel->execute() " << toString(request);
    // Execute.
    // TODO: What happens to the Event if the service dies abnormally
    // -- won't that keep the Event live forever, because the service
    // never has the opportunity to bump the reference count down? Or
    // maybe the HIDL infrastructure handles this magically? At worst,
    // it seems like this is a small memory leak, if the Event stays
    // alive forever.
    if (preparedModel->execute(request, eventSp) != ErrorStatus::NONE) {
        LOG(DEBUG) << "**Execute failed**";
        return ANEURALNETWORKS_OP_FAILED;
    }

    // TODO: Remove this synchronization point when the block of code below is
    // removed.
    Event::Status status = eventSp->wait();
    if (status != Event::Status::SUCCESS) {
        LOG(DEBUG) << "**Execute async failed**";
        return ANEURALNETWORKS_OP_FAILED;
    }

    // Copy the output data from shared memory to the output buffers.
    // TODO: Move this block of code somewhere else. It should not be in the
    // startCompute function.
    // TODO: outputMemory->update(); outputMemory->commit()
    for (auto& info : mOutputs) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndDimension.location;
            uint8_t* data = nullptr;
            int n = outputPointerArguments.getPointer(&data);
            if (n != ANEURALNETWORKS_NO_ERROR) {
                return n;
            }
            memcpy(info.buffer, data + loc.offset, loc.length);
        }
    }
    LOG(DEBUG) << "ExecutionBuilder::startComputeOnDevice completed";

    *event = eventSp;
    return ANEURALNETWORKS_NO_ERROR;
}

static void asyncStartComputeOnCpu(const Model& model, const Request& request,
                                   const std::vector<RunTimePoolInfo>& runTimePoolInfos,
                                   const sp<IEvent>& event) {
    CpuExecutor executor;
    int err = executor.run(model, request, runTimePoolInfos);
    ErrorStatus status = err == ANEURALNETWORKS_NO_ERROR ?
            ErrorStatus::NONE : ErrorStatus::GENERAL_FAILURE;
    event->notify(status);
}

int ExecutionBuilder::startComputeOnCpu(sp<Event>* event) {
    // TODO: use a thread pool

    Model model;
    mModel->setHidlModel(&model);

    // Prepare the event for asynchronous execution. The sp<Event> object is
    // returned when the execution has been successfully launched, otherwise a
    // nullptr is returned.
    sp<Event> eventSp = new Event();
    *event = nullptr;

    std::vector<RunTimePoolInfo> runTimePoolInfos;
    uint32_t count = mMemories.size();
    runTimePoolInfos.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        const Memory* mem = mMemories[i];
        runTimePoolInfos[i].set(mem->getHidlMemory());
    }
    // Create as many pools as there are input / output.
    auto fixPointerArguments = [&runTimePoolInfos](std::vector<ModelArgumentInfo>& argumentInfos) {
        for (ModelArgumentInfo& argumentInfo : argumentInfos) {
            if (argumentInfo.state == ModelArgumentInfo::POINTER) {
                RunTimePoolInfo runTimeInfo = {
                            .buffer = static_cast<uint8_t*>(argumentInfo.buffer)};
                argumentInfo.locationAndDimension.location.poolIndex =
                            static_cast<uint32_t>(runTimePoolInfos.size());
                argumentInfo.locationAndDimension.location.offset = 0;
                runTimePoolInfos.push_back(runTimeInfo);
            }
        }
    };
    fixPointerArguments(mInputs);
    fixPointerArguments(mOutputs);

    Request request;
    copyLocationAndDimension(mInputs, &request.inputs);
    copyLocationAndDimension(mOutputs, &request.outputs);

    // TODO: should model be moved with a std::cref?
    std::thread thread(asyncStartComputeOnCpu, model, std::move(request),
                       std::move(runTimePoolInfos), eventSp);
    eventSp->bind_thread(std::move(thread));

    *event = eventSp;
    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace nn
}  // namespace android
