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

#define LOG_TAG "RequestBuilder"

#include "RequestBuilder.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"
#include "Manager.h"
#include "ModelBuilder.h"

namespace android {
namespace nn {

RequestBuilder::RequestBuilder(const ModelBuilder* model)
      : mModel(model),
        mInputs(model->inputCount()),
        mOutputs(model->outputCount()),
        mMemories(model->getMemories()) {
    LOG(DEBUG) << "RequestBuilder::RequestBuilder";
    for (auto& p : mInputs) {
        p.state = ModelArgumentInfo::MISSING;
    }
    for (auto& p : mOutputs) {
        p.state = ModelArgumentInfo::MISSING;
    }
}

int RequestBuilder::setInput(uint32_t index, const ANeuralNetworksOperandType* type,
                             const void* buffer, uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksRequest_setInput bad index " << index << " " << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    ModelArgumentInfo& info = mInputs[index];
    info.state = ModelArgumentInfo::POINTER;
    info.locationAndDimension.location = {.poolIndex = RUN_TIME, .offset = 0, .length = length};
    updateDimensionInfo(&info, type, mModel->getInputOperandIndex(index));
    info.buffer = const_cast<void*>(buffer);
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                       const Memory* memory, uint32_t offset, uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksRequest_setInputFromMemory bad index " << index << " "
                   << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    ModelArgumentInfo& info = mInputs[index];
    info.state = ModelArgumentInfo::MEMORY;
    info.locationAndDimension.location = {.poolIndex = mMemories.add(memory),
                                          .offset = offset,
                                          .length = length};
    updateDimensionInfo(&info, type, mModel->getInputOperandIndex(index));
    info.buffer = nullptr;
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                              uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksRequest_setOutput bad index " << index << " " << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    ModelArgumentInfo& info = mOutputs[index];
    info.state = ModelArgumentInfo::POINTER;
    info.locationAndDimension.location = {.poolIndex = RUN_TIME, .offset = 0, .length = length};
    updateDimensionInfo(&info, type, mModel->getOutputOperandIndex(index));
    info.buffer = const_cast<void*>(buffer);
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                        const Memory* memory, uint32_t offset, uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksRequest_setOutputFromMemory bad index " << index << " "
                   << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    ModelArgumentInfo& info = mOutputs[index];
    info.state = ModelArgumentInfo::MEMORY;
    info.locationAndDimension.location = {.poolIndex = mMemories.add(memory),
                                          .offset = offset,
                                          .length = length};
    updateDimensionInfo(&info, type, mModel->getOutputOperandIndex(index));
    info.buffer = nullptr;
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::updateDimensionInfo(ModelArgumentInfo* info,
                                        const ANeuralNetworksOperandType* newType,
                                        uint32_t operandIndex) {
    if (newType == nullptr) {
        info->locationAndDimension.dimensions = hidl_vec<uint32_t>();
    } else {
        const Operand& operand = mModel->getOperand(operandIndex);
        uint32_t count = newType->dimensions.count;
        if (static_cast<OperandType>(newType->type) != operand.type ||
            count != operand.dimensions.size()) {
            LOG(ERROR) << "ANeuralNetworksRequest_setInput/Output incompatible types";
            return ANEURALNETWORKS_BAD_DATA;
        }
        for (uint32_t i = 0; i < count; i++) {
            info->locationAndDimension.dimensions[i] = newType->dimensions.data[i];
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::startCompute(Event** event) {
    // TODO validate that we have full types for all inputs and outputs,
    // that the graph is not cyclic,
    for (auto& p : mInputs) {
        if (p.state == ModelArgumentInfo::MISSING) {
            LOG(ERROR) << "ANeuralNetworksRequest_startCompute not all inputs specified";
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    for (auto& p : mOutputs) {
        if (p.state == ModelArgumentInfo::MISSING) {
            LOG(ERROR) << "ANeuralNetworksRequest_startCompute not all outputs specified";
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    LOG(DEBUG) << "RequestBuilder::startCompute";

    std::shared_ptr<Device> device = DeviceManager::get()->getAvailableDriver();
    Model model;
    mModel->setHidlModel(&model);

    return device == nullptr ? startComputeOnCpu(event, model)
                             : startComputeOnDevice(device->getInterface(), model, event);
}

// Figures out how to place each of the input or outputs in a buffer. This just does the layout,
// it does not copy data.  Aligns each input a bit.
int RequestBuilder::allocatePointerArgumentsToPool(std::vector<ModelArgumentInfo>* args,
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
        LOG(ERROR) << "ANeuralNetworksRequest_startCompute Size of all inputs or outputs exceeds "
                      "2^32.";
        return ANEURALNETWORKS_BAD_DATA;
    }
    hidl_memory hidlMemory;
    if (total > 0) {
        memory->create(total); // TODO check error
        mMemories.add(memory);
    }
    return ANEURALNETWORKS_NO_ERROR;
}

static void copyLocationAndDimension(const std::vector<ModelArgumentInfo>& argumentInfos,
                                     hidl_vec<InputOutputInfo>* ioInfos) {
    size_t count = argumentInfos.size();
    ioInfos->resize(count);
    for (size_t i = 0; i < count; i++) {
        (*ioInfos)[i] = argumentInfos[i].locationAndDimension;
    }
}

int RequestBuilder::startComputeOnDevice(sp<IDevice> driver, const Model& model, Event** event) {
    LOG(DEBUG) << "RequestBuilder::startComputeOnDevice1";
    // TODO Dangerous!  In async, the model will outlive it here. Safe for now
    sp<IPreparedModel> preparedModel = driver->prepareModel(model);
    if (preparedModel == nullptr) {
        return ANEURALNETWORKS_OP_FAILED;
    }

    // Layout the input and output data
    int n = allocatePointerArgumentsToPool(&mInputs, &mInputPointerArguments);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = allocatePointerArgumentsToPool(&mOutputs, &mOutputPointerArguments);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    // Copy the input data that was specified via a pointer.
    // mInputPointerArguments.update();
    for (auto& info : mInputs) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndDimension.location;
            uint8_t* data = mInputPointerArguments.getPointer();
            memcpy(data + loc.offset, info.buffer, loc.length);
        }
    }
    // TODO: Add mInputPointerArguments.commit() and .update() at all the right places

    Request request;
    copyLocationAndDimension(mInputs, &request.inputs);
    copyLocationAndDimension(mOutputs, &request.outputs);
    uint32_t count = mMemories.size();
    request.pools.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        request.pools[i] = mMemories[i]->getHidlMemory();
    }

    LOG(DEBUG) << "Before preparedModel->execute() " << toString(request);
    // Execute the request.
    if (!preparedModel->execute(request)) {
        LOG(DEBUG) << "**Execute failed**";
        return ANEURALNETWORKS_OP_FAILED;
    }

    // Copy the output data from shared memory to the output buffers.
    // TODO: outputMemory->update();
    for (auto& info : mOutputs) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndDimension.location;
            uint8_t* data = mOutputPointerArguments.getPointer();
            memcpy(info.buffer, data + loc.offset, loc.length);
        }
    }
    LOG(DEBUG) << "RequestBuilder::startComputeOnDevice completed";

    *event = new Event(); // TODO pass ievent
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::startComputeOnCpu(Event** event, [[maybe_unused]] const Model& model) {
    // TODO: use a thread pool
    Event* e = new Event();
    *event = e;

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
                RunTimePoolInfo runTimeInfo = {.buffer = static_cast<uint8_t*>(argumentInfo.buffer)};
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

    CpuExecutor executor;
    return executor.run(model, request, runTimePoolInfos);
}

} // namespace nn
} // namespace android
