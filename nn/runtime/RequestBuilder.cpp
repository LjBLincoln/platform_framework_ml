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

RequestBuilder::RequestBuilder(const ModelBuilder* model) : mModel(model) {
    LOG(DEBUG) << "RequestBuilder::RequestBuilder";
    mInputs.resize(model->inputCount());
    mOutputs.resize(model->outputCount());
    mInputBuffers.resize(model->inputCount());
    mOutputBuffers.resize(model->outputCount());
}

int RequestBuilder::setInput(uint32_t index, const ANeuralNetworksOperandType* type,
                             const void* buffer, uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksRequest_setInput bad index " << index << " " << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    mInputBuffers[index] = buffer;
    updateModelInputOutputInfo(&mInputs[index], type, length, mModel->getInputOperandIndex(index));
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::setInputFromHardwareBuffer(
        [[maybe_unused]] uint32_t index, [[maybe_unused]] const ANeuralNetworksOperandType* type,
        [[maybe_unused]] const AHardwareBuffer* buffer) {
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
}

int RequestBuilder::setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                              uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
        LOG(ERROR) << "ANeuralNetworksRequest_setOutput bad index " << index << " " << count;
        return ANEURALNETWORKS_BAD_DATA;
    }
    mOutputBuffers[index] = buffer;
    updateModelInputOutputInfo(&mOutputs[index], type, length,
                               mModel->getOutputOperandIndex(index));
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::setOutputFromHardwareBuffer(
        [[maybe_unused]] uint32_t index, [[maybe_unused]] const ANeuralNetworksOperandType* type,
        [[maybe_unused]] const AHardwareBuffer* buffer) {
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
}

int RequestBuilder::updateModelInputOutputInfo(InputOutputInfo* info,
                                               const ANeuralNetworksOperandType* newType,
                                               uint32_t length, uint32_t operandIndex) {
    info->location.length = length;
    if (newType == nullptr) {
        info->dimensions = hidl_vec<uint32_t>();
    } else {
        const Operand& operand = mModel->getOperand(operandIndex);
        uint32_t count = newType->dimensions.count;
        if (static_cast<OperandType>(newType->type) != operand.type ||
            count != operand.dimensions.size()) {
            LOG(ERROR) << "ANeuralNetworksRequest_setInput/Output incompatible types";
            return ANEURALNETWORKS_BAD_DATA;
        }
        for (uint32_t i = 0; i < count; i++) {
            info->dimensions[i] = newType->dimensions.data[i];
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::startCompute(Event** event) {
    // TODO validate that we have full types for all inputs and outputs,
    // that the graph is not cyclic,
    for (auto p : mInputBuffers) {
        if (p == nullptr) {
            LOG(ERROR) << "ANeuralNetworksRequest_startCompute not all inputs specified";
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    for (auto p : mOutputBuffers) {
        if (p == nullptr) {
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
// it does not copy data.  Returns the total size.  Aligns each input a bit.
template <typename T>
static int allocateToPool(uint32_t poolId, std::vector<InputOutputInfo>* ioInfos,
                          hidl_memory* pool) {
    int64_t total = 0;
    for (auto& ioInfo : *ioInfos) {
        // TODO Good enough alignment?
        total += alignBytesNeeded(static_cast<uint32_t>(total), ioInfo.location.length);
        ioInfo.location.poolIndex = poolId;
        ioInfo.location.offset = static_cast<uint32_t>(total);
        total += ioInfo.location.length;
    }
    if (total > 0xFFFFFFFF) {
        LOG(ERROR) << "ANeuralNetworksRequest_startCompute Size of all inputs or outputs exceeds "
                      "2^32.";
        return ANEURALNETWORKS_BAD_DATA;
    }
    // Copy the input data to a shared memory buffer.
    *pool = allocateSharedMemory(total); // TODO check error
    return ANEURALNETWORKS_NO_ERROR;
}

int RequestBuilder::startComputeOnDevice(sp<IDevice> driver, const Model& model, Event** event) {
    LOG(DEBUG) << "RequestBuilder::startComputeOnDevice0";
    // TODO Dangerous!  In async, the model will outlive it here. Safe for now
    sp<IPreparedModel> preparedModel = driver->prepareModel(model);
    if (preparedModel == nullptr) {
        return ANEURALNETWORKS_OP_FAILED;
    }

    // We have two pools:
    // 0: input data
    // 1: output data
    // TODO: Revise this once we support pools for data
    const int INPUT = 0;
    const int OUTPUT = 1;
    hidl_vec<hidl_memory> pools;
    pools.resize(2);

    // Layout the input and output data.
    int n = allocateToPool<const void*>(INPUT, &mInputs, &pools[INPUT]);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    n = allocateToPool<void*>(OUTPUT, &mOutputs, &pools[OUTPUT]);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    // Copy the input data to the shared memory.
    sp<IMemory> inputMemory = mapMemory(pools[INPUT]);
    if (inputMemory == nullptr) {
        LOG(ERROR) << "ANeuralNetworksRequest_startCompute Can't create shared memory.";
        return ANEURALNETWORKS_OP_FAILED;
    }

    inputMemory->update();
    void* data = static_cast<void*>(inputMemory->getPointer());
    for (size_t i = 0; i < mInputs.size(); i++) {
        auto& info = mInputs[i];
        memcpy(reinterpret_cast<uint8_t*>(data) + info.location.offset, mInputBuffers[i],
               info.location.length);
    }

    inputMemory->commit();

    // Map the output shared memory.
    sp<IMemory> outputMemory = mapMemory(pools[OUTPUT]);
    if (outputMemory == nullptr) {
        LOG(ERROR) << "ANeuralNetworksRequest_startCompute Can't create shared memory.";
        return ANEURALNETWORKS_OP_FAILED;
    }

    Request request;
    request.inputs = mInputs;
    request.outputs = mOutputs;
    request.pools = pools;

    LOG(DEBUG) << "Before preparedModel->execute()";
    LOG(DEBUG) << "With inputs " << toString(mInputs);
    LOG(DEBUG) << "With outputs " << toString(mOutputs);
    LOG(DEBUG) << "With pools " << toString(pools);
    // Execute the request.
    if (!preparedModel->execute(request)) {
        LOG(DEBUG) << "**Execute failed**";
        return ANEURALNETWORKS_OP_FAILED;
    }

    // Copy the output data from shared memory to the output buffers.
    outputMemory->update();
    data = static_cast<void*>(outputMemory->getPointer());
    for (size_t i = 0; i < mOutputs.size(); i++) {
        auto& info = mOutputs[i];
        memcpy(mOutputBuffers[i], reinterpret_cast<uint8_t*>(data) + info.location.offset,
               info.location.length);
    }
    LOG(DEBUG) << "RequestBuilder::startComputeOnDevice completed";

    *event = new Event(); // TODO pass ievent
    return ANEURALNETWORKS_NO_ERROR;
}

template <typename T>
static void assignOnePerPool(const std::vector<T> buffers, hidl_vec<InputOutputInfo>* ioInfos,
                             std::vector<RunTimePoolInfo>* runTimePoolInfos, uint32_t* poolIndex) {
    for (size_t i = 0; i < ioInfos->size(); i++) {
        (*ioInfos)[i].location = {.poolIndex = *poolIndex, .offset = 0};
        (*runTimePoolInfos)[*poolIndex].buffer =
                reinterpret_cast<uint8_t*>(const_cast<void*>(buffers[i]));
        (*poolIndex)++;
    }
}

int RequestBuilder::startComputeOnCpu(Event** event, const Model& model) {
    // TODO: use a thread pool
    Event* e = new Event();
    *event = e;

    Request request;
    request.inputs = mInputs;
    request.outputs = mOutputs;
    // Create as many pools as there are input / output.
    // TODO This will need to be revised once we accept pool location
    // for input values.
    const size_t totalSize = mInputs.size() + mOutputs.size();
    request.pools.resize(totalSize);
    std::vector<RunTimePoolInfo> runTimePoolInfos;
    runTimePoolInfos.resize(totalSize);

    uint32_t poolIndex = 0;
    assignOnePerPool<const void*>(mInputBuffers, &request.inputs, &runTimePoolInfos, &poolIndex);
    assignOnePerPool<void*>(mOutputBuffers, &request.outputs, &runTimePoolInfos, &poolIndex);

    CpuExecutor executor;
    return executor.run(model, request, runTimePoolInfos);
}

} // namespace nn
} // namespace android
