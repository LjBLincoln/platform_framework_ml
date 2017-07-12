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

#define LOG_TAG "Request"

#include "Request.h"

#include "CpuExecutor.h"
#include "Manager.h"
#include "ModelBuilder.h"

namespace android {
namespace nn {

Request::Request(const ModelBuilder* model) : mModel(model) {
    mInputs.resize(model->inputCount());
    for (auto& info : mInputs) {
        info.buffer = nullptr;
        info.length = 0;
    }
    mOutputs.resize(model->outputCount());
    for (auto& info : mOutputs) {
        info.buffer = nullptr;
        info.length = 0;
    }
}

int Request::setInput(uint32_t index, const ANeuralNetworksOperandType* type, const void* buffer,
                      uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
        ALOGE("ANeuralNetworksRequest_setInput bad index %u %u", index, count);
        return ANEURALNETWORKS_BAD_DATA;
    }
    updateModelInputOutputInfo(&mInputs[index], type, const_cast<void*>(buffer), length,
                               mModel->getInputOperandIndex(index));
    return ANEURALNETWORKS_NO_ERROR;
}

int Request::setInputFromHardwareBuffer([[maybe_unused]] uint32_t index,
                                        [[maybe_unused]] const ANeuralNetworksOperandType* type,
                                        [[maybe_unused]] const AHardwareBuffer* buffer) {
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
}

int Request::setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                       uint32_t length) {
    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
        ALOGE("ANeuralNetworksRequest_setOutput bad index %u %u", index, count);
        return ANEURALNETWORKS_BAD_DATA;
    }
    updateModelInputOutputInfo(&mOutputs[index], type, buffer, length,
                               mModel->getOutputOperandIndex(index));
    return ANEURALNETWORKS_NO_ERROR;
}

int Request::setOutputFromHardwareBuffer([[maybe_unused]] uint32_t index,
                                         [[maybe_unused]] const ANeuralNetworksOperandType* type,
                                         [[maybe_unused]] const AHardwareBuffer* buffer) {
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
}

int Request::updateModelInputOutputInfo(InputOutputInfo* info,
                                        const ANeuralNetworksOperandType* newType, void* buffer,
                                        uint32_t length, uint32_t operandIndex) {
    info->buffer = buffer;
    info->length = length;
    info->dimensionChanged = newType != nullptr;
    if (info->dimensionChanged) {
        uint32_t count = newType->dimensions.count;
        if (newType->type != mModel->getOperandType(operandIndex) ||
            count != mModel->getOperandNumberOfDimensions(operandIndex)) {
            ALOGE("ANeuralNetworksRequest_setInput/Output incompatible types");
            return ANEURALNETWORKS_BAD_DATA;
        }

        info->dimensions.clear();
        info->dimensions.resize(count);
        info->dimensions.insert(info->dimensions.begin(), newType->dimensions.data,
                                newType->dimensions.data + count);
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int Request::startCompute(Event** event) {
    // TODO validate that we have full types for all inputs and outputs,
    // that the graph is not cyclic,
    std::shared_ptr<IDevice> driver = DriverManager::get()->getAvailableDriver();
    return driver == nullptr ? startComputeOnCpu(event) : startComputeOnDevice(driver, event);
}

int Request::startComputeOnDevice(std::shared_ptr<IDevice> driver, Event** event) {
    SerializedModel model;
    mModel->serialize(&model.memory);

    IRequest* request = nullptr;
    // TODO Dangerous!  In async, the model will outlive it here. Safe for now
    int n = driver->prepareRequest(&model, &request);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    IEvent* ievent = nullptr;
    std::vector<int> inputsAndOutputs;
    n = request->execute(mInputs, mOutputs, &ievent);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *event = new Event();  // TODO pass ievent
    return ANEURALNETWORKS_NO_ERROR;
}

int Request::startComputeOnCpu(Event** event) {
    // TODO: use a thread pool
    Event* e = new Event();
    *event = e;

    CpuExecutor executor(mModel, mInputs, mOutputs);
    return executor.run();
}

}  // namespace nn
}  // namespace android
