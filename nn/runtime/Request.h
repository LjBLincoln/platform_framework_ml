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

#ifndef ANDROID_ML_NN_RUNTIME_REQUEST_H
#define ANDROID_ML_NN_RUNTIME_REQUEST_H

#include <vector>
#include "NeuralNetworks.h"

namespace android {
namespace nn {

// TODO Have a real implementation for this class.
class Event {
public:
    void wait() {}
};

class IDevice;
class ModelBuilder;
struct InputOutputInfo;

class Request {
public:
    Request(const ModelBuilder* model);

    void setPreference(uint32_t preference) { mPreference = preference; }

    int setInput(uint32_t index, const ANeuralNetworksOperandType* type, const void* buffer,
                 uint32_t length);
    int setInputFromHardwareBuffer(uint32_t index, const ANeuralNetworksOperandType* type,
                                   const AHardwareBuffer* buffer);
    int setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                  uint32_t length);
    int setOutputFromHardwareBuffer(uint32_t index, const ANeuralNetworksOperandType* type,
                                    const AHardwareBuffer* buffer);
    int startCompute(Event** event);

private:
    int updateModelInputOutputInfo(InputOutputInfo* info, const ANeuralNetworksOperandType* newType,
                                   void* buffer, uint32_t length, uint32_t operandIndex);

    int startComputeOnDevice(std::shared_ptr<IDevice> driver, Event** event);
    int startComputeOnCpu(Event** event);

    const ModelBuilder* mModel;
    // Whether the application prefers to go fast or use low power for this request.
    uint32_t mPreference = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;

    // The collected list of inputs and outputs of this request.
    std::vector<InputOutputInfo> mInputs;
    std::vector<InputOutputInfo> mOutputs;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_REQUEST_H
