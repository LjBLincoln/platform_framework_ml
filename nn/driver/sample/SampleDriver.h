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

#ifndef ANDROID_ML_NN_C_API_SAMPLE_DRIVER_H
#define ANDROID_ML_NN_C_API_SAMPLE_DRIVER_H

#include "HalAbstraction.h"
#include "NeuralNetworks.h"

namespace android {
namespace nn {

class BinaryModel;

// This class provides an example of how to implement a driver for the NN HAL.
// Since it's a simulated driver, it must run the computations on the CPU.
// An actual driver would not do that.
class SampleDriver : public IDevice {
public:
    virtual ~SampleDriver(){};
    virtual void initialize(Capabilities* capabilities);
    virtual void getSupportedSubgraph(void* graph, std::vector<bool>& canDo);

    virtual int prepareRequest(const SerializedModel* model, IRequest** request);
    virtual Status getStatus();
};

class SampleRequest : public IRequest {
public:
    SampleRequest(BinaryModel* model) : mModel(model) {}
    virtual ~SampleRequest() {}
    virtual int execute(const std::vector<InputOutputInfo>& inputs,
                        const std::vector<InputOutputInfo>& outputs, IEvent** event);
    virtual void releaseTempMemory() {}

private:
    BinaryModel* mModel;
};

class SampleEvent : public IEvent {
public:
    virtual ~SampleEvent() {}
    virtual uint32_t wait() { return ANEURALNETWORKS_NO_ERROR; }
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_C_API_SAMPLE_DRIVER_H
