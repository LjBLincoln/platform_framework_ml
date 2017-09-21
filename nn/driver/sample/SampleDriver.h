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

#ifndef ANDROID_ML_NN_SAMPLE_DRIVER_SAMPLE_DRIVER_H
#define ANDROID_ML_NN_SAMPLE_DRIVER_SAMPLE_DRIVER_H

#include "HalInterfaces.h"
#include "NeuralNetworks.h"

namespace android {
namespace nn {
namespace sample_driver {

// This class provides an example of how to implement a driver for the NN HAL.
// Since it's a simulated driver, it must run the computations on the CPU.
// An actual driver would not do that.
class SampleDriver : public IDevice {
public:
    ~SampleDriver() override;
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) override;
    Return<void> getSupportedOperations(const Model& model,
                                        getSupportedOperations_cb _hidl_cb) override;
    Return<void> prepareModel(const Model& model, const sp<IEvent>& event,
                              prepareModel_cb _hidl_cb) override;
    Return<DeviceStatus> getStatus() override;
};

class SamplePreparedModel : public IPreparedModel {
public:
    SamplePreparedModel(const Model& model);
    ~SamplePreparedModel() override;
    Return<ErrorStatus> execute(const Request& request, const sp<IEvent>& event) override;

private:
    void asyncExecute(const Request& request, const sp<IEvent>& event);
    Model mModel;
};

} // namespace sample_driver
} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_SAMPLE_DRIVER_SAMPLE_DRIVER_H
