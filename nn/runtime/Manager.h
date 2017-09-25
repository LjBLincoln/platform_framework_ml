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

#ifndef ANDROID_ML_NN_RUNTIME_MANAGER_H
#define ANDROID_ML_NN_RUNTIME_MANAGER_H

#include "HalInterfaces.h"
#include "Utils.h"

#include <map>
#include <unordered_set>
#include <vector>

namespace android {
namespace nn {

class ModelBuilder;

class Device {
public:
    Device(const std::string& name, const sp<IDevice>& device) : mName(name), mInterface(device) {}
    sp<IDevice> getInterface() { return mInterface; }
    const std::string& getName() { return mName; }
    void initialize();

    void getSupportedOperations(const Model& hidlModel, hidl_vec<bool>* supportedOperations) const;

    PerformanceInfo getFloat32Performance() const { return mFloat32Performance; }
    PerformanceInfo getQuantized8Performance() const { return mQuantized8Performance; }
private:
    std::string mName;
    sp<IDevice> mInterface;
    PerformanceInfo mFloat32Performance;
    PerformanceInfo mQuantized8Performance;

    // For debugging: behavior of IDevice::getSupportedOperations for SampleDriver.
    // 0 - all operations reported by IDevice::getSupportedOperations() supported
    // 1 - some operations reported by IDevice::getSupportedOperations() supported
    uint32_t mSupported;
};

// Manages the NN HAL devices.  Only one instance of this class will exist.
// Use get() to retrieve it.
class DeviceManager {
public:
    // TODO For now, just return the first one.
    // TODO deprecate
    std::shared_ptr<Device> getAvailableDriver() const {
        return mUseCpuOnly || mDevices.empty() ? nullptr : mDevices[0];
    }

    const std::vector<std::shared_ptr<Device>>& getDrivers() const {
        if (mUseCpuOnly) {
            return mNoDevices;
        }
        return mDevices;
    }

    // For testing only:
    void setUseCpuOnly(bool useCpuOnly) { mUseCpuOnly = useCpuOnly; }

    // Returns the singleton manager.
    static DeviceManager* get();

private:
    // Builds the list of available drivers and queries their capabilities.
    DeviceManager();

    // Adds a device for the manager to use.
    void registerDevice(const char* name, const sp<IDevice>& device) {
        auto d = std::make_shared<Device>(name, device);
        mDevices.push_back(d);
        d->initialize();
    }

    void findAvailableDevices();

    // List of all the devices we discovered.
    std::vector<std::shared_ptr<Device>> mDevices;

    // We leave this one always empty. To be used when mUseCpuOnly is true.
    std::vector<std::shared_ptr<Device>> mNoDevices;

    // If true, we'll ignore the drivers that are on the device and run everything
    // on the CPU.
    bool mUseCpuOnly = false;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_MANAGER_H
