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

#include <vector>

namespace android {
namespace nn {

class Device {
public:
    Device(const std::string& name, const sp<IDevice>& device) : mName(name), mInterface(device) {}
    sp<IDevice> getInterface() { return mInterface; }
    const std::string& getName() { return mName; }
    void initialize();

private:
    std::string mName;
    sp<IDevice> mInterface;

    /*
    std::array<bool, supportedOpsSize> mSupportedOperationTypes;
    bool mCachesCompilation;
    float mBootupTime;
    PerformanceInfo mFloat16Performance;
    PerformanceInfo mFloat32Performance;
    PerformanceInfo mQuantized8Performance;
    */
};

// Manages the NN HAL devices.  Only one instance of this class will exist.
// Use get() to retrieve it.
class DeviceManager {
public:
    // Initializes the manager: discover devices, query for their capabilities, etc.
    // This can be expensive, so we do it only when requested by the application.
    void initialize();
    void shutdown();

    // TODO For now, just return the first one.
    std::shared_ptr<Device> getAvailableDriver() const {
        return mUseCpuOnly || mDevices.empty() ? nullptr : mDevices[0];
    }

    // For testing only:
    void setUseCpuOnly(bool useCpuOnly) { mUseCpuOnly = useCpuOnly; }

    // Returns the singleton manager.
    static DeviceManager* get();

private:
    // Adds a device for the manager to use.
    void registerDevice(const char* name, const sp<IDevice>& device) {
        auto d = std::make_shared<Device>(name, device);
        mDevices.push_back(d);
        d->initialize();
    }

    void findAvailableDevices();

    // List of all the devices we discovered.
    std::vector<std::shared_ptr<Device>> mDevices;

    // The number of times initialise() has been called.  We will reset the content
    // of the manager when the equivalent number of shutdown() have been called.
    // This is done so that a library can call initialize and shutdown without
    // interfering with other code.
    //
    // TODO Need to revisit this whole section when integrating with HIDL and
    // ensuring multithreading is good.  Consider std::atomic<int>.
    int mUsageCount = 0;

    // If we true, we'll ignore the drivers that are on the device and run everything
    // on the CPU.
    bool mUseCpuOnly = false;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_MANAGER_H
