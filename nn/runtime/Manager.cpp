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

#define LOG_TAG "Manager"

#include "Manager.h"
#include "HalInterfaces.h"
#include "Utils.h"

#include <android/hidl/manager/1.0/IServiceManager.h>
#include <hidl/HidlTransportSupport.h>
#include <hidl/ServiceManagement.h>

namespace android {
namespace nn {

void Device::initialize() {
    mInterface->initialize([&]([[maybe_unused]] const Capabilities& capabilities) {
        LOG(DEBUG) << "Capab " << capabilities.float16Performance.execTime;
        LOG(DEBUG) << "Capab " << capabilities.float32Performance.execTime;
        /*
        supportedOperationTypes  = capabilities.supportedOperationTypes;
        cachesCompilation       = capabilities.cachesCompilation;
        bootupTime              = capabilities.bootupTime;
        float16Performance      = capabilities.float16Performance;
        float32Performance      = capabilities.float32Performance;
        quantized8Performance   = capabilities.quantized8Performance;
        */
    });
}

DeviceManager* DeviceManager::get() {
    static DeviceManager manager;
    return &manager;
}

void DeviceManager::findAvailableDevices() {
    using ::android::hardware::neuralnetworks::V1_0::IDevice;
    using ::android::hidl::manager::V1_0::IServiceManager;
    LOG(DEBUG) << "findAvailableDevices";

    sp<IServiceManager> manager = hardware::defaultServiceManager();
    if (manager == nullptr) {
        LOG(ERROR) << "Unable to open defaultServiceManager";
        return;
    }

    manager->listByInterface(IDevice::descriptor, [this](const hidl_vec<hidl_string>& names) {
        for (const auto& name : names) { // int i = 0; i < (int)names.size(); ++i) {
            LOG(DEBUG) << "Found interface " << name.c_str();
            sp<IDevice> device = IDevice::getService(name);
            if (device == nullptr) {
                LOG(ERROR) << "Got a null IDEVICE for " << name.c_str();
                continue;
            }
            registerDevice(name.c_str(), device);
        }
    });
}

void DeviceManager::initialize() {
    if (mUsageCount++ == 0) {
        findAvailableDevices();
    }
}

void DeviceManager::shutdown() {
    nnAssert(mUsageCount > 0);
    if (mUsageCount > 0) {
        if (--mUsageCount == 0) {
            mDevices.clear();
        }
    }
}

} // namespace nn
} // namespace android
