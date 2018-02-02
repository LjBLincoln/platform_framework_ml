/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef ANDROID_ML_NN_RUNTIME_VERSIONED_IDEVICE_H
#define ANDROID_ML_NN_RUNTIME_VERSIONED_IDEVICE_H

#include "HalInterfaces.h"

#include <android-base/macros.h>
#include <string>
#include <tuple>

namespace android {
namespace nn {

// TODO(butlermichael): document VersionedIDevice class
class VersionedIDevice {
    DISALLOW_IMPLICIT_CONSTRUCTORS(VersionedIDevice);
public:
    VersionedIDevice(sp<V1_0::IDevice> device);

    std::pair<ErrorStatus, Capabilities> getCapabilities();

    ErrorStatus prepareModel(const Model& model, const sp<IPreparedModelCallback>& callback);

    std::pair<ErrorStatus, hidl_vec<bool>> getSupportedOperations(const Model& model);

    DeviceStatus getStatus();

    bool operator==(nullptr_t);
    bool operator!=(nullptr_t);

private:
    // Both versions of IDevice are necessary because the driver could v1.0,
    // v1.1, or a later version. These two pointers logically represent the same
    // object.
    //
    // The general strategy is: HIDL returns a V1_0 device object, which
    // (if not nullptr) could be v1.0, v1.1, or a greater version. The V1_0
    // object is then "dynamically cast" to a V1_1 object. If successful,
    // mDeviceV1_1 will point to the same object as mDeviceV1_0; otherwise,
    // mDeviceV1_1 will be nullptr.
    //
    // In general:
    // * If the device is truly v1.0, mDeviceV1_0 will point to a valid object
    //   and mDeviceV1_1 will be nullptr.
    // * If the device is truly v1.1 or later, both mDeviceV1_0 and mDeviceV1_1
    //   will point to the same valid object.
    sp<V1_0::IDevice> mDeviceV1_0;
    sp<V1_1::IDevice> mDeviceV1_1;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_VERSIONED_IDEVICE_H
