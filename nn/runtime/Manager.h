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

#include "HalAbstraction.h"

#include <vector>

namespace android {
namespace nn {

// Manages the NN HAL drivers.  Only one instance of this class will exist.
// Use get() to retrieve it.
class DriverManager {
public:
    // Initializes the manager: discover drivers, query for their capabilities, etc.
    // This can be expensive, so we do it only when requested by the application.
    void initialize();
    void shutdown();

    // Adds a driver for the manager to use.
    void registerDriver(std::shared_ptr<IDevice> device) { mDrivers.push_back(device); }

    // TODO For now, just return the first one.
    std::shared_ptr<IDevice> getAvailableDriver() const { return mDrivers.empty() ? nullptr : mDrivers[0]; }

    // Returns the singleton manager.
    static DriverManager* get() { return &manager; }

private:
    // List of all the drivers currently discovered.
    std::vector<std::shared_ptr<IDevice>> mDrivers;

    // The number of times initialise() has been called.  We will reset the content
    // of the manager when the equivalent number of shutdown() have been called.
    // This is done so that a library can call initialize and shutdown without
    // interfering with other code.
    //
    // TODO Need to revisit this whole section when integrating with HIDL and
    // ensuring multithreading is good.  Consider std::atomic<int>.
    int mUsageCount = 0;

    static DriverManager manager;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_MANAGER_H
