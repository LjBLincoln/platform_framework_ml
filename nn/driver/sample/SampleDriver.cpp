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

#define LOG_TAG "SampleDriver"

#include "SampleDriver.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"

#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>

namespace android {
namespace nn {
namespace sample_driver {

Return<void> SampleDriver::prepareModel(const Model& model, const sp<IEvent>& event,
                                        prepareModel_cb cb) {
    LOG(DEBUG) << "prepareModel(" << toString(model) << ")"; // TODO errror
    if (validateModel(model)) {
        // TODO: make asynchronous later
        cb(ErrorStatus::NONE, new SamplePreparedModel(model));
    } else {
        cb(ErrorStatus::INVALID_ARGUMENT, nullptr);
    }

    // TODO: notify errors if they occur
    event->notify(ErrorStatus::NONE);
    return Void();
}

Return<DeviceStatus> SampleDriver::getStatus() {
    LOG(DEBUG) << "getStatus()";
    return DeviceStatus::AVAILABLE;
}

int SampleDriver::run() {
    android::hardware::configureRpcThreadpool(4, true);
    if (registerAsService(mName) != android::OK) {
        LOG(ERROR) << "Could not register service";
        return 1;
    }
    android::hardware::joinRpcThreadpool();
    LOG(ERROR) << "Service exited!";
    return 1;
}

static bool mapPools(std::vector<RunTimePoolInfo>* poolInfos, const hidl_vec<hidl_memory>& pools) {
    poolInfos->resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = (*poolInfos)[i];
        if (!poolInfo.set(pools[i])) {
            return false;
        }
    }
    return true;
}

void SamplePreparedModel::asyncExecute(const Request& request, const sp<IEvent>& event) {
    if (event.get() == nullptr) {
        LOG(ERROR) << "asyncExecute: invalid event";
        return;
    }

    std::vector<RunTimePoolInfo> poolInfo;
    if (!mapPools(&poolInfo, request.pools)) {
        event->notify(ErrorStatus::GENERAL_FAILURE);
        return;
    }

    CpuExecutor executor;
    int n = executor.run(mModel, request, poolInfo);
    LOG(DEBUG) << "executor.run returned " << n;
    ErrorStatus executionStatus =
            n == ANEURALNETWORKS_NO_ERROR ? ErrorStatus::NONE : ErrorStatus::GENERAL_FAILURE;
    Return<void> returned = event->notify(executionStatus);
    if (!returned.isOk()) {
        LOG(ERROR) << " hidl callback failed to return properly: " << returned.description();
    }
}

Return<ErrorStatus> SamplePreparedModel::execute(const Request& request, const sp<IEvent>& event) {
    LOG(DEBUG) << "execute(" << toString(request) << ")";
    if (!validateRequest(request, mModel)) {
        event->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the sample driver service
    // is expected to live forever.
    std::thread([this, request, event] { asyncExecute(request, event); }).detach();

    return ErrorStatus::NONE;
}

} // namespace sample_driver
} // namespace nn
} // namespace android
