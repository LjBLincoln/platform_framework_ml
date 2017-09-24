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

SampleDriver::~SampleDriver() {}

Return<void> SampleDriver::getCapabilities(getCapabilities_cb cb) {
    SetMinimumLogSeverity(base::VERBOSE);
    LOG(DEBUG) << "SampleDriver::getCapabilities()";

    // Our driver supports every op.
    static hidl_vec<OperationTuple> supportedOperationTuples{
            {OperationType::ADD, OperandType::TENSOR_FLOAT32},
            {OperationType::AVERAGE_POOL_2D, OperandType::TENSOR_FLOAT32},
            {OperationType::CONCATENATION, OperandType::TENSOR_FLOAT32},
            {OperationType::CONV_2D, OperandType::TENSOR_FLOAT32},
            {OperationType::DEPTHWISE_CONV_2D, OperandType::TENSOR_FLOAT32},
            {OperationType::DEPTH_TO_SPACE, OperandType::TENSOR_FLOAT32},
            {OperationType::DEQUANTIZE, OperandType::TENSOR_FLOAT32},
            {OperationType::EMBEDDING_LOOKUP, OperandType::TENSOR_FLOAT32},
            {OperationType::FLOOR, OperandType::TENSOR_FLOAT32},
            {OperationType::FULLY_CONNECTED, OperandType::TENSOR_FLOAT32},
            {OperationType::HASHTABLE_LOOKUP, OperandType::TENSOR_FLOAT32},
            {OperationType::L2_NORMALIZATION, OperandType::TENSOR_FLOAT32},
            {OperationType::L2_POOL_2D, OperandType::TENSOR_FLOAT32},
            {OperationType::LOCAL_RESPONSE_NORMALIZATION, OperandType::TENSOR_FLOAT32},
            {OperationType::LOGISTIC, OperandType::TENSOR_FLOAT32},
            {OperationType::LSH_PROJECTION, OperandType::TENSOR_FLOAT32},
            {OperationType::LSTM, OperandType::TENSOR_FLOAT32},
            {OperationType::MAX_POOL_2D, OperandType::TENSOR_FLOAT32},
            {OperationType::MUL, OperandType::TENSOR_FLOAT32},
            {OperationType::RELU, OperandType::TENSOR_FLOAT32},
            {OperationType::RELU1, OperandType::TENSOR_FLOAT32},
            {OperationType::RELU6, OperandType::TENSOR_FLOAT32},
            {OperationType::RESHAPE, OperandType::TENSOR_FLOAT32},
            {OperationType::RESIZE_BILINEAR, OperandType::TENSOR_FLOAT32},
            {OperationType::RNN, OperandType::TENSOR_FLOAT32},
            {OperationType::SOFTMAX, OperandType::TENSOR_FLOAT32},
            {OperationType::SPACE_TO_DEPTH, OperandType::TENSOR_FLOAT32},
            {OperationType::SVDF, OperandType::TENSOR_FLOAT32},
            {OperationType::TANH, OperandType::TENSOR_FLOAT32},

            {OperationType::ADD, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::AVERAGE_POOL_2D, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::CONCATENATION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::CONV_2D, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::DEPTHWISE_CONV_2D, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::DEPTH_TO_SPACE, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::DEQUANTIZE, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::EMBEDDING_LOOKUP, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::FLOOR, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::FULLY_CONNECTED, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::HASHTABLE_LOOKUP, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::L2_NORMALIZATION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::L2_POOL_2D, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LOCAL_RESPONSE_NORMALIZATION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LOGISTIC, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LSH_PROJECTION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LSTM, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::MAX_POOL_2D, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::MUL, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RELU, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RELU1, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RELU6, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RESHAPE, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RESIZE_BILINEAR, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RNN, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::SOFTMAX, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::SPACE_TO_DEPTH, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::SVDF, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::TANH, OperandType::TENSOR_QUANT8_ASYMM},
    };

    // TODO: These numbers are completely arbitrary.  To be revised.
    PerformanceInfo float32Performance = {
            .execTime = 132.0f, // nanoseconds?
            .powerUsage = 1.0f, // picoJoules
    };

    PerformanceInfo quantized8Performance = {
            .execTime = 100.0f, // nanoseconds?
            .powerUsage = 1.0f, // picoJoules
    };

    Capabilities capabilities = {
            .supportedOperationTuples = supportedOperationTuples,
            .cachesCompilation = false,
            .float32Performance = float32Performance,
            .quantized8Performance = quantized8Performance,
    };

    // return
    cb(ErrorStatus::NONE, capabilities);
    return Void();
}

Return<void> SampleDriver::getSupportedOperations(const Model& model,
                                                  getSupportedOperations_cb cb) {
    LOG(DEBUG) << "SampleDriver::getSupportedOperations()";
    if (validateModel(model)) {
        std::vector<bool> supported(model.operations.size(), true);
        cb(ErrorStatus::NONE, supported);
    }
    else {
        std::vector<bool> supported;
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
    }
    return Void();
}

Return<void> SampleDriver::prepareModel(const Model& model, const sp<IEvent>& event,
                                        prepareModel_cb cb) {
    LOG(DEBUG) << "SampleDriver::prepareModel(" << toString(model) << ")"; // TODO errror
    if (validateModel(model)) {
        // TODO: make asynchronous later
        cb(ErrorStatus::NONE, new SamplePreparedModel(model));
    }
    else {
        cb(ErrorStatus::INVALID_ARGUMENT, nullptr);
    }

    // TODO: notify errors if they occur
    event->notify(ErrorStatus::NONE);
    return Void();
}

Return<DeviceStatus> SampleDriver::getStatus() {
    LOG(DEBUG) << "SampleDriver::getStatus()";
    return DeviceStatus::AVAILABLE;
}

SamplePreparedModel::SamplePreparedModel(const Model& model) {
    // Make a copy of the model, as we need to preserve it.
    mModel = model;
}

SamplePreparedModel::~SamplePreparedModel() {}

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
    ErrorStatus executionStatus = n == ANEURALNETWORKS_NO_ERROR ?
            ErrorStatus::NONE : ErrorStatus::GENERAL_FAILURE;
    Return<void> returned = event->notify(executionStatus);
    if (!returned.isOk()) {
        LOG(ERROR) << "hidl callback failed to return properly: " << returned.description();
    }
}

Return<ErrorStatus> SamplePreparedModel::execute(const Request& request, const sp<IEvent>& event) {
    LOG(DEBUG) << "SampleDriver::execute(" << toString(request) << ")";
    if (!validateRequest(request, mModel)) {
        event->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the sample driver service
    // is expected to live forever.
    std::thread([this, request, event]{ asyncExecute(request, event); }).detach();

    return ErrorStatus::NONE;
}

} // namespace sample_driver
} // namespace nn
} // namespace android

using android::nn::sample_driver::SampleDriver;

int main() {
    android::sp<SampleDriver> driver = new SampleDriver();
    android::hardware::configureRpcThreadpool(4, true /* will join */);
    if (driver->registerAsService("sample") != android::OK) {
        ALOGE("Could not register service");
        return 1;
    }
    android::hardware::joinRpcThreadpool();
    ALOGE("Service exited!");
    return 1;
}
