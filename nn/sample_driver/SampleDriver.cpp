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

namespace android {
namespace nn {
namespace sample_driver {

Return<void> SampleDriver::initialize(initialize_cb cb) {
    LOG(DEBUG) << "SampleDriver::initialize()";

    // Our driver supports every op.
    hidl_vec<OperationType> supportedOperationTypes{
            OperationType::AVERAGE_POOL_FLOAT32,
            OperationType::CONCATENATION_FLOAT32,
            OperationType::CONV_FLOAT32,
            OperationType::DEPTHWISE_CONV_FLOAT32,
            OperationType::MAX_POOL_FLOAT32,
            OperationType::L2_POOL_FLOAT32,
            OperationType::DEPTH_TO_SPACE_FLOAT32,
            OperationType::SPACE_TO_DEPTH_FLOAT32,
            OperationType::LOCAL_RESPONSE_NORMALIZATION_FLOAT32,
            OperationType::SOFTMAX_FLOAT32,
            OperationType::RESHAPE_FLOAT32,
            OperationType::SPLIT_FLOAT32,
            OperationType::FAKE_QUANT_FLOAT32,
            OperationType::ADD_FLOAT32,
            OperationType::FULLY_CONNECTED_FLOAT32,
            OperationType::CAST_FLOAT32,
            OperationType::MUL_FLOAT32,
            OperationType::L2_NORMALIZATION_FLOAT32,
            OperationType::LOGISTIC_FLOAT32,
            OperationType::RELU_FLOAT32,
            OperationType::RELU6_FLOAT32,
            OperationType::RELU1_FLOAT32,
            OperationType::TANH_FLOAT32,
            OperationType::DEQUANTIZE_FLOAT32,
            OperationType::FLOOR_FLOAT32,
            OperationType::GATHER_FLOAT32,
            OperationType::RESIZE_BILINEAR_FLOAT32,
            OperationType::LSH_PROJECTION_FLOAT32,
            OperationType::LSTM_FLOAT32,
            OperationType::SVDF_FLOAT32,
            OperationType::RNN_FLOAT32,
            OperationType::N_GRAM_FLOAT32,
            OperationType::LOOKUP_FLOAT32,
    };

    // TODO: These numbers are completely arbitrary.  To be revised.
    PerformanceInfo float16Performance = {
            .execTime = 116.0f, // nanoseconds?
            .powerUsage = 1.0f, // picoJoules
    };

    PerformanceInfo float32Performance = {
            .execTime = 132.0f, // nanoseconds?
            .powerUsage = 1.0f, // picoJoules
    };

    PerformanceInfo quantized8Performance = {
            .execTime = 100.0f, // nanoseconds?
            .powerUsage = 1.0f, // picoJoules
    };

    Capabilities capabilities = {
            .supportedOperationTypes = supportedOperationTypes,
            .cachesCompilation = false,
            .bootupTime = 1e-3f,
            .float16Performance = float16Performance,
            .float32Performance = float32Performance,
            .quantized8Performance = quantized8Performance,
    };

    // return
    cb(capabilities);
    return Void();
}

Return<void> SampleDriver::getSupportedSubgraph([[maybe_unused]] const Model& model,
                                                getSupportedSubgraph_cb cb) {
    LOG(DEBUG) << "SampleDriver::getSupportedSubgraph()";
    std::vector<bool> canDo; // TODO implement
    if (validateModel(model)) {
        // TODO
    }
    cb(canDo);
    return Void();
}

Return<sp<IPreparedModel>> SampleDriver::prepareModel(const Model& model) {
    if (!validateModel(model)) {
        return nullptr;
    }
    LOG(DEBUG) << "SampleDriver::prepareModel(" << toString(model) << ")";
    return new SamplePreparedModel(model);
}

Return<DeviceStatus> SampleDriver::getStatus() {
    LOG(DEBUG) << "SampleDriver::getStatus()";
    return DeviceStatus::AVAILABLE;
}

SamplePreparedModel::SamplePreparedModel(const Model& model) {
    // Make a copy of the model, as we need to preserve it.
    mModel = model;
}

static bool mapPools(std::vector<RunTimePoolInfo>* poolInfos, const hidl_vec<hidl_memory>& pools) {
    poolInfos->resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = (*poolInfos)[i];
        poolInfo.memory = mapMemory(pools[i]);
        if (poolInfo.memory == nullptr) {
            LOG(ERROR) << "SampleDriver Can't create shared memory.";
            return false;
        }
        poolInfo.memory->update();
        poolInfo.buffer =
                reinterpret_cast<uint8_t*>(static_cast<void*>(poolInfo.memory->getPointer()));
        if (poolInfo.buffer == nullptr) {
            LOG(ERROR) << "SamplePreparedModel::execute Can't create shared memory.";
            return false;
        }
    }
    return true;
}

Return<bool> SamplePreparedModel::execute(const Request& request) {
    LOG(DEBUG) << "SampleDriver::prepareRequest(" << toString(request) << ")";

    std::vector<RunTimePoolInfo> poolInfo;
    if (!mapPools(&poolInfo, request.pools)) {
        return false;
    }

    CpuExecutor executor;
    int n = executor.run(mModel, request, poolInfo);
    LOG(DEBUG) << "executor.run returned " << n;
    return n == ANEURALNETWORKS_NO_ERROR;
}

} // namespace sample_driver
} // namespace nn
} // namespace android
