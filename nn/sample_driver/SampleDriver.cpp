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
    SetMinimumLogSeverity(base::VERBOSE);
    LOG(DEBUG) << "SampleDriver::initialize()";

    // Our driver supports every op.
    static hidl_vec<OperationTuple> supportedOperationTuples{
            {OperationType::AVERAGE_POOL, OperandType::TENSOR_FLOAT32},
            {OperationType::CONCATENATION, OperandType::TENSOR_FLOAT32},
            {OperationType::CONV, OperandType::TENSOR_FLOAT32},
            {OperationType::DEPTHWISE_CONV, OperandType::TENSOR_FLOAT32},
            {OperationType::MAX_POOL, OperandType::TENSOR_FLOAT32},
            {OperationType::L2_POOL, OperandType::TENSOR_FLOAT32},
            {OperationType::DEPTH_TO_SPACE, OperandType::TENSOR_FLOAT32},
            {OperationType::SPACE_TO_DEPTH, OperandType::TENSOR_FLOAT32},
            {OperationType::LOCAL_RESPONSE_NORMALIZATION, OperandType::TENSOR_FLOAT32},
            {OperationType::SOFTMAX, OperandType::TENSOR_FLOAT32},
            {OperationType::RESHAPE, OperandType::TENSOR_FLOAT32},
            {OperationType::SPLIT, OperandType::TENSOR_FLOAT32},
            {OperationType::FAKE_QUANT, OperandType::TENSOR_FLOAT32},
            {OperationType::ADD, OperandType::TENSOR_FLOAT32},
            {OperationType::FULLY_CONNECTED, OperandType::TENSOR_FLOAT32},
            {OperationType::CAST, OperandType::TENSOR_FLOAT32},
            {OperationType::MUL, OperandType::TENSOR_FLOAT32},
            {OperationType::L2_NORMALIZATION, OperandType::TENSOR_FLOAT32},
            {OperationType::LOGISTIC, OperandType::TENSOR_FLOAT32},
            {OperationType::RELU, OperandType::TENSOR_FLOAT32},
            {OperationType::RELU6, OperandType::TENSOR_FLOAT32},
            {OperationType::RELU1, OperandType::TENSOR_FLOAT32},
            {OperationType::TANH, OperandType::TENSOR_FLOAT32},
            {OperationType::DEQUANTIZE, OperandType::TENSOR_FLOAT32},
            {OperationType::FLOOR, OperandType::TENSOR_FLOAT32},
            {OperationType::GATHER, OperandType::TENSOR_FLOAT32},
            {OperationType::RESIZE_BILINEAR, OperandType::TENSOR_FLOAT32},
            {OperationType::LSH_PROJECTION, OperandType::TENSOR_FLOAT32},
            {OperationType::LSTM, OperandType::TENSOR_FLOAT32},
            {OperationType::SVDF, OperandType::TENSOR_FLOAT32},
            {OperationType::RNN, OperandType::TENSOR_FLOAT32},
            {OperationType::N_GRAM, OperandType::TENSOR_FLOAT32},
            {OperationType::EMBEDDING_LOOKUP, OperandType::TENSOR_FLOAT32},
            {OperationType::HASHTABLE_LOOKUP, OperandType::TENSOR_FLOAT32},

            {OperationType::AVERAGE_POOL, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::CONCATENATION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::CONV, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::DEPTHWISE_CONV, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::MAX_POOL, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::L2_POOL, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::DEPTH_TO_SPACE, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::SPACE_TO_DEPTH, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LOCAL_RESPONSE_NORMALIZATION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::SOFTMAX, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RESHAPE, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::SPLIT, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::FAKE_QUANT, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::ADD, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::FULLY_CONNECTED, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::CAST, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::MUL, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::L2_NORMALIZATION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LOGISTIC, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RELU, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RELU6, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RELU1, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::TANH, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::DEQUANTIZE, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::FLOOR, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::GATHER, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RESIZE_BILINEAR, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LSH_PROJECTION, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::LSTM, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::SVDF, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::RNN, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::N_GRAM, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::EMBEDDING_LOOKUP, OperandType::TENSOR_QUANT8_ASYMM},
            {OperationType::HASHTABLE_LOOKUP, OperandType::TENSOR_QUANT8_ASYMM},
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
            .supportedOperationTuples = supportedOperationTuples,
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
    LOG(DEBUG) << "SampleDriver::prepareModel(" << toString(model) << ")"; // TODO errror
    if (!validateModel(model)) {
        return nullptr;
    }
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
        if (!poolInfo.set(pools[i])) {
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
