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

#define LOG_TAG "Utils"

#include "Utils.h"

#include "NeuralNetworks.h"

namespace android {
namespace nn {

const char* typeNames[ANEURALNETWORKS_NUMBER_DATA_TYPES] = {
            "FLOAT16",
            "FLOAT32",
            "INT8",
            "UINT8",
            "INT16",
            "UINT16",
            "INT32",
            "UINT32",
            "TENSOR_FLOAT16",
            "TENSOR_FLOAT32",
            "TENSOR_SIMMETRICAL_QUANT8",
};

const char* errorNames[] = {
            "NO_ERROR",        "OUT_OF_MEMORY", "INCOMPLETE", "NULL", "BAD_DATA",
            "NOT_IMPLEMENTED",  // TODO remove
};

const char* kOperationNames[ANEURALNETWORKS_NUMBER_OPERATION_TYPES] = {
            "AVERAGE_POOL_FLOAT32",
            "CONCATENATION_FLOAT32",
            "CONV_FLOAT32",
            "DEPTHWISE_CONV_FLOAT32",
            "MAX_POOL_FLOAT32",
            "L2_POOL_FLOAT32",
            "DEPTH_TO_SPACE_FLOAT32",
            "SPACE_TO_DEPTH_FLOAT32",
            "LOCAL_RESPONSE_NORMALIZATION_FLOAT32",
            "SOFTMAX_FLOAT32",
            "RESHAPE_FLOAT32",
            "SPLIT_FLOAT32",
            "FAKE_QUANT_FLOAT32",
            "ADD_FLOAT32",
            "FULLY_CONNECTED_FLOAT32",
            "CAST_FLOAT32",
            "MUL_FLOAT32",
            "L2_NORMALIZATION_FLOAT32",
            "LOGISTIC_FLOAT32",
            "RELU_FLOAT32",
            "RELU6_FLOAT32",
            "RELU1_FLOAT32",
            "TANH_FLOAT32",
            "DEQUANTIZE_FLOAT32",
            "FLOOR_FLOAT32",
            "GATHER_FLOAT32",
            "RESIZE_BILINEAR_FLOAT32",
            "LSH_PROJECTION_FLOAT32",
            "LSTM_FLOAT32",
            "SVDF_FLOAT32",
            "RNN_FLOAT32",
            "N_GRAM_FLOAT32",
            "LOOKUP_FLOAT32",
};

const char* getOperationName(uint32_t opCode) {
    return kOperationNames[opCode];
}

uint32_t sizeOfDataType[ANEURALNETWORKS_NUMBER_DATA_TYPES]{
            2,  // ANEURALNETWORKS_FLOAT16
            4,  // ANEURALNETWORKS_FLOAT32
            1,  // ANEURALNETWORKS_INT8
            1,  // ANEURALNETWORKS_UINT8
            2,  // ANEURALNETWORKS_INT16
            2,  // ANEURALNETWORKS_UINT16
            4,  // ANEURALNETWORKS_INT32
            4,  // ANEURALNETWORKS_UINT32
            2,  // ANEURALNETWORKS_TENSOR_FLOAT16
            4,  // ANEURALNETWORKS_TENSOR_FLOAT32
            1   // ANEURALNETWORKS_TENSOR_SIMMETRICAL_QUANT8
};

uint32_t sizeOfData(uint32_t type, const Range<uint32_t>& dimensions) {
    nnAssert(type < ANEURALNETWORKS_NUMBER_DATA_TYPES);

    uint32_t size = sizeOfDataType[type];
    for (auto d : dimensions) {
        size *= d;
    }
    return size;
}

}  // namespace nn
}  // namespace android
