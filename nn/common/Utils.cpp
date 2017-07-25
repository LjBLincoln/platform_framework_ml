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

#include <android-base/logging.h>

using ::android::hidl::allocator::V1_0::IAllocator;

namespace android {
namespace nn {

const char* kTypeNames[ANEURALNETWORKS_NUMBER_DATA_TYPES] = {
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

// TODO Check if this useful
const char* kErrorNames[] = {
        "NO_ERROR",        "OUT_OF_MEMORY", "INCOMPLETE", "NULL", "BAD_DATA",
        "NOT_IMPLEMENTED", // TODO remove
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

const char* getOperationName(OperationType type) {
    uint32_t n = static_cast<uint32_t>(type);
    nnAssert(n < ANEURALNETWORKS_NUMBER_OPERATION_TYPES);
    return kOperationNames[n];
}

const uint32_t kSizeOfDataType[ANEURALNETWORKS_NUMBER_DATA_TYPES]{
        2, // ANEURALNETWORKS_FLOAT16
        4, // ANEURALNETWORKS_FLOAT32
        1, // ANEURALNETWORKS_INT8
        1, // ANEURALNETWORKS_UINT8
        2, // ANEURALNETWORKS_INT16
        2, // ANEURALNETWORKS_UINT16
        4, // ANEURALNETWORKS_INT32
        4, // ANEURALNETWORKS_UINT32
        2, // ANEURALNETWORKS_TENSOR_FLOAT16
        4, // ANEURALNETWORKS_TENSOR_FLOAT32
        1  // ANEURALNETWORKS_TENSOR_SIMMETRICAL_QUANT8
};

uint32_t sizeOfData(OperandType type, const std::vector<uint32_t>& dimensions) {
    int n = static_cast<int>(type);
    nnAssert(n < ANEURALNETWORKS_NUMBER_DATA_TYPES);

    uint32_t size = kSizeOfDataType[n];
    for (auto d : dimensions) {
        size *= d;
    }
    return size;
}

hidl_memory allocateSharedMemory(int64_t size) {
    hidl_memory memory;

    // TODO: should we align memory size to nearest page? doesn't seem necessary...
    const std::string& type = "ashmem";
    sp<IAllocator> allocator = IAllocator::getService(type);
    allocator->allocate(size, [&](bool success, const hidl_memory& mem) {
        if (!success) {
            LOG(ERROR) << "unable to allocate " << size << " bytes of " << type;
        } else {
            memory = mem;
        }
    });

    return memory;
}

uint32_t alignBytesNeeded(uint32_t index, size_t length) {
    uint32_t pattern;
    if (length < 2) {
        pattern = 0; // No alignment necessary
    } else if (length < 4) {
        pattern = 1; // Align on 2-byte boundary
    } else {
        pattern = 3; // Align on 4-byte boundary
    }
    uint32_t extra = (~(index - 1)) & pattern;
    return extra;
}

static bool validOperandIndexes(const hidl_vec<uint32_t> indexes, size_t operandCount) {
    for (uint32_t i : indexes) {
        if (i >= operandCount) {
            LOG(ERROR) << "Index out of range " << i << "/" << operandCount;
            return false;
        }
    }
    return true;
}

static bool validOperands(const hidl_vec<Operand>& operands, const hidl_vec<uint8_t>& operandValues,
                          size_t poolCount) {
    for (auto& operand : operands) {
        if (static_cast<uint32_t>(operand.type) >= HAL_NUM_OPERAND_TYPES) {
            LOG(ERROR) << "Invalid operand type " << toString(operand.type);
            return false;
        }
        /* TODO validate dim with type
        if (!validOperandIndexes(operand.dimensions, mDimensions)) {
            return false;
        }
        */
        if (operand.location.poolIndex ==
            static_cast<uint32_t>(LocationValues::LOCATION_SAME_BLOCK)) {
            if (operand.location.offset + operand.location.length > operandValues.size()) {
                LOG(ERROR) << "OperandValue location out of range.  Starts at "
                           << operand.location.offset << ", length " << operand.location.length
                           << ", max " << operandValues.size();
                return false;
            }
        } else if (operand.location.poolIndex ==
                   static_cast<uint32_t>(LocationValues::LOCATION_AT_RUN_TIME)) {
            if (operand.location.offset != 0 || operand.location.length != 0) {
                LOG(ERROR) << "Unexpected offset " << operand.location.offset << " or length "
                           << operand.location.length << " for runtime location.";
                return false;
            }
        } else if (operand.location.poolIndex >= poolCount) {
            // TODO: Revisit when we add support for multiple pools.
            LOG(ERROR) << "Invalid poolIndex " << operand.location.poolIndex << "/" << poolCount;
            return false;
        }
    }
    return true;
}

static bool validOperations(const hidl_vec<Operation>& operations, size_t operandCount) {
    for (auto& op : operations) {
        if (static_cast<uint32_t>(op.type) >= HAL_NUM_OPERATION_TYPES) {
            LOG(ERROR) << "Invalid operation type " << toString(op.type);
            return false;
        }
        if (!validOperandIndexes(op.inputs, operandCount) ||
            !validOperandIndexes(op.outputs, operandCount)) {
            return false;
        }
    }
    return true;
}

// TODO doublecheck
// TODO also do validateRequest
bool validateModel(const Model& model) {
    const size_t operandCount = model.operands.size();
    return (validOperands(model.operands, model.operandValues, model.pools.size()) &&
            validOperations(model.operations, operandCount) &&
            validOperandIndexes(model.inputIndexes, operandCount) &&
            validOperandIndexes(model.outputIndexes, operandCount));
}

} // namespace nn
} // namespace android
