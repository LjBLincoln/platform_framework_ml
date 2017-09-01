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

// Contains all the entry points to the C Neural Networks API.
// We do basic validation of the operands and then call the class
// that implements the functionality.

#define LOG_TAG "NeuralNetworks"

#include "NeuralNetworks.h"
#include "Manager.h"
#include "Memory.h"
#include "ModelBuilder.h"
#include "RequestBuilder.h"

#include <memory>
#include <vector>

// Make sure the constants defined in the header file have not changed values.
// IMPORTANT: When adding new values, update kNumberOfDataTypes in Utils.h.
static_assert(ANEURALNETWORKS_FLOAT16 == 0, "ANEURALNETWORKS_FLOAT16 may have changed");
static_assert(ANEURALNETWORKS_FLOAT32 == 1, "ANEURALNETWORKS_FLOAT32 may have changed");
static_assert(ANEURALNETWORKS_INT8 == 2, "ANEURALNETWORKS_INT8 may have changed");
static_assert(ANEURALNETWORKS_UINT8 == 3, "ANEURALNETWORKS_UINT8 may have changed");
static_assert(ANEURALNETWORKS_INT16 == 4, "ANEURALNETWORKS_INT16 may have changed");
static_assert(ANEURALNETWORKS_UINT16 == 5, "ANEURALNETWORKS_UINT16 may have changed");
static_assert(ANEURALNETWORKS_INT32 == 6, "ANEURALNETWORKS_INT32 may have changed");
static_assert(ANEURALNETWORKS_UINT32 == 7, "ANEURALNETWORKS_UINT32 may have changed");
static_assert(ANEURALNETWORKS_TENSOR_FLOAT16 == 8,
              "ANEURALNETWORKS_TENSOR_FLOAT16 may have changed");
static_assert(ANEURALNETWORKS_TENSOR_FLOAT32 == 9,
              "ANEURALNETWORKS_TENSOR_FLOAT32 may have changed");
static_assert(ANEURALNETWORKS_TENSOR_INT32 == 10,
              "ANEURALNETWORKS_TENSOR_INT32 may have changed");
static_assert(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM == 11,
              "ANEURALNETWORKS_TENSOR_QUANT8_ASYMM may have changed");

// IMPORTANT: When adding new values, update kNumberOfOperationTypes in Utils.h.
static_assert(ANEURALNETWORKS_OEM_OPERATION == 0, "ANEURALNETWORKS_OEM_OPERATION may have changed");
static_assert(ANEURALNETWORKS_ADD == 1, "ANEURALNETWORKS_ADD may have changed");
static_assert(ANEURALNETWORKS_AVERAGE_POOL == 2, "ANEURALNETWORKS_AVERAGE_POOL may have changed");
static_assert(ANEURALNETWORKS_CAST == 3, "ANEURALNETWORKS_CAST may have changed");
static_assert(ANEURALNETWORKS_CONCATENATION == 4, "ANEURALNETWORKS_CONCATENATION may have changed");
static_assert(ANEURALNETWORKS_CONV == 5, "ANEURALNETWORKS_CONV may have changed");
static_assert(ANEURALNETWORKS_DEPTHWISE_CONV == 6,
              "ANEURALNETWORKS_DEPTHWISE_CONV may have changed");
static_assert(ANEURALNETWORKS_DEPTH_TO_SPACE == 7,
              "ANEURALNETWORKS_DEPTH_TO_SPACE may have changed");
static_assert(ANEURALNETWORKS_DEQUANTIZE == 8, "ANEURALNETWORKS_DEQUANTIZE may have changed");
static_assert(ANEURALNETWORKS_EMBEDDING_LOOKUP == 9,
              "ANEURALNETWORKS_EMBEDDING_LOOKUP may have changed");
static_assert(ANEURALNETWORKS_FAKE_QUANT == 10, "ANEURALNETWORKS_FAKE_QUANT may have changed");
static_assert(ANEURALNETWORKS_FLOOR == 11, "ANEURALNETWORKS_FLOOR may have changed");
static_assert(ANEURALNETWORKS_FULLY_CONNECTED == 12,
              "ANEURALNETWORKS_FULLY_CONNECTED may have changed");
static_assert(ANEURALNETWORKS_GATHER == 13, "ANEURALNETWORKS_GATHER may have changed");
static_assert(ANEURALNETWORKS_HASHTABLE_LOOKUP == 14,
              "ANEURALNETWORKS_HASHTABLE_LOOKUP may have changed");
static_assert(ANEURALNETWORKS_L2_NORMALIZATION == 15,
              "ANEURALNETWORKS_L2_NORMALIZATION may have changed");
static_assert(ANEURALNETWORKS_L2_POOL == 16, "ANEURALNETWORKS_L2_POOL may have changed");
static_assert(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION == 17,
              "ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION may have changed");
static_assert(ANEURALNETWORKS_LOGISTIC == 18, "ANEURALNETWORKS_LOGISTIC may have changed");
static_assert(ANEURALNETWORKS_LSH_PROJECTION == 19,
              "ANEURALNETWORKS_LSH_PROJECTION may have changed");
static_assert(ANEURALNETWORKS_LSTM == 20, "ANEURALNETWORKS_LSTM may have changed");
static_assert(ANEURALNETWORKS_MAX_POOL == 21, "ANEURALNETWORKS_MAX_POOL may have changed");
static_assert(ANEURALNETWORKS_MUL == 22, "ANEURALNETWORKS_MUL may have changed");
static_assert(ANEURALNETWORKS_RELU == 23, "ANEURALNETWORKS_RELU may have changed");
static_assert(ANEURALNETWORKS_RELU1 == 24, "ANEURALNETWORKS_RELU1 may have changed");
static_assert(ANEURALNETWORKS_RELU6 == 25, "ANEURALNETWORKS_RELU6 may have changed");
static_assert(ANEURALNETWORKS_RESHAPE == 26, "ANEURALNETWORKS_RESHAPE may have changed");
static_assert(ANEURALNETWORKS_RESIZE_BILINEAR == 27,
              "ANEURALNETWORKS_RESIZE_BILINEAR may have changed");
static_assert(ANEURALNETWORKS_RNN == 28, "ANEURALNETWORKS_RNN may have changed");
static_assert(ANEURALNETWORKS_SOFTMAX == 29, "ANEURALNETWORKS_SOFTMAX may have changed");
static_assert(ANEURALNETWORKS_SPACE_TO_DEPTH == 30,
              "ANEURALNETWORKS_SPACE_TO_DEPTH may have changed");
static_assert(ANEURALNETWORKS_SPLIT == 31, "ANEURALNETWORKS_SPLIT may have changed");
static_assert(ANEURALNETWORKS_SVDF == 32, "ANEURALNETWORKS_SVDF may have changed");
static_assert(ANEURALNETWORKS_TANH == 33, "ANEURALNETWORKS_TANH may have changed");

static_assert(ANEURALNETWORKS_PREFER_LOW_POWER == 0,
              "ANEURALNETWORKS_PREFER_LOW_POWER may have changed");
static_assert(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER == 1,
              "ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER may have changed");
static_assert(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED == 2,
              "ANEURALNETWORKS_PREFER_SUSTAINED_SPEED may have changed");

static_assert(ANEURALNETWORKS_NO_ERROR == 0, "ANEURALNETWORKS_NO_ERROR may have changed");
static_assert(ANEURALNETWORKS_OUT_OF_MEMORY == 1, "ANEURALNETWORKS_OUT_OF_MEMORY may have changed");
static_assert(ANEURALNETWORKS_INCOMPLETE == 2, "ANEURALNETWORKS_INCOMPLETE may have changed");
static_assert(ANEURALNETWORKS_UNEXPECTED_NULL == 3,
              "ANEURALNETWORKS_UNEXPECTED_NULL may have changed");
static_assert(ANEURALNETWORKS_BAD_DATA == 4, "ANEURALNETWORKS_BAD_DATA may have changed");
static_assert(ANEURALNETWORKS_OP_FAILED == 5, "ANEURALNETWORKS_OP_FAILED may have changed");

// Make sure that the constants are compatible with the values defined in
// hardware/interfaces/neuralnetworks/1.0/types.hal.
static_assert(static_cast<uint32_t>(OperandType::FLOAT16) == ANEURALNETWORKS_FLOAT16,
              "FLOAT16 != ANEURALNETWORKS_FLOAT16");
static_assert(static_cast<uint32_t>(OperandType::FLOAT32) == ANEURALNETWORKS_FLOAT32,
              "FLOAT32 != ANEURALNETWORKS_FLOAT32");
static_assert(static_cast<uint32_t>(OperandType::INT8) == ANEURALNETWORKS_INT8,
              "INT8 != ANEURALNETWORKS_INT8");
static_assert(static_cast<uint32_t>(OperandType::UINT8) == ANEURALNETWORKS_UINT8,
              "UINT8 != ANEURALNETWORKS_UINT8");
static_assert(static_cast<uint32_t>(OperandType::INT16) == ANEURALNETWORKS_INT16,
              "INT16 != ANEURALNETWORKS_INT16");
static_assert(static_cast<uint32_t>(OperandType::UINT16) == ANEURALNETWORKS_UINT16,
              "UINT16 != ANEURALNETWORKS_UINT16");
static_assert(static_cast<uint32_t>(OperandType::INT32) == ANEURALNETWORKS_INT32,
              "INT32 != ANEURALNETWORKS_INT32");
static_assert(static_cast<uint32_t>(OperandType::UINT32) == ANEURALNETWORKS_UINT32,
              "UINT32 != ANEURALNETWORKS_UINT32");
static_assert(static_cast<uint32_t>(OperandType::TENSOR_FLOAT16) == ANEURALNETWORKS_TENSOR_FLOAT16,
              "TENSOR_FLOAT16 != ANEURALNETWORKS_TENSOR_FLOAT16");
static_assert(static_cast<uint32_t>(OperandType::TENSOR_FLOAT32) == ANEURALNETWORKS_TENSOR_FLOAT32,
              "TENSOR_FLOAT32 != ANEURALNETWORKS_TENSOR_FLOAT32");
static_assert(static_cast<uint32_t>(OperandType::TENSOR_QUANT8_ASYMM) ==
                          ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
              "TENSOR_QUANT8_ASYMM != ANEURALNETWORKS_TENSOR_QUANT8_ASYMM");

static_assert(static_cast<uint32_t>(OperationType::ADD) == ANEURALNETWORKS_ADD,
              "OperationType::ADD != ANEURALNETWORKS_ADD");
static_assert(static_cast<uint32_t>(OperationType::AVERAGE_POOL) == ANEURALNETWORKS_AVERAGE_POOL,
              "OperationType::AVERAGE_POOL != ANEURALNETWORKS_AVERAGE_POOL");
static_assert(static_cast<uint32_t>(OperationType::CAST) == ANEURALNETWORKS_CAST,
              "OperationType::CAST != ANEURALNETWORKS_CAST");
static_assert(static_cast<uint32_t>(OperationType::CONCATENATION) == ANEURALNETWORKS_CONCATENATION,
              "OperationType::CONCATENATION != ANEURALNETWORKS_CONCATENATION");
static_assert(static_cast<uint32_t>(OperationType::CONV) == ANEURALNETWORKS_CONV,
              "OperationType::CONV != ANEURALNETWORKS_CONV");
static_assert(static_cast<uint32_t>(OperationType::DEPTHWISE_CONV) == ANEURALNETWORKS_DEPTHWISE_CONV,
              "OperationType::DEPTHWISE_CONV != ANEURALNETWORKS_DEPTHWISE_CONV");
static_assert(static_cast<uint32_t>(OperationType::DEPTH_TO_SPACE) == ANEURALNETWORKS_DEPTH_TO_SPACE,
              "OperationType::DEPTH_TO_SPACE != ANEURALNETWORKS_DEPTH_TO_SPACE");
static_assert(static_cast<uint32_t>(OperationType::DEQUANTIZE) == ANEURALNETWORKS_DEQUANTIZE,
              "OperationType::DEQUANTIZE != ANEURALNETWORKS_DEQUANTIZE");
static_assert(static_cast<uint32_t>(OperationType::EMBEDDING_LOOKUP) ==
                          ANEURALNETWORKS_EMBEDDING_LOOKUP,
              "OperationType::EMBEDDING_LOOKUP != ANEURALNETWORKS_EMBEDDING_LOOKUP");
static_assert(static_cast<uint32_t>(OperationType::FAKE_QUANT) == ANEURALNETWORKS_FAKE_QUANT,
              "OperationType::FAKE_QUANT != ANEURALNETWORKS_FAKE_QUANT");
static_assert(static_cast<uint32_t>(OperationType::FLOOR) == ANEURALNETWORKS_FLOOR,
              "OperationType::FLOOR != ANEURALNETWORKS_FLOOR");
static_assert(static_cast<uint32_t>(OperationType::FULLY_CONNECTED) ==
                          ANEURALNETWORKS_FULLY_CONNECTED,
              "OperationType::FULLY_CONNECTED != ANEURALNETWORKS_FULLY_CONNECTED");
static_assert(static_cast<uint32_t>(OperationType::GATHER) == ANEURALNETWORKS_GATHER,
              "OperationType::GATHER != ANEURALNETWORKS_GATHER");
static_assert(static_cast<uint32_t>(OperationType::HASHTABLE_LOOKUP) ==
                          ANEURALNETWORKS_HASHTABLE_LOOKUP,
              "OperationType::HASHTABLE_LOOKUP != ANEURALNETWORKS_HASHTABLE_LOOKUP");
static_assert(static_cast<uint32_t>(OperationType::L2_NORMALIZATION) ==
                          ANEURALNETWORKS_L2_NORMALIZATION,
              "OperationType::L2_NORMALIZATION != ANEURALNETWORKS_L2_NORMALIZATION");
static_assert(static_cast<uint32_t>(OperationType::L2_POOL) == ANEURALNETWORKS_L2_POOL,
              "OperationType::L2_POOL != ANEURALNETWORKS_L2_POOL");
static_assert(static_cast<uint32_t>(OperationType::LOCAL_RESPONSE_NORMALIZATION) ==
                          ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION,
              "OperationType::LOCAL_RESPONSE_NORMALIZATION != "
              "ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION");
static_assert(static_cast<uint32_t>(OperationType::LOGISTIC) == ANEURALNETWORKS_LOGISTIC,
              "OperationType::LOGISTIC != ANEURALNETWORKS_LOGISTIC");
static_assert(static_cast<uint32_t>(OperationType::LSH_PROJECTION) == ANEURALNETWORKS_LSH_PROJECTION,
              "OperationType::LSH_PROJECTION != ANEURALNETWORKS_LSH_PROJECTION");
static_assert(static_cast<uint32_t>(OperationType::LSTM) == ANEURALNETWORKS_LSTM,
              "OperationType::LSTM != ANEURALNETWORKS_LSTM");
static_assert(static_cast<uint32_t>(OperationType::MAX_POOL) == ANEURALNETWORKS_MAX_POOL,
              "OperationType::MAX_POOL != ANEURALNETWORKS_MAX_POOL");
static_assert(static_cast<uint32_t>(OperationType::MUL) == ANEURALNETWORKS_MUL,
              "OperationType::MUL != ANEURALNETWORKS_MUL");
static_assert(static_cast<uint32_t>(OperationType::RELU) == ANEURALNETWORKS_RELU,
              "OperationType::RELU != ANEURALNETWORKS_RELU");
static_assert(static_cast<uint32_t>(OperationType::RELU1) == ANEURALNETWORKS_RELU1,
              "OperationType::RELU1 != ANEURALNETWORKS_RELU1");
static_assert(static_cast<uint32_t>(OperationType::RELU6) == ANEURALNETWORKS_RELU6,
              "OperationType::RELU6 != ANEURALNETWORKS_RELU6");
static_assert(static_cast<uint32_t>(OperationType::RESHAPE) == ANEURALNETWORKS_RESHAPE,
              "OperationType::RESHAPE != ANEURALNETWORKS_RESHAPE");
static_assert(static_cast<uint32_t>(OperationType::RESIZE_BILINEAR) ==
                          ANEURALNETWORKS_RESIZE_BILINEAR,
              "OperationType::RESIZE_BILINEAR != ANEURALNETWORKS_RESIZE_BILINEAR");
static_assert(static_cast<uint32_t>(OperationType::RNN) == ANEURALNETWORKS_RNN,
              "OperationType::RNN != ANEURALNETWORKS_RNN");
static_assert(static_cast<uint32_t>(OperationType::SOFTMAX) == ANEURALNETWORKS_SOFTMAX,
              "OperationType::SOFTMAX != ANEURALNETWORKS_SOFTMAX");
static_assert(static_cast<uint32_t>(OperationType::SPACE_TO_DEPTH) == ANEURALNETWORKS_SPACE_TO_DEPTH,
              "OperationType::SPACE_TO_DEPTH != ANEURALNETWORKS_SPACE_TO_DEPTH");
static_assert(static_cast<uint32_t>(OperationType::SPLIT) == ANEURALNETWORKS_SPLIT,
              "OperationType::SPLIT != ANEURALNETWORKS_SPLIT");
static_assert(static_cast<uint32_t>(OperationType::SVDF) == ANEURALNETWORKS_SVDF,
              "OperationType::SVDF != ANEURALNETWORKS_SVDF");
static_assert(static_cast<uint32_t>(OperationType::TANH) == ANEURALNETWORKS_TANH,
              "OperationType::TANH != ANEURALNETWORKS_TANH");

using namespace android::nn;

// Validates the type. The used dimensions can be underspecified.
static int ValidateOperandType(const ANeuralNetworksOperandType& type, const char* tag,
                               bool allowPartial) {
    if (!allowPartial) {
        for (uint32_t i = 0; i < type.dimensions.count; i++) {
            if (type.dimensions.data[i] == 0) {
                LOG(ERROR) << tag << " OperandType invalid dimensions[" << i
                           << "] = " << type.dimensions.data[i];
                return ANEURALNETWORKS_BAD_DATA;
            }
        }
    }
    if (type.type >= kNumberOfDataTypes) {
        LOG(ERROR) << tag << " OperandType invalid type " << type.type;
        return ANEURALNETWORKS_BAD_DATA;
    }
    /* TODO validate the quantization info.
    if (type.offset != 0.f && type.scale == 0.f) {
        LOG(ERROR) << ("%s OperandType invalid offset %f and scale %f", tag, type.offset,
    type.scale); return ANEURALNETWORKS_BAD_DATA;
    }
    if (type.scale != 0.f &&
        (type.type == ANEURALNETWORKS_FLOAT16 ||
         type.type != ANEURALNETWORKS_FLOAT32)) {
            LOG(ERROR) << ("%s OperandType scale %f with float type %u", tag, type.scale,
    type.type); return ANEURALNETWORKS_BAD_DATA;
        }
     */
    return ANEURALNETWORKS_NO_ERROR;
}

static int ValidateOperandList(const ANeuralNetworksIntList& list, uint32_t count,
                               const char* tag) {
    for (uint32_t i = 0; i < list.count; i++) {
        if (list.data[i] >= count) {
            LOG(ERROR) << tag << " invalid operand index at " << i << " = " << list.data[i]
                       << ", count " << count;
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksInitialize() {
    DeviceManager::get()->initialize();
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksShutdown() {
    DeviceManager::get()->shutdown();
}

int ANeuralNetworksMemory_create(size_t size, ANeuralNetworksMemory** memory) {
    if (!memory) {
        LOG(ERROR) << "ANeuralNetworksMemory_create passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (size > 0xFFFFFFFF) {
        LOG(ERROR) << "ANeuralNetworksMemory_create size exceeds max " << size;
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t size32 = static_cast<uint32_t>(size);
    *memory = nullptr;
    std::unique_ptr<Memory> m = std::make_unique<Memory>(Memory());
    if (m == nullptr) {
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    int n = m->create(size32);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(m.release());
    return ANEURALNETWORKS_NO_ERROR;
}

/* TODO
int ANeuralNetworksMemory_createFromHidlMemory(hidl_memory hidlMemory,
                                               ANeuralNetworksMemory** memory) {
    if (!memory) {
        LOG(ERROR) << "ANeuralNetworksMemory_create passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    *memory = nullptr;
    std::unique_ptr<Memory> m = std::make_unique<Memory>(Memory());
    if (m == nullptr) {
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    int n = m->setFromHidlMemory(hidlMemory);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(m.release());
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromFd(int fd, ANeuralNetworksMemory** memory) {
    if (fd < 0) {
        LOG(ERROR) << "ANeuralNetworksMemory_createFromFd invalid fd " << fd;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    *memory = nullptr;
    std::unique_ptr<Memory> m = std::make_unique<Memory>(Memory());
    if (m == nullptr) {
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    int n = m->setFromFd(fd);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(m.release());
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromGrallocBuffer(buffer_handle_t buffer,
                                                  ANeuralNetworksMemory** memory) {
    *memory = nullptr;
    // TODO implement
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksMemory_createFromHardwareBuffer(AHardwareBuffer* buffer,
                                                   ANeuralNetworksMemory** memory) {
    *memory = nullptr;
    // TODO implement
    return ANEURALNETWORKS_NO_ERROR;
}
*/

uint8_t* ANeuralNetworksMemory_getPointer(ANeuralNetworksMemory* memory) {
    if (!memory) {
        LOG(ERROR) << "ANeuralNetworksMemory_getPoiter passed a nullptr";
        return nullptr;
    }
    Memory* m = reinterpret_cast<Memory*>(memory);
    return m->getPointer();
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
    // No validation.  Free of nullptr is valid.
    Memory* m = reinterpret_cast<Memory*>(memory);
    delete m;
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
    if (!model) {
        LOG(ERROR) << "ANeuralNetworksModel_create passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = new ModelBuilder();
    if (m == nullptr) {
        *model = nullptr;
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    *model = reinterpret_cast<ANeuralNetworksModel*>(m);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
    // No validation.  Free of nullptr is valid.
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    delete m;
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type) {
    if (!model || !type) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperand passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    int n = ValidateOperandType(*type, "ANeuralNetworksModel_addOperand", true);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    return m->addOperand(*type);
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length) {
    if (!model || !buffer) {
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValue passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->setOperandValue(index, buffer, length);
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   uint32_t offset, size_t length) {
    if (!model || !memory) {
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValue passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    const Memory* mem = reinterpret_cast<const Memory*>(memory);
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->setOperandValueFromMemory(index, mem, offset, length);
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type,
                                      ANeuralNetworksIntList* inputs,
                                      ANeuralNetworksIntList* outputs) {
    if (!model || !inputs || !outputs) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperation passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    if (type >= kNumberOfOperationTypes) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperation invalid operations type " << type;
        return ANEURALNETWORKS_BAD_DATA;
    }
    int n = ValidateOperandList(*inputs, m->operandCount(),
                                "ANeuralNetworksModel_addOperation inputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = ValidateOperandList(*outputs, m->operandCount(),
                            "ANeuralNetworksModel_addOperation outputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    return m->addOperation(type, inputs, outputs);
}

int ANeuralNetworksModel_setInputsAndOutputs(ANeuralNetworksModel* model,
                                             ANeuralNetworksIntList* inputs,
                                             ANeuralNetworksIntList* outputs) {
    if (!model || !inputs || !outputs) {
        LOG(ERROR) << ("ANeuralNetworksModel_setInputsAndOutputs passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    int n = ValidateOperandList(*inputs, m->operandCount(),
                                "ANeuralNetworksModel_setInputsAndOutputs inputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = ValidateOperandList(*outputs, m->operandCount(),
                            "ANeuralNetworksModel_setInputsAndOutputs outputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    return m->setInputsAndOutputs(inputs, outputs);
}

int ANeuralNetworksRequest_create(ANeuralNetworksModel* model, ANeuralNetworksRequest** request) {
    if (!model || !request) {
        LOG(ERROR) << "ANeuralNetworksRequest_create passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    RequestBuilder* r = m->createRequest();
    if (r == nullptr) {
        *request = nullptr;
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    *request = reinterpret_cast<ANeuralNetworksRequest*>(r);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksRequest_free(ANeuralNetworksRequest* request) {
    // No validation.  Free of nullptr is valid.
    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    delete r;
}

int ANeuralNetworksRequest_setPreference(ANeuralNetworksRequest* request, uint32_t preference) {
    if (!request) {
        LOG(ERROR) << "ANeuralNetworksRequest_setPreference passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (preference >= kNumberOfPreferences) {
        LOG(ERROR) << "ANeuralNetworksRequest_setPreference invalid preference " << preference;
        return ANEURALNETWORKS_BAD_DATA;
    }

    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    r->setPreference(preference);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksRequest_setInput(ANeuralNetworksRequest* request, int32_t index,
                                    const ANeuralNetworksOperandType* type, const void* buffer,
                                    size_t length) {
    // TODO: For a non-optional input, also verify that buffer is not null.
    if (!request) {
        LOG(ERROR) << "ANeuralNetworksRequest_setInput passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (type != nullptr) {
        int n = ValidateOperandType(*type, "ANeuralNetworksRequest_setInput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
        LOG(ERROR) << "ANeuralNetworksRequest_setInput input exceeds max length " << length;
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t l = static_cast<uint32_t>(length);
    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    return r->setInput(index, type, buffer, l);
}

int ANeuralNetworksRequest_setInputFromMemory(ANeuralNetworksRequest* request, int32_t index,
                                              const ANeuralNetworksOperandType* type,
                                              const ANeuralNetworksMemory* memory, uint32_t offset,
                                              uint32_t length) {
    if (!request || !memory) {
        LOG(ERROR) << "ANeuralNetworksRequest_setInputFromMemory passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    const Memory* m = reinterpret_cast<const Memory*>(memory);
    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    return r->setInputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksRequest_setOutput(ANeuralNetworksRequest* request, int32_t index,
                                     const ANeuralNetworksOperandType* type, void* buffer,
                                     size_t length) {
    if (!request || !buffer) {
        LOG(ERROR) << "ANeuralNetworksRequest_setOutput passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (type != nullptr) {
        int n = ValidateOperandType(*type, "ANeuralNetworksRequest_setOutput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
        LOG(ERROR) << "ANeuralNetworksRequest_setOutput input exceeds max length " << length;
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t l = static_cast<uint32_t>(length);

    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    return r->setOutput(index, type, buffer, l);
}

int ANeuralNetworksRequest_setOutputFromMemory(ANeuralNetworksRequest* request, int32_t index,
                                               const ANeuralNetworksOperandType* type,
                                               const ANeuralNetworksMemory* memory, uint32_t offset,
                                               uint32_t length) {
    if (!request || !memory) {
        LOG(ERROR) << "ANeuralNetworksRequest_setOutputFromMemory passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    const Memory* m = reinterpret_cast<const Memory*>(memory);
    return r->setOutputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksRequest_startCompute(ANeuralNetworksRequest* request,
                                        ANeuralNetworksEvent** event) {
    if (!request || !event) {
        LOG(ERROR) << "ANeuralNetworksRequest_startCompute passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    Event* e = nullptr;
    int n = r->startCompute(&e);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *event = reinterpret_cast<ANeuralNetworksEvent*>(e);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
    if (event == nullptr) {
        LOG(ERROR) << "ANeuralNetworksEvent_wait passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    Event* e = reinterpret_cast<Event*>(event);
    e->wait();
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
    // No validation.  Free of nullptr is valid.
    Event* e = reinterpret_cast<Event*>(event);
    delete e;
}
