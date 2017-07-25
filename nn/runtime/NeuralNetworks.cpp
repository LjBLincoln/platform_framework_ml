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
#include "ModelBuilder.h"
#include "RequestBuilder.h"

#include <vector>

// Make sure the constants defined in the header file have not changed values.
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
static_assert(ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8 == 10,
              "ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8 may have changed");

// Ensure that the constants are compatible with the values defined in the hal files.
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
static_assert(static_cast<uint32_t>(OperandType::TENSOR_SYMMETRICAL_QUANT8) ==
                      ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8,
              "TENSOR_SYMMETRICAL_QUANT8 != ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8");

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
    if (type.type >= ANEURALNETWORKS_NUMBER_DATA_TYPES) {
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

int ANeuralNetworksModel_createBaselineModel(ANeuralNetworksModel** model, uint32_t modelId) {
    if (!model) {
        LOG(ERROR) << "ANeuralNetworksModel_create passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (modelId >= ANEURALNETWORKS_NUMBER_BASELINE_MODELS) {
        LOG(ERROR) << "ANeuralNetworksModel_createBaselineModel invalid modelId " << modelId;
        return ANEURALNETWORKS_BAD_DATA;
    }

    ModelBuilder* m = new ModelBuilder();
    if (m == nullptr) {
        *model = nullptr;
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    /* TODO uint32_t n = m->loadBaseLineModel(modelId);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        delete m;
        return n;
    }
     */
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

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type,
                                      ANeuralNetworksIntList* inputs,
                                      ANeuralNetworksIntList* outputs) {
    if (!model || !inputs || !outputs) {
        LOG(ERROR) << "ANeuralNetworksModel_addOperation passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    if (type >= ANEURALNETWORKS_NUMBER_OPERATION_TYPES) {
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

int ANeuralNetworksModel_addSubModel(ANeuralNetworksModel* model,
                                     const ANeuralNetworksModel* submodel,
                                     ANeuralNetworksIntList* inputs,
                                     ANeuralNetworksIntList* outputs) {
    if (!model || !submodel) {
        LOG(ERROR) << "ANeuralNetworksModel_addSubModel passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    int n = ValidateOperandList(*inputs, m->operandCount(),
                                "ANeuralNetworksModel_addSubModel inputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = ValidateOperandList(*outputs, m->operandCount(),
                            "ANeuralNetworksModel_addSubModel outputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
}

int ANeuralNetworksModel_setBaselineId(ANeuralNetworksModel* model, uint32_t baseLineId) {
    if (!model) {
        LOG(ERROR) << "ANeuralNetworksModel_setBaselineId passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (baseLineId >= ANEURALNETWORKS_NUMBER_BASELINE_MODELS) {
        LOG(ERROR) << "ANeuralNetworksModel_setBaselineId invalid baselineId " << baseLineId;
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO implement
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
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
    if (preference >= ANEURALNETWORKS_NUMBER_PREFERENCES) {
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
    if (!request || !buffer) {
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
    }
    uint32_t l = static_cast<uint32_t>(length);
    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    return r->setInput(index, type, buffer, l);
}

int ANeuralNetworksRequest_setInputFromHardwareBuffer(ANeuralNetworksRequest* request,
                                                      int32_t index,
                                                      const ANeuralNetworksOperandType* type,
                                                      const AHardwareBuffer* buffer) {
    if (!request || !type || !buffer) {
        LOG(ERROR) << "ANeuralNetworksRequest_setInputFromHardwareBuffer passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    return r->setInputFromHardwareBuffer(index, type, buffer);
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
    }
    uint32_t l = static_cast<uint32_t>(length);

    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    return r->setOutput(index, type, buffer, l);
}

int ANeuralNetworksRequest_setOutputFromHardwareBuffer(ANeuralNetworksRequest* request,
                                                       int32_t index,
                                                       const ANeuralNetworksOperandType* type,
                                                       const AHardwareBuffer* buffer) {
    if (!request || !type || !buffer) {
        LOG(ERROR) << "ANeuralNetworksRequest_setOutputFromHardwareBuffer passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    RequestBuilder* r = reinterpret_cast<RequestBuilder*>(request);
    return r->setOutputFromHardwareBuffer(index, type, buffer);
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
