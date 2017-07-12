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
#include "Request.h"

#include <vector>

using namespace android::nn;

// Validates the type. The used dimensions can be underspecified.
static int ValidateOperandType(const ANeuralNetworksOperandType& type, const char* tag,
                               bool allowPartial) {
    if (!allowPartial) {
        for (uint32_t i = 0; i < type.dimensions.count; i++) {
            if (type.dimensions.data[i] == 0) {
                ALOGE("%s OperandType invalid dimensions[%u] = %u", tag, i,
                      type.dimensions.data[i]);
                return ANEURALNETWORKS_BAD_DATA;
            }
        }
    }
    if (type.type >= ANEURALNETWORKS_NUMBER_DATA_TYPES) {
        ALOGE("%s OperandType invalid type %u", tag, type.type);
        return ANEURALNETWORKS_BAD_DATA;
    }
    /* TODO validate the quantization info.
    if (type.offset != 0.f && type.scale == 0.f) {
        ALOGE("%s OperandType invalid offset %f and scale %f", tag, type.offset,
    type.scale); return ANEURALNETWORKS_BAD_DATA;
    }
    if (type.scale != 0.f &&
        (type.type == ANEURALNETWORKS_FLOAT16 ||
         type.type != ANEURALNETWORKS_FLOAT32)) {
            ALOGE("%s OperandType scale %f with float type %u", tag, type.scale,
    type.type); return ANEURALNETWORKS_BAD_DATA;
        }
     */
    return ANEURALNETWORKS_NO_ERROR;
}

static int ValidateOperandList(const ANeuralNetworksIntList& list, uint32_t count,
                               const char* tag) {
    for (uint32_t i = 0; i < list.count; i++) {
        if (list.data[i] >= count) {
            ALOGE("%s invalid operand index at %u = %u, count %u", tag, i, list.data[i], count);
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksInitialize() {
    DriverManager::get()->initialize();
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksShutdown() {
    DriverManager::get()->shutdown();
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
    if (!model) {
        ALOGE("ANeuralNetworksModel_create passed a nullptr");
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
        ALOGE("ANeuralNetworksModel_create passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (modelId >= ANEURALNETWORKS_NUMBER_BASELINE_MODELS) {
        ALOGE("ANeuralNetworksModel_createBaselineModel invalid modelId %u", modelId);
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
        ALOGE("ANeuralNetworksModel_addOperand passed a nullptr");
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
        ALOGE("ANeuralNetworksModel_setOperandValue passed a nullptr");
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
        ALOGE("ANeuralNetworksModel_addOperation passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    if (type >= ANEURALNETWORKS_NUMBER_OPERATION_TYPES) {
        ALOGE("ANeuralNetworksModel_addOperation invalid operations type %u", type);
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
        ALOGE("ANeuralNetworksModel_setInputsAndOutputs passed a nullptr");
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
        ALOGE("ANeuralNetworksModel_addSubModel passed a nullptr");
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
        ALOGE("ANeuralNetworksModel_setBaselineId passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (baseLineId >= ANEURALNETWORKS_NUMBER_BASELINE_MODELS) {
        ALOGE("ANeuralNetworksModel_setBaselineId invalid baselineId %u", baseLineId);
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO implement
    return ANEURALNETWORKS_NOT_IMPLEMENTED;
}

int ANeuralNetworksRequest_create(ANeuralNetworksModel* model, ANeuralNetworksRequest** request) {
    if (!model || !request) {
        ALOGE("ANeuralNetworksRequest_create passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    Request* r = m->createRequest();
    if (r == nullptr) {
        *request = nullptr;
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    *request = reinterpret_cast<ANeuralNetworksRequest*>(r);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksRequest_free(ANeuralNetworksRequest* request) {
    // No validation.  Free of nullptr is valid.
    Request* r = reinterpret_cast<Request*>(request);
    delete r;
}

int ANeuralNetworksRequest_setPreference(ANeuralNetworksRequest* request, uint32_t preference) {
    if (!request) {
        ALOGE("ANeuralNetworksRequest_setPreference passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (preference >= ANEURALNETWORKS_NUMBER_PREFERENCES) {
        ALOGE("ANeuralNetworksRequest_setPreference invalid preference %u", preference);
        return ANEURALNETWORKS_BAD_DATA;
    }

    Request* r = reinterpret_cast<Request*>(request);
    r->setPreference(preference);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksRequest_setInput(ANeuralNetworksRequest* request, int32_t index,
                                    const ANeuralNetworksOperandType* type, const void* buffer,
                                    size_t length) {
    if (!request || !buffer) {
        ALOGE("ANeuralNetworksRequest_setInput passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (type != nullptr) {
        int n = ValidateOperandType(*type, "ANeuralNetworksRequest_setInput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
        ALOGE("ANeuralNetworksRequest_setInput input exceeds max length %zu", length);
    }
    uint32_t l = static_cast<uint32_t>(length);
    Request* r = reinterpret_cast<Request*>(request);
    return r->setInput(index, type, buffer, l);
}

int ANeuralNetworksRequest_setInputFromHardwareBuffer(ANeuralNetworksRequest* request,
                                                      int32_t index,
                                                      const ANeuralNetworksOperandType* type,
                                                      const AHardwareBuffer* buffer) {
    if (!request || !type || !buffer) {
        ALOGE("ANeuralNetworksRequest_setInputFromHardwareBuffer passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    Request* r = reinterpret_cast<Request*>(request);
    return r->setInputFromHardwareBuffer(index, type, buffer);
}

int ANeuralNetworksRequest_setOutput(ANeuralNetworksRequest* request, int32_t index,
                                     const ANeuralNetworksOperandType* type, void* buffer,
                                     size_t length) {
    if (!request || !buffer) {
        ALOGE("ANeuralNetworksRequest_setOutput passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (type != nullptr) {
        int n = ValidateOperandType(*type, "ANeuralNetworksRequest_setOutput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
        ALOGE("ANeuralNetworksRequest_setOutput input exceeds max length %zu", length);
    }
    uint32_t l = static_cast<uint32_t>(length);

    Request* r = reinterpret_cast<Request*>(request);
    return r->setOutput(index, type, buffer, l);
}

int ANeuralNetworksRequest_setOutputFromHardwareBuffer(ANeuralNetworksRequest* request,
                                                       int32_t index,
                                                       const ANeuralNetworksOperandType* type,
                                                       const AHardwareBuffer* buffer) {
    if (!request || !type || !buffer) {
        ALOGE("ANeuralNetworksRequest_setOutputFromHardwareBuffer passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    Request* r = reinterpret_cast<Request*>(request);
    return r->setOutputFromHardwareBuffer(index, type, buffer);
}

int ANeuralNetworksRequest_startCompute(ANeuralNetworksRequest* request,
                                        ANeuralNetworksEvent** event) {
    if (!request || !event) {
        ALOGE("ANeuralNetworksRequest_startCompute passed a nullptr");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    Request* r = reinterpret_cast<Request*>(request);
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
        ALOGE("ANeuralNetworksEvent_wait passed a nullptr");
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
