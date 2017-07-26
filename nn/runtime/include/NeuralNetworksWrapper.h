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

// Provides C++ classes to more easily use the Neural Networks API.

#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H

#include "NeuralNetworks.h"

#include <vector>

namespace android {
namespace nn {
namespace wrapper {

enum class Type {
    FLOAT16 = ANEURALNETWORKS_FLOAT16,
    FLOAT32 = ANEURALNETWORKS_FLOAT32,
    INT8 = ANEURALNETWORKS_INT8,
    UINT8 = ANEURALNETWORKS_UINT8,
    INT16 = ANEURALNETWORKS_INT16,
    UINT16 = ANEURALNETWORKS_UINT16,
    INT32 = ANEURALNETWORKS_INT32,
    UINT32 = ANEURALNETWORKS_UINT32,
    TENSOR_FLOAT16 = ANEURALNETWORKS_TENSOR_FLOAT16,
    TENSOR_FLOAT32 = ANEURALNETWORKS_TENSOR_FLOAT32,
    TENSOR_SYMMETRICAL_QUANT8 = ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8,
};

enum class ExecutePreference {
    PREFER_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER,
    PREFER_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
    PREFER_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED
};

enum class Result {
    NO_ERROR = ANEURALNETWORKS_NO_ERROR,
    OUT_OF_MEMORY = ANEURALNETWORKS_OUT_OF_MEMORY,
    INCOMPLETE = ANEURALNETWORKS_INCOMPLETE,
    UNEXPECTED_NULL = ANEURALNETWORKS_UNEXPECTED_NULL,
    BAD_DATA = ANEURALNETWORKS_BAD_DATA,
};

struct OperandType {
    ANeuralNetworksOperandType operandType;
    // uint32_t type;
    std::vector<uint32_t> dimensions;

    OperandType(Type type, const std::vector<uint32_t>& d) : dimensions(d) {
        operandType.type = static_cast<uint32_t>(type);
        operandType.dimensions.count = static_cast<uint32_t>(dimensions.size());
        operandType.dimensions.data = dimensions.data();
    }
};

inline Result Initialize() {
    return static_cast<Result>(ANeuralNetworksInitialize());
}

inline void Shutdown() {
    ANeuralNetworksShutdown();
}

class Model {
public:
    Model() {
        // TODO handle the value returned by this call
        ANeuralNetworksModel_create(&mModel);
    }
    ~Model() { ANeuralNetworksModel_free(mModel); }

    uint32_t addOperand(const OperandType* type) {
        if (ANeuralNetworksModel_addOperand(mModel, &(type->operandType)) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
        return mNextOperandId++;
    }

    void setOperandValue(uint32_t index, const void* buffer, size_t length) {
        if (ANeuralNetworksModel_setOperandValue(mModel, index, buffer, length) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    void addOperation(ANeuralNetworksOperationType type, const std::vector<uint32_t>& inputs,
                      const std::vector<uint32_t>& outputs) {
        ANeuralNetworksIntList in, out;
        Set(&in, inputs);
        Set(&out, outputs);
        if (ANeuralNetworksModel_addOperation(mModel, type, &in, &out) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }
    void setInputsAndOutputs(const std::vector<uint32_t>& inputs,
                             const std::vector<uint32_t>& outputs) {
        ANeuralNetworksIntList in, out;
        Set(&in, inputs);
        Set(&out, outputs);
        if (ANeuralNetworksModel_setInputsAndOutputs(mModel, &in, &out) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }
    ANeuralNetworksModel* getHandle() const { return mModel; }
    bool isValid() const { return mValid; }
    static Model* createBaselineModel(uint32_t modelId) {
        Model* model = new Model();
        if (ANeuralNetworksModel_createBaselineModel(&model->mModel, modelId) !=
            ANEURALNETWORKS_NO_ERROR) {
            delete model;
            model = nullptr;
        }
        return model;
    }

private:
    /**
     * WARNING list won't be valid once vec is destroyed or modified.
     */
    void Set(ANeuralNetworksIntList* list, const std::vector<uint32_t>& vec) {
        list->count = static_cast<uint32_t>(vec.size());
        list->data = vec.data();
    }

    ANeuralNetworksModel* mModel = nullptr;
    // We keep track of the operand ID as a convenience to the caller.
    uint32_t mNextOperandId = 0;
    bool mValid = true;
};

class Event {
public:
    ~Event() { ANeuralNetworksEvent_free(mEvent); }
    Result wait() { return static_cast<Result>(ANeuralNetworksEvent_wait(mEvent)); }
    void set(ANeuralNetworksEvent* newEvent) {
        ANeuralNetworksEvent_free(mEvent);
        mEvent = newEvent;
    }

private:
    ANeuralNetworksEvent* mEvent = nullptr;
};

class Request {
public:
    Request(const Model* model) {
        int result = ANeuralNetworksRequest_create(model->getHandle(), &mRequest);
        if (result != 0) {
            // TODO Handle the error
        }
    }

    ~Request() { ANeuralNetworksRequest_free(mRequest); }

    Result setPreference(ExecutePreference preference) {
        return static_cast<Result>(
                ANeuralNetworksRequest_setPreference(mRequest, static_cast<uint32_t>(preference)));
    }

    Result setInput(uint32_t index, const void* buffer, size_t length,
                    const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(
                ANeuralNetworksRequest_setInput(mRequest, index, type, buffer, length));
    }

    Result setInputFromHardwareBuffer(uint32_t index, const AHardwareBuffer* buffer,
                                      const ANeuralNetworksOperandType* type) {
        return static_cast<Result>(
                ANeuralNetworksRequest_setInputFromHardwareBuffer(mRequest, index, type, buffer));
    }

    Result setOutput(uint32_t index, void* buffer, size_t length,
                     const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(
                ANeuralNetworksRequest_setOutput(mRequest, index, type, buffer, length));
    }

    Result setOutputFromHardwareBuffer(uint32_t index, const AHardwareBuffer* buffer,
                                       const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(
                ANeuralNetworksRequest_setOutputFromHardwareBuffer(mRequest, index, type, buffer));
    }

    Result startCompute(Event* event) {
        ANeuralNetworksEvent* ev = nullptr;
        Result result = static_cast<Result>(ANeuralNetworksRequest_startCompute(mRequest, &ev));
        event->set(ev);
        return result;
    }

    Result compute() {
        ANeuralNetworksEvent* event = nullptr;
        Result result = static_cast<Result>(ANeuralNetworksRequest_startCompute(mRequest, &event));
        if (result != Result::NO_ERROR) {
            return result;
        }
        // TODO how to manage the lifetime of events when multiple waiters is not
        // clear.
        return static_cast<Result>(ANeuralNetworksEvent_wait(event));
    }

private:
    ANeuralNetworksRequest* mRequest = nullptr;
};

} // namespace wrapper
} // namespace nn
} // namespace android

#endif //  ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_WRAPPER_H
