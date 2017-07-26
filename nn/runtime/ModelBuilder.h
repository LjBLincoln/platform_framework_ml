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

// Class used to build a model through a succession of successive calls
// to the NN API.

#ifndef ANDROID_ML_NN_RUNTIME_MODEL_BUILDER_H
#define ANDROID_ML_NN_RUNTIME_MODEL_BUILDER_H

#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include "Utils.h"

namespace android {
namespace nn {

class RequestBuilder;

class ModelBuilder {
public:
    virtual ~ModelBuilder() {}
    // Adds an operand to the model.
    int addOperand(const ANeuralNetworksOperandType& type);
    int setOperandValue(uint32_t index, const void* buffer, size_t length);

    int addOperation(ANeuralNetworksOperationType type, const ANeuralNetworksIntList* inputs,
                     const ANeuralNetworksIntList* outputs);
    int setInputsAndOutputs(const ANeuralNetworksIntList* inputs,
                            const ANeuralNetworksIntList* outputs);
    int loadBaseLineModel(uint32_t modelId);

    RequestBuilder* createRequest();

    void setHidlModel(Model* model) const;

    uint32_t operandCount() const {
        // We don't allow more than uint32_t worth of operands
        return static_cast<uint32_t>(mOperands.size());
    }
    uint32_t operationCount() const {
        // We don't allow more than uint32_t worth of operations
        return static_cast<uint32_t>(mOperations.size());
    }
    uint32_t inputCount() const { return static_cast<uint32_t>(mInputIndexes.size()); }
    uint32_t outputCount() const { return static_cast<uint32_t>(mOutputIndexes.size()); }
    uint32_t getInputOperandIndex(uint32_t i) const { return mInputIndexes[i]; }
    uint32_t getOutputOperandIndex(uint32_t i) const { return mOutputIndexes[i]; }
    const Operand& getOperand(uint32_t index) const { return mOperands[index]; }

private:
    // Sorts the operations to be in the correct order for single threaded
    // node-at-a-time execution.
    void sortIntoRunOrder();
    /*
    int32_t getOperandIndex(const ArrayInfo& info, uint32_t listIndex) const {
        nnAssert(listIndex < info.count);
        return mOperandIndexes[info.offset + listIndex];
    }
    */
    void finishTheModel();

    // The operations of the graph.
    std::vector<Operation> mOperations;
    // The description of the operands of the graph.
    std::vector<Operand> mOperands;
    // Specifies where to find the list of indexes identifying
    // the inputs and outputs of the model.  The offset is into
    // the mOperandIndexes table.
    std::vector<uint32_t> mInputIndexes;
    std::vector<uint32_t> mOutputIndexes;

    // The value of the operands that are defined at model
    // creation time.
    // TODO We are copying all the values.  Once we support memory
    // pools, revisit.
    std::vector<uint8_t> mOperandValues;

    // Once the request has been created, we should not allow further
    // modifications to the model.
    mutable bool mCompletedModel = false;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_MODEL_BUILDER_H
