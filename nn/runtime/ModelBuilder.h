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

#ifndef ANDROID_ML_NN_RUNTIME_MODEL_H
#define ANDROID_ML_NN_RUNTIME_MODEL_H

#include "Model.h"
#include "NeuralNetworks.h"
#include "Utils.h"

namespace android {
namespace nn {

class Request;

class ModelBuilder : public IModel {
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
    Request* createRequest();

    // Serialize the model into the buffer.
    // TODO This should be a shared memory buffer instead, or a file.
    void serialize(std::vector<uint8_t>* buffer) const;

    uint32_t operandCount() const {
        // We don't allow more than uint32_t worth of operands
        return static_cast<uint32_t>(mOperands.size());
    }
    uint32_t operationCount() const {
        // We don't allow more than uint32_t worth of operations
        return static_cast<uint32_t>(mOperations.size());
    }
    uint32_t inputCount() const { return mModelInputs.count; }
    uint32_t outputCount() const { return mModelOutputs.count; }
    uint32_t getOperandType(uint32_t index) const { return mOperands[index].type; }
    uint32_t getOperandNumberOfDimensions(uint32_t index) const {
        return mOperands[index].dimensions.count;
    }

    // From IModel
    virtual Range<OperationEntry> getOperations() const {
        return Range<OperationEntry>(mOperations);
    }
    virtual Range<OperandEntry> getOperands() const { return Range<OperandEntry>(mOperands); }
    virtual Range<uint32_t> getOperandIndexes(const ArrayInfo& info) const {
        return Range<uint32_t>(mOperandIndexes, info);
    }
    virtual void copyDimensionStorage(std::vector<uint32_t>* dimensions) const {
        *dimensions = mDimensions;
    }
    virtual uint32_t getInputOperandIndex(uint32_t listIndex) const {
        return getOperandIndex(mModelInputs, listIndex);
    }
    virtual uint32_t getOutputOperandIndex(uint32_t listIndex) const {
        return getOperandIndex(mModelOutputs, listIndex);
    }
    virtual const void* getDataPointer(uint32_t offset) const {
        return mOperandValues.data() + offset;
    }

private:
    // Sorts the operations to be in the correct order for single threaded
    // node-at-a-time execution.
    void sortIntoRunOrder();

    int32_t getOperandIndex(const ArrayInfo& info, uint32_t listIndex) const {
        nnAssert(listIndex < info.count);
        return mOperandIndexes[info.offset + listIndex];
    }
    void finishTheModel();

    // The operations of the graph.
    std::vector<OperationEntry> mOperations;
    // The description of the operands of the graph.
    std::vector<OperandEntry> mOperands;
    // Used by OperandEntry to store arrays of dimension values.
    std::vector<uint32_t> mDimensions;
    // Usded to store arrays of indexes into the mOperands table.
    std::vector<uint32_t> mOperandIndexes;
    // The value of the operands that are defined at model
    // creation time.
    // TODO We are copying all the values.  Once we support memory
    // pools, revisit.
    std::vector<uint8_t> mOperandValues;
    // Specifies where to find the list of indexes identifying
    // the inputs and outputs of the model.  The offset is into
    // the mOperandIndexes table.
    ArrayInfo mModelInputs;
    ArrayInfo mModelOutputs;

    // Once the request has been created, we should not allow further
    // modifications to the model.
    mutable bool mCompletedModel = false;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_MODEL_H
