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

// Classes used to plan how to execute a model across multiple devices.

#ifndef ANDROID_ML_NN_RUNTIME_EXECUTION_PLAN_H
#define ANDROID_ML_NN_RUNTIME_EXECUTION_PLAN_H

#include "HalInterfaces.h"
#include "Memory.h"
#include "NeuralNetworks.h"
#include "Utils.h"

#include <set>

namespace android {
namespace nn {

class CompilationBuilder;
class Device;
class ExecutionPlan;
class Memory;
class ModelBuilder;

class ExecutionStep {
private:
    typedef std::vector<std::pair<uint32_t, uint32_t>> RemapVectorType;

public:
    enum OperandKind { INPUT, OUTPUT };

    ExecutionStep(ExecutionPlan* plan,
                  uint32_t stepIndex,
                  std::shared_ptr<ModelBuilder> model,
                  std::shared_ptr<Device> device);
    int addOperation(int operationIndex, const ModelBuilder& fromModel);
    int addOperand(uint32_t fromOperandIndex, uint32_t* toOperandIndex,
                   const ModelBuilder& fromModel, OperandKind kind);

    // Each vector entry is of the form (fromModel index, subModel index)
    const RemapVectorType& getSubModelInputs() const {
        return mSubModelInputs;
    }

    void recordSubModelOutput(uint32_t fromModelIndex) {
        const auto it = mOperandMap.find(fromModelIndex);
        nnAssert(it != mOperandMap.end());
        mSubModelOutputs.insert(std::make_pair(fromModelIndex, it->second));
    }

    void finishSubModel();

    void dump() const;
private:
    // TODO: Some of the data is working state information that
    // shouldn't be needed after we've constructed but not executed
    // the step.

    ExecutionPlan* mPlan;
    uint32_t mIndex;  // index of step within plan
    std::shared_ptr<ModelBuilder> mSubModel;
    std::shared_ptr<Device> mDevice;  // nullptr signifies CPU

    // Inputs of original model that are also inputs of this submodel:
    //     (fromModel index, subModel index)
    RemapVectorType mModelInputs;
    // Outputs of original model that are also outputs of this submodel:
    //     (fromModel index, subModel index)
    RemapVectorType mModelOutputs;
    // Temporaries of original model that are inputs of this submodel:
    //     (fromModel index, subModel index)
    RemapVectorType mSubModelInputs;
    // Temporaries of original model that are outputs of this submodel:
    //     (fromModel index, subModel index)
    std::set<std::pair<uint32_t, uint32_t>> mSubModelOutputs;
    // Converts operand indexes from the main model to the submodel.
    std::unordered_map<uint32_t, uint32_t> mOperandMap;
};

class ExecutionPlan {
public:
    std::shared_ptr<ExecutionStep> newStep(const std::shared_ptr<Device> device);

    void finishSubModels();

    void recordTemporaryDef(uint32_t fromModelIndex, uint32_t stepIndex) {
        nnAssert(mTemporaryToDefiningStep.count(fromModelIndex) == 0);
        mTemporaryToDefiningStep.insert(std::make_pair(fromModelIndex, stepIndex));
    }

    void dump() const;

private:
    void findSubModelOutputs();

    // TODO: Some of the data is working state information that
    // shouldn't be needed after we've constructed but not executed
    // the plan.

    std::vector<std::shared_ptr<ExecutionStep>> mSteps;

    // Map from original operand index to defining step index.
    // Used for all (and only) TEMPORARY_VARIABLEs.
    std::unordered_map<uint32_t, uint32_t> mTemporaryToDefiningStep;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_EXECUTION_PLAN_H
