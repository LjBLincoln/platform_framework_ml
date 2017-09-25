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

namespace android {
namespace nn {

class CompilationBuilder;
class Device;
class ExecutionStep;
class Memory;
class ModelBuilder;

class ExecutionStep {
public:
    ExecutionStep(std::shared_ptr<ModelBuilder> model, std::shared_ptr<Device> device);
    int addOperation(int operationIndex, const ModelBuilder& fromModel);
    int addOperand(uint32_t fromOperandIndex, uint32_t* toOperandIndex,
                   const ModelBuilder& fromModel);

private:
    std::shared_ptr<ModelBuilder> mSubModel;
    std::shared_ptr<Device> mDevice;
    // Result
    std::vector<uint32_t> mModelInputsFrom;
    std::vector<uint32_t> mModelInputsTo;
    // Converts operand indexes from the main model to the submodel.
    std::unordered_map<uint32_t, uint32_t> mOperandMap;
};

class ExecutionPlan {
  public:
    void addStep(std::shared_ptr<ExecutionStep> step) {
        mSteps.push_back(step);
    }
  private:
    std::vector<std::shared_ptr<ExecutionStep>> mSteps;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_EXECUTION_PLAN_H
