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

#ifndef ANDROID_ML_NN_COMMON_CPU_EXECUTOR_H
#define ANDROID_ML_NN_COMMON_CPU_EXECUTOR_H

#include "HalAbstraction.h"
#include "OperationsUtils.h"
#include "Utils.h"

#include <vector>

namespace android {
namespace nn {

class IModel;

// Information we maintain about each operand during execution.
struct RunTimeOperandInfo {
    // The type and dimensions of the operand.  The dimensions can
    // change at runtime.  We include the type because it's useful
    // to pass together with the dimension to the functions implementing
    // the operators.
    Shape shape;
    // Where the operand's data is stored.  Check the corresponding
    // location information in the model to figure out if this points
    // to memory we have allocated for an temporary operand.
    void* buffer;
    // The length of the buffer.
    uint32_t length;
    // Keeps track of how many operations have yet to make use
    // of this temporary variable.  When the count is decremented to 0,
    // we free the buffer.  For non-temporary variables, this count is
    // always 0.
    uint32_t numberOfUsesLeft;
};

// This class is used to execute a model on the CPU.
class CpuExecutor {
public:
    // The model must outlive the executor.  We prevent it from being modified
    // while this is executing.
    CpuExecutor(const IModel* model, const std::vector<InputOutputInfo>& modelInputs,
                const std::vector<InputOutputInfo>& modelOutputs);
    // Executes the model. The results will be stored at the locations
    // specified in the constructor.
    int run();

private:
    // Runs one operation of the graph.
    int executeOperation(const OperationEntry& entry);
    // Decrement the usage count for the operands listed.  Frees the memory
    // allocated for any temporary variable with a count of zero.
    void freeNoLongerUsedOperands(const Range<uint32_t>& inputs);

    // The operand is a model input or output.  Override the information that
    // came with the model with the one passed by the calling program.
    void overrideOperand(uint32_t operandIndex, const InputOutputInfo& info);

    // The model that we'll execute.
    const IModel* mModel;
    // We're copying the list of all the dimensions from the model, as
    // these may be modified when we run the operatins.  Since we're
    // making a full copy, the indexes used in the operand description
    // stay valid.
    std::vector<uint32_t> mDimensions;
    // Runtime information about all the operands.
    std::vector<RunTimeOperandInfo> mOperands;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_COMMON_CPU_EXECUTOR_H
