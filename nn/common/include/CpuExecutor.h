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

#include "HalInterfaces.h"
#include "OperationsUtils.h"
#include "Utils.h"

#include <vector>

namespace android {
namespace nn {

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    // TODO Storing the type here is redundant, as it won't change during execution.
    OperandType type;
    // The type and dimensions of the operand.  The dimensions can
    // change at runtime.  We include the type because it's useful
    // to pass together with the dimension to the functions implementing
    // the operators.
    std::vector<uint32_t> dimensions;
    // Where the operand's data is stored.  Check the corresponding
    // location information in the model to figure out if this points
    // to memory we have allocated for an temporary operand.
    uint8_t* buffer;
    // The length of the buffer.
    uint32_t length;
    // Keeps track of how many operations have yet to make use
    // of this temporary variable.  When the count is decremented to 0,
    // we free the buffer.  For non-temporary variables, this count is
    // always 0.
    uint32_t numberOfUsesLeft;

    Shape shape() const {
        return Shape{.type = type,
                     .dimensions = dimensions};
    }
};

struct RunTimePoolInfo {
    sp<IMemory> memory;
    uint8_t* buffer;
};

// This class is used to execute a model on the CPU.
class CpuExecutor {
public:
    // Executes the model. The results will be stored at the locations
    // specified in the constructor.
    // The model must outlive the executor.  We prevent it from being modified
    // while this is executing.
    int run(const Model& model, const Request& request,
            const std::vector<RunTimePoolInfo>& runTimePoolInfos);

private:
    bool initializeRunTimeInfo(const std::vector<RunTimePoolInfo>& runTimePoolInfos);
    // Runs one operation of the graph.
    int executeOperation(const Operation& entry);
    // Decrement the usage count for the operands listed.  Frees the memory
    // allocated for any temporary variable with a count of zero.
    void freeNoLongerUsedOperands(const std::vector<uint32_t>& inputs);
    void setLocationAndUses(RunTimeOperandInfo* to, const DataLocation& location,
                            const std::vector<RunTimePoolInfo>& runTimePoolInfos);
    bool setRunTimeOperandInfo(uint32_t operandIndex, const std::vector<uint32_t>& dimensions,
                               const DataLocation& location, uint32_t useCount,
                               const std::vector<RunTimePoolInfo>& runTimePoolInfos);
    // The operand is a model input or output.  Override the information that
    // came with the model with the one passed by the calling program.
    // void overrideOperand(uint32_t operandIndex, const InputOutputInfo& info);
    //  void overrideAddress(uint32_t operandIndex, void* buffer);

    // The model and the request that we'll execute. Only valid while run()
    // is being executed.
    const Model* mModel = nullptr;
    const Request* mRequest = nullptr;

    // We're copying the list of all the dimensions from the model, as
    // these may be modified when we run the operatins.  Since we're
    // making a full copy, the indexes used in the operand description
    // stay valid.
    //    std::vector<uint32_t> mDimensions;
    // Runtime information about all the operands.
    std::vector<RunTimeOperandInfo> mOperands;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_CPU_EXECUTOR_H
