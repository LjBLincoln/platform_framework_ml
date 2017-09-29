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

#ifndef ANDROID_ML_NN_RUNTIME_EXECUTION_BUILDER_H
#define ANDROID_ML_NN_RUNTIME_EXECUTION_BUILDER_H

#include "Event.h"
#include "HalInterfaces.h"
#include "Memory.h"
#include "NeuralNetworks.h"

#include <unordered_map>
#include <vector>

using ::android::hardware::neuralnetworks::V1_0::implementation::Event;

namespace android {
namespace nn {

class CompilationBuilder;
class ExecutionPlan;
class Memory;
class ModelBuilder;

// TODO move length out of DataLocation
struct ModelArgumentInfo {
    // Whether the arguement was specified as being in a Memory, as a pointer,
    // or has not been specified.
    // If POINTER then:
    //   locationAndDimension.location.length is valid.
    //   locationAndDimension.dimension is valid.
    //   buffer is valid
    // If MEMORY then:
    //   locationAndDimension.location.{poolIndex, offset, length} is valid.
    //   locationAndDimension.dimension is valid.
    enum { POINTER, MEMORY, UNSPECIFIED } state;
    RequestArgument locationAndDimension;
    void* buffer;

    int setFromPointer(const Operand& operand, const ANeuralNetworksOperandType* type, void* buffer,
                       uint32_t length);
    int setFromMemory(const Operand& operand, const ANeuralNetworksOperandType* type,
                      uint32_t poolIndex, uint32_t offset, uint32_t length);
    int updateDimensionInfo(const Operand& operand, const ANeuralNetworksOperandType* newType);
};

class ExecutionBuilder {
public:
    ExecutionBuilder(const CompilationBuilder* compilation);

    int setInput(uint32_t index, const ANeuralNetworksOperandType* type, const void* buffer,
                 size_t length);
    int setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                           const Memory* memory, size_t offset, size_t length);
    int setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                  size_t length);
    int setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                            const Memory* memory, size_t offset, size_t length);
    int startCompute(sp<Event>* event);

private:
    int allocatePointerArgumentsToPool(std::vector<ModelArgumentInfo>* args, Memory* memory);
    int updateDimensionInfo(ModelArgumentInfo* info, const ANeuralNetworksOperandType* newType,
                            const Operand& operand);
    int startComputeOnDevice(sp<Event>* event, sp<IDevice> driver,
                             sp<IPreparedModel> preparedModel = nullptr);
    int startComputeOnCpu(sp<Event>* event);

    const ModelBuilder* mModel;
    const ExecutionPlan* mPlan;

    // The information we'll send to the driver about the inputs and outputs.
    // Note that we build this in two steps:
    // 1. As the arguments are specified, set the corresponding mInputs or mOutputs element.
    //    If set from a pointer, don't set the location in the RequestArgument but store it
    //    instead in mInputBuffers or mOutputBuffers.
    // 2. Once we have all the inputs and outputs, if needed, allocate shared memory for
    //    the m*Buffers entries.  Copy the input values into the shared memory.
    // We do this to avoid creating a lot of shared memory objects if we have a lot of
    // parameters specified via pointers.  We also avoid copying in the case where
    // some of the nodes will interpreted on the CPU anyway.
    std::vector<ModelArgumentInfo> mInputs;
    std::vector<ModelArgumentInfo> mOutputs;
    // We separate the input & output pools so that we reduce the copying done if we
    // do an eventual remoting (hidl_memory->update()).  We could also use it to set
    // protection on read only memory but that's not currently done.
    Memory mInputPointerArguments;
    Memory mOutputPointerArguments;
    MemoryTracker mMemories;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_EXECUTION_BUILDER_H
