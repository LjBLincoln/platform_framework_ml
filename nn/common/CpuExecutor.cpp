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

#define LOG_TAG "CpuExecutor"

#include "CpuExecutor.h"

#include "NeuralNetworks.h"
#include "Operations.h"

namespace android {
namespace nn {

// If we don't have a buffer, allocate it.
static bool allocateIfNeeded(RunTimeOperandInfo* info, const Shape& shape) {
    info->type = shape.type;
    info->dimensions = shape.dimensions;
    if (info->buffer == nullptr) {
        uint32_t length = sizeOfData(info->type, info->dimensions);
        info->buffer = new uint8_t[length];
        if (info->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

static int32_t getInt32ScalarData(RunTimeOperandInfo& info) {
    int32_t * data = reinterpret_cast<int32_t*>(info.buffer);
    return data[0];
}

// Ignore the .pools entry in model and request.  This will have been taken care of
// by the caller.
int CpuExecutor::run(const Model& model, const Request& request,
                     const std::vector<RunTimePoolInfo>& runTimePoolInfos) {
    LOG(DEBUG) << "CpuExecutor::run()";
    LOG(DEBUG) << "model: " << toString(model);
    LOG(DEBUG) << "request: " << toString(request);

    mModel = &model;
    mRequest = &request; // TODO check if mRequest is needed
    initializeRunTimeInfo(runTimePoolInfos);
    // The model has serialized the operation in execution order.
    for (const auto& operation : model.operations) {
        int n = executeOperation(operation);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    mModel = nullptr;
    mRequest = nullptr;
    LOG(DEBUG) << "Completed run normally";
    return ANEURALNETWORKS_NO_ERROR;
}

bool CpuExecutor::initializeRunTimeInfo(const std::vector<RunTimePoolInfo>& runTimePoolInfos) {
    LOG(DEBUG) << "CpuExecutor::initializeRunTimeInfo";
    const size_t count = mModel->operands.size();
    mOperands.resize(count);
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel->operands[i];
        if (!setRunTimeOperandInfo(i, from.dimensions, from.location, from.numberOfConsumers,
                                   runTimePoolInfos)) {
            return false;
        }
        mOperands[i].type = from.type;
    }

    nnAssert(mModel->inputIndexes.size() == mRequest->inputs.size());
    for (size_t i = 0; i < mModel->inputIndexes.size(); i++) {
        const InputOutputInfo& from = mRequest->inputs[i];
        if (!setRunTimeOperandInfo(mModel->inputIndexes[i], from.dimensions, from.location, 0,
                                   runTimePoolInfos)) {
            return false;
        }
    }
    nnAssert(mModel->outputIndexes.size() == mRequest->outputs.size());
    for (size_t i = 0; i < mModel->outputIndexes.size(); i++) {
        const InputOutputInfo& from = mRequest->outputs[i];
        if (!setRunTimeOperandInfo(mModel->outputIndexes[i], from.dimensions, from.location, 0,
                                   runTimePoolInfos)) {
            return false;
        }
    }
    return true;
}

bool CpuExecutor::setRunTimeOperandInfo(uint32_t operandIndex,
                                        const std::vector<uint32_t>& dimensions,
                                        const DataLocation& location, uint32_t useCount,
                                        const std::vector<RunTimePoolInfo>& runTimePoolInfos) {
    LOG(DEBUG) << "CpuExecutor::setRunTimeOperand(" << operandIndex << ", " << toString(dimensions)
               << ", " << toString(location) << ")";

    RunTimeOperandInfo& to = mOperands[operandIndex];
    if (dimensions.size() > 0) {
        to.dimensions = dimensions;
    }
    if (location.poolIndex == static_cast<uint32_t>(LocationValues::LOCATION_AT_RUN_TIME)) {
        to.buffer = nullptr;
        to.numberOfUsesLeft = useCount;
    } else if (location.poolIndex == static_cast<uint32_t>(LocationValues::LOCATION_SAME_BLOCK)) {
        to.buffer = const_cast<uint8_t*>(&mModel->operandValues[location.offset]);
        to.numberOfUsesLeft = 0;
    } else {
        if (location.poolIndex >= runTimePoolInfos.size()) {
            LOG(ERROR) << "For operand " << operandIndex << ", got a poolIndex id "
                       << location.poolIndex << " which is >= " << runTimePoolInfos.size();
            return false;
        }
        auto& r = runTimePoolInfos[location.poolIndex];
        to.buffer = r.buffer + location.offset;
        to.numberOfUsesLeft = 0;
    }
    to.length = location.length;
    return true;
}

void CpuExecutor::freeNoLongerUsedOperands(const std::vector<uint32_t>& inputs) {
    for (uint32_t i : inputs) {
        auto& info = mOperands[i];
        // Check if it's a static or model input/output.
        if (info.numberOfUsesLeft == 0) {
            continue;
        }
        nnAssert(mModel->operands[i].location.poolIndex ==
                 static_cast<uint32_t>(LocationValues::LOCATION_AT_RUN_TIME));
        info.numberOfUsesLeft--;
        if (info.numberOfUsesLeft == 0) {
            nnAssert(info.buffer != nullptr);
            delete[] info.buffer;
            info.buffer = nullptr;
        }
    }
}

int CpuExecutor::executeOperation(const Operation& operation) {
    LOG(DEBUG) << "CpuExecutor::executeOperation(" << toString(operation) << ")";
    const auto& ins = operation.inputs;
    const auto& outs = operation.outputs;
    bool success = false;

    // Function to verify that the number of input and output parameters
    // matches what is expected.
    auto parameterCountIs = [&ins, &outs, &operation](size_t expectedIns,
                                                      size_t expectedOuts) -> bool {
        if (ins.size() != expectedIns || outs.size() != expectedOuts) {
            LOG(ERROR) << getOperationName(operation.type) << ": Invalid number of ins "
                       << ins.size() << " / " << expectedIns << " and outs " << outs.size() << " / "
                       << expectedOuts;
            return false;
        }
        return true;
    };

    switch (operation.type) { // static_cast<OperationType>(operation.type)) {
        case OperationType::ADD_FLOAT32: {
            if (!parameterCountIs(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            RunTimeOperandInfo& out = mOperands[outs[0]];
            Shape outShape = out.shape();

            success = addTensorsFloat32Prepare(in1.shape(), in2.shape(), &outShape) &&
                    allocateIfNeeded(&out, outShape) &&
                    addTensorsFloat32(reinterpret_cast<const float*>(in1.buffer),
                                      reinterpret_cast<const float*>(in2.buffer),
                                      reinterpret_cast<float*>(out.buffer), outShape);
        } break;
        case OperationType::DEPTHWISE_CONV_FLOAT32: {
            if (!parameterCountIs(8, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            const RunTimeOperandInfo& filter = mOperands[ins[1]];
            const RunTimeOperandInfo& bias   = mOperands[ins[2]];

            int32_t padding          = getInt32ScalarData(mOperands[ins[3]]);
            int32_t stride_width     = getInt32ScalarData(mOperands[ins[4]]);
            int32_t stride_height    = getInt32ScalarData(mOperands[ins[5]]);
            int32_t depth_multiplier = getInt32ScalarData(mOperands[ins[6]]);
            int32_t activation       = getInt32ScalarData(mOperands[ins[7]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = depthwiseConvFloat32Prepare(input.shape(), filter.shape(), bias.shape(),
                                                  padding, stride_width, stride_height,
                                                  &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      depthwiseConvFloat32(reinterpret_cast<const float*>(input.buffer),
                                           input.shape(),
                                           reinterpret_cast<const float*>(filter.buffer),
                                           filter.shape(),
                                           reinterpret_cast<const float*>(bias.buffer),
                                           bias.shape(),
                                           padding, stride_width, stride_height,
                                           depth_multiplier, activation,
                                           reinterpret_cast<float*>(output.buffer),
                                           outShape);

        } break;
        case OperationType::CONV_FLOAT32: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            const RunTimeOperandInfo& filter = mOperands[ins[1]];
            const RunTimeOperandInfo& bias   = mOperands[ins[2]];

            int32_t padding          = getInt32ScalarData(mOperands[ins[3]]);
            int32_t stride_width     = getInt32ScalarData(mOperands[ins[4]]);
            int32_t stride_height    = getInt32ScalarData(mOperands[ins[5]]);
            int32_t activation       = getInt32ScalarData(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = convFloat32Prepare(input.shape(), filter.shape(), bias.shape(),
                                         padding, stride_width, stride_height,
                                         &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      convFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                  reinterpret_cast<const float*>(filter.buffer), filter.shape(),
                                  reinterpret_cast<const float*>(bias.buffer), bias.shape(),
                                  padding, stride_width, stride_height, activation,
                                  reinterpret_cast<float*>(output.buffer), outShape);

        } break;
        case OperationType::AVERAGE_POOL_FLOAT32: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];

            int32_t padding          = getInt32ScalarData(mOperands[ins[1]]);
            int32_t stride_width     = getInt32ScalarData(mOperands[ins[2]]);
            int32_t stride_height    = getInt32ScalarData(mOperands[ins[3]]);
            int32_t filter_width     = getInt32ScalarData(mOperands[ins[4]]);
            int32_t filter_height    = getInt32ScalarData(mOperands[ins[5]]);
            int32_t activation       = getInt32ScalarData(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = genericPoolingFloat32Prepare(input.shape(),
                                                   padding, stride_width, stride_height,
                                                   filter_width, filter_height,
                                                   &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      averagePoolFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                         padding, stride_width, stride_height,
                                         filter_width, filter_height, activation,
                                         reinterpret_cast<float*>(output.buffer), outShape);

        } break;
        case OperationType::L2_POOL_FLOAT32: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];

            int32_t padding          = getInt32ScalarData(mOperands[ins[1]]);
            int32_t stride_width     = getInt32ScalarData(mOperands[ins[2]]);
            int32_t stride_height    = getInt32ScalarData(mOperands[ins[3]]);
            int32_t filter_width     = getInt32ScalarData(mOperands[ins[4]]);
            int32_t filter_height    = getInt32ScalarData(mOperands[ins[5]]);
            int32_t activation       = getInt32ScalarData(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = genericPoolingFloat32Prepare(input.shape(),
                                                   padding, stride_width, stride_height,
                                                   filter_width, filter_height,
                                                   &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      l2PoolFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                    padding, stride_width, stride_height,
                                    filter_width, filter_height, activation,
                                    reinterpret_cast<float*>(output.buffer), outShape);

        } break;
        case OperationType::MAX_POOL_FLOAT32: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];

            int32_t padding          = getInt32ScalarData(mOperands[ins[1]]);
            int32_t stride_width     = getInt32ScalarData(mOperands[ins[2]]);
            int32_t stride_height    = getInt32ScalarData(mOperands[ins[3]]);
            int32_t filter_width     = getInt32ScalarData(mOperands[ins[4]]);
            int32_t filter_height    = getInt32ScalarData(mOperands[ins[5]]);
            int32_t activation       = getInt32ScalarData(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = genericPoolingFloat32Prepare(input.shape(),
                                                   padding, stride_width, stride_height,
                                                   filter_width, filter_height,
                                                   &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      maxPoolFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                     padding, stride_width, stride_height,
                                     filter_width, filter_height, activation,
                                     reinterpret_cast<float*>(output.buffer), outShape);

        } break;
        case OperationType::RELU_FLOAT32: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = genericActivationFloat32Prepare(input.shape(), &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      reluFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                  reinterpret_cast<float*>(output.buffer), outShape);
        } break;
        case OperationType::RELU6_FLOAT32: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = genericActivationFloat32Prepare(input.shape(), &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      relu6Float32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                   reinterpret_cast<float*>(output.buffer), outShape);
        } break;
        case OperationType::TANH_FLOAT32: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = genericActivationFloat32Prepare(input.shape(), &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      tanhFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                  reinterpret_cast<float*>(output.buffer), outShape);
        } break;
        case OperationType::LOGISTIC_FLOAT32: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = genericActivationFloat32Prepare(input.shape(), &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      logisticFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                      reinterpret_cast<float*>(output.buffer), outShape);
        } break;
        default:
            nnAssert(false);
            break;
    }
    if (!success) {
        LOG(ERROR) << getOperationName(operation.type) << " failed.";
        return ANEURALNETWORKS_OP_FAILED;
    }

    freeNoLongerUsedOperands(ins);
    return ANEURALNETWORKS_NO_ERROR;
}

} // namespace nn
} // namespace android
