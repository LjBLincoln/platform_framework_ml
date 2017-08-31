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

#include "EmbeddingLookup.h"
#include "HashtableLookup.h"
#include "LSHProjection.h"
#include "LSTM.h"
#include "NeuralNetworks.h"
#include "Operations.h"
#include "RNN.h"
#include "SVDF.h"

namespace android {
namespace nn {

bool RunTimePoolInfo::set(const hidl_memory& hidlMemory) {
    memory = mapMemory(hidlMemory);
    if (memory == nullptr) {
        LOG(ERROR) << "Can't map shared memory.";
        return false;
    }
    memory->update();
    buffer = reinterpret_cast<uint8_t*>(static_cast<void*>(memory->getPointer()));
    if (buffer == nullptr) {
        LOG(ERROR) << "Can't access shared memory.";
        return false;
    }
    return true;
}

// If we don't have a buffer, allocate it.
static bool allocateIfNeeded(RunTimeOperandInfo* info, const Shape& shape) {
    info->type = shape.type;
    info->dimensions = shape.dimensions;
    info->scale = shape.scale;
    info->offset = shape.offset;
    if (info->buffer == nullptr) {
        uint32_t length = sizeOfData(info->type, info->dimensions);
        info->buffer = new uint8_t[length];
        if (info->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

template <typename T>
static T getScalarData(RunTimeOperandInfo& info) {
    T* data = reinterpret_cast<T*>(info.buffer);
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
        mOperands[i].scale = from.scale;
        mOperands[i].offset = from.zeroPoint;
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
    if (location.poolIndex == RUN_TIME) {
        to.buffer = nullptr;
        to.numberOfUsesLeft = useCount;
    } else if (location.poolIndex == SAME_BLOCK) {
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
        nnAssert(mModel->operands[i].location.poolIndex == RUN_TIME);
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
            LOG(ERROR) << getOperationName(operation.opTuple.operationType)
                       << ": Invalid number of ins "
                       << ins.size() << " / " << expectedIns
                       << " and outs " << outs.size() << " / "
                       << expectedOuts;
            return false;
        }
        return true;
    };

    switch (operation.opTuple.operationType) {
        case OperationType::OEM_OPERATION: {
            LOG(ERROR) << "OEM operation not supported for CPU execution";
            success = false;
        } break;
        case OperationType::ADD: {
            if (!parameterCountIs(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& out = mOperands[outs[0]];
            Shape outShape = out.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = addPrepare(in1.shape(), in2.shape(), &outShape) &&
                          allocateIfNeeded(&out, outShape) &&
                          addFloat32(reinterpret_cast<const float*>(in1.buffer),
                                     reinterpret_cast<const float*>(in2.buffer),
                                     activation,
                                     reinterpret_cast<float*>(out.buffer),
                                     outShape);
            }
        } break;
        case OperationType::MUL: {
            if (!parameterCountIs(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& out = mOperands[outs[0]];
            Shape outShape = out.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = mulPrepare(in1.shape(), in2.shape(), &outShape) &&
                          allocateIfNeeded(&out, outShape) &&
                          mulFloat32(reinterpret_cast<const float*>(in1.buffer),
                                     reinterpret_cast<const float*>(in2.buffer),
                                     activation,
                                     reinterpret_cast<float*>(out.buffer),
                                     outShape);
            }
        } break;
        case OperationType::FLOOR: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = floorPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          floorFloat32(reinterpret_cast<const float*>(input.buffer),
                                       reinterpret_cast<float*>(output.buffer),
                                       outShape);
            }
        } break;
        case OperationType::DEQUANTIZE: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = dequantizePrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          dequantizeQuant8ToFloat32(
                                  reinterpret_cast<const uint8_t*>(input.buffer),
                                  reinterpret_cast<float*>(output.buffer),
                                  outShape);
            }
        } break;
        case OperationType::DEPTHWISE_CONV: {
            if (!parameterCountIs(8, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            const RunTimeOperandInfo& filter = mOperands[ins[1]];
            const RunTimeOperandInfo& bias   = mOperands[ins[2]];

            int32_t padding          = getScalarData<int32_t>(mOperands[ins[3]]);
            int32_t stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            int32_t stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            int32_t depth_multiplier = getScalarData<int32_t>(mOperands[ins[6]]);
            int32_t activation       = getScalarData<int32_t>(mOperands[ins[7]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = depthwiseConvPrepare(input.shape(), filter.shape(), bias.shape(),
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
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = depthwiseConvPrepare(input.shape(), filter.shape(), bias.shape(),
                                               padding, stride_width, stride_height,
                                               &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          depthwiseConvQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                              input.shape(),
                                              reinterpret_cast<const uint8_t*>(filter.buffer),
                                              filter.shape(),
                                              reinterpret_cast<const int32_t*>(bias.buffer),
                                              bias.shape(),
                                              padding, stride_width, stride_height,
                                              depth_multiplier, activation,
                                              reinterpret_cast<uint8_t*>(output.buffer),
                                              outShape);
            }

        } break;
        case OperationType::CONV: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            const RunTimeOperandInfo& filter = mOperands[ins[1]];
            const RunTimeOperandInfo& bias   = mOperands[ins[2]];

            int32_t padding          = getScalarData<int32_t>(mOperands[ins[3]]);
            int32_t stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            int32_t stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            int32_t activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = convPrepare(input.shape(), filter.shape(), bias.shape(),
                                      padding, stride_width, stride_height,
                                      &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          convFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                      reinterpret_cast<const float*>(filter.buffer), filter.shape(),
                                      reinterpret_cast<const float*>(bias.buffer), bias.shape(),
                                      padding, stride_width, stride_height, activation,
                                      reinterpret_cast<float*>(output.buffer), outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = convPrepare(input.shape(), filter.shape(), bias.shape(),
                                      padding, stride_width, stride_height,
                                      &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          convQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<const uint8_t*>(filter.buffer),
                                     filter.shape(),
                                     reinterpret_cast<const int32_t*>(bias.buffer),
                                     bias.shape(),
                                     padding, stride_width, stride_height, activation,
                                     reinterpret_cast<uint8_t*>(output.buffer),
                                     outShape);
            }
        } break;
        case OperationType::AVERAGE_POOL: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];

            int32_t padding          = getScalarData<int32_t>(mOperands[ins[1]]);
            int32_t stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
            int32_t stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
            int32_t filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            int32_t filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            int32_t activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericPoolingPrepare(input.shape(),
                                                padding, stride_width, stride_height,
                                                filter_width, filter_height,
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          averagePoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                             input.shape(),
                                             padding, stride_width, stride_height,
                                             filter_width, filter_height, activation,
                                             reinterpret_cast<float*>(output.buffer),
                                             outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericPoolingPrepare(input.shape(),
                                                padding, stride_width, stride_height,
                                                filter_width, filter_height,
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          averagePoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                            input.shape(),
                                            padding, stride_width, stride_height,
                                            filter_width, filter_height, activation,
                                            reinterpret_cast<uint8_t*>(output.buffer),
                                            outShape);
            }
        } break;
        case OperationType::L2_POOL: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];

            int32_t padding          = getScalarData<int32_t>(mOperands[ins[1]]);
            int32_t stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
            int32_t stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
            int32_t filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            int32_t filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            int32_t activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericPoolingPrepare(input.shape(),
                                                padding, stride_width, stride_height,
                                                filter_width, filter_height,
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          l2PoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                        input.shape(),
                                        padding, stride_width, stride_height,
                                        filter_width, filter_height, activation,
                                        reinterpret_cast<float*>(output.buffer),
                                        outShape);
            }
        } break;
        case OperationType::MAX_POOL: {
            if (!parameterCountIs(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];

            int32_t padding          = getScalarData<int32_t>(mOperands[ins[1]]);
            int32_t stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
            int32_t stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
            int32_t filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
            int32_t filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
            int32_t activation       = getScalarData<int32_t>(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericPoolingPrepare(input.shape(),
                                                padding, stride_width, stride_height,
                                                filter_width, filter_height,
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          maxPoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                         input.shape(),
                                         padding, stride_width, stride_height,
                                         filter_width, filter_height, activation,
                                         reinterpret_cast<float*>(output.buffer),
                                         outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericPoolingPrepare(input.shape(),
                                                padding, stride_width, stride_height,
                                                filter_width, filter_height,
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          maxPoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                        input.shape(),
                                        padding, stride_width, stride_height,
                                        filter_width, filter_height, activation,
                                        reinterpret_cast<uint8_t*>(output.buffer),
                                        outShape);
            }

        } break;
        case OperationType::RELU: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          reluFloat32(reinterpret_cast<const float*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<float*>(output.buffer),
                                      outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          reluQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<uint8_t*>(output.buffer),
                                     outShape);
            }
        } break;
        case OperationType::RELU1: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          relu1Float32(reinterpret_cast<const float*>(input.buffer),
                                       input.shape(),
                                       reinterpret_cast<float*>(output.buffer),
                                       outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          relu1Quant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<uint8_t*>(output.buffer),
                                      outShape);
            }
        } break;
        case OperationType::RELU6: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          relu6Float32(reinterpret_cast<const float*>(input.buffer),
                                       input.shape(),
                                       reinterpret_cast<float*>(output.buffer),
                                       outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          relu6Quant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<uint8_t*>(output.buffer),
                                      outShape);
            }
        } break;
        case OperationType::TANH: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          tanhFloat32(reinterpret_cast<const float*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<float*>(output.buffer),
                                      outShape);
            }
        } break;
        case OperationType::LOGISTIC: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          logisticFloat32(reinterpret_cast<const float*>(input.buffer),
                                          input.shape(),
                                          reinterpret_cast<float*>(output.buffer),
                                          outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          logisticQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                         input.shape(),
                                         reinterpret_cast<uint8_t*>(output.buffer),
                                         outShape);
            }
        } break;
        case OperationType::SOFTMAX: {
            if (!parameterCountIs(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            RunTimeOperandInfo& input = mOperands[ins[0]];
            float beta = getScalarData<float>(mOperands[ins[1]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          softmaxFloat32(reinterpret_cast<const float*>(input.buffer),
                                         input.shape(),
                                         beta,
                                         reinterpret_cast<float*>(output.buffer),
                                         output.shape());
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          softmaxQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                        input.shape(),
                                        beta,
                                        reinterpret_cast<uint8_t*>(output.buffer),
                                        output.shape());
            }
        } break;
        case OperationType::FULLY_CONNECTED: {
            if (!parameterCountIs(4, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            RunTimeOperandInfo& input   = mOperands[ins[0]];
            RunTimeOperandInfo& weights = mOperands[ins[1]];
            RunTimeOperandInfo& bias    = mOperands[ins[2]];

            int32_t activation = getScalarData<int32_t>(mOperands[ins[3]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = fullyConnectedPrepare(input.shape(), weights.shape(), bias.shape(),
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          fullyConnectedFloat32(reinterpret_cast<const float*>(input.buffer),
                                                input.shape(),
                                                reinterpret_cast<const float*>(weights.buffer),
                                                weights.shape(),
                                                reinterpret_cast<const float*>(bias.buffer),
                                                bias.shape(),
                                                activation,
                                                reinterpret_cast<float*>(output.buffer),
                                                outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = fullyConnectedPrepare(input.shape(), weights.shape(), bias.shape(),
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          fullyConnectedQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                               input.shape(),
                                               reinterpret_cast<const uint8_t*>(weights.buffer),
                                               weights.shape(),
                                               reinterpret_cast<const int32_t*>(bias.buffer),
                                               bias.shape(),
                                               activation,
                                               reinterpret_cast<uint8_t*>(output.buffer),
                                               outShape);
            }
        } break;
        case OperationType::CONCATENATION: {
            if (outs.size() != 1 || ins.size() < 3) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            int numInputTensors = ins.size() - 2;
            int32_t axis = getScalarData<int32_t>(mOperands[ins[numInputTensors]]);
            int32_t activation = getScalarData<int32_t>(mOperands[ins[numInputTensors+1]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                std::vector<Shape> inputShapes(numInputTensors);
                std::vector<const float*> inputDataPtrs(numInputTensors);

                for (int i=0; i<numInputTensors; i++) {
                    RunTimeOperandInfo& input = mOperands[ins[i]];
                    inputShapes[i] = input.shape();
                    inputDataPtrs[i] = reinterpret_cast<const float*>(input.buffer);
                }
                success = concatenationPrepare(inputShapes, axis, &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          concatenationFloat32(inputDataPtrs, inputShapes,
                                               axis, activation,
                                               reinterpret_cast<float*>(output.buffer), outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                std::vector<Shape> inputShapes(numInputTensors);
                std::vector<const uint8_t*> inputDataPtrs(numInputTensors);

                for (int i=0; i<numInputTensors; i++) {
                    RunTimeOperandInfo& input = mOperands[ins[i]];
                    inputShapes[i] = input.shape();
                    inputDataPtrs[i] = reinterpret_cast<const uint8_t*>(input.buffer);
                }
                success = concatenationPrepare(inputShapes, axis, &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          concatenationQuant8(inputDataPtrs, inputShapes,
                                              axis, activation,
                                              reinterpret_cast<uint8_t*>(output.buffer),
                                              outShape);
            }
        } break;
        case OperationType::L2_NORMALIZATION: {
            if (!parameterCountIs(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericNormalizationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          l2normFloat32(reinterpret_cast<const float*>(input.buffer),
                                        input.shape(),
                                        reinterpret_cast<float*>(output.buffer),
                                        outShape);
            } else if (operation.opTuple.operandType == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericNormalizationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          l2normQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                       input.shape(),
                                       reinterpret_cast<uint8_t*>(output.buffer),
                                       outShape);
            }
        } break;
        case OperationType::LOCAL_RESPONSE_NORMALIZATION: {
            if (!parameterCountIs(5, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t radius = getScalarData<int32_t>(mOperands[ins[1]]);
            float bias = getScalarData<float>(mOperands[ins[2]]);
            float alpha = getScalarData<float>(mOperands[ins[3]]);
            float beta = getScalarData<float>(mOperands[ins[4]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = genericNormalizationPrepare(input.shape(), &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          localResponseNormFloat32(reinterpret_cast<const float*>(input.buffer),
                                                   input.shape(),
                                                   radius, bias, alpha, beta,
                                                   reinterpret_cast<float*>(output.buffer),
                                                   outShape);
            }
        } break;
        case OperationType::RESHAPE: {
            if (!parameterCountIs(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& targetShape = mOperands[ins[1]];

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = reshapePrepare(input.shape(),
                                     reinterpret_cast<const int32_t*>(targetShape.buffer),
                                     getNumberOfElements(targetShape.shape()),
                                     &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      reshapeGeneric(reinterpret_cast<const void*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<void*>(output.buffer),
                                     outShape);
        } break;
        case OperationType::RESIZE_BILINEAR: {
            if (!parameterCountIs(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t height = getScalarData<int32_t>(mOperands[ins[1]]);
            int32_t width = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (operation.opTuple.operandType == OperandType::TENSOR_FLOAT32) {
                success = resizeBilinearPrepare(input.shape(),
                                                height, width,
                                                &outShape) &&
                          allocateIfNeeded(&output, outShape) &&
                          resizeBilinearFloat32(reinterpret_cast<const float*>(input.buffer),
                                                input.shape(),
                                                reinterpret_cast<float*>(output.buffer),
                                                outShape);
            }
        } break;
        case OperationType::DEPTH_TO_SPACE: {
            if (!parameterCountIs(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t blockSize = getScalarData<int32_t>(mOperands[ins[1]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = depthToSpacePrepare(input.shape(),
                                          blockSize,
                                          &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      depthToSpaceGeneric(input.buffer,
                                          input.shape(),
                                          blockSize,
                                          output.buffer,
                                          outShape);
        } break;
        case OperationType::SPACE_TO_DEPTH: {
            if (!parameterCountIs(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t blockSize = getScalarData<int32_t>(mOperands[ins[1]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = spaceToDepthPrepare(input.shape(),
                                          blockSize,
                                          &outShape) &&
                      allocateIfNeeded(&output, outShape) &&
                      spaceToDepthGeneric(input.buffer,
                                          input.shape(),
                                          blockSize,
                                          output.buffer,
                                          outShape);
        } break;
        case OperationType::EMBEDDING_LOOKUP: {
            EmbeddingLookup lookup(operation, mOperands);
            success = lookup.Eval();
        } break;
        case OperationType::HASHTABLE_LOOKUP: {
            HashtableLookup lookup(operation, mOperands);
            success = lookup.Eval();
        } break;
        case OperationType::LSH_PROJECTION: {
            LSHProjection lsh(operation, mOperands);
            success = lsh.Eval();
        } break;
        case OperationType::LSTM: {
            LSTMCell lstm_cell(operation, mOperands);
            success = lstm_cell.Eval();
        } break;
        case OperationType::RNN: {
            RNN rnn_cell(operation, mOperands);
            success = rnn_cell.Eval();
        } break;
        case OperationType::SVDF: {
            SVDF svdf(operation, mOperands);
            success = svdf.Eval();
        } break;
        default:
            nnAssert(false);
            break;
    }
    if (!success) {
        LOG(ERROR) << getOperationName(operation.opTuple.operationType) << " failed.";
        return ANEURALNETWORKS_OP_FAILED;
    }

    freeNoLongerUsedOperands(ins);
    return ANEURALNETWORKS_NO_ERROR;
}

} // namespace nn
} // namespace android
