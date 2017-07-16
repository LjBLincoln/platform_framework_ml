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

#ifndef ANDROID_ML_NN_COMMON_HAL_ABSTRACTION_H
#define ANDROID_ML_NN_COMMON_HAL_ABSTRACTION_H

#include <vector>

// This class is used to abstract the HAL interface that will be created
// HIDL gen from the HIDL files.  We may not need this long term, although
// it is useful for running on a desktop without the HIDL compiler.

namespace android {
namespace nn {

// The types the operands can take.  These must be the same value as the NN API>
// TODO Use a single file for both.
enum class DataType {
    FLOAT16 = 0,
    FLOAT32 = 1,
    INT8 = 2,
    UINT8 = 3,
    INT16 = 4,
    UINT16 = 5,
    INT32 = 6,
    UINT32 = 7,
    TENSOR_FLOAT16 = 8,
    TENSOR_FLOAT32 = 9,
    TENSOR_SIMMETRICAL_QUANT8 = 10,

    NUM_DATA_TYPES = 11
};

// TODO There's currently a 1:1 mapping with the NN API constants.
// This will no longer be the case once an op supports more than one type.
// We'll need to add a conversion when finalizing the model.
enum class OperatorType {
    AVERAGE_POOL_FLOAT32 = 0,
    CONCATENATION_FLOAT32 = 1,
    CONV_FLOAT32 = 2,
    DEPTHWISE_CONV_FLOAT32 = 3,
    MAX_POOL_FLOAT32 = 4,
    L2_POOL_FLOAT32 = 5,
    DEPTH_TO_SPACE_FLOAT32 = 6,
    SPACE_TO_DEPTH_FLOAT32 = 7,
    LOCAL_RESPONSE_NORMALIZATION_FLOAT32 = 8,
    SOFTMAX_FLOAT32 = 9,
    RESHAPE_FLOAT32 = 10,
    SPLIT_FLOAT32 = 11,
    FAKE_QUANT_FLOAT32 = 12,
    ADD_FLOAT32 = 13,
    FULLY_CONNECTED_FLOAT32 = 14,
    CAST_FLOAT32 = 15,
    MUL_FLOAT32 = 16,
    L2_NORMALIZATION_FLOAT32 = 17,
    LOGISTIC_FLOAT32 = 18,
    RELU_FLOAT32 = 19,
    RELU6_FLOAT32 = 20,
    RELU1_FLOAT32 = 21,
    TANH_FLOAT32 = 22,
    DEQUANTIZE_FLOAT32 = 23,
    FLOOR_FLOAT32 = 24,
    GATHER_FLOAT32 = 25,
    RESIZE_BILINEAR_FLOAT32 = 26,
    LSH_PROJECTION_FLOAT32 = 27,
    LSTM_FLOAT32 = 28,
    SVDF_FLOAT32 = 29,
    RNN_FLOAT32 = 30,
    N_GRAM_FLOAT32 = 31,
    LOOKUP_FLOAT32 = 32,

    NUM_OPERATOR_TYPES = 33
};

// Status of a driver.
enum Status { AVAILABLE, BUSY, OFFLINE, UNKNOWN };

// Used by a driver to report its performance characteristics.
// TODO revisit the data types and scales.
struct PerformanceInfo {
    float execTime;    // in nanoseconds
    float powerUsage;  // in picoJoules
};

// Serialized representation of the model.
struct SerializedModel {
    std::vector<uint8_t> memory;
};

// The capabilities of a driver.
struct Capabilities {
    bool supportedOperatorTypes[static_cast<size_t>(OperatorType::NUM_OPERATOR_TYPES)];
    // TODO Do the same for baseline model IDs
    bool cachesCompilation;
    // TODO revisit the data types and scales.
    float bootupTime;  // in nanoseconds
    PerformanceInfo float16Performance;
    PerformanceInfo float32Performance;
    PerformanceInfo quantized8Performance;
};

// Informaton about one input or output operand of a model.
struct InputOutputInfo {
    void* buffer;
    uint32_t length;  // In bytes.
    // If true, the calling program has provided different dimensions for the
    // operand than was specified in the model.
    bool dimensionChanged;
    // The dimensions to use if the dimensions have been changed.
    std::vector<uint32_t> dimensions;
};

// See the HAL files for documentation on these interfaces.
class IEvent {
public:
    virtual ~IEvent(){}
    virtual uint32_t wait() = 0;
};

class IRequest {
public:
    virtual ~IRequest(){}
    virtual int execute(const std::vector<InputOutputInfo>& inputs,
                        const std::vector<InputOutputInfo>& outputs, IEvent** event) = 0;
    virtual void releaseTempMemory() = 0;
};

class IDevice {
public:
    virtual ~IDevice(){}
    virtual void initialize(Capabilities* capabilities) = 0;
    virtual void getSupportedSubgraph(void* graph, std::vector<bool>& canDo) = 0;
    virtual int prepareRequest(const SerializedModel* model, IRequest** request) = 0;
    virtual Status getStatus() = 0;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_COMMON_HAL_ABSTRACTION_H
