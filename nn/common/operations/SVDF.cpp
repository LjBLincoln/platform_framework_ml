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

#include "SVDF.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"

namespace android {
namespace nn {

namespace {

// TODO: Implement this using circular buffer instead.
// This is here temporarily only to show the logic.
void svdf_right_shift_state(float* state, int state_len, float shift_value) {
  for (int i = 0; i < state_len - 1; i++) {
    state[i] = state[i + 1];
  }
  state[state_len - 1] = shift_value;
}

int32_t getInt32ScalarData(RunTimeOperandInfo& info) {
    int32_t * data = reinterpret_cast<int32_t*>(info.buffer);
    return data[0];
}

}

SVDF::SVDF(const Operation& operation,
           std::vector<RunTimeOperandInfo>& operands) {
    auto GetInput = [&operation,
                     &operands](uint32_t index) -> const RunTimeOperandInfo* {
        const std::vector<uint32_t>& inputs = operation.inputs;
        const int index_of_operand = inputs[index];
        if (index_of_operand < 0) {
            return nullptr;
        }
        return &operands[index_of_operand];
    };

    auto GetOutput = [&operation,
                      &operands](uint32_t index) -> RunTimeOperandInfo* {
        const std::vector<uint32_t>& outputs = operation.outputs;
        const int index_of_operand = outputs[index];
        // Expects index of operand in range.
        return &operands[index_of_operand];
    };

    input_ = GetInput(kInputTensor);
    weights_feature_ = GetInput(kWeightsFeatureTensor);
    weights_time_ = GetInput(kWeightsTimeTensor);
    bias_ = GetInput(kBiasTensor);

    params_.rank_ = getInt32ScalarData(operands[operation.inputs[kRankParam]]);
    params_.activation_ = static_cast<ActivationFn>(getInt32ScalarData(operands[operation.inputs[kActivationParam]]));

    state_ = GetOutput(kStateTensor);
    output_ = GetOutput(kOutputTensor);
}

bool SVDF::Eval() {
    const int batch_size = input_->shape().dimensions[0];
    const int input_size = input_->shape().dimensions[1];
    const int num_units = weights_feature_->shape().dimensions[0];
    const int memory_size = weights_time_->shape().dimensions[1];
    const int weights_feature_stride = weights_feature_->shape().dimensions[1];
    const int weights_time_stride = weights_time_->shape().dimensions[1];

    // Initialize weights_feature and weights_time pointers.
    const float* weights_feature_ptr = reinterpret_cast<float *>(weights_feature_->buffer);
    const float* weights_time_ptr = reinterpret_cast<float *>(weights_time_->buffer);

    // For each batch
    for (int b = 0; b < batch_size; b++) {
        // Initialize the pointer to input, output and bias.
        const float* input_ptr_batch = reinterpret_cast<float *>(input_->buffer) + b * input_size;
        float* output_ptr_batch = reinterpret_cast<float*>(output_->buffer) + b * num_units;
        float* state_ptr_batch = reinterpret_cast<float*>(state_->buffer) + b * (memory_size - 1) * num_units;

        // For each unit
        for (int c = 0; c < num_units; c++) {
            float activation = 0.0;

            // tf.nn.conv1d(inputs, weights_feature, feature_dim, "VALID")
            for (int j = 0; j < input_size; j++) {
                activation += input_ptr_batch[j] * weights_feature_ptr[j];
            }

            // Initialize state pointer for unit 'c'.
            float* state_ptr = state_ptr_batch + c * (memory_size - 1);

            // Apply bias if bias tensor exists.
            output_ptr_batch[c] = bias_->buffer ? reinterpret_cast<float *>(bias_->buffer)[c] : 0.f;

            // output = tf.matmul(state, weights_time)
            output_ptr_batch[c] += weights_time_ptr[memory_size - 1] * activation;
            for (int j = 0; j < memory_size - 1; j++) {
                output_ptr_batch[c] += weights_time_ptr[j] * state_ptr[j];
            }

            // Apply activation.
            output_ptr_batch[c] =
                    (ActivationFunctor(params_.activation_))(output_ptr_batch[c]);

            // Right shift the state and concatenate with activation.
            svdf_right_shift_state(state_ptr, memory_size - 1, activation);

            // Update weight pointers.
            weights_feature_ptr += weights_feature_stride;
            weights_time_ptr += weights_time_stride;
        }
        // Reset weight pointers for next batch.
        weights_feature_ptr = reinterpret_cast<float*>(weights_feature_->buffer);
        weights_time_ptr = reinterpret_cast<float*>(weights_time_->buffer);
    }
    return true;
}

}  // namespace nn
}  // namespace android
