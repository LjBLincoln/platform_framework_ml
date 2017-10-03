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
    input_ = GetInput(operation, operands, kInputTensor);
    weights_feature_ = GetInput(operation, operands, kWeightsFeatureTensor);
    weights_time_ = GetInput(operation, operands, kWeightsTimeTensor);
    bias_ = GetInput(operation, operands, kBiasTensor);

    params_.rank_ = getInt32ScalarData(*GetInput(operation, operands, kRankParam));
    params_.activation_ = static_cast<ActivationFn>(getInt32ScalarData(
        *GetInput(operation, operands, kActivationParam)));

    state_ = GetOutput(operation, operands, kStateTensor);
    output_ = GetOutput(operation, operands, kOutputTensor);
}

bool SVDF::Prepare(const Operation &operation,
                   std::vector<RunTimeOperandInfo> &operands,
                   Shape *stateShape,
                   Shape *outputShape) {
  // Check we have all the inputs and outputs we need.
  const int num_inputs = NumInputsWithValues(operation, operands);
  NN_CHECK(num_inputs == 5 || num_inputs == 6);
  NN_CHECK_EQ(NumOutputs(operation), 2);

  const RunTimeOperandInfo *input =
      GetInput(operation, operands, SVDF::kInputTensor);
  const RunTimeOperandInfo *weights_feature =
      GetInput(operation, operands, SVDF::kWeightsFeatureTensor);
  const RunTimeOperandInfo *weights_time =
      GetInput(operation, operands, SVDF::kWeightsTimeTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const uint32_t batch_size = SizeOfDimension(input, 0);
  const uint32_t num_units = SizeOfDimension(weights_feature, 0);
  const uint32_t memory_size = SizeOfDimension(weights_time, 1);
  NN_CHECK_EQ(SizeOfDimension(input, 1), SizeOfDimension(weights_feature, 1));
  NN_CHECK_EQ(SizeOfDimension(weights_time, 0), num_units);

  const RunTimeOperandInfo *bias =
      GetInput(operation, operands, kBiasTensor);
  if (!IsNullInput(bias)) {
    NN_CHECK_EQ(SizeOfDimension(bias, 0), num_units);
  }

  const RunTimeOperandInfo *state =
      GetInput(operation, operands, SVDF::kStateTensor);
  const RunTimeOperandInfo *output =
      GetInput(operation, operands, SVDF::kOutputTensor);

  // Resize state.
  const Shape &inputShape = input->shape();
  stateShape->type = inputShape.type;
  stateShape->dimensions = { batch_size, memory_size * num_units };
  stateShape->offset = inputShape.offset;
  stateShape->scale = inputShape.scale;

  // Resize output.
  outputShape->type = inputShape.type;
  outputShape->dimensions = { batch_size, num_units };
  outputShape->offset = inputShape.offset;
  outputShape->scale = inputShape.scale;

  return true;
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
