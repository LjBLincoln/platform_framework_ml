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

#include "RNN.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"

namespace android {
namespace nn {

namespace {

template <typename T>
T getScalarData(RunTimeOperandInfo& info) {
  T* data = reinterpret_cast<T*>(info.buffer);
  return data[0];
}

}  // anonymous namespace

RNN::RNN(const Operation& operation,
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
  weights_ = GetInput(kWeightsTensor);
  recurrent_weights_ = GetInput(kRecurrentWeightsTensor);
  bias_ = GetInput(kBiasTensor);
  hidden_state_ = GetInput(kHiddenStateTensor);

  activation_ = static_cast<ActivationFn>(
      getScalarData<int32_t>(operands[operation.inputs[kActivationParam]]));

  output_ = GetOutput(kOutputTensor);
}

bool RNN::Eval() {
  const float* bias_ptr = reinterpret_cast<float*>(bias_->buffer);

  const uint32_t batch_size = input_->shape().dimensions[0];
  const uint32_t num_units = weights_->shape().dimensions[0];
  const uint32_t input_size = input_->shape().dimensions[1];
  const uint32_t input_weights_stride = weights_->shape().dimensions[1];
  const uint32_t recurrent_weights_stride =
      recurrent_weights_->shape().dimensions[1];

  // For each batch
  for (uint32_t b = 0; b < batch_size; b++) {
    // Initialize the pointer to input, output and bias.
    const float* input_ptr_batch =
        reinterpret_cast<float*>(input_->buffer) + b * input_size;
    float* output_ptr_batch =
        reinterpret_cast<float*>(output_->buffer) + b * num_units;
    float* hidden_state_ptr_batch =
        reinterpret_cast<float*>(hidden_state_->buffer) + b * num_units;

    // Initialize input_weights and recurrent_weights.
    const float* input_weights_ptr = reinterpret_cast<float*>(weights_->buffer);
    const float* recurrent_weights_ptr =
        reinterpret_cast<float*>(recurrent_weights_->buffer);

    // Output = bias
    for (uint32_t o = 0; o < num_units; o++) {
      output_ptr_batch[o] = bias_ptr[o];
    }

    // Output += input * input_weights
    for (uint32_t o = 0; o < num_units; o++) {
      for (uint32_t i = 0; i < input_size; i++) {
        output_ptr_batch[o] += input_ptr_batch[i] * input_weights_ptr[i];
      }
      input_weights_ptr += input_weights_stride;
    }

    // Output += recurrent_weights * hidden_state
    for (uint32_t o = 0; o < num_units; o++) {
      for (uint32_t h = 0; h < num_units; h++) {
        output_ptr_batch[o] +=
            hidden_state_ptr_batch[h] * recurrent_weights_ptr[h];
      }
      recurrent_weights_ptr += recurrent_weights_stride;
    }

    // Output = activation(Output) and update hidden_state
    for (uint32_t o = 0; o < num_units; o++) {
      output_ptr_batch[o] =
          (ActivationFunctor(activation_))(output_ptr_batch[o]);
      hidden_state_ptr_batch[o] = output_ptr_batch[o];
    }
  }

  return true;
}

}  // namespace nn
}  // namespace android
