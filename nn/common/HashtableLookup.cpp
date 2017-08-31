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

#include "HashtableLookup.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"
#include "Operations.h"

namespace android {
namespace nn {

namespace {

int greater(const void* a, const void* b) {
  return *static_cast<const float*>(a) - *static_cast<const float*>(b);
}

}  // anonymous namespace

HashtableLookup::HashtableLookup(const Operation& operation,
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

  lookup_ = GetInput(kLookupTensor);
  key_ = GetInput(kKeyTensor);
  value_ = GetInput(kValueTensor);

  output_ = GetOutput(kOutputTensor);
  hits_ = GetOutput(kHitsTensor);
}

bool HashtableLookup::Eval() {
  const int num_rows = value_->shape().dimensions[0];
  const int row_bytes = sizeof(float) * value_->shape().dimensions[1];
  void* pointer = nullptr;

  for (int i = 0; i < static_cast<int>(lookup_->shape().dimensions[0]); i++) {
    int idx = -1;
    pointer = bsearch(lookup_->buffer + sizeof(float) * i, key_->buffer,
                      num_rows, sizeof(float), greater);
    if (pointer != nullptr) {
      idx =
          (reinterpret_cast<uint8_t*>(pointer) - key_->buffer) / sizeof(float);
    }

    if (idx >= num_rows || idx < 0) {
      memset(output_->buffer + i * row_bytes, 0, row_bytes);
      (reinterpret_cast<float*>(hits_->buffer))[i] = 0;
    } else {
      memcpy(output_->buffer + i * row_bytes, value_->buffer + idx * row_bytes,
             row_bytes);
      (reinterpret_cast<float*>(hits_->buffer))[i] = 1;
    }
  }

  return true;
}

}  // namespace nn
}  // namespace android
