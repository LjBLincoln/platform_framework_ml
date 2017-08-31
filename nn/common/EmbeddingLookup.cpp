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

#include "EmbeddingLookup.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"
#include "Operations.h"

namespace android {
namespace nn {

EmbeddingLookup::EmbeddingLookup(const Operation& operation,
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

  value_ = GetInput(kValueTensor);
  lookup_ = GetInput(kLookupTensor);

  output_ = GetOutput(kOutputTensor);
}

bool EmbeddingLookup::Eval() {
  auto multiAll = [](const std::vector<uint32_t> &dims) -> uint32_t {
    uint32_t sz = 1;
    for (uint32_t d : dims) { sz *= d; }
    return sz;
  };
  const int row_size = value_->shape().dimensions[0];
  const int total_bytes = sizeof(float) * multiAll(value_->shape().dimensions);
  const int row_bytes = total_bytes/row_size;

  for (uint32_t i = 0; i < lookup_->shape().dimensions[0]; i++) {
    int idx = static_cast<int>((reinterpret_cast<float*>(lookup_->buffer))[i]);
    LOG(INFO) << "idx=" << idx;
    if (idx >= row_size || idx < 0) {
      LOG(ERROR) << "Embedding Lookup: index out of bounds.";
      return false;
    } else {
      memcpy(output_->buffer + i * row_bytes, value_->buffer + idx * row_bytes,
             row_bytes);
    }
  }

  return true;
}

}  // namespace nn
}  // namespace android
