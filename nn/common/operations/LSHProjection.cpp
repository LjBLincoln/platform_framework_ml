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

#include "LSHProjection.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"
#include "util/hash/farmhash.h"
//#include "farmhash.h"

namespace android {
namespace nn {

namespace {

template <typename T>
T getScalarData(RunTimeOperandInfo& info) {
  T* data = reinterpret_cast<T*>(info.buffer);
  return data[0];
}

}  // anonymous namespace

LSHProjection::LSHProjection(const Operation& operation,
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
  weight_ = GetInput(kWeightTensor);
  hash_ = GetInput(kHashTensor);

  type_ = static_cast<LSHProjectionType>(
      getScalarData<int32_t>(operands[operation.inputs[kTypeParam]]));

  output_ = GetOutput(kOutputTensor);
}

int SizeOfDimension(const RunTimeOperandInfo* operand, int dim) {
  return operand->shape().dimensions[dim];
}

// Compute sign bit of dot product of hash(seed, input) and weight.
// NOTE: use float as seed, and convert it to double as a temporary solution
//       to match the trained model. This is going to be changed once the new
//       model is trained in an optimized method.
//
int running_sign_bit(const RunTimeOperandInfo* input,
                     const RunTimeOperandInfo* weight, float seed) {
  double score = 0.0;
  int input_item_bytes = sizeOfData(input->type, input->dimensions) /
      SizeOfDimension(input, 0);
  char* input_ptr = (char*)(input->buffer);

  const size_t seed_size = sizeof(float);
  const size_t key_bytes = sizeof(float) + input_item_bytes;
  std::unique_ptr<char[]> key(new char[key_bytes]);

  for (int i = 0; i < SizeOfDimension(input, 0); ++i) {
    // Create running hash id and value for current dimension.
    memcpy(key.get(), &seed, seed_size);
    memcpy(key.get() + seed_size, input_ptr, input_item_bytes);

    int64_t hash_signature = farmhash::Fingerprint64(key.get(), key_bytes);
    double running_value = static_cast<double>(hash_signature);
    input_ptr += input_item_bytes;
    if (weight->buffer == nullptr) {
      score += running_value;
    } else {
      score += reinterpret_cast<float*>(weight->buffer)[i] * running_value;
    }
  }

  return (score > 0) ? 1 : 0;
}

void SparseLshProjection(const RunTimeOperandInfo* hash,
                         const RunTimeOperandInfo* input,
                         const RunTimeOperandInfo* weight, int32_t* out_buf) {
  int num_hash = SizeOfDimension(hash, 0);
  int num_bits = SizeOfDimension(hash, 1);
  for (int i = 0; i < num_hash; i++) {
    int32_t hash_signature = 0;
    for (int j = 0; j < num_bits; j++) {
      float seed = reinterpret_cast<float*>(hash->buffer)[i * num_bits + j];
      int bit = running_sign_bit(input, weight, seed);
      hash_signature = (hash_signature << 1) | bit;
    }
    *out_buf++ = hash_signature;
  }
}

void DenseLshProjection(const RunTimeOperandInfo* hash,
                        const RunTimeOperandInfo* input,
                        const RunTimeOperandInfo* weight, int32_t* out_buf) {
  int num_hash = SizeOfDimension(hash, 0);
  int num_bits = SizeOfDimension(hash, 1);
  for (int i = 0; i < num_hash; i++) {
    for (int j = 0; j < num_bits; j++) {
      float seed = reinterpret_cast<float*>(hash->buffer)[i * num_bits + j];
      int bit = running_sign_bit(input, weight, seed);
      *out_buf++ = bit;
    }
  }
}

bool LSHProjection::Eval() {
  int32_t* out_buf = reinterpret_cast<int32_t*>(output_->buffer);

  switch (type_) {
    case LSHProjectionType_DENSE:
      DenseLshProjection(hash_, input_, weight_, out_buf);
      break;
    case LSHProjectionType_SPARSE:
      SparseLshProjection(hash_, input_, weight_, out_buf);
      break;
    default:
      return false;
  }
  return true;
}

}  // namespace nn
}  // namespace android
