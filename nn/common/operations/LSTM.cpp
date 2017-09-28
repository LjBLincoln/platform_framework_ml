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

#include "LSTM.h"

#include "CpuExecutor.h"
#include "HalInterfaces.h"

namespace android {
namespace nn {

// TODO: move the kernels to a separate file as soon as we have the
// optimized version ready.
namespace {

// Limit a float input f between +abs_limit and -abs_limit.
inline float Clip(float f, float abs_limit) {
  float result = (abs_limit < f) ? abs_limit : f;
  result = (-abs_limit > result) ? -abs_limit : result;
  return result;
}

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector.
void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result) {
  for (int b = 0; b < n_batch; b++) {
    float* result_in_batch = result + b * m_rows;
    const float* matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++) {
      const float* vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++) {
        *result_in_batch += *matrix_ptr++ * *vector_in_batch++;
      }
      result_in_batch++;
    }
  }
}

// Cwise product of two vectors.
void VectorVectorCwiseProduct(const float* vector1, const float* vector2,
                              int v_size, float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = *vector1++ * *vector2++;
  }
}

// Cwise product and accumulation of two vectors. Since it's a MAC operation, the
// assumption here is that result array is initialized to valid values.
void VectorVectorCwiseProductAccumulate(const float* vector1,
                                        const float* vector2, int v_size,
                                        float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ += *vector1++ * *vector2++;
  }
}

// Cwise product and accumulation of a vector and a batch-vector. Since it's a MAC
// operation, the assumption here is that result array is initialized to valid
// values.
void VectorBatchVectorCwiseProductAccumulate(const float* vector, int v_size,
                                             const float* batch_vector,
                                             int n_batch, float* result) {
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < v_size; v++) {
      *result++ += vector[v] * *batch_vector++;
    }
  }
}

// Batch vector initialization with another vector.
void VectorBatchVectorAssign(const float* vector, int v_size, int n_batch,
                             float* batch_vector) {
  for (int b = 0; b < n_batch; b++) {
    memcpy(batch_vector + b * v_size, vector, v_size * sizeof(float));
  }
}

// Apply sigmoid to elements of a vector.
void ApplySigmoidToVector(const float* vector, int v_size, float* result) {
  auto sigmoid_func = ActivationFunctor(kActivationSigmoid);
  for (int v = 0; v < v_size; v++) {
    *result++ = (sigmoid_func)(*vector++);
  }
}

// Apply activation function to elements of a vector.
void ApplyActivationToVector(const float* vector, int v_size,
                             ActivationFn activation, float* result) {
  auto activation_func = ActivationFunctor(activation);
  for (int v = 0; v < v_size; v++) {
    *result++ = (activation_func)(*vector++);
  }
}

// Copy vector to another vector.
inline void CopyVector(const float* vector, int v_size, float* result) {
  memcpy(result, vector, v_size * sizeof(float));
}

// Compute "1.0f - elements of vector" (used in CIFG).
void Sub1Vector(const float* vector, int v_size, float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = 1.0f - *vector++;
  }
}

// Fill vector with 0.f.
void ZeroVector(float* vector, int v_size) {
  memset(vector, 0, v_size * sizeof(float));
}

// Clip elements of a vector using a abs_limit value.
void ClipVector(const float* vector, int v_size, float abs_limit,
                float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = Clip(*vector++, abs_limit);
  }
}

template <typename T>
inline T *GetBuffer(RunTimeOperandInfo* operand) {
  return reinterpret_cast<T*>(operand->buffer);
}

template <typename T>
inline const T *GetBuffer(const RunTimeOperandInfo* operand) {
  return reinterpret_cast<const T*>(operand->buffer);
}

}  // anonymous namespace

LSTMCell::LSTMCell(const Operation& operation,
                   std::vector<RunTimeOperandInfo>& operands) {
  input_ = GetInput(operation, operands, kInputTensor);

  input_to_input_weights_ = GetInput(operation, operands, kInputToInputWeightsTensor);  // optional
  input_to_forget_weights_ = GetInput(operation, operands, kInputToForgetWeightsTensor);
  input_to_cell_weights_ = GetInput(operation, operands, kInputToCellWeightsTensor);
  input_to_output_weights_ = GetInput(operation, operands, kInputToOutputWeightsTensor);

  recurrent_to_input_weights_ =
      GetInput(operation, operands, kRecurrentToInputWeightsTensor);  // optional
  recurrent_to_forget_weights_ = GetInput(operation, operands, kRecurrentToForgetWeightsTensor);
  recurrent_to_cell_weights_ = GetInput(operation, operands, kRecurrentToCellWeightsTensor);
  recurrent_to_output_weights_ = GetInput(operation, operands, kRecurrentToOutputWeightsTensor);

  cell_to_input_weights_ = GetInput(operation, operands, kCellToInputWeightsTensor);    // optional
  cell_to_forget_weights_ = GetInput(operation, operands, kCellToForgetWeightsTensor);  // optional
  cell_to_output_weights_ = GetInput(operation, operands, kCellToOutputWeightsTensor);  // optional

  input_gate_bias_ = GetInput(operation, operands, kInputGateBiasTensor);
  forget_gate_bias_ = GetInput(operation, operands, kForgetGateBiasTensor);
  cell_bias_ = GetInput(operation, operands, kCellGateBiasTensor);
  output_gate_bias_ = GetInput(operation, operands, kOutputGateBiasTensor);

  projection_weights_ = GetInput(operation, operands, kProjectionWeightsTensor);  // optional
  projection_bias_ = GetInput(operation, operands, kProjectionBiasTensor);        // optional

  params_.activation_ = static_cast<ActivationFn>(getScalarData<int32_t>(
      *GetInput(operation, operands, kActivationParam)));
  params_.cell_clip_ = getScalarData<float>(*GetInput(operation, operands, kCellClipParam));
  params_.proj_clip_ = getScalarData<float>(*GetInput(operation, operands, kProjClipParam));

  output_state_ = GetOutput(operation, operands, kOutputStateTensor);
  cell_state_ = GetOutput(operation, operands, kCellStateTensor);
  output_ = GetOutput(operation, operands, kOutputTensor);

  scratch_buffer_ = GetOutput(operation, operands, kScratchBufferTensor);
}

bool LSTMCell::CheckInputTensorDimensions(
    const Operation &operation, std::vector<RunTimeOperandInfo> &operands,
    uint32_t n_input, uint32_t n_output, uint32_t n_cell) {
  LSTMParams params = {
    .activation_ = static_cast<ActivationFn>(getScalarData<int32_t>(*GetInput(operation, operands, LSTMCell::kActivationParam))),
    .cell_clip_  = getScalarData<float>(*GetInput(operation, operands, LSTMCell::kCellClipParam)),
    .proj_clip_  = getScalarData<float>(*GetInput(operation, operands, LSTMCell::kProjClipParam))
  };

  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  NN_CHECK(params.cell_clip_ >= 0);
  NN_CHECK(params.proj_clip_ >= 0);

  const RunTimeOperandInfo *input_to_input_weights =
      GetInput(operation, operands, LSTMCell::kInputToInputWeightsTensor);
  if (!IsNullInput(input_to_input_weights)) {
    NN_CHECK_EQ(NumDimensions(input_to_input_weights), 2);
    NN_CHECK_EQ(SizeOfDimension(input_to_input_weights, 0), n_cell);
    NN_CHECK_EQ(SizeOfDimension(input_to_input_weights, 1), n_input);
  }

  const RunTimeOperandInfo *input_to_forget_weights =
      GetInput(operation, operands, LSTMCell::kInputToForgetWeightsTensor);
  NN_CHECK_EQ(NumDimensions(input_to_forget_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(input_to_forget_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(input_to_forget_weights, 1), n_input);

  const RunTimeOperandInfo *input_to_cell_weights =
      GetInput(operation, operands, LSTMCell::kInputToCellWeightsTensor);
  NN_CHECK_EQ(NumDimensions(input_to_cell_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(input_to_cell_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(input_to_cell_weights, 1), n_input);

  const RunTimeOperandInfo *recurrent_to_input_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToInputWeightsTensor);
  if (!IsNullInput(recurrent_to_input_weights)) {
    NN_CHECK_EQ(NumDimensions(recurrent_to_input_weights), 2);
    NN_CHECK_EQ(SizeOfDimension(recurrent_to_input_weights, 0), n_cell);
    NN_CHECK_EQ(SizeOfDimension(recurrent_to_input_weights, 1), n_output);
  }

  const RunTimeOperandInfo *recurrent_to_forget_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToForgetWeightsTensor);
  NN_CHECK_EQ(NumDimensions(recurrent_to_forget_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_forget_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_forget_weights, 1), n_output);

  const RunTimeOperandInfo *recurrent_to_cell_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToCellWeightsTensor);
  NN_CHECK_EQ(NumDimensions(recurrent_to_cell_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_cell_weights, 0), n_cell);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_cell_weights, 1), n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
      (!IsNullInput(input_to_input_weights) &&
       !IsNullInput(recurrent_to_input_weights)) ||
      (IsNullInput(input_to_input_weights) &&
       IsNullInput(recurrent_to_input_weights));
  NN_CHECK(cifg_weights_all_or_none);

  const RunTimeOperandInfo *cell_to_input_weights =
      GetInput(operation, operands, LSTMCell::kCellToInputWeightsTensor);
  if (!IsNullInput(cell_to_input_weights)) {
    NN_CHECK_EQ(NumDimensions(cell_to_input_weights), 1);
    NN_CHECK_EQ(SizeOfDimension(cell_to_input_weights, 0), n_cell);
  }

  const RunTimeOperandInfo *cell_to_forget_weights =
      GetInput(operation, operands, LSTMCell::kCellToForgetWeightsTensor);
  if (!IsNullInput(cell_to_forget_weights)) {
    NN_CHECK_EQ(NumDimensions(cell_to_forget_weights), 1);
    NN_CHECK_EQ(SizeOfDimension(cell_to_forget_weights, 0), n_cell);
  }

  const RunTimeOperandInfo *cell_to_output_weights =
      GetInput(operation, operands, LSTMCell::kCellToOutputWeightsTensor);
  if (!IsNullInput(cell_to_output_weights)) {
    NN_CHECK_EQ(NumDimensions(cell_to_output_weights), 1);
    NN_CHECK_EQ(SizeOfDimension(cell_to_output_weights, 0), n_cell);
  }

  // Making sure the peephole weights are there all or none.
  const bool use_cifg = IsNullInput(input_to_input_weights);
  const bool peephole_weights_all_or_none =
      ((!IsNullInput(cell_to_input_weights) || use_cifg) &&
       !IsNullInput(cell_to_forget_weights) &&
       !IsNullInput(cell_to_output_weights)) ||
      (IsNullInput(cell_to_input_weights) &&
       IsNullInput(cell_to_forget_weights) &&
       IsNullInput(cell_to_output_weights));
  NN_CHECK(peephole_weights_all_or_none);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  const RunTimeOperandInfo* input_gate_bias =
      GetInput(operation, operands, LSTMCell::kInputGateBiasTensor);
  if (use_cifg) {
    NN_CHECK(IsNullInput(input_gate_bias));
  } else {
    NN_CHECK_EQ(NumDimensions(input_gate_bias), 1);
    NN_CHECK_EQ(SizeOfDimension(input_gate_bias, 0), n_cell);
  }

  const RunTimeOperandInfo *forget_gate_bias =
      GetInput(operation, operands, LSTMCell::kForgetGateBiasTensor);
  NN_CHECK_EQ(NumDimensions(forget_gate_bias), 1);
  NN_CHECK_EQ(SizeOfDimension(forget_gate_bias, 0), n_cell);

  const RunTimeOperandInfo *cell_bias =
      GetInput(operation, operands, LSTMCell::kCellGateBiasTensor);
  NN_CHECK_EQ(NumDimensions(cell_bias), 1);
  NN_CHECK_EQ(SizeOfDimension(cell_bias, 0), n_cell);

  const RunTimeOperandInfo *output_gate_bias =
      GetInput(operation, operands, LSTMCell::kOutputGateBiasTensor);
  NN_CHECK_EQ(NumDimensions(output_gate_bias), 1);
  NN_CHECK_EQ(SizeOfDimension(output_gate_bias, 0), n_cell);

  const RunTimeOperandInfo *projection_weights =
      GetInput(operation, operands, LSTMCell::kProjectionWeightsTensor);
  if (!IsNullInput(projection_weights)) {
    NN_CHECK_EQ(NumDimensions(projection_weights), 2);
    NN_CHECK_EQ(SizeOfDimension(projection_weights, 0), n_output);
    NN_CHECK_EQ(SizeOfDimension(projection_weights, 1), n_cell);
  }

  const RunTimeOperandInfo *projection_bias =
      GetInput(operation, operands, LSTMCell::kProjectionBiasTensor);
  if (!IsNullInput(projection_bias)) {
    NN_CHECK_EQ(NumDimensions(projection_bias), 1);
    NN_CHECK_EQ(SizeOfDimension(projection_bias, 0), n_output);
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  // TODO: make sure this is correct.
  const bool projecton_tensors_consistent =
      (!IsNullInput(projection_weights) || IsNullInput(projection_bias));
  NN_CHECK(projecton_tensors_consistent == true);

  return true;
}

bool LSTMCell::Prepare(const Operation &operation,
                       std::vector<RunTimeOperandInfo> &operands,
                       Shape *scratchShape,
                       Shape *outputStateShape,
                       Shape *cellStateShape,
                       Shape *outputShape) {
  // Check we have all the inputs and outputs we need.
  NN_CHECK(NumInputsWithValues(operation, operands) >= 13 &&
           NumInputsWithValues(operation, operands) <= 21);
  NN_CHECK_EQ(NumOutputs(operation), 4);

  // Inferring batch size, number of outputs and number of cells from the
  // input tensors.
  const RunTimeOperandInfo *input =
      GetInput(operation, operands, LSTMCell::kInputTensor);
  NN_CHECK(NumDimensions(input) > 1);
  const uint32_t n_batch = SizeOfDimension(input, 0);
  const uint32_t n_input = SizeOfDimension(input, 1);

  const RunTimeOperandInfo *input_to_output_weights =
      GetInput(operation, operands, LSTMCell::kInputToOutputWeightsTensor);
  const uint32_t n_cell = SizeOfDimension(input_to_output_weights, 0);
  NN_CHECK_EQ(NumDimensions(input_to_output_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(input_to_output_weights, 1), n_input);

  const RunTimeOperandInfo *recurrent_to_output_weights =
      GetInput(operation, operands, LSTMCell::kRecurrentToOutputWeightsTensor);
  NN_CHECK_EQ(NumDimensions(recurrent_to_output_weights), 2);
  NN_CHECK_EQ(SizeOfDimension(recurrent_to_output_weights, 0),
                    n_cell);
  const uint32_t n_output = SizeOfDimension(recurrent_to_output_weights, 1);

  // Check that input tensor dimensions matches with each other.
  if (!CheckInputTensorDimensions(operation, operands, n_input, n_output, n_cell)) {
    return false;
  }

  // Resize the output and output_state tensors.
  const Shape &inputShape = input->shape();

  outputShape->type = inputShape.type;
  outputShape->dimensions = { n_batch, n_output };
  outputShape->offset = inputShape.offset;
  outputShape->scale = inputShape.scale;

  outputStateShape->type = inputShape.type;
  outputStateShape->dimensions = { n_batch, n_output };
  outputStateShape->offset = inputShape.offset;
  outputStateShape->scale = inputShape.scale;

  cellStateShape->type = inputShape.type;
  cellStateShape->dimensions = { n_batch, n_cell };
  cellStateShape->offset = inputShape.offset;
  cellStateShape->scale = inputShape.scale;

  const RunTimeOperandInfo *input_to_input_weights =
      GetInput(operation, operands, LSTMCell::kInputToInputWeightsTensor);
  const bool use_cifg = IsNullInput(input_to_input_weights);
  if (use_cifg) {
    // Reserving space for Cell, Forget, Output gates
    scratchShape->dimensions = { n_batch, n_cell * 3 };
  } else {
    // Reserving space for Input, Cell, Forget, Output gates
    scratchShape->dimensions = { n_batch, n_cell * 4 };
  }
  scratchShape->type = inputShape.type;
  scratchShape->offset = inputShape.offset;
  scratchShape->scale = inputShape.scale;

  return true;
}

bool LSTMCell::Eval() {
  const uint32_t n_batch = input_->shape().dimensions[0];
  const uint32_t n_input = input_->shape().dimensions[1];
  // n_cell and n_output will be the same size when there is no projection.
  const uint32_t n_cell = input_to_output_weights_->shape().dimensions[0];
  const uint32_t n_output = recurrent_to_output_weights_->shape().dimensions[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_->lifetime == OperandLifeTime::NO_VALUE);
  const bool use_peephole = (cell_to_output_weights_->lifetime != OperandLifeTime::NO_VALUE);

  // Index the scratch buffers pointers to the global scratch buffer.
  float* input_gate_scratch = nullptr;
  float* cell_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_scratch = reinterpret_cast<float*>(scratch_buffer_->buffer);
    forget_gate_scratch = cell_scratch + n_cell * n_batch;
    output_gate_scratch = cell_scratch + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = reinterpret_cast<float*>(scratch_buffer_->buffer);
    cell_scratch = input_gate_scratch + n_cell * n_batch;
    forget_gate_scratch = input_gate_scratch + 2 * n_cell * n_batch;
    output_gate_scratch = input_gate_scratch + 3 * n_cell * n_batch;
  }

  // Initialize scratch buffers with bias.
  if (!use_cifg) {
    VectorBatchVectorAssign(GetBuffer<float>(input_gate_bias_), n_cell, n_batch,
                            input_gate_scratch);
  }
  VectorBatchVectorAssign(GetBuffer<float>(forget_gate_bias_), n_cell, n_batch,
                          forget_gate_scratch);
  VectorBatchVectorAssign(GetBuffer<float>(cell_bias_), n_cell, n_batch,
                          cell_scratch);
  VectorBatchVectorAssign(GetBuffer<float>(output_gate_bias_), n_cell, n_batch,
                          output_gate_scratch);

  // For each batch and cell: compute input_weight * input.
  if (!use_cifg) {
    MatrixBatchVectorMultiplyAccumulate(
        GetBuffer<float>(input_to_input_weights_), n_cell, n_input,
        GetBuffer<float>(input_), n_batch, input_gate_scratch);
  }
  MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(input_to_forget_weights_), n_cell, n_input,
      GetBuffer<float>(input_), n_batch, forget_gate_scratch);
  MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(input_to_cell_weights_), n_cell, n_input,
      GetBuffer<float>(input_), n_batch, cell_scratch);
  MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(input_to_output_weights_), n_cell, n_input,
      GetBuffer<float>(input_), n_batch, output_gate_scratch);

  // For each batch and cell: compute recurrent_weight * output_state.
  if (!use_cifg) {
    MatrixBatchVectorMultiplyAccumulate(
        GetBuffer<float>(recurrent_to_input_weights_), n_cell, n_output,
        GetBuffer<float>(output_state_), n_batch, input_gate_scratch);
  }
  MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(recurrent_to_forget_weights_), n_cell, n_output,
      GetBuffer<float>(output_state_), n_batch, forget_gate_scratch);
  MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(recurrent_to_cell_weights_), n_cell, n_output,
      GetBuffer<float>(output_state_), n_batch, cell_scratch);
  MatrixBatchVectorMultiplyAccumulate(
      GetBuffer<float>(recurrent_to_output_weights_), n_cell, n_output,
      GetBuffer<float>(output_state_), n_batch, output_gate_scratch);

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole) {
      VectorBatchVectorCwiseProductAccumulate(
          GetBuffer<float>(cell_to_input_weights_), n_cell,
          GetBuffer<float>(cell_state_), n_batch, input_gate_scratch);
    }
    ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                         input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole) {
    VectorBatchVectorCwiseProductAccumulate(
        GetBuffer<float>(cell_to_forget_weights_), n_cell,
        GetBuffer<float>(cell_state_), n_batch, forget_gate_scratch);
  }
  ApplySigmoidToVector(forget_gate_scratch, n_cell * n_batch,
                       forget_gate_scratch);

  // For each batch and cell: update the cell.
  VectorVectorCwiseProduct(forget_gate_scratch, GetBuffer<float>(cell_state_),
                           n_batch * n_cell, GetBuffer<float>(cell_state_));
  ApplyActivationToVector(cell_scratch, n_batch * n_cell, params_.activation_,
                          cell_scratch);
  if (use_cifg) {
    Sub1Vector(forget_gate_scratch, n_batch * n_cell, forget_gate_scratch);
    VectorVectorCwiseProductAccumulate(cell_scratch, forget_gate_scratch,
                                       n_batch * n_cell,
                                       GetBuffer<float>(cell_state_));
  } else {
    VectorVectorCwiseProductAccumulate(cell_scratch, input_gate_scratch,
                                       n_batch * n_cell,
                                       GetBuffer<float>(cell_state_));
  }
  if (params_.cell_clip_ > 0.0) {
    ClipVector(GetBuffer<float>(cell_state_), n_batch * n_cell,
               params_.cell_clip_, GetBuffer<float>(cell_state_));
  }

  // For each batch and cell: update the output gate.
  if (use_peephole) {
    VectorBatchVectorCwiseProductAccumulate(
        GetBuffer<float>(cell_to_output_weights_), n_cell,
        GetBuffer<float>(cell_state_), n_batch, output_gate_scratch);
  }
  ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                       output_gate_scratch);
  ApplyActivationToVector(GetBuffer<float>(cell_state_), n_batch * n_cell,
                          params_.activation_, cell_scratch);
  VectorVectorCwiseProduct(output_gate_scratch, cell_scratch, n_batch * n_cell,
                           output_gate_scratch);

  // For each batch: update the projection and output_state.
  const bool use_projection_weight =
          (projection_weights_->lifetime != OperandLifeTime::NO_VALUE);
  const bool use_projection_bias = (projection_bias_->lifetime != OperandLifeTime::NO_VALUE);
  if (use_projection_weight) {
    if (use_projection_bias) {
      VectorBatchVectorAssign(GetBuffer<float>(projection_bias_), n_output,
                              n_batch, GetBuffer<float>(output_));
    } else {
      ZeroVector(GetBuffer<float>(output_), n_batch * n_output);
    }
    MatrixBatchVectorMultiplyAccumulate(GetBuffer<float>(projection_weights_),
                                        n_output, n_cell, output_gate_scratch,
                                        n_batch, GetBuffer<float>(output_));
    if (params_.proj_clip_ > 0.0) {
      ClipVector(GetBuffer<float>(output_), n_batch * n_output,
                 params_.proj_clip_, GetBuffer<float>(output_));
    }
  } else {
    CopyVector(output_gate_scratch, n_batch * n_output,
               GetBuffer<float>(output_));
  }
  CopyVector(GetBuffer<float>(output_), n_batch * n_output,
             GetBuffer<float>(output_state_));

  return true;
}

}  // namespace nn
}  // namespace android
