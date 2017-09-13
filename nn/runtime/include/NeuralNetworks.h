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

/**
 * @addtogroup NeuralNetworks
 * @{
 */

/**
 * @file NeuralNetworks.h
 */

#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_H

/******************************************************************
 *
 * IMPORTANT NOTICE:
 *
 *   This file is part of Android's set of stable system headers
 *   exposed by the Android NDK (Native Development Kit).
 *
 *   Third-party source AND binary code relies on the definitions
 *   here to be FROZEN ON ALL UPCOMING PLATFORM RELEASES.
 *
 *   - DO NOT MODIFY ENUMS (EXCEPT IF YOU ADD NEW 32-BIT VALUES)
 *   - DO NOT MODIFY CONSTANTS OR FUNCTIONAL MACROS
 *   - DO NOT CHANGE THE SIGNATURE OF FUNCTIONS IN ANY WAY
 *   - DO NOT CHANGE THE LAYOUT OR SIZE OF STRUCTURES
 */

#if __ANDROID_API__ >= __ANDROID_API_O_MR1__

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * Operand types.
 *
 * The type of operands that can be added to a model.
 *
 * Although we define many types, most operators accept just a few
 * types. Most used are {@link ANEURALNETWORKS_TENSOR_FLOAT32},
 * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
 * and {@link ANEURALNETWORKS_INT32}.
 */
typedef enum {
    /** The following entries are used to declare scalars. */

    /** A 16 bit floating point scalar value. */
    ANEURALNETWORKS_FLOAT16 = 0,
    /** A 32 bit floating point scalar value. */
    ANEURALNETWORKS_FLOAT32 = 1,
    /** A signed 8 bit integer scalar value. */
    ANEURALNETWORKS_INT8 = 2,
    /** An unsigned 8 bit integer scalar value. */
    ANEURALNETWORKS_UINT8 = 3,
    /** A signed 16 bit integer scalar value. */
    ANEURALNETWORKS_INT16 = 4,
    /** An unsigned 16 bit integer scalar value. */
    ANEURALNETWORKS_UINT16 = 5,
    /** A signed 32 bit integer scalar value. */
    ANEURALNETWORKS_INT32 = 6,
    /** An unsigned 32 bit integer scalar value. */
    ANEURALNETWORKS_UINT32 = 7,

    /** The following entries are used to declare tensors. */

    /** A tensor of 16 bit floating point values. */
    ANEURALNETWORKS_TENSOR_FLOAT16 = 8,
    /** A tensor of 32 bit floating point values. */
    ANEURALNETWORKS_TENSOR_FLOAT32 = 9,
    /** A tensor of 32 bit integer values. */
    ANEURALNETWORKS_TENSOR_INT32 = 10,
    /** A tensor of 8 bit integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that can be used to convert
     * the 8 bit integer to the real value and vice versa. These two numbers are:
     * - scale: a 32 bit floating point value
     * - zero_value: an 32 bit integer
     *
     * The formula is:
     * real_value = (integer_value - zero_value) * scale.
     */
    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 11,
} OperandCode;

/**
 * Operation types.
 *
 * The type of operations that can be added to a model.
 */
typedef enum {
    /** OEM specific operation.
     *
     * This operation is OEM specific. It should only be used for OEM applications.
     */
    ANEURALNETWORKS_OEM_OPERATION = 0,
    /** Adds two tensors, elment-wise.
     *
     * Takes two input tensors of identical type and compatible dimensions. The output
     * is the sum of both input tensors, optionally modified by an activation function.
     *
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     *
     * The size of the output is the maximum size along each dimension of the input operands.
     * It starts with the trailing dimensions, and works its way forward.
     *
     * Example:
     *
     *     input1.dimension = {4, 1, 2}
     *     input2.dimension = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same type, and compatible dimensions as input0.
     * * 2: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The sum, a tensor of the same type as input0.
     */
    ANEURALNETWORKS_ADD = 1,
    /** Performs a 2-D average pooling operation.
     *
     * The output dimensions are functions of the filter dimensions, stride, and padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[batch, row, col, channel] =
     *         sum_{i, j}(input[batch, row + i, col + j, channel]) / sum(1)
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
     * * 1: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
     * * 2: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
     * * 3: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
     * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
     * * 5: An INT32 value, specifying the output stride in the ‘width’ dimension.
     * * 6: An INT32 value, specifying the output stride in the ‘height’ dimension.
     * * 7: An INT32 value, specifying the filter width.
     * * 8: An INT32 value, specifying the filter height.
     * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
     */
    ANEURALNETWORKS_AVERAGE_POOL_2D = 2,
    /** Concatenates the input tensors along the given dimension.
     *
     * The input tensors must have identical type and the same dimensions except the
     * dimension along the concatenation axis.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * 0 ~ n: The list on n input tensors, of shape [D0, D1, ..., Daxis(i), ..., Dm]
     * n+1: An INT32 value, specifying the concatenation axis.
     * n+2: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The output, a tensor of the same type as the input tensors.
     *      The output shape is [D0, D1, ..., sum(Daxis(i)), ..., Dm].
     */
    ANEURALNETWORKS_CONCATENATION = 3,
    /** Performs an 2-D convolution operation.
     *
     * The CONV_2D op sweeps a 2-D filter that can mix channels together over a batch of
     * images, applying the filter to each window of each image of the appropriate size.
     *
     * The output dimensions are functions of the filter dimensions, stride, and padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[batch, row, col, channel] =
     *         sum_{i, j} (
     *             input[batch, row + i, col + j, k] *
     *             filter[channel, row + i, col + j, k] +
     *             bias[channel]
     *         )
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
     * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
     *      specifying the filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32} type, the bias should
     *      also be of {@link ANEURALNETWORKS_TENSOR_FLOAT32}.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} type, the bias
     *      should be of {@link ANEURALNETWORKS_TENSOR_INT32}.
     * * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
     * * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
     * * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
     * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
     * * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
     * * 8: An INT32 value, specifying the output stride in the ‘height’ dimension.
     * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth_out].
     */
    ANEURALNETWORKS_CONV_2D = 4,
    /** Performs a depthwise 2-D convolution operation.
     *
     * Given an input tensor of shape [batches, height, width, depth_in] and a filter
     * tensor of shape [depth_out, filter_height, filter_width, depth_in] containing
     * in_channels convolutional filters of depth 1, DEPTHWISE_CONV applies a different
     * filter to each input channel (expanding from 1 channel to channel_multiplier channels
     * for each), then concatenates the results together.
     *
     * The output has depth_out = depth_in * depth_multiplier channels.
     * The output dimensions are functions of the filter dimensions, stride, and padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, i, j, k * channel_multiplier + q] =
     *         sum_{di, dj} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
     *             filter[di, dj, k, q]
     *         )
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
     * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
     *      specifying the filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32} type, the bias should
     *      also be of {@link ANEURALNETWORKS_TENSOR_FLOAT32}.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} type, the bias
     *      should be of {@link ANEURALNETWORKS_TENSOR_INT32}.
     * * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
     * * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
     * * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
     * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
     * * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
     * * 8: An INT32 value, specifying the output stride in the ‘height’ dimension.
     * * 9: An INT32 value, specifying the depthwise multiplier.
     * * 10: An INT32 value, and has to be one of the {@link FuseCode} values.
     *       Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth_out].
     */
    ANEURALNETWORKS_DEPTHWISE_CONV_2D = 5,
    /** Rearranges data from depth into blocks of spatial data.
     *
     * More specifically, this op outputs a copy of the input tensor where values from
     * the depth dimension are moved in spatial blocks to the height and width dimensions.
     * The value block_size indicates the input block size and how the data is moved.
     *
     * Chunks of data of size block_size * block_size from depth are rearranged into
     * non-overlapping blocks of size block_size x block_size.
     *
     * The width of the output tensor is input_depth * block_size, whereas the height is
     * input_height * block_size.
     * The depth of the input tensor must be divisible by block_size * block_size
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
     * * 1: An INT32 value, specifying the block_size. block_size must be >=1 and
     *      block_size * block_size must be a divisor of the input depth.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batch, height*block_size, width*block_size,
     *      depth/(block_size*block_size)].
     */
    ANEURALNETWORKS_DEPTH_TO_SPACE = 6,
    /** Dequantizes the input tensor.
     *
     * The formula is:
     *
     *     output = (input - zero_value) * scale.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0, but with type
     *      {@link ANEURALNETWORKS_TENSOR_FLOAT32}.
     */
    ANEURALNETWORKS_DEQUANTIZE = 7,
    ANEURALNETWORKS_EMBEDDING_LOOKUP = 8,
    ANEURALNETWORKS_FAKE_QUANT = 9,
    /** Computes element-wise floor() on the input tensor.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor.
     *
     * Outputs:
     * * 0: The output, a tensor of the same type and dimensions as input0.
     */
    ANEURALNETWORKS_FLOOR = 10,
    /** Denotes a fully (densely) connected layer, which connects all elements in the input
     * tensor with each element in the output tensor.
     *
     * This layer implements the operation:
     *
     *     outputs = activation(inputs * weights’ + bias)
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input. If rank is greater than 2, then it gets flattened to
     *      a 2-D Tensor. The 2-D Tensor is handled as if dimensions corresponded to shape
     *      [batch_size, input_size], where “batch_size” corresponds to the batching dimension,
     *      and “input_size” is the size of the input.
     * * 1: A 2-D tensor, specifying the weights, of shape [num_units, input_size], where “num_units”
     *      corresponds to the number of output nodes.
     * * 2: A 1-D tensor, of shape [num_units], specifying the bias.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32} type, the bias should
     *      also be of {@link ANEURALNETWORKS_TENSOR_FLOAT32}.
     *      For input tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} type, the bias
     *      should be of {@link ANEURALNETWORKS_TENSOR_INT32}.
     * * 3: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The output tensor, of shape [batch_size, num_units].
     */
    ANEURALNETWORKS_FULLY_CONNECTED = 11,
    ANEURALNETWORKS_HASHTABLE_LOOKUP = 12,
    /** Applies L2 normalization along the depth dimension.
     *
     * The values in the output tensor are computed as:
     *
     *     output[batch, row, col, channel] =
     *         input[batch, row, col, channel] /
     *         sqrt(sum_{c} pow(input[batch, row, col, c], 2))
     *
     * For x with more dimensions, independently normalizes each 1-D slice along dimension dim.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
     */
    ANEURALNETWORKS_L2_NORMALIZATION = 13,
    /** Performs an 2-D L2 pooling operation.
     *
     * The output dimensions are functions of the filter dimensions, stride, and padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[batch, row, col, channel] =
     *         sqrt(sum_{i, j} pow(input[batch, row + i, col + j, channel], 2) / sum(1))
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
     * * 1: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
     * * 2: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
     * * 3: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
     * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
     * * 5: An INT32 value, specifying the output stride in the ‘width’ dimension.
     * * 6: An INT32 value, specifying the output stride in the ‘height’ dimension.
     * * 7: An INT32 value, specifying the filter width.
     * * 8: An INT32 value, specifying the filter height.
     * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
     */
    ANEURALNETWORKS_L2_POOL_2D = 14,
    /** Applies Local Response Normalization along the depth dimension.
     *
     * The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last
     * dimension), and each vector is normalized independently. Within a given vector,
     * each component is divided by the weighted, squared sum of inputs within depth_radius.
     *
     * The output is calculated using this formula:
     *
     *     sqr_sum[a, b, c, d] =
     *         sum(pow(input[a, b, c, d - depth_radius : d + depth_radius + 1], 2)
     *     output = input / pow((bias + alpha * sqr_sum), beta)
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
     * * 1: An INT32 value, specifying the radius of the normalization window.
     * * 2: A FLOAT32 value, specifying the bias, must not be zero.
     * * 3: A FLOAT32 value, specifying the scale factor, alpha.
     * * 4: A FLOAT32 value, specifying the exponent, beta.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     */
    ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION = 15,
    /** Computes sigmoid activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = 1 / (1 + exp(-input))
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     */
    ANEURALNETWORKS_LOGISTIC = 16,
    ANEURALNETWORKS_LSH_PROJECTION = 17,
    ANEURALNETWORKS_LSTM = 18,
    /** Performs an 2-D max pooling operation.
     *
     * The output dimensions are functions of the filter dimensions, stride, and padding.
     *
     * The values in the output tensor are computed as:
     *
     *     output[batch, row, col, channel] =
     *         max_{i, j} (input[batch, row + i, col + j, channel])
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
     * * 1: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
     * * 2: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
     * * 3: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
     * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
     * * 5: An INT32 value, specifying the output stride in the ‘width’ dimension.
     * * 6: An INT32 value, specifying the output stride in the ‘height’ dimension.
     * * 7: An INT32 value, specifying the filter width.
     * * 8: An INT32 value, specifying the filter height.
     * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
     */
    ANEURALNETWORKS_MAX_POOL_2D = 19,
    /** Multiplies two tensors, elment-wise.
     *
     * Takes two input tensors of identical type and compatible dimensions. The output
     * is the product of both input tensors, optionally modified by an activation function.
     *
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     *
     * The size of the resulting output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its way forward.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     *
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same type, and compatible dimensions as input0.
     * * 2: An INT32 value, and has to be one of the {@link FuseCode} values.
     *      Specifies the activation to invoke on the result of each addition.
     *
     * Outputs:
     * * 0: The product, a tensor of the same type as input0.
     */
    ANEURALNETWORKS_MUL = 20,
    /** Computes rectified linear activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = max(0, input)
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     */
    ANEURALNETWORKS_RELU = 21,
    /** Computes rectified linear 1 activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = min(1.f, max(-1.f, input))
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     */
    ANEURALNETWORKS_RELU1 = 22,
    /** Computes rectified linear 6 activation on the input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = min(6, max(0, input))
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     */
    ANEURALNETWORKS_RELU6 = 23,
    /** Reshapes a tensor.
     *
     * Given tensor, this operation returns a tensor that has the same values as tensor,
     * but with a newly specified shape.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the tensor to be reshaped.
     * * 1: A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32}, defining the shape
     *      of the output tensor. The number of elements implied by shape must be the same
     *      as the number of elements in the input tensor.
     *
     * Outputs:
     * * 0: The output tensor, of shape specified by the input shape.
     */
    ANEURALNETWORKS_RESHAPE = 24,
    /** Resizes images to given size using the bilinear interpretation.
     *
     * Resized images will be distorted if their original aspect ratio is not the same as input.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
     * * 1: An INT32 value, specifying the output width of the output tensor.
     * * 2: An INT32 value, specifying the output height of the output tensor.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, new_height, new_width, depth].
     */
    ANEURALNETWORKS_RESIZE_BILINEAR = 25,
    ANEURALNETWORKS_RNN = 26,
    /** Computes the softmax activation on the input tensor element-wise, per batch, by
     * normalizing the input vector so the maximum coefficient is zero.
     *
     * The output is calculated using this formula:
     *
     *     output[batch, i] =
     *         exp((input[batch, i] - max(input[batch, :])) * beta) /
     *         sum_{k}{exp((input[batch, k] - max(input[batch, :])) * beta)}
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 2 or 4.
     *
     * Inputs:
     * * 0: A 2-D or 4-D tensor, specifying the tensor to be reshaped.
     * * 1: A FLOAT32 value, specifying the scaling factor for the exponent, beta.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     */
    ANEURALNETWORKS_SOFTMAX = 27,
    /** Rearranges blocks of spatial data, into depth.
     *
     * More specifically, this op outputs a copy of the input tensor where values from
     * the height and width dimensions are moved to the depth dimension.
     * The value block_size indicates the input block size and how the data is moved.
     *
     * Chunks of data of size block_size * block_size from depth are rearranged into
     * non-overlapping blocks of size block_size x block_size.
     *
     * The depth of the output tensor is input_depth * block_size * block_size.
     * The input tensor's height and width must be divisible by block_size.
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" data layout.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
     * * 1: An INT32 value, specifying the block_size. block_size must be >=1 and
     *      block_size must be a divisor of both the input height and width.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batch, height/block_size, width/block_size,
     *      depth*block_size*block_size].
     */
    ANEURALNETWORKS_SPACE_TO_DEPTH = 28,
    ANEURALNETWORKS_SVDF = 29,
    /** Computes hyperbolic tangent of input tensor element-wise.
     *
     * The output is calculated using this formula:
     *
     *     output = tanh(input)
     *
     * Supported tensor types:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * * 0: A tensor, specifying the input.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     */
    ANEURALNETWORKS_TANH = 30,
} OperationCode;

/**
 * Fused activation function types.
 *
 */
typedef enum {
    /** NO fused activation function. */
    ANEURALNETWORKS_FUSED_NONE = 0,
    /** Fused ReLU activation function. */
    ANEURALNETWORKS_FUSED_RELU = 1,
    /** Fused ReLU1 activation function. */
    ANEURALNETWORKS_FUSED_RELU1 = 2,
    /** Fused ReLU6 activation function. */
    ANEURALNETWORKS_FUSED_RELU6 = 3,
} FuseCode;

/**
 * Execution preferences.
 */
typedef enum {
    /**
     * Prefer executing in a way that minimizes battery drain.
     * This is desirable for compilations that will be executed often.
     */
    ANEURALNETWORKS_PREFER_LOW_POWER = 0,
    /**
     * Prefer returning a single answer as fast as possible, even if this causes
     * more power consumption.
     */
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
    /**
     * Prefer maximizing the throughput of successive frames, for example when
     * processing successive frames coming from the camera.
     */
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,
} PreferenceCode;

/**
 * Result codes.
 */
typedef enum {
    ANEURALNETWORKS_NO_ERROR = 0,
    ANEURALNETWORKS_OUT_OF_MEMORY = 1,
    ANEURALNETWORKS_INCOMPLETE = 2,
    ANEURALNETWORKS_UNEXPECTED_NULL = 3,
    ANEURALNETWORKS_BAD_DATA = 4,
    ANEURALNETWORKS_OP_FAILED = 5,
    ANEURALNETWORKS_UNMAPPABLE = 5,
    ANEURALNETWORKS_BAD_STATE = 6,
} ResultCode;

/**
 * ANeuralNetworksMemory is an opaque type that represents memory.
 *
 * This type is used to represent shared memory, memory mapped files,
 * and similar memories.
 *
 * By using shared memory, a program can efficiently communicate to the
 * runtime and drivers the tensors that define a model. See
 * {@link ANeuralNetworksModel_setOperandValueFromMemory}. An application
 * should typically create one shared memory object that contains every tensor
 * needed to define a model. {@link ANeuralNetworksMemory_createFromFd} can be
 * used to create shared memory from a file handle. {@link ANeuralNetworksMemory_createShared}
 * can be used to directly created shared memory.
 *
 * Memory objects can also be used to specify the input and output arguments of
 * a request. See {@link ANeuralNetworksRequest_setInputFromMemory}
 * and {@link ANeuralNetworksRequest_setOutputFromMemory}. This is a typical
 * usage for hardware buffers. See {@link ANeuralNetworksMemory_createFromHardwareBuffer}.
 */
typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;

/**
 * ANeuralNetworksModel is an opaque type that contains a description of the
 * mathematical operations that constitute the model.
 *
 * <p>The model will be built by calling<ul>
 * <li>{@link ANeuralNetworksModel_create},</li>
 * <li>{@link ANeuralNetworksModel_addOperation},</li>
 * <li>{@link ANeuralNetworksModel_addOperand},</li>
 * </ul>
 *
 * A model is completed by calling {@link ANeuralNetworksModel_finish}.
 * A model is destroyed by calling {@link ANeuralNetworksModel_free}.
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a model at a given time. It is however safe for more than one
 * thread to use the model once {@link ANeuralNetworksModel_finish} has returned.</p>
 */
typedef struct ANeuralNetworksModel ANeuralNetworksModel;

/**
 * ANeuralNetworksCompilation is an opaque type that can be used to compile
 * a machine learning model.
 *
 * <p>To use:<ul>
 *    <li>Create a new compilation instance by calling the
 *        {@link ANeuralNetworksCompilation_create} function.</li>
 *    <li>Perform the compilation with {@link ANeuralNetworksCompilation_start}.</li>
 *    <li>Wait for the compilation to complete with {@link ANeuralNetworksCompilation_wait}.</li>
 *    <li>Use the compilation as many times as needed
 *        with {@link ANeuralNetworksRequest_create}.</li>
 *    <li>Destroy the compilation with {@link ANeuralNetworksCompilation_free}
 *        once all requests using the compilation have completed.</li></ul></p>
 *
 * <p>A compilation cannot be modified once {@link ANeuralNetworksCompilation_start}
 * has been called on it.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a compilation at a given time. It is however safe for more than one
 * thread to use {@link ANeuralNetworksCompilation_wait} at the same time.
 * It is also safe for multiple threads to use a compilation object once
 * {@link ANeuralNetworksCompilation_wait} has completed.</p>
 */
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;

/**
 * ANeuralNetworksRequest is an opaque type that can be used to apply a machine
 * learning model to a set of inputs.
 *
 * <p>To use:<ul>
 *    <li>Create a new request instance by calling the
 *        {@link ANeuralNetworksRequest_create} function.</li>
 *    <li>Associate data to the model inputs with
 *        {@link ANeuralNetworksRequest_setInput} or
 *        {@link ANeuralNetworksRequest_setInputFromMemory}.</li>
 *    <li>Associate output buffers to the model outputs with
 *        {@link ANeuralNetworksRequest_setOutput} or
 *        {@link ANeuralNetworksRequest_setOutputFromMemory}.</li>
 *    <li>Apply the model with {@link ANeuralNetworksRequest_startCompute}.</li>
 *    <li>Wait for the request to complete with {@link
 *        ANeuralNetworksRequest_wait}.</li>
 *    <li>Destroy the request with
 *        {@link ANeuralNetworksRequest_free}.</li></ul></p>
 *
 * <p>A request cannot be modified once {@link ANeuralNetworksRequest_start}
 * has been called on it.</p>
 *
 * <p>A request can be applied to a model with
 * {@link ANeuralNetworksRequest_startCompute} only once. Create new requests
 * to do new evaluations of the model.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a request at a given time. It is however safe for more than one
 * thread to use {@link ANeuralNetworksRequest_wait} at the same time.</p>
 */
typedef struct ANeuralNetworksRequest ANeuralNetworksRequest;

typedef struct ANeuralNetworksIntList {
    uint32_t count;
    const uint32_t* data;
} ANeuralNetworksIntList;

/**
 * ANeuralNetworksOperandType describes the type of an operand.
 * This structure is used to describe both scalars and tensors.
 */
typedef struct ANeuralNetworksOperandType {
    /** The data type, e.g ANEURALNETWORKS_INT8. */
    uint32_t type;
    /** Count and size of each dimension. The count should be 0 for scalars. */
    ANeuralNetworksIntList dimensions;
    /** These two fields are only used for quantized tensors.
     * They should be zero for scalars and non-fixed point tensors.
     * The dequantized value of each entry is (value - offset) * scale.
     */
    float scale;
    int32_t offset;
} ANeuralNetworksOperandType;

typedef uint32_t ANeuralNetworksOperationType;

/**
 * Initializes the machine learning runtime.
 *
 * This should be called before any other ANeuralNetworks functions.
 * This function may start work threads, may clean up part of the
 * cache, and query the capabilities of the drivers.
 *
 * As the initialization may take some time, you may want to call
 * this function outside of the initialization path of your application,
 * so that your application starts quickly.
 *
 * Your application should call {@link ANeuralNetworksShutdown} to tear
 * down the runtime.
 *
 * It is safe for a process to call this function multiple times.
 * The first call performs the initialization. Successive calls increase
 * an internal reference count. An equivalent number of calls to
 * ANeuralNetworksShutdown must be performed for the runtime to be
 * destroyed. This enables libraries to safely call Initialize and Shutdown.
 *
 * This function is thread safe.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksInitialize();

/**
 * Destroys the machine learning runtime.
 *
 * This function frees any resource used by the runtime. It will wait
 * until in flight requests have completed and will prevent new ones
 * from being started with {@link ANeuralNetworksRequest_startCompute}.
 *
 * Threads blocked on {@link ANeuralNetworksRequest_wait} calls will be
 * released before this function terminates.
 *
 * See {@link ANeuralNetworksInitialize} for details on how multiple calls
 * to Initialize and Shutdown work.
 *
 * This function is thread safe.
 */
void ANeuralNetworksShutdown();

/**
 * Creates a shared memory object.
 *
 * Creates a shared memory region of the specified size in bytes.
 * See {@link ANeuralNetworksMemory} for a description on how to use
 * this shared memory.
 *
 * @param size The requested size in bytes.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 */
int ANeuralNetworksMemory_createShared(size_t size, ANeuralNetworksMemory** memory);

/**
 * Creates a shared memory object from a file descriptor.
 *
 * The shared memory is backed by a file descriptor via mmap.
 * See {@link ANeuralNetworksMemory} for a description on how to use
 * this shared memory.
 *
 * @param size The requested size in bytes.
 *             Must not be larger than the file size.
 * @param prot The desired memory protection for mmap.
 * @param fd The requested file descriptor.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 */
int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd,
                                       ANeuralNetworksMemory** memory);

/**
 * Returns pointer to the memory.
 *
 * Returns a pointer to the underlying memory. Not all memories represented by
 * {@link ANeuralNetworksMemory} can return a CPU addressable pointer, so be sure to
 * check the return value.
 *
 * @param memory The memory object we are inquiring about.
 * @param buffer A pointer to where the buffer pointer is returned. *buffer is set
 *               to NULL in case of error.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 *         ANEURALNETWORKS_UNMAPPABLE is returned if the memory can't be accessed
 *         directly by the CPU. Other error codes are possible.
 */
int ANeuralNetworksMemory_getPointer(ANeuralNetworksMemory* memory, uint8_t** buffer);

/**
 * Delete a memory object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The memory object to be freed.
 */
void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory);

/**
 * Create an empty {@link ANeuralNetworksModel}.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksRequest_startCompute} is invoked.
 *
 * The model should be constructed with calls to
 * {@link ANeuralNetworksModel_addOperation} and
 * {@link ANeuralNetworksModel_addOperand}
 *
 * <p>{@link ANeuralNetworksModel_finish} should be called once the model
 * has been fully constructed.</p>
 *
 * <p>{@link ANeuralNetworksModel_free} should be called once the model
 * is no longer needed.</p>
 *
 * This function is thread safe.
 *
 * @param model The {@link ANeuralNetworksModel} to be created.
 *              Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_create(ANeuralNetworksModel** model);

/**
 * Destroy a model.
 *
 * The model need not have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
void ANeuralNetworksModel_free(ANeuralNetworksModel* model);

/**
 * Indicate that we have finished modifying a model. Required before
 * calling {@link ANeuralNetworksCompilation_compile}.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be finished.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_finish(ANeuralNetworksModel* model);

/**
 * Add an operand to a model.
 *
 * The order in which the operands are added is important. The first one added
 * to a model will have the index value 0, the second 1, etc. These indexes are
 * used as operand identifiers in {@link ANeuralNetworksModel_addOperation},
 * {@link ANeuralNetworksRequest_setInput},
 * {@link ANeuralNetworksRequest_setInputFromMemory},
 * {@link ANeuralNetworksRequest_setOutput},
 * {@link ANeuralNetworksRequest_setOutputFromMemory} and
 * {@link ANeuralNetworksRequest_setOperandValue}.
 *
 * To build a model that can accomodate inputs of various sizes, as you may want
 * to do for a CNN, set the size of the dimensions that will vary at run time to 0.
 * If you do so, provide the full dimensions when calling
 * {@link ANeuralNetworksRequest_setInput} or {@link ANeuralNetworksRequest_setInputFromMemory}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has been
 * called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param type The {@link ANeuralNetworksOperandType} that describes the shape
 * of the operand.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type);

/**
 * Sets an operand to a constant value.
 *
 * For scalar values, the content of buffer is copied into the model.
 *
 * For tensor values, a pointer to the buffer is stored within the model.
 * The application is responsible for not changing the content of this region
 * until all requests using this model have completed. As the data may
 * be copied during processing, modifying the data after this call yields
 * undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has been
 * called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length);

/**
 * Sets an operand to a value stored in a memory object.
 *
 * The content of the memory is not copied. A reference to that memory is stored
 * inside the model. The application is responsible for not changing the content
 * of the memory region until all requests using this model have completed.
 * As the data may be copied during processing, modifying the data after this call
 * yields undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has been
 * called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   uint32_t offset, size_t length);

/**
 * Add an operation to a model.
 *
 * @param model The model to be modified.
 * @param type The type of the operation.
 * @param inputs An array of indexes identifying each an operand.
 * @param outputs An array of indexes identifying each an operand.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has been
 * called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type,
                                      ANeuralNetworksIntList* inputs,
                                      ANeuralNetworksIntList* outputs);

/**
 * Specfifies which operands will be the model's inputs and outputs.
 *
 * An operand cannot be used for both input and output. Doing so will
 * return an error.
 *
 * @param model The model to be modified.
 * @param inputs An array of indexes identifying the input operands.
 * @param outputs An array of indexes identifying the output operands.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has been
 * called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 */
int ANeuralNetworksModel_setInputsAndOutputs(ANeuralNetworksModel* model,
                                             ANeuralNetworksIntList* inputs,
                                             ANeuralNetworksIntList* outputs);

/**
 * Create a {@link ANeuralNetworksCompilation} to compile the given model.
 * This only creates the object. Compilation is only performed once
 * {@link ANeuralNetworksCompilation_start} is invoked.
 *
 * <p>The provided model must outlive the compilation.</p>
 *
 * The model must already have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded usage.
 *
 * @param model The {@link ANeuralNetworksModel} to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the model is invalid.
 */
int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation);

/**
 * Destroy a compilation.
 *
 * <p>If called on a compilation for which
 * {@link ANeuralNetworksCompilation_start} has been called, the
 * function will return immediately but will mark the compilation to be deleted
 * once the compilation completes. The {@link ANeuralNetworksCompilation_wait}
 * will return ERROR_DELETED.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded usage.
 *
 * @param compilation The compilation to be destroyed. Passing NULL is acceptable and
 *                    results in no operation.
 */
void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation);

/**
 * Sets the execution preference.
 *
 * <p>Provides guidance to the runtime when trade-offs are possible.</p>
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded usage.
 *
 * @param compilation The compilation to be modified.
 * @param preference Either {@link PREFER_LOW_POWER},
 *                  {@link PREFER_SINGLE_FAST_ANSWER}, or
 *                  {@link PREFER_SUSTAINED_SPEED}.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             uint32_t preference);

/**
 * Schedule the compilation to be performed.
 *
 * <p>Schedules the compilation to be performed. Once the model has been
 * compiled and the result is available for {@link ANeuralNetworksReques_create},
 * the compilation will be signaled. Use {@link ANeuralNetworksompilation_wait}
 * to wait for that signal.</p>
 *
 * Multiple compilations can be scheduled and performed concurrently, and
 * compilations can be performed concurrently with execution of requests.
 * The runtime makes no guarantee on the ordering of the completion of compilations
 * and requests. If it's important to the application, the application should enforce
 * the ordering by using
 * {@link ANeuralNetworksCompilation_wait} and {@link ANeuralNetworksRequest_wait}.
 *
 * ANeuralNetworksCompilation_wait must be called to recuperate the resources used
 * by the compilation.
 *
 * This function must only be called once for a given compilation.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded usage.
 *
 * @param compilation The compilation to be scheduled.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksCompilation_start(ANeuralNetworksCompilation* compilation);

/**
 * Waits until the compilation completes.
 *
 * More than one thread can wait on a compilation. When the compilation completes,
 * all threads will be released.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the compilation completed normally.
 */
int ANeuralNetworksCompilation_wait(ANeuralNetworksCompilation* compilation);

/**
 * Create a {@link ANeuralNetworksRequest} to apply the given compilation.
 * This only creates the object. Computation is only performed once
 * {@link ANeuralNetworksRequest_startCompute} is invoked.
 *
 * <p>The provided compilation must outlive the request.</p>
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
 * @param request The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the compilation is invalid.
 */
int ANeuralNetworksRequest_create(ANeuralNetworksCompilation* compilation,
                                  ANeuralNetworksRequest** request);

/**
 * Destroy a request.
 *
 * <p>If called on a request for which
 * {@link ANeuralNetworksRequest_startCompute} has been called, the
 * function will return immediately but will mark the request to be deleted
 * once the computation completes.   The {link ANeuralNetworksRequest_wait}
 * will return ANEURALNETWORKS_ERROR_DELETED.
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @param request The request to be destroyed. Passing NULL is acceptable and
 *                results in no operation.
 */
void ANeuralNetworksRequest_free(ANeuralNetworksRequest* request);

/**
 * Associate a user buffer with an input of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided buffer must outlive the request.</p>
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the input to.
 * @param type The type of the operand. This should be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other properties of the type must be the same as
 *             specified in the model. If the type is the same as specified
 *             when the model was built, NULL can be passed.
 * @param buffer The buffer containing the data.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if the
 *         name is not recognized or the buffer is too small for the input.
 */
int ANeuralNetworksRequest_setInput(ANeuralNetworksRequest* request, int32_t index,
                                    const ANeuralNetworksOperandType* type, const void* buffer,
                                    size_t length);

/**
 * Associate part of a memory object with an input of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided memory must outlive the request.</p>
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the input to.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data whithin the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if the
 *         name is not recognized or the buffer is too small for the input.
 */
int ANeuralNetworksRequest_setInputFromMemory(ANeuralNetworksRequest* request, int32_t index,
                                              const ANeuralNetworksOperandType* type,
                                              const ANeuralNetworksMemory* memory, uint32_t offset,
                                              uint32_t length);

/**
 * Associate a user buffer with an output of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided buffer must outlive the request.</p>
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the output to.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param buffer The buffer where the data is to be written.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if the
 *         name is not recognized or the buffer is too small for the output.
 */
int ANeuralNetworksRequest_setOutput(ANeuralNetworksRequest* request, int32_t index,
                                     const ANeuralNetworksOperandType* type, void* buffer,
                                     size_t length);

/**
 * Associate part of a memory object with an output of the model of the
 * {@link ANeuralNetworksRequest}.
 *
 * <p>The provided memory must outlive the request.</p>
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @param request The request to be modified.
 * @param index The index of the model operand we're associating the input to.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory where the data is to be stored.
 * @param offset This specifies the location of the data whithin the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The length in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if the
 *         name is not recognized or the buffer is too small for the output.
 */
int ANeuralNetworksRequest_setOutputFromMemory(ANeuralNetworksRequest* request, int32_t index,
                                               const ANeuralNetworksOperandType* type,
                                               const ANeuralNetworksMemory* memory, uint32_t offset,
                                               uint32_t length);

/**
 * Schedule the request for execution.
 *
 * <p>Schedules the request for execution. Once the model has been
 * applied and the outputs are ready to be consumed, the request will be
 * signaled. Use {@link ANeuralNetworksRequest_wait} to wait for that signal.
 * </p>
 *
 * Multiple requests can be scheduled and executed concurrently, and compilations
 * can be performed concurrently with execution of requests. The runtime makes
 * no guarantee on the ordering of the completion of compilations and requests.
 * If it's important to the application, the application should enforce the ordering
 * by using {@link ANeuralNetworksCompilation_wait} and {@link ANeuralNetworksRequest_wait}.
 *
 * ANeuralNetworksRequest_wait must be called to recuperate the resources used
 * by the request.
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @param request The request to be scheduled and executed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuralNetworksRequest_startCompute(ANeuralNetworksRequest* request);

/**
 * Waits until the request completes.
 *
 * More than one thread can wait on a request.  When the request completes,
 * all threads will be released.
 *
 * See {@link ANeuralNetworksRequest} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 */
int ANeuralNetworksRequest_wait(ANeuralNetworksRequest* request);

__END_DECLS

#endif  //  __ANDROID_API__ >= 27

#endif  // ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_H

/** @} */
