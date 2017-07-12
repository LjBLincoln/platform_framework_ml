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

#ifndef ANDROID_ML_NN_COMMON_HAL_MODEL_H
#define ANDROID_ML_NN_COMMON_HAL_MODEL_H

// This file contains the data structures that used to access the

//namespace android {
//namespace nn {

#include <cstdint>
#include <sys/cdefs.h>

__BEGIN_DECLS

// The location will be specified at runtime. It's either a temporary
// variable, an input, or an output.
const uint32_t LOCATION_AT_RUN_TIME = 0xFFFFFFFF;
// The operand's value is in the same memory pool as the model.
const uint32_t LOCATION_SAME_BLOCK = 0xFFFFFFFE;

// Used to represent a variable length array.
struct ArrayInfo {
    // The number of elements of the array.
    uint32_t count;
    // The offset in whichevere data structure to find the first
    // element of the array.  The unit type of the offset depends
    // on the data structure it is indexing.
    uint32_t offset;
};

// A serialized model starts with this block of memory.
// TODO Look into alignment or padding issues.
struct ModelHeader {
    // Size and location of the operation table, an array of OperationEntry.
    // The offset is the distance in bytes from the start of the header.
    ArrayInfo operations;
    // Size and location of the operand table, an array of OperandEntry.
    // The offset is the distance in bytes from the start of the header.
    ArrayInfo operands;
    // Size and location of the table of dimensions, an array of uint32_t.
    // The offset is the distance in bytes from the start of the header.
    ArrayInfo dimensions;
    // Size and location of the table of operand indexes, an array of uint32_t.
    // The offset is the distance in bytes from the start of the header.
    ArrayInfo operandIndexes;
    // Size and location of the memory block containing all the fixed
    // operand values.  The element type is uint8_t.
    // The offset is the distance in bytes from the start of the header.
    ArrayInfo operandValues;

    // The list of operand indexes for the inputs of the model.
    // The offset is an index in the operandIndexes table.
    ArrayInfo modelInputs;
    // The list of operand indexes for the outputs of the model.
    // The offset is an index in the operandIndexes table.
    ArrayInfo modelOutputs;
};

// Describes one operation of the graph.
struct OperationEntry {
    // The type of operation.
    uint32_t opCode;
    // Describes the table that contains the indexes of the inputs of the
    // operation. The offset is the index in the operandIndexes table.
    ArrayInfo inputs;
    // Describes the table that contains the indexes of the outputs of the
    // operation. The offset is the index in the operandIndexes table.
    ArrayInfo outputs;
};

// Describes the location of a data object.
struct DataLocation {
    // The index of the memory pool where this location is found.
    // Two special values can also be used.  See the LOCATION_* constants above.
    uint32_t pool;
    // Offset in bytes from the start of the pool.
    uint32_t offset;
};

// Describes one operand of the graph.
struct OperandEntry {
    uint32_t type;
    // The number of operations that uses this operand as input.
    uint32_t numberOfConsumers;
    // TODO handle quantization params.

    // The following three fields maybe superseded at runtime.

    // Dimensions of the operand.  The offset is an index in the dimensions table.
    ArrayInfo dimensions;
    // Where to find the data for this operand.
    DataLocation location;
    // The length of the data, in bytes.
    uint32_t length;
};

__END_DECLS

//}  // namespace nn
//}  // namespace android

#endif  // ANDROID_ML_NN_COMMON_HAL_MODEL_H
