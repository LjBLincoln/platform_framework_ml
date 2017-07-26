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

#ifndef ANDROID_ML_NN_COMMON_UTILS_H
#define ANDROID_ML_NN_COMMON_UTILS_H

#include "HalInterfaces.h"
#include "NeuralNetworks.h"

#include <android-base/logging.h>
#include <vector>

namespace android {
namespace nn {

// TODO Remove all the LOG(DEBUG) statements in all the files.

// Assert macro, as Android does not generally support assert.
#define nnAssert(v)                                                                            \
    do {                                                                                       \
        if (!(v)) {                                                                            \
            LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                       << "'\n";                                                               \
            abort();                                                                           \
        }                                                                                      \
    } while (0)

// Returns the the amount of space needed to store a tensor of the specified
// dimensions and type.
uint32_t sizeOfData(OperandType type, const std::vector<uint32_t>& dimensions);

// Returns the name of the operation in ASCII.
const char* getOperationName(OperationType opCode);

hidl_memory allocateSharedMemory(int64_t size);

// Returns the number of padding bytes needed to align data of the
// specified length.  It aligns object of length:
// 2, 3 on a 2 byte boundary,
// 4+ on a 4 byte boundary.
// We may want to have different alignments for tensors.
// TODO: This is arbitrary, more a proof of concept.  We need
// to determine what this should be.
uint32_t alignBytesNeeded(uint32_t index, size_t length);

inline void setFromIntList(hidl_vec<uint32_t>* vec, const ANeuralNetworksIntList& list) {
    vec->resize(list.count);
    for (uint32_t i = 0; i < list.count; i++) {
        (*vec)[i] = list.data[i];
    }
}

inline void setFromIntList(std::vector<uint32_t>* vec, const ANeuralNetworksIntList& list) {
    vec->resize(list.count);
    for (uint32_t i = 0; i < list.count; i++) {
        (*vec)[i] = list.data[i];
    }
}

inline std::string toString(uint32_t obj) {
    return std::to_string(obj);
}

template <typename Type>
std::string toString(const std::vector<Type>& range) {
    std::string os = "[";
    for (size_t i = 0; i < range.size(); ++i) {
        os += (i == 0 ? "" : ", ") + toString(range[i]);
    }
    return os += "]";
}

bool validateModel(const Model& model);

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_UTILS_H
