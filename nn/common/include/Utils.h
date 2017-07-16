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

#include "HalModel.h"

#include <stdio.h>
#include <vector>

namespace android {
namespace nn {

// TODO Replace with the real Android logging macros.
#define ALOGE(format, ...) printf(LOG_TAG ": ERROR " format "\n", ##__VA_ARGS__)
#define ALOGI(format, ...) printf(LOG_TAG ": " format "\n", ##__VA_ARGS__)

// Assert macro, as Android does not generally support assert.
#define nnAssert(v)                                                                       \
    do {                                                                                  \
        if (!(v)) {                                                                       \
            fprintf(stderr, "nnAssert failed at %s:%d - '%s'\n", __FILE__, __LINE__, #v); \
            abort();                                                                      \
        }                                                                                 \
    } while (0)

// Represent a list of items.  Handy to iterate over lists and sublists.
template <typename T>
class Range {
public:
    // The default constructor should only be used when followed by a call
    // to setFromBuffer.
    Range() {}
    // Range over all the elements of the vector.
    Range(const std::vector<T>& data) {
        mCount = static_cast<uint32_t>(data.size());
        mBegin = data.data();
    }
    // Range over the sublist of elements of the vector, as specified by info.
    Range(const std::vector<T>& data, const ArrayInfo& info) {
        mCount = info.count;
        mBegin = data.data() + info.offset;
    }
    // Range over the sublist of the range, as specified by info.
    Range(const Range<T>& data, const ArrayInfo& info) {
        mCount = info.count;
        mBegin = data.begin() + info.offset;
    }
    // Range of the specified number of elements, starting at the specified value.
    Range(uint32_t count, T* start) {
        mCount = count;
        mBegin = start;
    }

    // Range over consecutive elements starting at buffer + info.offset.
    void setFromBuffer(const ArrayInfo& info, const uint8_t* buffer) {
        mCount = info.count;
        mBegin = reinterpret_cast<const T*>(buffer + info.offset);
    }

    // These two methods enable the use of for(x:Range(..)).
    const T* begin() const { return mBegin; }
    const T* end() const { return mBegin + mCount; }

    // Returns the element at the specifed index.
    T operator[](uint32_t index) const {
        nnAssert(index < mCount);
        return mBegin[index];
    }
    // All our ranges are read-only.  If we need to write, use this:
    // uint32_t& operator[] (uint32_t index) {
    //    nnAssert(index < mCount);
    //    return mBegin[index];
    // }

    uint32_t count() const { return mCount; }

private:
    const T* mBegin = nullptr;  // The start of the range.
    uint32_t mCount = 0;        // The number of elements in the range.
};

// Returns the the amount of space needed to store a tensor of the specified
// dimensions and type.
uint32_t sizeOfData(uint32_t type, const Range<uint32_t>& dimensions);

// Returns the name of the operation in ASCII.
const char* getOperationName(uint32_t opCode);

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_COMMON_UTILS_H
