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

#ifndef ANDROID_ML_NN_RUNTIME_MEMORY_H
#define ANDROID_ML_NN_RUNTIME_MEMORY_H

#include "NeuralNetworks.h"
#include "Utils.h"

#include <unordered_map>

namespace android {
namespace nn {

class ModelBuilder;

// Represents a shared memory region.
class Memory {
public:
    // Creates a shared memory object of the size specified in bytes.
    int create(uint32_t size);

    /* TODO implement
    int setFromHidlMemory(hardware::hidl_memory hidlMemory) {
        mHidlMemory = hidlMemory;
        mMemory = mapMemory(hidlMemory);
        if (mMemory == nullptr) {
            LOG(ERROR) << "setFromHidlMemory failed";
            return ANEURALNETWORKS_OP_FAILED;
        }
        return ANEURALNETWORKS_NO_ERROR;
    }
    int setFromFd(int fd) {
        return ANEURALNETWORKS_NO_ERROR;
    }
    int setFromGrallocBuffer(buffer_handle_t buffer,
                             ANeuralNetworksMemory** memory) {
        return ANEURALNETWORKS_NO_ERROR;
    }
    int setFromHardwareBuffer(AHardwareBuffer* buffer,
                              ANeuralNetworksMemory** memory) {
        return ANEURALNETWORKS_NO_ERROR;
    }
    */

    hardware::hidl_memory getHidlMemory() const { return mHidlMemory; }

	// Returns a pointer to the underlying memory of this shared memory.
    uint8_t* getPointer() const {
        return static_cast<uint8_t*>(static_cast<void*>(mMemory->getPointer()));
    }

private:
    // The hidl_memory handle for this shared memory.  We will pass this value when
    // communicating with the drivers.
    hardware::hidl_memory mHidlMemory;
    sp<IMemory> mMemory;
};

// A utility class to accumulate mulitple Memory objects and assign each
// a distinct index number, starting with 0.
class MemoryTracker {
public:
    // Adds the memory, if it does not already exists.  Returns its index.
    // The memories should survive the tracker.
    uint32_t add(const Memory* memory);
    // Returns the number of memories contained.
    uint32_t size() const { return static_cast<uint32_t>(mKnown.size()); }
    // Returns the ith memory.
    const Memory* operator[](size_t i) const { return mMemories[i]; }

private:
    // The vector of Memory pointers we are building.
    std::vector<const Memory*> mMemories;
    // A faster way to see if we already have a memory than doing find().
    std::unordered_map<const Memory*, uint32_t> mKnown;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_MEMORY_H
