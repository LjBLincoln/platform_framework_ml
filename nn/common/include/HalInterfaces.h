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

#ifndef ANDROID_ML_NN_COMMON_HAL_INTERFACES_H
#define ANDROID_ML_NN_COMMON_HAL_INTERFACES_H

#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.0/types.h>
#include <android/hidl/allocator/1.0/IAllocator.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>

using ::android::hardware::Return;
using ::android::hardware::Void;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
using ::android::hardware::hidl_vec;
using ::android::hardware::neuralnetworks::V1_0::Capabilities;
using ::android::hardware::neuralnetworks::V1_0::DataLocation;
using ::android::hardware::neuralnetworks::V1_0::DeviceStatus;
using ::android::hardware::neuralnetworks::V1_0::IDevice;
using ::android::hardware::neuralnetworks::V1_0::IPreparedModel;
using ::android::hardware::neuralnetworks::V1_0::InputOutputInfo;
using ::android::hardware::neuralnetworks::V1_0::LocationValues;
using ::android::hardware::neuralnetworks::V1_0::Model;
using ::android::hardware::neuralnetworks::V1_0::Operand;
using ::android::hardware::neuralnetworks::V1_0::OperandType;
using ::android::hardware::neuralnetworks::V1_0::Operation;
using ::android::hardware::neuralnetworks::V1_0::OperationType;
using ::android::hardware::neuralnetworks::V1_0::PerformanceInfo;
using ::android::hardware::neuralnetworks::V1_0::Request;
using ::android::hidl::allocator::V1_0::IAllocator;
using ::android::hidl::memory::V1_0::IMemory;

const uint32_t HAL_NUM_OPERAND_TYPES = 11;
const uint32_t HAL_NUM_OPERATION_TYPES = 33;

#endif // ANDROID_ML_NN_COMMON_HAL_INTERFACES_H
