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

#define LOG_TAG "CompilationBuilder"

#include "CompilationBuilder.h"

#include "ExecutionBuilder.h"

namespace android {
namespace nn {

CompilationBuilder::CompilationBuilder(const ModelBuilder* model) :
    mModel(model) {
    LOG(DEBUG) << "CompilationBuilder::CompilationBuilder";
}

int CompilationBuilder::compile() {
    // TODO mechanism for tracking state: creating, compiling, compiled
    return ANEURALNETWORKS_NO_ERROR;
}

int CompilationBuilder::createExecution(ExecutionBuilder **execution) {
    *execution = new ExecutionBuilder(this);
    return (*execution ? ANEURALNETWORKS_NO_ERROR : ANEURALNETWORKS_OUT_OF_MEMORY);
}

}  // namespace nn
}  // namespace android
