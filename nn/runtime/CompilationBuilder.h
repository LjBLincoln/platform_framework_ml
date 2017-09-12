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

#ifndef ANDROID_ML_NN_RUNTIME_COMPILATION_BUILDER_H
#define ANDROID_ML_NN_RUNTIME_COMPILATION_BUILDER_H

#include "NeuralNetworks.h"

namespace android {
namespace nn {

class ModelBuilder;
class RequestBuilder;

class CompilationBuilder {
public:
    friend class RequestBuilder;  // TODO remove this

    CompilationBuilder(const ModelBuilder* model);

    void setPreference(uint32_t preference) { mPreference = preference; }

    int compile();  // TODO: Asynchronous (startCompile?)

    int createRequest(RequestBuilder** request);

private:
    // int startComputeOnCpu(const Model& model, sp<Event>* event);

    const ModelBuilder* mModel;
    // Whether the application prefers to go fast or use low power for this request.
    uint32_t mPreference = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_COMPILATION_BUILDER_H
