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

#ifndef ANDROID_FRAMEWORK_ML_NN_RUNTIME_TEST_GENERATEDUTILS_H
#define ANDROID_FRAMEWORK_ML_NN_RUNTIME_TEST_GENERATEDUTILS_H

#include "NeuralNetworksWrapper.h"
#include "TestHarness.h"

namespace generated_tests {
using namespace android::nn::wrapper;
// Mixed-typed examples
typedef test_helper::MixedTypedExampleType MixedTypedExample;

void execute(std::function<void(Model*)> createModel,
             std::function<bool(int)> isIgnored,
             std::vector<MixedTypedExample>& examples,
             std::string dumpFile = "");

}  // namespace generated_tests


#endif  // ANDROID_FRAMEWORK_ML_NN_RUNTIME_TEST_GENERATEDUTILS_H
