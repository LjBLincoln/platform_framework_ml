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

// Top level driver for models and examples converted from TFLite tests

#include "NeuralNetworksWrapper.h"

#include <gtest/gtest.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>

typedef std::pair<std::map<int, std::vector<float>>,
                  std::map<int, std::vector<float>>>
    Example;

using namespace android::nn::wrapper;

namespace add {
std::vector<Example> examples = {
// Generated add
#include "generated/examples/add_tests.example.cc"
};
// Generated model constructor
#include "generated/models/add.model.cpp"
}  // add

namespace conv_1_h3_w2_SAME {
std::vector<Example> examples = {
// Converted examples
#include "generated/examples/conv_1_h3_w2_SAME_tests.example.cc"
};
// Generated model constructor
#include "generated/models/conv_1_h3_w2_SAME.model.cpp"
}  // namespace conv_1_h3_w2_SAME

namespace conv_1_h3_w2_VALID {
std::vector<Example> examples = {
// Converted examples
#include "generated/examples/conv_1_h3_w2_VALID_tests.example.cc"
};
// Generated model constructor
#include "generated/models/conv_1_h3_w2_VALID.model.cpp"
}  // namespace conv_1_h3_w2_VALID

namespace conv_3_h3_w2_SAME {
std::vector<Example> examples = {
// Converted examples
#include "generated/examples/conv_3_h3_w2_SAME_tests.example.cc"
};
// Generated model constructor
#include "generated/models/conv_3_h3_w2_SAME.model.cpp"
}  // namespace conv_3_h3_w2_SAME

namespace conv_3_h3_w2_VALID {
std::vector<Example> examples = {
// Converted examples
#include "generated/examples/conv_3_h3_w2_VALID_tests.example.cc"
};
// Generated model constructor
#include "generated/models/conv_3_h3_w2_VALID.model.cpp"
}  // namespace conv_3_h3_w2_VALID

namespace depthwise_conv {
std::vector<Example> examples = {
// Converted examples
#include "generated/examples/depthwise_conv_tests.example.cc"
};
// Generated model constructor
#include "generated/models/depthwise_conv.model.cpp"
}  // namespace depthwise_conv

namespace mobilenet {
std::vector<Example> examples = {
// Converted examples
#include "generated/examples/mobilenet_224_gender_basic_fixed_tests.example.cc"
};
// Generated model constructor
#include "generated/models/mobilenet_224_gender_basic_fixed.model.cpp"
}  // namespace mobilenet

namespace {
bool Execute(void (*create_model)(Model*), std::vector<Example>& examples) {
    Model model;
    create_model(&model);

    int example_no = 1;
    bool error = false;

    for (auto& example : examples) {
        Request request(&model);

        // Go through all inputs
        for (auto& i : example.first) {
            std::vector<float>& input = i.second;
            request.setInput(i.first, (const void*)input.data(),
                             input.size() * sizeof(float));
        }

        std::map<int, std::vector<float>> test_outputs;

        assert(example.second.size() == 1);
        int output_no = 0;
        for (auto& i : example.second) {
            std::vector<float>& output = i.second;
            test_outputs[i.first].resize(output.size());
            std::vector<float>& test_output = test_outputs[i.first];
            request.setOutput(output_no++, (void*)test_output.data(),
                              test_output.size() * sizeof(float));
        }
        Result r = request.compute();
        if (r != Result::NO_ERROR)
            std::cerr << "Request was not completed normally\n";
        bool mismatch = false;
        for (auto& i : example.second) {
            std::vector<float>& test = test_outputs[i.first];
            std::vector<float>& golden = i.second;
            for (unsigned i = 0; i < golden.size(); i++) {
                if (std::fabs(golden[i] - test[i]) > 1.5e-5f) {
                    std::cerr << " output[" << i << "] = " << test[i]
                              << " (should be " << golden[i] << ")\n";
                    error = error || true;
                    mismatch = mismatch || true;
                }
            }
        }
        if (mismatch) {
            std::cerr << "Example: " << example_no++;
            std::cerr << " failed\n";
        }
    }
    return error;
}

class GeneratedTests : public ::testing::Test {
   protected:
    virtual void SetUp() {
        ASSERT_EQ(android::nn::wrapper::Initialize(),
                  android::nn::wrapper::Result::NO_ERROR);
    }

    virtual void TearDown() { android::nn::wrapper::Shutdown(); }
};
}  // namespace

TEST_F(GeneratedTests, add) {
    ASSERT_EQ(
        Execute(add::CreateModel, add::examples),
        0);
}

TEST_F(GeneratedTests, conv_1_h3_w2_SAME) {
    ASSERT_EQ(
        Execute(conv_1_h3_w2_SAME::CreateModel, conv_1_h3_w2_SAME::examples),
        0);
}

TEST_F(GeneratedTests, conv_1_h3_w2_VALID) {
    ASSERT_EQ(
        Execute(conv_1_h3_w2_VALID::CreateModel, conv_1_h3_w2_VALID::examples),
        0);
}

TEST_F(GeneratedTests, conv_3_h3_w2_SAME) {
    ASSERT_EQ(
        Execute(conv_3_h3_w2_SAME::CreateModel, conv_3_h3_w2_SAME::examples),
        0);
}

TEST_F(GeneratedTests, conv_3_h3_w2_VALID) {
    ASSERT_EQ(
        Execute(conv_3_h3_w2_VALID::CreateModel, conv_3_h3_w2_VALID::examples),
        0);
}

TEST_F(GeneratedTests, depthwise_conv) {
    ASSERT_EQ(Execute(depthwise_conv::CreateModel, depthwise_conv::examples),
              0);
}

TEST_F(GeneratedTests, mobilenet) {
    ASSERT_EQ(Execute(mobilenet::CreateModel, mobilenet::examples), 0);
}
