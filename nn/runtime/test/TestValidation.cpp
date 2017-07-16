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

#include "NeuralNetworks.h"

#include <gtest/gtest.h>
#include <string>

// This file tests all the validations done by the Neural Networks API.

namespace {
class ValidationTest : public ::testing::Test {
protected:
    virtual void SetUp() { ASSERT_EQ(ANeuralNetworksInitialize(), ANEURALNETWORKS_NO_ERROR); }
    virtual void TearDown() { ANeuralNetworksShutdown(); }
};

class ValidationTestModel : public ::testing::Test {
protected:
    virtual void SetUp() {
        ASSERT_EQ(ANeuralNetworksInitialize(), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_create(&mModel), ANEURALNETWORKS_NO_ERROR);
    }
    virtual void TearDown() {
        ANeuralNetworksModel_free(mModel);
        ANeuralNetworksShutdown();
    }
    ANeuralNetworksModel* mModel = nullptr;
};

class ValidationTestRequest : public ::testing::Test {
protected:
    virtual void SetUp() {
        ASSERT_EQ(ANeuralNetworksInitialize(), ANEURALNETWORKS_NO_ERROR);

        ASSERT_EQ(ANeuralNetworksModel_create(&mModel), ANEURALNETWORKS_NO_ERROR);
        uint32_t dimensions[]{1};
        ANeuralNetworksOperandType tensorType{.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                              .dimensions = {.count = 1, .data = dimensions}};
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        uint32_t inList[2]{0, 1};
        uint32_t outList[1]{2};
        ANeuralNetworksIntList inputs{.count = 2, .data = inList};
        ANeuralNetworksIntList outputs{.count = 1, .data = outList};
        ASSERT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_ADD, &inputs, &outputs),
                  ANEURALNETWORKS_NO_ERROR);

        ASSERT_EQ(ANeuralNetworksRequest_create(mModel, &mRequest), ANEURALNETWORKS_NO_ERROR);
    }
    virtual void TearDown() {
        ANeuralNetworksRequest_free(mRequest);
        ANeuralNetworksModel_free(mModel);
        ANeuralNetworksShutdown();
    }
    ANeuralNetworksModel* mModel = nullptr;
    ANeuralNetworksRequest* mRequest = nullptr;
};

TEST_F(ValidationTest, CreateModel) {
    EXPECT_EQ(ANeuralNetworksModel_create(nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);

    EXPECT_EQ(ANeuralNetworksModel_createBaselineModel(nullptr,
                                                       ANEURALNETWORKS_INCEPTION_SMALL_20_20),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    ANeuralNetworksModel* model = nullptr;
    EXPECT_EQ(ANeuralNetworksModel_createBaselineModel(&model,
                                                       ANEURALNETWORKS_NUMBER_BASELINE_MODELS),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestModel, AddOperand) {
    ANeuralNetworksOperandType floatType{.type = ANEURALNETWORKS_FLOAT32,
                                         .dimensions = {.count = 0, .data = nullptr}};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(nullptr, &floatType),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);
    // TODO more types,
}

TEST_F(ValidationTestModel, SetOperandValue) {
    ANeuralNetworksOperandType floatType{.type = ANEURALNETWORKS_FLOAT32,
                                         .dimensions = {.count = 0, .data = nullptr}};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &floatType), ANEURALNETWORKS_NO_ERROR);

    char buffer[20];
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(nullptr, 0, buffer, sizeof(buffer)),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, nullptr, sizeof(buffer)),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    // This should fail, since buffer is not the size of a float32.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, buffer, sizeof(buffer)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, as this operand does not exist.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 1, buffer, 4), ANEURALNETWORKS_BAD_DATA);

    // TODO lots of validation of type
    // EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, buffer,
    // sizeof(buffer)), ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestModel, AddOperation) {
    ANeuralNetworksIntList inputs{};
    ANeuralNetworksIntList outputs{};
    EXPECT_EQ(ANeuralNetworksModel_addOperation(nullptr, ANEURALNETWORKS_AVERAGE_POOL, &inputs,
                                                &outputs),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_AVERAGE_POOL, nullptr,
                                                &outputs),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_AVERAGE_POOL, &inputs,
                                                nullptr),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    // EXPECT_EQ(ANeuralNetworksModel_addOperation(mModel,
    // ANEURALNETWORKS_AVERAGE_POOL, &inputs,
    //                                            &outputs),
    //          ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestModel, AddSubModel) {
    ANeuralNetworksIntList inputs;
    ANeuralNetworksIntList outputs;
    ANeuralNetworksModel* submodel;
    EXPECT_EQ(ANeuralNetworksModel_addSubModel(nullptr, submodel, &inputs, &outputs),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_addSubModel(mModel, nullptr, &inputs, &outputs),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    // EXPECT_EQ(ANeuralNetworksModel_addSubModel(mModel, &submodel,
    //                                           &inputs, &outputs),
    //          ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestModel, SetInputsAndOutputs) {
    ANeuralNetworksIntList inputs;
    ANeuralNetworksIntList outputs;
    EXPECT_EQ(ANeuralNetworksModel_setInputsAndOutputs(nullptr, &inputs, &outputs),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_setInputsAndOutputs(mModel, nullptr, &outputs),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_setInputsAndOutputs(mModel, &inputs, nullptr),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestModel, SetBaselineId) {
    EXPECT_EQ(ANeuralNetworksModel_setBaselineId(nullptr, ANEURALNETWORKS_INCEPTION_SMALL_20_20),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_setBaselineId(mModel, ANEURALNETWORKS_NUMBER_BASELINE_MODELS),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestModel, CreateRequest) {
    ANeuralNetworksRequest* request = nullptr;
    EXPECT_EQ(ANeuralNetworksRequest_create(nullptr, &request), ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksRequest_create(mModel, nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);
    // EXPECT_EQ(ANeuralNetworksRequest_create(mModel, ANeuralNetworksRequest *
    // *request),
    //          ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestRequest, SetPreference) {
    EXPECT_EQ(ANeuralNetworksRequest_setPreference(nullptr, ANEURALNETWORKS_PREFER_LOW_POWER),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    EXPECT_EQ(ANeuralNetworksRequest_setPreference(mRequest, ANEURALNETWORKS_NUMBER_PREFERENCES),
              ANEURALNETWORKS_BAD_DATA);
}

#if 0
// TODO do more..
TEST_F(ValidationTestRequest, SetInput) {
    EXPECT_EQ(ANeuralNetworksRequest_setInput(ANeuralNetworksRequest * request, int32_t index,
                                              const ANeuralNetworksOperandType* type,
                                              const void* buffer, size_t length),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestRequest, SetInputFromHardwareBuffer) {
    EXPECT_EQ(ANeuralNetworksRequest_setInputFromHardwareBuffer(ANeuralNetworksRequest * request,
                                                                int32_t index,
                                                                const ANeuralNetworksOperandType*
                                                                        type,
                                                                const AHardwareBuffer* buffer),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestRequest, SetOutput) {
    EXPECT_EQ(ANeuralNetworksRequest_setOutput(ANeuralNetworksRequest * request, int32_t index,
                                               const ANeuralNetworksOperandType* type, void* buffer,
                                               size_t length),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestRequest, SetOutputFromHardwareBuffer) {
    EXPECT_EQ(ANeuralNetworksRequest_setOutputFromHardwareBuffer(ANeuralNetworksRequest * request,
                                                                 int32_t index,
                                                                 const ANeuralNetworksOperandType*
                                                                         type,
                                                                 const AHardwareBuffer* buffer),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestRequest, StartCompute) {
    EXPECT_EQ(ANeuralNetworksRequest_startCompute(ANeuralNetworksRequest * request,
                                                  ANeuralNetworksEvent * *event),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestEvent, Wait) {
    EXPECT_EQ(ANeuralNetworksEvent_wait(ANeuralNetworksEvent * event),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestEvent, Free) {
    EXPECT_EQ(d ANeuralNetworksEvent_free(ANeuralNetworksEvent * event),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}
#endif

} // namespace
