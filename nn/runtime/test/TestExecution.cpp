/*
 * Copyright (C) 2018 The Android Open Source Project
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

#undef NDEBUG

#include "Callbacks.h"
#include "CompilationBuilder.h"
#include "Manager.h"
#include "ModelBuilder.h"
#include "NeuralNetworks.h"
#include "NeuralNetworksWrapper.h"
#include "SampleDriver.h"
#include "ValidateHal.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include <gtest/gtest.h>

namespace android {

using CompilationBuilder = nn::CompilationBuilder;
using Device = nn::Device;
using DeviceManager = nn::DeviceManager;
using HidlModel = hardware::neuralnetworks::V1_1::Model;
using PreparedModelCallback = hardware::neuralnetworks::V1_0::implementation::PreparedModelCallback;
using Result = nn::wrapper::Result;
using SampleDriver = nn::sample_driver::SampleDriver;
using WrapperCompilation = nn::wrapper::Compilation;
using WrapperEvent = nn::wrapper::Event;
using WrapperExecution = nn::wrapper::Execution;
using WrapperModel = nn::wrapper::Model;
using WrapperOperandType = nn::wrapper::OperandType;
using WrapperType = nn::wrapper::Type;

namespace {

// Wraps an IPreparedModel to allow dummying up the execution status.
class TestPreparedModel : public IPreparedModel {
public:
    // If errorStatus is NONE, then execute behaves normally (and sends back
    // the actual execution status).  Otherwise, don't bother to execute, and
    // just send back errorStatus (as the execution status, not the launch
    // status).
    TestPreparedModel(sp<IPreparedModel> preparedModel, ErrorStatus errorStatus) :
            mPreparedModel(preparedModel), mErrorStatus(errorStatus) {}

    Return<ErrorStatus> execute(const Request& request,
                                const sp<IExecutionCallback>& callback) override {
        if (mErrorStatus == ErrorStatus::NONE) {
            return mPreparedModel->execute(request, callback);
        } else {
            callback->notify(mErrorStatus);
            return ErrorStatus::NONE;
        }
    }
private:
    sp<IPreparedModel> mPreparedModel;
    ErrorStatus mErrorStatus;
};

// Behaves like SampleDriver, except that it produces wrapped IPreparedModel.
class TestDriver11 : public SampleDriver {
public:
    // Allow dummying up the error status for execution of all models
    // prepared from this driver.  If errorStatus is NONE, then
    // execute behaves normally (and sends back the actual execution
    // status).  Otherwise, don't bother to execute, and just send
    // back errorStatus (as the execution status, not the launch
    // status).
    TestDriver11(const std::string& name, ErrorStatus errorStatus) :
                 SampleDriver(name.c_str()), mErrorStatus(errorStatus) { }

    Return<void> getCapabilities_1_1(getCapabilities_1_1_cb _hidl_cb) override {
        android::nn::initVLogMask();
        Capabilities capabilities =
                {.float32Performance = {.execTime = 0.75f, .powerUsage = 0.75f},
                 .quantized8Performance = {.execTime = 0.75f, .powerUsage = 0.75f},
                 .relaxedFloat32toFloat16Performance = {.execTime = 0.75f, .powerUsage = 0.75f}};
        _hidl_cb(ErrorStatus::NONE, capabilities);
        return Void();
    }

    Return<void> getSupportedOperations_1_1(const HidlModel& model,
                                            getSupportedOperations_cb cb) override {
        if (nn::validateModel(model)) {
            std::vector<bool> supported(model.operations.size(), true);
            cb(ErrorStatus::NONE, supported);
        } else {
            std::vector<bool> supported;
            cb(ErrorStatus::INVALID_ARGUMENT, supported);
        }
        return Void();
    }

    Return<ErrorStatus> prepareModel_1_1(
        const HidlModel& model,
        ExecutionPreference preference,
        const sp<IPreparedModelCallback>& actualCallback) override {

        sp<PreparedModelCallback> localCallback = new PreparedModelCallback;
        Return<ErrorStatus> prepareModelReturn =
                SampleDriver::prepareModel_1_1(model, preference, localCallback);
        if (!prepareModelReturn.isOkUnchecked()) {
            return prepareModelReturn;
        }
        if (prepareModelReturn != ErrorStatus::NONE) {
            actualCallback->notify(localCallback->getStatus(), localCallback->getPreparedModel());
            return prepareModelReturn;
        }
        localCallback->wait();
        if (localCallback->getStatus() != ErrorStatus::NONE) {
            actualCallback->notify(localCallback->getStatus(), localCallback->getPreparedModel());
        } else {
            actualCallback->notify(ErrorStatus::NONE,
                                   new TestPreparedModel(localCallback->getPreparedModel(),
                                                         mErrorStatus));
        }
        return prepareModelReturn;
    }

private:
    ErrorStatus mErrorStatus;
};

// Like TestDriver, but implementing 1.0
class TestDriver10 : public V1_0::IDevice {
public:
    TestDriver10(const std::string& name, ErrorStatus errorStatus) : m11Driver(name, errorStatus) {}
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) override {
        return m11Driver.getCapabilities(_hidl_cb);
    }
    Return<void> getSupportedOperations(const V1_0::Model& model,
                                        getSupportedOperations_cb _hidl_cb) override {
        return m11Driver.getSupportedOperations(model, _hidl_cb);
    }
    Return<ErrorStatus> prepareModel(
        const V1_0::Model& model,
        const sp<IPreparedModelCallback>& actualCallback) override {
        return m11Driver.prepareModel(model, actualCallback);
    }
    Return<DeviceStatus> getStatus() override {
        return m11Driver.getStatus();
    }
private:
    TestDriver11 m11Driver;
};

// This class adds some simple utilities on top of
// ::android::nn::wrapper::Compilation in order to provide access to
// certain features from CompilationBuilder that are not exposed by
// the base class.
template<typename DriverClass>
class TestCompilation : public WrapperCompilation {
public:
    TestCompilation(const WrapperModel* model) : WrapperCompilation(model) {
        // We need to ensure that we use our TestDriver and do not
        // fall back to CPU.  (If we allow CPU fallback, then when our
        // TestDriver reports an execution failure, we'll re-execute
        // on CPU, and will not see the failure.)
        builder()->setPartitioning(DeviceManager::kPartitioningWithoutFallback);
    }

    // Allow dummying up the error status for all executions from this
    // compilation.  If errorStatus is NONE, then execute behaves
    // normally (and sends back the actual execution status).
    // Otherwise, don't bother to execute, and just send back
    // errorStatus (as the execution status, not the launch status).
    Result finish(const std::string& deviceName, ErrorStatus errorStatus) {
        std::vector<std::shared_ptr<Device>> devices;
        auto device = std::make_shared<Device>(deviceName,
                                               new DriverClass(deviceName, errorStatus));
        assert(device->initialize());
        devices.push_back(device);
        return static_cast<Result>(builder()->finish(devices));
    }

private:
    CompilationBuilder* builder() {
        return reinterpret_cast<CompilationBuilder*>(getHandle());
    }
};

template<class DriverClass>
class ExecutionTestTemplate :
            public ::testing::TestWithParam<std::tuple<ErrorStatus, Result>> {
public:
    ExecutionTestTemplate() :
            kName(toString(std::get<0>(GetParam()))),
            kForceErrorStatus(std::get<0>(GetParam())),
            kExpectResult(std::get<1>(GetParam())),
            mModel(makeModel()),
            mCompilation(&mModel) { }

protected:
    // Unit test method
    void TestWait();

    const std::string kName;

    // Allow dummying up the error status for execution.  If
    // kForceErrorStatus is NONE, then execution behaves normally (and
    // sends back the actual execution status).  Otherwise, don't
    // bother to execute, and just send back kForceErrorStatus (as the
    // execution status, not the launch status).
    const ErrorStatus kForceErrorStatus;

    // What result do we expect from the execution?  (The Result
    // equivalent of kForceErrorStatus.)
    const Result kExpectResult;

    WrapperModel mModel;
    TestCompilation<DriverClass> mCompilation;

    void setInputOutput(WrapperExecution* execution) {
        ASSERT_EQ(execution->setInput(0, &mInputBuffer, sizeof(mInputBuffer)), Result::NO_ERROR);
        ASSERT_EQ(execution->setOutput(0, &mOutputBuffer, sizeof(mOutputBuffer)), Result::NO_ERROR);
    }

    float mInputBuffer  = 3.14;
    float mOutputBuffer = 0;
    const float kOutputBufferExpected = 3;

private:
    static WrapperModel makeModel() {
        static const WrapperOperandType tensorType(WrapperType::TENSOR_FLOAT32, { 1 });

        WrapperModel model;
        uint32_t input = model.addOperand(&tensorType);
        uint32_t output = model.addOperand(&tensorType);
        model.addOperation(ANEURALNETWORKS_FLOOR, { input }, { output });
        model.identifyInputsAndOutputs({ input }, { output } );
        assert(model.finish() == Result::NO_ERROR);

        return model;
    }
};

template<class DriverClass> void ExecutionTestTemplate<DriverClass>::TestWait() {
    SCOPED_TRACE(kName);
    ASSERT_EQ(mCompilation.finish(kName, kForceErrorStatus), Result::NO_ERROR);
    WrapperExecution execution(&mCompilation);
    ASSERT_NO_FATAL_FAILURE(setInputOutput(&execution));
    WrapperEvent event;
    ASSERT_EQ(execution.startCompute(&event), Result::NO_ERROR);
    ASSERT_EQ(event.wait(), kExpectResult);
    if (kExpectResult == Result::NO_ERROR) {
        ASSERT_EQ(mOutputBuffer, kOutputBufferExpected);
    }
}

auto kTestValues = ::testing::Values(std::make_tuple(ErrorStatus::NONE,
                                                     Result::NO_ERROR),
                                     std::make_tuple(ErrorStatus::DEVICE_UNAVAILABLE,
                                                     Result::OP_FAILED),
                                     std::make_tuple(ErrorStatus::GENERAL_FAILURE,
                                                     Result::OP_FAILED),
                                     std::make_tuple(ErrorStatus::OUTPUT_INSUFFICIENT_SIZE,
                                                     Result::OP_FAILED),
                                     std::make_tuple(ErrorStatus::INVALID_ARGUMENT,
                                                     Result::BAD_DATA));

class ExecutionTest11 : public ExecutionTestTemplate<TestDriver11> {};
TEST_P(ExecutionTest11, Wait) {
    TestWait();
}
INSTANTIATE_TEST_CASE_P(Flavor, ExecutionTest11, kTestValues);

class ExecutionTest10 : public ExecutionTestTemplate<TestDriver10> {};
TEST_P(ExecutionTest10, Wait) {
    TestWait();
}
INSTANTIATE_TEST_CASE_P(Flavor, ExecutionTest10, kTestValues);

}  // namespace
}  // namespace android
