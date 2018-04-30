/**
 * Copyright 2017 The Android Open Source Project
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

#include "run_tflite.h"

#include "tensorflow/contrib/lite/kernels/register.h"

#include <android/log.h>
#include <cstdio>
#include <sys/time.h>

#define LOG_TAG "NN_BENCHMARK"

#define FATAL(fmt,...) do { \
  __android_log_print(ANDROID_LOG_FATAL, LOG_TAG, fmt, ##__VA_ARGS__); \
    assert(false); \
} while(0)

namespace {

long long currentTimeInUsec() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return ((tv.tv_sec * 1000000L) + tv.tv_usec);
}

} // namespace

BenchmarkModel::BenchmarkModel(const char* modelfile) {
    // Memory map the model. NOTE this needs lifetime greater than or equal
    // to interpreter context.
    mTfliteModel = tflite::FlatBufferModel::BuildFromFile(modelfile);
    if (!mTfliteModel) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to load model %s", modelfile);
        return;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*mTfliteModel, resolver)(&mTfliteInterpreter);
    if (!mTfliteInterpreter) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to create TFlite interpreter");
        return;
    }
}

BenchmarkModel::~BenchmarkModel() {
}

bool BenchmarkModel::setInput(const uint8_t* dataPtr, size_t length) {
    int input = mTfliteInterpreter->inputs()[0];
    auto* input_tensor = mTfliteInterpreter->tensor(input);

    switch (input_tensor->type) {
        case kTfLiteFloat32:
        case kTfLiteUInt8: {
            void* raw = input_tensor->data.raw;
            memcpy(raw, dataPtr, length);
            break;
        }
        default:
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "Input tensor type not supported");
            return false;
    }
    return true;
}


float BenchmarkModel::getOutputError(const uint8_t* expected_data, size_t length) {
      int output = mTfliteInterpreter->outputs()[0];
      auto* output_tensor = mTfliteInterpreter->tensor(output);
      if (output_tensor->bytes != length) {
          FATAL("Wrong size of output tensor, expected %zu, is %zu", output_tensor->bytes, length);
      }

      float err_sum = 0.0;
      switch (output_tensor->type) {
          case kTfLiteUInt8: {
              uint8_t* output_raw = mTfliteInterpreter->typed_tensor<uint8_t>(output);
              for (size_t i = 0;i < output_tensor->bytes; ++i) {
                  float err = ((float)output_raw[i]) - ((float)expected_data[i]);
                  err_sum += err*err;
              }
              break;
          }
          case kTfLiteFloat32: {
              const float* expected = reinterpret_cast<const float*>(expected_data);
              float* output_raw = mTfliteInterpreter->typed_tensor<float>(output);
              for (size_t i = 0;i < output_tensor->bytes / sizeof(float); ++i) {
                  float err = output_raw[i] - expected[i];
                  err_sum += err*err;
              }
              break;
          }
          default:
              FATAL("Output sensor type %d not supported", output_tensor->type);
      }

      return err_sum;
}

bool BenchmarkModel::resizeInputTensors(std::vector<int> shape) {
    // The benchmark only expects single input tensor, hardcoded as 0.
    int input = mTfliteInterpreter->inputs()[0];
    mTfliteInterpreter->ResizeInputTensor(input, shape);
    if (mTfliteInterpreter->AllocateTensors() != kTfLiteOk) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to allocate tensors!");
        return false;
    }
    return true;
}

bool BenchmarkModel::runInference(bool use_nnapi) {
    mTfliteInterpreter->UseNNAPI(use_nnapi);

    auto status = mTfliteInterpreter->Invoke();
    if (status != kTfLiteOk) {
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to invoke: %d!", (int)status);
      return false;
    }
    return true;
}

bool BenchmarkModel::benchmark(const std::vector<InferenceInOut> &inOutData,
                               int inferencesMaxCount,
                               float timeout,
                               std::vector<InferenceResult> *result) {

    if (inOutData.size() == 0) {
        FATAL("Input/output vector is empty");
    }

    float inferenceTotal = 0.0;
    for(int i = 0;i < inferencesMaxCount; i++) {
        const InferenceInOut & data = inOutData[i % inOutData.size()];
        setInput(data.input, data.input_size);

        long long startTime = currentTimeInUsec();
        if (!runInference(true)) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Inference %d failed", i);
            return false;
        }
        long long endTime = currentTimeInUsec();

        float inferenceTime = static_cast<float>(endTime - startTime) / 1000000.0f;
        result->push_back( {inferenceTime, getOutputError(data.output,data.output_size) } );

        // Timeout?
        inferenceTotal += inferenceTime;
        if (inferenceTotal > timeout) {
            return true;
        }
    }
    return true;
}
