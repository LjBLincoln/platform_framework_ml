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

#include <jni.h>
#include <string>
#include <iomanip>
#include <sstream>
#include <fcntl.h>

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/sharedmem.h>
#include <sys/mman.h>

extern "C"
JNIEXPORT jlong
JNICALL
Java_com_example_android_nn_benchmark_NNTestBase_initModel(
        JNIEnv *env,
        jobject /* this */,
        jstring _modelFileName) {
    const char *modelFileName = env->GetStringUTFChars(_modelFileName, NULL);
    void* handle = new BenchmarkModel(modelFileName);
    env->ReleaseStringUTFChars(_modelFileName, modelFileName);

    return (jlong)(uintptr_t)handle;
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_example_android_nn_benchmark_NNTestBase_destroyModel(
        JNIEnv *env,
        jobject /* this */,
        jlong _modelHandle) {
    BenchmarkModel* model = (BenchmarkModel *) _modelHandle;
    delete(model);
}

extern "C"
JNIEXPORT jboolean
JNICALL
Java_com_example_android_nn_benchmark_NNTestBase_resizeInputTensors(
        JNIEnv *env,
        jobject /* this */,
        jlong _modelHandle,
        jintArray _inputShape) {
    BenchmarkModel* model = (BenchmarkModel *) _modelHandle;
    jint* shapePtr = env->GetIntArrayElements(_inputShape, nullptr);
    jsize shapeLen = env->GetArrayLength(_inputShape);

    std::vector<int> shape(shapePtr, shapePtr + shapeLen);
    return model->resizeInputTensors(std::move(shape));
}

extern "C"
JNIEXPORT jboolean
JNICALL
Java_com_example_android_nn_benchmark_NNTestBase_runBenchmark(
        JNIEnv *env,
        jobject /* this */,
        jlong _modelHandle,
        jobject inOutDataList,
        jobject resultList,
        jint inferencesMaxCount,
        jfloat timeoutSec) {

    jclass list_class = env->FindClass("java/util/List");
    if (list_class == nullptr) {return false;}
    jmethodID list_size = env->GetMethodID(list_class, "size", "()I");
    if (list_size == nullptr) {return false;}
    jmethodID list_get = env->GetMethodID(list_class, "get", "(I)Ljava/lang/Object;");
    if (list_get == nullptr) {return false;}
    jmethodID list_add = env->GetMethodID(list_class, "add", "(Ljava/lang/Object;)Z");
    if (list_add == nullptr) {return false;}

    jclass inout_class = env->FindClass("com/example/android/nn/benchmark/InferenceInOut");
    if (inout_class == nullptr) {return false;}
    jfieldID inout_input = env->GetFieldID(inout_class, "mInput", "[B");
    if (inout_input == nullptr) {return false;}
    jfieldID inout_expectedOutput = env->GetFieldID(inout_class, "mExpectedOutput", "[B");
    if (inout_expectedOutput == nullptr) {return false;}

    jclass result_class = env->FindClass("com/example/android/nn/benchmark/InferenceResult");
    if (result_class == nullptr) {return false;}
    jmethodID result_ctor = env->GetMethodID(result_class, "<init>", "(FF)V");
    if (result_ctor == nullptr) {return false;}

    BenchmarkModel* model = (BenchmarkModel *) _modelHandle;

    std::vector<InferenceResult> result;
    std::vector<InferenceInOut> data;

    size_t data_count = env->CallIntMethod(inOutDataList, list_size);
    if (env->ExceptionCheck()) {return false;}

    // Fetch input/output arrays
    for (int i = 0;i < data_count; ++i) {
        jobject inout = env->CallObjectMethod(inOutDataList, list_get, i);
        if (env->ExceptionCheck()) {return false;}

        jbyteArray input = static_cast<jbyteArray>(
            env->GetObjectField(inout, inout_input));
        uint8_t *input_data = reinterpret_cast<uint8_t*>(
            env->GetByteArrayElements(input, NULL));
        size_t input_len = env->GetArrayLength(input);

        jbyteArray expectedOutput = static_cast<jbyteArray>(
            env->GetObjectField(inout, inout_expectedOutput));
        uint8_t *expectedOutput_data = reinterpret_cast<uint8_t*>(
            env->GetByteArrayElements(expectedOutput, NULL));
        size_t expectedOutput_len = env->GetArrayLength(expectedOutput);

        data.push_back( { input_data, input_len, expectedOutput_data, expectedOutput_len} );
    }

    // TODO: Remove success boolean from this mehtod and throw an exception in case of problems
    bool success = model->benchmark(data, inferencesMaxCount, timeoutSec, &result);

    // Release arrays
    for (int i = 0;i < data_count; ++i) {
        jobject inout = env->CallObjectMethod(inOutDataList, list_get, i);
        if (env->ExceptionCheck()) {return false;}

        jbyteArray input = static_cast<jbyteArray>(
            env->GetObjectField(inout, inout_input));
        jbyteArray expectedOutput = static_cast<jbyteArray>(
            env->GetObjectField(inout, inout_expectedOutput));
        env->ReleaseByteArrayElements(
            input, reinterpret_cast<jbyte*>(data[i].input), JNI_ABORT);
        env->ReleaseByteArrayElements(
            expectedOutput, reinterpret_cast<jbyte*>(data[i].output), JNI_ABORT);
    }

    // Generate results
    if (success) {
        for (const InferenceResult &rentry : result) {
            jobject object = env->NewObject(
                result_class, result_ctor, rentry.computeTimeSec, rentry.squareError);
            if (env->ExceptionCheck() || object == NULL) {return false;}

            env->CallBooleanMethod(resultList, list_add, object);
            if (env->ExceptionCheck()) {return false;}
        }
    }

    return success;
}
