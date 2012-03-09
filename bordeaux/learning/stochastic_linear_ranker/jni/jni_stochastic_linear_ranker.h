/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef LEARNING_JNI_STOCHASTIC_LINEAR_RANKER_H
#define LEAENING_JNI_STOCHASTIC_LINEAR_RANKER_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL
Java_android_bordeaux_learning_StochasticLinearRanker_initNativeClassifier(
    JNIEnv* env,
    jobject thiz);


JNIEXPORT jboolean JNICALL
Java_android_bordeaux_learning_StochasticLinearRanker_deleteNativeClassifier(
    JNIEnv* env,
    jobject thiz,
    jint paPtr);

JNIEXPORT jboolean JNICALL
Java_android_bordeaux_learning_StochasticLinearRanker_nativeUpdateClassifier(
    JNIEnv* env,
    jobject thiz,
    jobjectArray key_array_positive,
    jfloatArray value_array_positive,
    jobjectArray key_array_negative,
    jfloatArray value_array_negative,
    jint paPtr);

JNIEXPORT jfloat JNICALL
Java_android_bordeaux_learning_StochasticLinearRanker_nativeScoreSample(
    JNIEnv* env,
    jobject thiz,
    jobjectArray key_array,
    jfloatArray value_array,
    jint paPtr);

JNIEXPORT void JNICALL
Java_android_bordeaux_learning_StochasticLinearRanker_nativeGetClassifier(
    JNIEnv* env,
    jobject thiz,
    jobjectArray key_array_model,
    jfloatArray value_array_model,
    jfloatArray value_array_param,
    jint paPtr);

JNIEXPORT jint JNICALL
Java_android_bordeaux_learning_StochasticLinearRanker_nativeGetLengthClassifier(
    JNIEnv* env,
    jobject thiz,
    jint paPtr);

JNIEXPORT jboolean JNICALL
Java_android_bordeaux_learning_StochasticLinearRanker_nativeLoadClassifier(
    JNIEnv* env,
    jobject thiz,
    jobjectArray key_array_model,
    jfloatArray value_array_model,
    jfloatArray value_array_param,
    jint paPtr);

#ifdef __cplusplus
}
#endif

#endif /* ANDROID_LERNING_JNI_STOCHASTIC_LINEAR_RANKER_H */
