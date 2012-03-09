/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "jni/jni_stochastic_linear_ranker.h"
#include "native/common_defs.h"
#include "native/sparse_weight_vector.h"
#include "native/stochastic_linear_ranker.h"

#include <vector>
#include <string>
using std::string;
using std::vector;
using std::hash_map;
using learning_stochastic_linear::StochasticLinearRanker;
using learning_stochastic_linear::SparseWeightVector;

void CreateSparseWeightVector(JNIEnv* env, const jobjectArray keys, const float* values,
    const int length, SparseWeightVector<string> * sample) {
  for (int i = 0; i < length; ++i) {
    jboolean iscopy;
    jstring s = (jstring) env->GetObjectArrayElement(keys, i);
    const char *key = env->GetStringUTFChars(s, &iscopy);
    sample->SetElement(key, static_cast<double>(values[i]));
    env->ReleaseStringUTFChars(s,key);
  }
}

void DecomposeSparseWeightVector(JNIEnv* env, jobjectArray *keys, jfloatArray *values,
    const int length, SparseWeightVector<string> *sample) {

  SparseWeightVector<string>::Wmap w_ = sample->GetMap();
  int i=0;
  for ( SparseWeightVector<string>::Witer_const iter = w_.begin();
    iter != w_.end(); ++iter) {
    std::string key = iter->first;
    jstring jstr = env->NewStringUTF(key.c_str());
    env->SetObjectArrayElement(*keys, i, jstr);
    double value = iter->second;
    jfloat s[1];
    s[0] = value;
    env->SetFloatArrayRegion(*values, i, 1, s);
    i++;
  }
}

jboolean Java_android_bordeaux_learning_StochasticLinearRanker_nativeLoadClassifier(
    JNIEnv* env,
    jobject thiz,
    jobjectArray key_array_model,
    jfloatArray value_array_model,
    jfloatArray value_array_param,
    jint paPtr) {

  StochasticLinearRanker<string>* classifier = (StochasticLinearRanker<string>*) paPtr;
  if (classifier && key_array_model && value_array_model && value_array_param) {
    const int keys_m_len = env->GetArrayLength(key_array_model);
    jfloat* values_m = env->GetFloatArrayElements(value_array_model, NULL);
    const int values_m_len = env->GetArrayLength(value_array_model);
    jfloat* param_m = env->GetFloatArrayElements(value_array_param, NULL);

    if (values_m && key_array_model && values_m_len == keys_m_len) {
      SparseWeightVector<string> model;
      CreateSparseWeightVector(env, key_array_model, values_m, values_m_len, &model);
      model.SetNormalizer((double) param_m[0]);
      classifier->LoadWeights(model);
      classifier->SetIterationNumber((uint64) param_m[1]);
      classifier->SetNormConstraint((double) param_m[2]);

      switch ((int) param_m[3]){
      case 0 :
        classifier->SetRegularizationType(learning_stochastic_linear::L0);
        break;
      case 1 :
        classifier->SetRegularizationType(learning_stochastic_linear::L1);
        break;
      case 2 :
        classifier->SetRegularizationType(learning_stochastic_linear::L2);
        break;
      case 3 :
        classifier->SetRegularizationType(learning_stochastic_linear::L1L2);
        break;
      case 4 :
        classifier->SetRegularizationType(learning_stochastic_linear::L1LInf);
        break;
      }

      classifier->SetLambda((double) param_m[4]);

      switch ((int) param_m[5]){
      case 0 :
        classifier->SetUpdateType(learning_stochastic_linear::FULL_CS);
        break;
      case 1 :
        classifier->SetUpdateType(learning_stochastic_linear::CLIP_CS);
        break;
      case 2 :
        classifier->SetUpdateType(learning_stochastic_linear::REG_CS);
        break;
      case 3 :
        classifier->SetUpdateType(learning_stochastic_linear::SL);
        break;
      case 4 :
        classifier->SetUpdateType(learning_stochastic_linear::ADAPTIVE_REG);
        break;
      }

      switch ((int) param_m[6]){
      case 0 :
        classifier->SetAdaptationMode(learning_stochastic_linear::CONST);
        break;
      case 1 :
        classifier->SetAdaptationMode(learning_stochastic_linear::INV_LINEAR);
        break;
      case 2 :
        classifier->SetAdaptationMode(learning_stochastic_linear::INV_QUADRATIC);
        break;
      case 3 :
        classifier->SetAdaptationMode(learning_stochastic_linear::INV_SQRT);
        break;
      }

      switch ((int) param_m[7]){
      case 0 :
        classifier->SetKernelType(learning_stochastic_linear::LINEAR, (double) param_m[8],
                                  (double) param_m[9],(double) param_m[10]);
        break;
      case 1 : classifier->SetKernelType(learning_stochastic_linear::POLY, (double) param_m[8],
                                         (double) param_m[9],(double) param_m[10]);
        break;
      case 2 : classifier->SetKernelType(learning_stochastic_linear::RBF, (double) param_m[8],
                                          (double) param_m[9],(double) param_m[10]);
        break;
      }

      switch ((int) param_m[11]){
      case 0 :
        classifier->SetRankLossType(learning_stochastic_linear::PAIRWISE);
        break;
      case 1 :
        classifier->SetRankLossType(learning_stochastic_linear::RECIPROCAL_RANK);
        break;
      }

      classifier->SetAcceptanceProbability((double) param_m[12]);
      classifier->SetMiniBatchSize((uint64)param_m[13]);
      classifier->SetGradientL0Norm((int32)param_m[14]);
      env->ReleaseFloatArrayElements(value_array_model, values_m, JNI_ABORT);
      env->ReleaseFloatArrayElements(value_array_param, param_m, JNI_ABORT);
      return JNI_TRUE;
    }
  }
  return JNI_FALSE;
}

jint Java_android_bordeaux_learning_StochasticLinearRanker_nativeGetLengthClassifier(
  JNIEnv* env,
  jobject thiz,
  jint paPtr) {

  StochasticLinearRanker<string>* classifier = (StochasticLinearRanker<string>*) paPtr;
  SparseWeightVector<string> M_weights;
  classifier->SaveWeights(&M_weights);

  SparseWeightVector<string>::Wmap w_map = M_weights.GetMap();
  int len = w_map.size();
  return len;
}

void Java_android_bordeaux_learning_StochasticLinearRanker_nativeGetClassifier(
  JNIEnv* env,
  jobject thiz,
  jobjectArray key_array_model,
  jfloatArray value_array_model,
  jfloatArray value_array_param,
  jint paPtr) {

  StochasticLinearRanker<string>* classifier = (StochasticLinearRanker<string>*) paPtr;

  SparseWeightVector<string> M_weights;
  classifier->SaveWeights(&M_weights);
  double Jni_weight_normalizer = M_weights.GetNormalizer();
  int Jni_itr_num = classifier->GetIterationNumber();
  double Jni_norm_cont = classifier->GetNormContraint();
  int Jni_reg_type = classifier->GetRegularizationType();
  double Jni_lambda = classifier->GetLambda();
  int Jni_update_type = classifier->GetUpdateType();
  int Jni_AdaptationMode = classifier->GetAdaptationMode();
  double Jni_kernel_param, Jni_kernel_gain, Jni_kernel_bias;
  int Jni_kernel_type = classifier->GetKernelType(&Jni_kernel_param, &Jni_kernel_gain, &Jni_kernel_bias);
  int Jni_rank_loss_type = classifier->GetRankLossType();
  double Jni_accp_prob = classifier->GetAcceptanceProbability();
  uint64 Jni_min_batch_size = classifier->GetMiniBatchSize();
  int32 Jni_GradL0Norm = classifier->GetGradientL0Norm();
  const int Var_num = 15;
  jfloat s[Var_num]= {  (float) Jni_weight_normalizer,
                        (float) Jni_itr_num,
                        (float) Jni_norm_cont,
                        (float) Jni_reg_type,
                        (float) Jni_lambda,
                        (float) Jni_update_type,
                        (float) Jni_AdaptationMode,
                        (float) Jni_kernel_type,
                        (float) Jni_kernel_param,
                        (float) Jni_kernel_gain,
                        (float) Jni_kernel_bias,
                        (float) Jni_rank_loss_type,
                        (float) Jni_accp_prob,
                        (float) Jni_min_batch_size,
                        (float) Jni_GradL0Norm};

  env->SetFloatArrayRegion(value_array_param, 0, Var_num, s);

  SparseWeightVector<string>::Wmap w_map = M_weights.GetMap();
  int array_len = w_map.size();

  DecomposeSparseWeightVector(env, &key_array_model, &value_array_model, array_len, &M_weights);
}

jint Java_android_bordeaux_learning_StochasticLinearRanker_initNativeClassifier(JNIEnv* env,
                             jobject thiz) {
  StochasticLinearRanker<string>* classifier = new StochasticLinearRanker<string>();
  classifier->SetUpdateType(learning_stochastic_linear::REG_CS);
  classifier->SetRegularizationType(learning_stochastic_linear::L2);
  return ((jint) classifier);
}


jboolean Java_android_bordeaux_learning_StochasticLinearRanker_deleteNativeClassifier(JNIEnv* env,
                               jobject thiz,
                               jint paPtr) {
  StochasticLinearRanker<string>* classifier = (StochasticLinearRanker<string>*) paPtr;
  delete classifier;
  return JNI_TRUE;
}

jboolean Java_android_bordeaux_learning_StochasticLinearRanker_nativeUpdateClassifier(
  JNIEnv* env,
  jobject thiz,
  jobjectArray key_array_positive,
  jfloatArray value_array_positive,
  jobjectArray key_array_negative,
  jfloatArray value_array_negative,
  jint paPtr) {
  StochasticLinearRanker<string>* classifier = (StochasticLinearRanker<string>*) paPtr;

  if (classifier && key_array_positive && value_array_positive &&
      key_array_negative && value_array_negative) {

    const int keys_p_len = env->GetArrayLength(key_array_positive);
    jfloat* values_p = env->GetFloatArrayElements(value_array_positive, NULL);
    const int values_p_len = env->GetArrayLength(value_array_positive);
    jfloat* values_n = env->GetFloatArrayElements(value_array_negative, NULL);
    const int values_n_len = env->GetArrayLength(value_array_negative);
    const int keys_n_len = env->GetArrayLength(key_array_negative);

    if (values_p && key_array_positive && values_p_len == keys_p_len &&
      values_n && key_array_negative && values_n_len == keys_n_len) {

      SparseWeightVector<string> sample_pos;
      SparseWeightVector<string> sample_neg;
      CreateSparseWeightVector(env, key_array_positive, values_p, values_p_len, &sample_pos);
      CreateSparseWeightVector(env, key_array_negative, values_n, values_n_len, &sample_neg);
      classifier->UpdateClassifier(sample_pos, sample_neg);
      env->ReleaseFloatArrayElements(value_array_negative, values_n, JNI_ABORT);
      env->ReleaseFloatArrayElements(value_array_positive, values_p, JNI_ABORT);

      return JNI_TRUE;
    }
    env->ReleaseFloatArrayElements(value_array_negative, values_n, JNI_ABORT);
    env->ReleaseFloatArrayElements(value_array_positive, values_p, JNI_ABORT);
  }
  return JNI_FALSE;
}


jfloat Java_android_bordeaux_learning_StochasticLinearRanker_nativeScoreSample(
  JNIEnv* env,
  jobject thiz,
  jobjectArray key_array,
  jfloatArray value_array,
  jint paPtr) {

  StochasticLinearRanker<string>* classifier = (StochasticLinearRanker<string>*) paPtr;

  if (classifier && key_array && value_array) {

    jfloat* values = env->GetFloatArrayElements(value_array, NULL);
    const int values_len = env->GetArrayLength(value_array);
    const int keys_len = env->GetArrayLength(key_array);

    if (values && key_array && values_len == keys_len) {
      SparseWeightVector<string> sample;
      CreateSparseWeightVector(env, key_array, values, values_len, &sample);
      env->ReleaseFloatArrayElements(value_array, values, JNI_ABORT);
      return classifier->ScoreSample(sample);
    }
  }
  return -1;
}
