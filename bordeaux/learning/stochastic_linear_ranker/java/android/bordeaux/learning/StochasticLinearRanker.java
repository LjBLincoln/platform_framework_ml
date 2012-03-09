/*
 * Copyright (C) 2011 The Android Open Source Project
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


package android.bordeaux.learning;
import android.util.Log;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

/**
 * Stochastic Linear Ranker, learns how to rank a sample. The learned rank score
 * can be used to compare samples.
 * This java class wraps the native StochasticLinearRanker class.
 * To update the ranker, call updateClassifier with two samples, with the first
 * one having higher rank than the second one.
 * To get the rank score of the sample call scoreSample.
 *  TODO: adding more interfaces for changing the learning parameters
 */
public class StochasticLinearRanker {
    String TAG = "StochasticLinearRanker";
    static int VAR_NUM = 15;
    public StochasticLinearRanker() {
        mNativeClassifier = initNativeClassifier();
    }

    /**
     * Train the ranker with a pair of samples. A sample,  a pair of arrays of
     * keys and values. The first sample should have higher rank than the second
     * one.
     */
    public boolean updateClassifier(String[] keys_positive,
                                    float[] values_positive,
                                    String[] keys_negative,
                                    float[] values_negative) {
        return nativeUpdateClassifier(keys_positive, values_positive,
                                      keys_negative, values_negative,
                                      mNativeClassifier);
    }

    /**
     * Get the rank score of the sample, a sample is a list of key, value pairs..
     */
    public float scoreSample(String[] keys, float[] values) {
        return nativeScoreSample(keys, values, mNativeClassifier);
    }

    /**
     * Get the current model and parameters of ranker
     */
    public void getModel(ArrayList keys_list, ArrayList values_list, ArrayList param_list){
        int len = nativeGetLengthClassifier(mNativeClassifier);
        String[] keys = new String[len];
        float[] values = new float[len];
        float[] param = new float[VAR_NUM];
        nativeGetClassifier(keys, values, param, mNativeClassifier);
        boolean add_flag;
        for (int  i=0; i< keys.length ; i++){
            add_flag = keys_list.add(keys[i]);
            add_flag = values_list.add(values[i]);
        }
        for (int  i=0; i< param.length ; i++)
            add_flag = param_list.add(param[i]);
    }

    /**
     * use the given model and parameters for ranker
     */
    public boolean loadModel(String [] keys, float[] values, float[] param){
        return nativeLoadClassifier(keys, values, param, mNativeClassifier);
    }

    @Override
    protected void finalize() throws Throwable {
        deleteNativeClassifier(mNativeClassifier);
    }

    static {
        System.loadLibrary("bordeaux");
    }

    private int mNativeClassifier;

    /*
     * The following methods are the java stubs for the jni implementations.
     */
    private native int initNativeClassifier();

    private native void deleteNativeClassifier(int classifierPtr);

    private native boolean nativeUpdateClassifier(
            String[] keys_positive,
            float[] values_positive,
            String[] keys_negative,
            float[] values_negative,
            int classifierPtr);

    private native float nativeScoreSample(String[] keys,
                                           float[] values,
                                           int classifierPtr);
    private native void nativeGetClassifier(String [] keys, float[] values, float[] param,
                                             int classifierPtr);
    private native boolean nativeLoadClassifier(String [] keys, float[] values,
                                                 float[] param, int classifierPtr);
    private native int nativeGetLengthClassifier(int classifierPtr);
}
