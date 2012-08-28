/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you my not use this file except in compliance with the License.
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

package android.bordeaux.services;

import android.os.IBinder;
import android.util.Log;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashSet;
import java.util.Iterator;
import java.io.Serializable;
import java.io.*;
import java.lang.Boolean;
import android.bordeaux.services.FeatureAssembly;
import android.bordeaux.learning.HistogramPredictor;

/**
 * This is interface to implement Prediction based on histogram that
 * uses predictor_histogram from learnerning section
 */
public class Predictor extends IPredictor.Stub
        implements IBordeauxLearner {
    private final String TAG = "Predictor";
    private ModelChangeCallback modelChangeCallback = null;

    private HistogramPredictor mPredictor = new HistogramPredictor();
    private FeatureAssembly mFeatureAssembly = new FeatureAssembly();

    public static final String SET_FEATURE = "Set Feature";
    public static final String USE_HISTORY = "Use History";

    public static final String PREVIOUS_SAMPLE = "Previous Sample";

    private boolean mUseHistory = false;
    private long mHistorySpan = 0;
    private String mPrevSample;
    private long mPrevSampleTime;

    /**
     * Reset the Predictor
     */
    public void resetPredictor(){
        mPredictor.resetPredictor();

        if (modelChangeCallback != null) {
            modelChangeCallback.modelChanged(this);
        }
    }

    /**
     * Input is a sampleName e.g.action name. This input is then augmented with requested build-in
     * features such as time and location to create sampleFeatures. The sampleFeatures is then
     * pushed to the histogram
     */
    public void pushNewSample(String sampleName) {
        Map<String, String> sampleFeatures = getSampleFeatures();
        Log.e(TAG, "pushNewSample " + sampleName + ": " + sampleFeatures);

        // TODO: move to the end of the function?
        mPrevSample = sampleName;
        mPrevSampleTime = System.currentTimeMillis();

        mPredictor.addSample(sampleName, sampleFeatures);
        if (modelChangeCallback != null) {
            modelChangeCallback.modelChanged(this);
        }
    }

    private Map<String, String> getSampleFeatures() {
        Map<String, String> sampleFeatures = mFeatureAssembly.getFeatureMap();
        long currTime = System.currentTimeMillis();

        if (mUseHistory && mPrevSample != null &&
            ((currTime - mPrevSampleTime) < mHistorySpan)) {
            sampleFeatures.put(PREVIOUS_SAMPLE, mPrevSample);
        }

        return sampleFeatures;
    }

    // TODO: getTopK samples instead get scord for debugging only
    /**
     * return probabilty of an exmple using the histogram
     */
    public List<StringFloat> getTopCandidates(int topK) {
        ArrayList<StringFloat> result = new ArrayList<StringFloat>(topK);
        Map<String, String> features = getSampleFeatures();

        List<Map.Entry<String, Double> > topApps = mPredictor.findTopClasses(features, topK);

        int listSize =  topApps.size();
        if (topK > 0) {
            listSize = Math.min(topK, listSize);
        }

        for (int i = 0; i < listSize; ++i) {
            Map.Entry<String, Double> entry = topApps.get(i);
            result.add(new StringFloat(entry.getKey(), entry.getValue().floatValue()));
        }
        return result;
    }

    /**
     * Set parameters for 1) using History in probability estimations e.g. consider the last event
     * and 2) featureAssembly e.g. time and location.
     */
    public boolean setPredictorParameter(String key, String value) {
        boolean result = true;
        if (key.equals(SET_FEATURE)) {
            result = mFeatureAssembly.registerFeature(value);
            if (result) {
                mPredictor.useFeature(value);
            } else {
               Log.e(TAG,"Setting on feauture: " + value + " which is not available");
            }
        } else if (key.equals(USE_HISTORY)) {
            mUseHistory = true;
            mHistorySpan = Long.valueOf(value);
            mPredictor.useFeature(PREVIOUS_SAMPLE);
        } else {
            Log.e(TAG,"Setting parameter " + key + " with " + value + " is not valid");
        }
        return result;
    }

    // Beginning of the IBordeauxLearner Interface implementation
    public byte [] getModel() {
      return mPredictor.getModel();
    }

    public boolean setModel(final byte [] modelData) {
      return mPredictor.setModel(modelData);
    }

    public IBinder getBinder() {
        return this;
    }

    public void setModelChangeCallback(ModelChangeCallback callback) {
        modelChangeCallback = callback;
    }
    // End of IBordeauxLearner Interface implemenation
}
