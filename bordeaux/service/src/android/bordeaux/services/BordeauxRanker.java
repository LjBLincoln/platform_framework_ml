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

package android.bordeaux.services;

import android.bordeaux.services.ILearning_StochasticLinearRanker;
import android.bordeaux.services.StringFloat;
import android.content.Context;
import android.os.RemoteException;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

/** Ranker for the Learning framework.
 *  For training: call updateClassifier with a pair of samples.
 *  For ranking: call scoreSample to the score of the rank
 *  Data is represented as sparse key, value pair. And key is a String, value
 *  is a float.
 *  Note: since the actual ranker is running in a remote the service.
 *  Sometimes the connection may be lost or not established.
 *
 */
public class BordeauxRanker {
    static final String TAG = "BordeauxRanker";
    static final String RANKER_NOTAVAILABLE = "Ranker not Available";
    private Context mContext;
    private String mName;
    private ILearning_StochasticLinearRanker mRanker;
    private ArrayList<StringFloat> getArrayList(final HashMap<String, Float> sample) {
        ArrayList<StringFloat> stringfloat_sample = new ArrayList<StringFloat>();
        for (Map.Entry<String, Float> x : sample.entrySet()) {
           StringFloat v = new StringFloat();
           v.key = x.getKey();
           v.value = x.getValue();
           stringfloat_sample.add(v);
        }
        return stringfloat_sample;
    }

    private boolean retrieveRanker() {
        if (mRanker == null)
            mRanker = BordeauxManagerService.getRanker(mContext, mName);
        // if classifier is not available, return false
        if (mRanker == null) {
            Log.e(TAG,"Ranker not available.");
            return false;
        }
        return true;
    }

    public BordeauxRanker(Context context) {
        mContext = context;
        mName = "defaultRanker";
        mRanker = BordeauxManagerService.getRanker(context, mName);
    }

    public BordeauxRanker(Context context, String name) {
        mContext = context;
        mName = name;
        mRanker = BordeauxManagerService.getRanker(context, mName);
    }

    // Update the ranker with two samples, sample1 has higher rank than
    // sample2.
    public boolean update(final HashMap<String, Float> sample1,
                          final HashMap<String, Float> sample2) {
        if (!retrieveRanker())
            return false;
        try {
            mRanker.UpdateClassifier(getArrayList(sample1), getArrayList(sample2));
        } catch (RemoteException e) {
            Log.e(TAG,"Exception: updateClassifier.");
            return false;
        }
        return true;
    }

    public float scoreSample(final HashMap<String, Float> sample) {
        if (!retrieveRanker())
            throw new RuntimeException(RANKER_NOTAVAILABLE);
        try {
            return mRanker.ScoreSample(getArrayList(sample));
        } catch (RemoteException e) {
            Log.e(TAG,"Exception: scoring the sample.");
            throw new RuntimeException(RANKER_NOTAVAILABLE);
        }
    }

    public void loadModel(String filename) {
        if (!retrieveRanker())
            throw new RuntimeException(RANKER_NOTAVAILABLE);
        try {
            mRanker.LoadModel(filename);
        } catch (RemoteException e) {
            Log.e(TAG,"Exception: loading model.");
            throw new RuntimeException(RANKER_NOTAVAILABLE);
        }
    }

    public String saveModel(String filename) {
        if (!retrieveRanker())
            throw new RuntimeException(RANKER_NOTAVAILABLE);
        try {
            return mRanker.SaveModel(filename);
        } catch (RemoteException e) {
            Log.e(TAG,"Exception: saving model.");
            throw new RuntimeException(RANKER_NOTAVAILABLE);
        }
    }
}
