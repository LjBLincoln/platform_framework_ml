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
import android.util.Pair;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;


/**
 * A histogram based predictor which records co-occurrences of applations with a speficic feature,
 * for example, location, * time of day, etc. The histogram is kept in a two level hash table.
 * The first level key is the feature value and the second level key is the app id.
 */

// TODO: Use Parceable or Serializable to load and save this class
public class HistogramPredictor {
    final static String TAG = "HistogramPredictor";

    private HashMap<String, HistogramCounter> mPredictor =
            new HashMap<String, HistogramCounter>();

    private static final double FEATURE_INACTIVE_LIKELIHOOD = 0.00000001;
    private final double logInactive = Math.log(FEATURE_INACTIVE_LIKELIHOOD);

    /*
     * This class keeps the histogram counts for each feature and provide the
     * joint probabilities of <feature, class>.
     */
    private class HistogramCounter {
        private HashMap<String, HashMap<String, Integer> > mCounter =
                new HashMap<String, HashMap<String, Integer> >();
        private int mTotalCount;

        public HistogramCounter() {
            resetCounter();
        }

        public void setCounter(HashMap<String, HashMap<String, Integer> > counter) {
            resetCounter();
            mCounter.putAll(counter);

            // get total count
            for (Map.Entry<String, HashMap<String, Integer> > entry : counter.entrySet()) {
                for (Integer value : entry.getValue().values()) {
                    mTotalCount += value.intValue();
                }
            }
        }

        public void resetCounter() {
            mCounter.clear();
            mTotalCount = 0;
        }

        public void addSample(String className, String featureValue) {
            HashMap<String, Integer> classCounts;

            if (!mCounter.containsKey(featureValue)) {
                classCounts = new HashMap<String, Integer>();
                mCounter.put(featureValue, classCounts);
            }
            classCounts = mCounter.get(featureValue);

            int count = (classCounts.containsKey(className)) ?
                    classCounts.get(className) + 1 : 1;
            classCounts.put(className, count);
            mTotalCount++;
        }

        public HashMap<String, Double> getClassScores(String featureValue) {
            HashMap<String, Double> classScores = new HashMap<String, Double>();

            double logTotalCount = Math.log((double) mTotalCount);
            if (mCounter.containsKey(featureValue)) {
                for(Map.Entry<String, Integer> entry :
                        mCounter.get(featureValue).entrySet()) {
                    double score =
                            Math.log((double) entry.getValue()) - logTotalCount;
                    classScores.put(entry.getKey(), score);
                }
            }
            return classScores;
        }

        public HashMap<String, HashMap<String, Integer> > getCounter() {
            return mCounter;
        }
    }

    /*
     * Given a map of feature name -value pairs returns the mostly likely apps to
     * be launched with corresponding likelihoods.
     */
    public List<Map.Entry<String, Double> > findTopClasses(Map<String, String> features, int topK) {
        // Most sophisticated function in this class
        HashMap<String, Double> appScores = new HashMap<String, Double>();
        double defaultLikelihood = mPredictor.size() * logInactive;

        // compute all app scores
        for (Map.Entry<String, HistogramCounter> entry : mPredictor.entrySet()) {
            String featureName = entry.getKey();
            HistogramCounter counter = entry.getValue();

            if (features.containsKey(featureName)) {
                String featureValue = features.get(featureName);
                HashMap<String, Double> scoreMap = counter.getClassScores(featureValue);

                for (Map.Entry<String, Double> item : scoreMap.entrySet()) {
                    String appName = item.getKey();
                    double appScore = item.getValue();

                    double score = (appScores.containsKey(appName)) ?
                        appScores.get(appName) : defaultLikelihood;
                    score += appScore - logInactive;

                    appScores.put(appName, score);
                }
            }
        }

        // sort app scores
        List<Map.Entry<String, Double> > appList =
               new ArrayList<Map.Entry<String, Double> >(appScores.size());
        appList.addAll(appScores.entrySet());
        Collections.sort(appList, new  Comparator<Map.Entry<String, Double> >() {
            public int compare(Map.Entry<String, Double> o1,
                               Map.Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });

        Log.e(TAG, "findTopApps appList: " + appList);
        return appList;
    }

    /*
     * Add a new observation of given sample id and features to the histograms
     */
    public void addSample(String sampleId, Map<String, String> features) {
        for (Map.Entry<String, HistogramCounter> entry : mPredictor.entrySet()) {
            String featureName = entry.getKey();
            HistogramCounter counter = entry.getValue();

            if (features.containsKey(featureName)) {
                String featureValue = features.get(featureName);
                counter.addSample(sampleId, featureValue);
            }
        }
    }

    /*
     * reset predictor to a empty model
     */
    public void resetPredictor() {
        // TODO: not sure this step would reduce memory waste
        for (HistogramCounter counter : mPredictor.values()) {
            counter.resetCounter();
        }
        mPredictor.clear();
    }

    /*
     * specify a feature to used for prediction
     */
    public void useFeature(String featureName) {
        if (!mPredictor.containsKey(featureName)) {
            mPredictor.put(featureName, new HistogramCounter());
        }
    }

    /*
     * convert the prediction model into a byte array
     */
    public byte[] getModel() {
        // TODO: convert model to a more memory efficient data structure.
        HashMap<String, HashMap<String, HashMap<String, Integer > > > model =
                new HashMap<String, HashMap<String, HashMap<String, Integer > > >();
        for(Map.Entry<String, HistogramCounter> entry : mPredictor.entrySet()) {
            model.put(entry.getKey(), entry.getValue().getCounter());
        }

        try {
            ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
            ObjectOutputStream objStream = new ObjectOutputStream(byteStream);
            objStream.writeObject(model);
            byte[] bytes = byteStream.toByteArray();
            //Log.i(TAG, "getModel: " + bytes);
            return bytes;
        } catch (IOException e) {
            throw new RuntimeException("Can't get model");
        }
    }

    /*
     * set the prediction model from a model data in the format of byte array
     */
    public boolean setModel(final byte[] modelData) {
        HashMap<String, HashMap<String, HashMap<String, Integer > > > model;

        try {
            ByteArrayInputStream input = new ByteArrayInputStream(modelData);
            ObjectInputStream objStream = new ObjectInputStream(input);
            model = (HashMap<String, HashMap<String, HashMap<String, Integer > > >)
                    objStream.readObject();
        } catch (IOException e) {
            throw new RuntimeException("Can't load model");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Learning class not found");
        }

        resetPredictor();
        for (Map.Entry<String, HashMap<String, HashMap<String, Integer> > > entry :
                model.entrySet()) {
            useFeature(entry.getKey());
            mPredictor.get(entry.getKey()).setCounter(entry.getValue());
        }
        return true;
    }
}
