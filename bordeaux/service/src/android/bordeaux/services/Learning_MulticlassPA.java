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

import android.bordeaux.learning.MulticlassPA;
import java.util.List;
import java.util.ArrayList;

public class Learning_MulticlassPA extends ILearning_MulticlassPA.Stub {
    private MulticlassPA mMulticlassPA_learner;

    class IntFloatArray {
        int[] indexArray;
        float[] floatArray;
    };

    private IntFloatArray splitIntFloatArray(List<IntFloat> sample) {
        IntFloatArray splited = new IntFloatArray();
        ArrayList<IntFloat> s = (ArrayList<IntFloat>)sample;
        splited.indexArray = new int[s.size()];
        splited.floatArray = new float[s.size()];
        for (int i = 0; i < s.size(); i++) {
            splited.indexArray[i] = s.get(i).index;
            splited.floatArray[i] = s.get(i).value;
        }
        return splited;
    }

    public Learning_MulticlassPA() {
        mMulticlassPA_learner = new MulticlassPA(2, 2, 0.001f);
    }

    // This implementation, combines training and prediction in one step.
    // The return value is the prediction value for the supplied sample. It
    // also update the model with the current sample.
    public void TrainOneSample(List<IntFloat> sample, int target) {
        IntFloatArray splited = splitIntFloatArray(sample);
        mMulticlassPA_learner.sparseTrainOneExample(splited.indexArray,
                                                    splited.floatArray,
                                                    target);
    }

    public int Classify(List<IntFloat> sample) {
        IntFloatArray splited = splitIntFloatArray(sample);
        int prediction = mMulticlassPA_learner.sparseGetClass(splited.indexArray,
                                                              splited.floatArray);
        return prediction;
    }

}
