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

import android.location.Location;
import android.text.format.Time;
import android.util.Log;

import java.lang.Math;
import java.util.ArrayList;

public class SemanticCluster extends BaseCluster {

    public static String TAG = "SemanticCluster";

    private String mSemanticId;

    public SemanticCluster(LocationCluster cluster, long avgInterval, long semanticIndex) {
        mCenter = new double[3];
        for (int i = 0; i < 3; ++i) {
            mCenter[i] = cluster.mCenter[i];
        }
        mDuration = cluster.mDuration;
        mAvgInterval = avgInterval;

        setSemanticId(semanticIndex);
    }

    public String getSemanticId() {
        return mSemanticId;
    }

    private void setSemanticId(long index) {
        mSemanticId = "cluser: " + String.valueOf(index);
    }
}
