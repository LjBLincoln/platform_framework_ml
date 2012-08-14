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

public class BaseCluster {

    public static String TAG = "BaseCluster";

    protected double[] mCenter;

    protected long mAvgInterval;
    protected long mDuration;

    protected static final double EARTH_RADIUS = 6378100f;

    public BaseCluster() {
    }

    public BaseCluster(Location location, long avgInterval) {
        mAvgInterval = avgInterval;
        mCenter = getLocationVector(location);

        mDuration = 0;
    }

    protected double[] getLocationVector(Location location) {
        double vector[] = new double[3];
        double lambda = Math.toRadians(location.getLongitude());
        double phi = Math.toRadians(location.getLatitude());
        vector[0] = Math.cos(lambda) * Math.cos(phi);
        vector[1] = Math.sin(lambda) * Math.cos(phi);
        vector[2] = Math.sin(phi);
        return vector;
    }

    private double computeDistance(double[] vector1, double[] vector2) {
        double product = 0f;
        for (int i = 0; i < 3; ++i) {
            product += vector1[i] * vector2[i];
        }
        double radian = Math.acos(Math.min(product, 1f));
        return radian * EARTH_RADIUS;
    }

    /*
     * This computes the distance from loation to the cluster center in meter.
     */
    public float distanceToCenter(Location location) {
        return (float) computeDistance(mCenter, getLocationVector(location));
    }

    public float distanceToCluster(BaseCluster cluster) {
        return (float) computeDistance(mCenter, cluster.mCenter);
    }

    public void absorbCluster(BaseCluster cluster) {
        if (cluster.mAvgInterval != mAvgInterval) {
            throw new RuntimeException(
                    "aborbing cluster failed: inconsistent average invergal ");
        }

        double currWeight = ((double) mDuration) / (mDuration + cluster.mDuration);
        double newWeight = 1f - currWeight;
        double norm = 0;
        for (int i = 0; i < 3; ++i) {
            mCenter[i] = currWeight * mCenter[i] + newWeight * cluster.mCenter[i];
            norm += mCenter[i] * mCenter[i];
        }
        // normalize
        for (int i = 0; i < 3; ++i) {
            mCenter[i] /= norm;
        }
        mDuration += cluster.mDuration;
    }

    public boolean passThreshold(long durationThreshold) {
        // TODO: might want to keep semantic cluster
        return mDuration > durationThreshold;
    }
}
