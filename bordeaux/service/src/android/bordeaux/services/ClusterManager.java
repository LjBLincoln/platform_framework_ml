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

import java.util.ArrayList;
import java.util.HashSet;

/**
 * ClusterManager incrementally indentify representitve clusters from the input location
 * stream. Clusters are updated online using leader based clustering algorithm. The input
 * locations initially are kept by the clusters. Periodially, a cluster consolidating
 * procedure is carried out to refine the cluster centers. After consolidation, the
 * location data are released.
 */
public class ClusterManager {

    private static String TAG = "ClusterManager";

    private static float LOCATION_CLUSTER_RADIUS = 25; // meter

    private static float SEMANTIC_CLUSTER_RADIUS = 50; // meter

    private static long CONSOLIDATE_INTERVAL = 90000; // is milliseconds

    private static long LOCATION_CLUSTER_THRESHOLD = 1000; // in milliseconds

    private static long SEMANTIC_CLUSTER_THRESHOLD = 30000; // in milliseconds

    private Location mLastLocation = null;

    private long mTimeRef = 0;

    private long mSemanticClusterCount = 0;

    private ArrayList<LocationCluster> mLocClusters = new ArrayList<LocationCluster>();

    private ArrayList<SemanticCluster> mSemanticClusters = new ArrayList<SemanticCluster>();

    public ClusterManager() {
    }

    public void addSample(Location location) {
        float bestClusterDistance = Float.MAX_VALUE;
        int bestClusterIndex = -1;
        long lastDuration;
        long currentTime = location.getTime();

        if (mLastLocation != null) {
            // get the duration spent in the last location
            long duration = location.getTime() - mLastLocation.getTime();
            // TODO: set duration is a separate field
            mLastLocation.setTime(duration);
            Log.v(TAG, "sample duration: " + duration +
                  ", number of clusters: " + mLocClusters.size());

            // add the last location to cluster.
            // first find the cluster it belongs to.
            for (int i = 0; i < mLocClusters.size(); ++i) {
                float distance = mLocClusters.get(i).distanceToCenter(mLastLocation);
                Log.v(TAG, "clulster " + i + " is within " + distance + " meters");
                if (distance < bestClusterDistance) {
                    bestClusterDistance = distance;
                    bestClusterIndex = i;
                }
            }

            // add the location to the selected cluster
            if (bestClusterDistance < LOCATION_CLUSTER_RADIUS) {
              Log.v(TAG, "add sample to cluster: " + bestClusterIndex + ",( " +
                    location.getLongitude() + ", " + location.getLatitude() + ")");
                mLocClusters.get(bestClusterIndex).addSample(mLastLocation);
            } else {
                // if it is far away from all existing clusters, create a new cluster.
                LocationCluster cluster =
                        new LocationCluster(mLastLocation, CONSOLIDATE_INTERVAL);
                // move the center of the new cluster if its covering region overlaps
                // with an existing cluster.
                if (bestClusterDistance < 2 * LOCATION_CLUSTER_RADIUS) {
                    cluster.moveAwayCluster(mLocClusters.get(bestClusterIndex),
                            ((float) 2 * LOCATION_CLUSTER_RADIUS));
                }
                mLocClusters.add(cluster);
            }
        } else {
            mTimeRef = currentTime;
        }

        long collectDuration = currentTime - mTimeRef;
        Log.e(TAG, "collect duration: " + collectDuration);
        if (collectDuration > CONSOLIDATE_INTERVAL) {
            // TODO : conslidation takes time. move this to a separate thread later.
            consolidateClusters(collectDuration);
            mTimeRef = currentTime;
        }

        /*
        // TODO: this could be removed
        Log.i(TAG, "location : " +  location.getLongitude() + ", " + location.getLatitude());
        if (mLastLocation != null) {
            Log.i(TAG, "mLastLocation: " +  mLastLocation.getLongitude() + ", " +
                  mLastLocation.getLatitude());
        }  // end of deletion
        */

        mLastLocation = location;
    }

    private void consolidateClusters(long duration) {
        Log.e(TAG, "considalating " + mLocClusters.size() + " clusters");
        LocationCluster cluster;

        // TODO: which should be first? considate or merge?
        for (int i = mLocClusters.size() - 1; i >= 0; --i) {
            cluster = mLocClusters.get(i);
            cluster.consolidate(duration);

            // TODO: currently set threshold to 1 sec so almost none of the location
            // clusters will be removed.
            if (!cluster.passThreshold(LOCATION_CLUSTER_THRESHOLD)) {
                mLocClusters.remove(cluster);
            }
        }

        // merge clusters whose regions are overlapped. note that after merge
        // translates the cluster center but keeps the region size unchanged.
        for (int i = mLocClusters.size() - 1; i >= 0; --i) {
            cluster = mLocClusters.get(i);
            for (int j = i - 1; j >= 0; --j) {
                float distance = mLocClusters.get(j).distanceToCluster(cluster);
                if (distance < LOCATION_CLUSTER_RADIUS) {
                    mLocClusters.get(j).absorbCluster(cluster);
                    mLocClusters.remove(cluster);
                }
            }
        }
        updateSemanticClusters();
    }

    private void updateSemanticClusters() {
        // select candidate location clusters
        ArrayList<LocationCluster> candidates = new ArrayList<LocationCluster>();
        for (LocationCluster cluster : mLocClusters) {
            if (cluster.passThreshold(SEMANTIC_CLUSTER_THRESHOLD)) {
                candidates.add(cluster);
            }
        }
        for (LocationCluster candidate : candidates) {
            float bestClusterDistance = Float.MAX_VALUE;
            SemanticCluster bestCluster = null;
            for (SemanticCluster cluster : mSemanticClusters) {
                float distance = cluster.distanceToCluster(candidate);

                Log.e(TAG, "distance to semantic cluster: " + cluster.getSemanticId());

                if (distance < bestClusterDistance) {
                    bestClusterDistance = distance;
                    bestCluster = cluster;
                }
            }

            // add the location to the selected cluster
            SemanticCluster semanticCluster;
            if (bestClusterDistance < SEMANTIC_CLUSTER_RADIUS) {
                bestCluster.absorbCluster(candidate);
            } else {
                // if it is far away from all existing clusters, create a new cluster.
                semanticCluster = new SemanticCluster(candidate, CONSOLIDATE_INTERVAL,
                        mSemanticClusterCount++);
                mSemanticClusters.add(semanticCluster);
            }
        }
        Log.e(TAG, "location: " + candidates.size() + ", semantic: " + mSemanticClusters.size());

        candidates.clear();
    }

    public String getSemanticLocation() {
        String label = "unknown";

        if (mLastLocation != null) {
            for (SemanticCluster cluster: mSemanticClusters) {
                if (cluster.distanceToCenter(mLastLocation) < SEMANTIC_CLUSTER_RADIUS) {
                    return cluster.getSemanticId();
                }
            }
        }
        return label;
    }
}
