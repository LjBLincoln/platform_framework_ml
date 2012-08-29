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

import android.content.Context;
import android.location.Location;
import android.text.format.Time;
import android.util.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * ClusterManager incrementally indentify representitve clusters from the input location
 * stream. Clusters are updated online using leader based clustering algorithm. The input
 * locations initially are kept by the clusters. Periodially, a cluster consolidating
 * procedure is carried out to refine the cluster centers. After consolidation, the
 * location data are released.
 */
public class ClusterManager {

    private static String TAG = "ClusterManager";

    private static float LOCATION_CLUSTER_RADIUS = 50; // meter

    private static float SEMANTIC_CLUSTER_RADIUS = 100; // meter

    // Consoliate location clusters (and check for new semantic clusters)
    // every 30 minutes (1800 seconds).
    private static final long CONSOLIDATE_INTERVAL = 1800;

    // Prune away clusters that are stayed for less than 3 minutes (180 seconds)
    private static long LOCATION_CLUSTER_THRESHOLD = 180;

    // A location cluster can be labeled as a semantic cluster if it has been
    // stayed for at least 10 minutes (600 seconds) within a day.
    private static final long SEMANTIC_CLUSTER_THRESHOLD = 600; // seconds

    // Reset location cluters every 6 hours (21600 seconds).
    private static final long LOCATION_REFRESH_PERIOD = 21600; // seconds

    private static String UNKNOWN_LOCATION = "Unknown Location";

    private static String HOME = "Home";

    private static String OFFICE = "Office";

    private Location mLastLocation = null;

    private long mClusterDuration;

    private long mTimeRef = 0;

    private long mSemanticClusterCount = 0;

    private ArrayList<LocationCluster> mLocationClusters = new ArrayList<LocationCluster>();

    private ArrayList<SemanticCluster> mSemanticClusters = new ArrayList<SemanticCluster>();

    private AggregatorRecordStorage mStorage;

    private static String SEMANTIC_TABLE = "SemanticTable";

    private static String SEMANTIC_ID = "ID";

    private static final String SEMANTIC_LONGITUDE = "Longitude";

    private static final String SEMANTIC_LATITUDE = "Latitude";

    private static final String[] SEMANTIC_COLUMNS =
        new String[]{ SEMANTIC_ID,
                      SEMANTIC_LONGITUDE,
                      SEMANTIC_LATITUDE,
                      TimeStatsAggregator.WEEKEND,
                      TimeStatsAggregator.WEEKDAY,
                      TimeStatsAggregator.MORNING,
                      TimeStatsAggregator.NOON,
                      TimeStatsAggregator.AFTERNOON,
                      TimeStatsAggregator.EVENING,
                      TimeStatsAggregator.NIGHT,
                      TimeStatsAggregator.LATENIGHT };

    private static final int mFeatureValueStart = 3;
    private static final int mFeatureValueEnd = 10;

    public ClusterManager(Context context) {
        mStorage = new AggregatorRecordStorage(context, SEMANTIC_TABLE, SEMANTIC_COLUMNS);

        loadSemanticClusters();
    }

    public void addSample(Location location) {
        float bestClusterDistance = Float.MAX_VALUE;
        int bestClusterIndex = -1;
        long lastDuration;
        long currentTime = location.getTime() / 1000; // measure time in seconds

        if (mLastLocation != null) {
            // get the duration spent in the last location
            long duration = (location.getTime() - mLastLocation.getTime()) / 1000;
            mClusterDuration += duration;

            Log.v(TAG, "sample duration: " + duration +
                  ", number of clusters: " + mLocationClusters.size());

            // add the last location to cluster.
            // first find the cluster it belongs to.
            for (int i = 0; i < mLocationClusters.size(); ++i) {
                float distance = mLocationClusters.get(i).distanceToCenter(mLastLocation);
                Log.v(TAG, "clulster " + i + " is within " + distance + " meters");
                if (distance < bestClusterDistance) {
                    bestClusterDistance = distance;
                    bestClusterIndex = i;
                }
            }

            // add the location to the selected cluster
            if (bestClusterDistance < LOCATION_CLUSTER_RADIUS) {
                mLocationClusters.get(bestClusterIndex).addSample(mLastLocation, duration);
            } else {
                // if it is far away from all existing clusters, create a new cluster.
                LocationCluster cluster = new LocationCluster(mLastLocation, duration);
                // move the center of the new cluster if its covering region overlaps
                // with an existing cluster.
                if (bestClusterDistance < 2 * LOCATION_CLUSTER_RADIUS) {
                    Log.v(TAG, "move away activated");
                    cluster.moveAwayCluster(mLocationClusters.get(bestClusterIndex),
                            ((float) 2 * LOCATION_CLUSTER_RADIUS));
                }
                mLocationClusters.add(cluster);
            }
        } else {
            mTimeRef = currentTime;

            if (mLocationClusters.isEmpty()) {
                mClusterDuration = 0;
            }
        }

        long collectDuration = currentTime - mTimeRef;
        Log.e(TAG, "collect duration: " + collectDuration);
        if (collectDuration > CONSOLIDATE_INTERVAL) {
            // TODO : conslidation takes time. move this to a separate thread later.
            consolidateClusters(collectDuration);
            mTimeRef = currentTime;
        }
        mLastLocation = location;
    }

    private void consolidateClusters(long duration) {
        LocationCluster cluster;
        for (int i = mLocationClusters.size() - 1; i >= 0; --i) {
            cluster = mLocationClusters.get(i);
            cluster.consolidate(duration);

            // remove transit clusters
            if (!cluster.passThreshold(LOCATION_CLUSTER_THRESHOLD)) {
                mLocationClusters.remove(cluster);
            }
        }

        // merge clusters whose regions are overlapped. note that after merge
        // cluster center changes but cluster size remains unchanged.
        for (int i = mLocationClusters.size() - 1; i >= 0; --i) {
            cluster = mLocationClusters.get(i);
            for (int j = i - 1; j >= 0; --j) {
                float distance = mLocationClusters.get(j).distanceToCluster(cluster);
                if (distance < LOCATION_CLUSTER_RADIUS) {
                    mLocationClusters.get(j).absorbCluster(cluster);
                    mLocationClusters.remove(cluster);
                }
            }
        }

        // check if new semantic clusters are found
        if (findNewSemanticClusters() &&
            mClusterDuration < LOCATION_REFRESH_PERIOD) {
            saveSemanticClusters();
        }

        if (mClusterDuration >  LOCATION_REFRESH_PERIOD) {
            updateSemanticClusters();
            mClusterDuration = 0;
        }
    }

    private boolean findNewSemanticClusters() {
        // select candidate location clusters
        ArrayList<LocationCluster> candidates = new ArrayList<LocationCluster>();
        for (LocationCluster cluster : mLocationClusters) {
            if (!cluster.hasSemanticId() &&
                cluster.passThreshold(SEMANTIC_CLUSTER_THRESHOLD)) {
                candidates.add(cluster);
            }
        }

        // assign each candidate to a semantic cluster
        boolean foundNewClusters = false;
        for (LocationCluster candidate : candidates) {
            // find the closest semantic cluster
            float bestClusterDistance = Float.MAX_VALUE;
            SemanticCluster bestCluster = null;
            for (SemanticCluster cluster : mSemanticClusters) {
                float distance = cluster.distanceToCluster(candidate);
                Log.v(TAG, "distance to semantic cluster: " + cluster.getSemanticId());

                if (distance < bestClusterDistance) {
                    bestClusterDistance = distance;
                    bestCluster = cluster;
                }
            }

            // if candidate doesn't belong to any semantic cluster, create a new
            // semantic cluster
            if (bestClusterDistance > SEMANTIC_CLUSTER_RADIUS) {
                // if it is far away from all existing clusters, create a new cluster.
                bestCluster = new SemanticCluster(candidate, mSemanticClusterCount++);
                String id = bestCluster.getSemanticId();

                // Add new semantic cluster to the list.
                mSemanticClusters.add(bestCluster);

                foundNewClusters = true;
            }
            candidate.setSemanticId(bestCluster.getSemanticId());
        }
        candidates.clear();
        return foundNewClusters;
    }

    private void updateSemanticClusters() {
        synchronized (mSemanticClusters) {
            // initialize semanticMap
            HashMap<String, ArrayList<BaseCluster> > semanticMap =
                new HashMap<String, ArrayList<BaseCluster> >();
            for (SemanticCluster cluster : mSemanticClusters) {
                String semanticId = cluster.getSemanticId();
                semanticMap.put(cluster.getSemanticId(), new ArrayList<BaseCluster>());
                semanticMap.get(semanticId).add(cluster);
            }

            // assign each candidate to a semantic cluster
            for (LocationCluster cluster : mLocationClusters) {
                if (cluster.hasSemanticId()) {
                    semanticMap.get(cluster.getSemanticId()).add(cluster);
                }
            }
            // reset location clusters.
            mLocationClusters.clear();
            mLastLocation = null;
            mTimeRef = 0;

            // use candidates semantic cluster
            BaseCluster newCluster = new BaseCluster();
            for (ArrayList<BaseCluster> clusterList : semanticMap.values()) {
                SemanticCluster semanticCluster = (SemanticCluster) clusterList.get(0);

                if (clusterList.size() > 1) {
                    newCluster.setCluster(clusterList.get(1));
                    for (int i = 2; i < clusterList.size(); i++) {
                      newCluster.absorbCluster(clusterList.get(i));
                    }
                    semanticCluster.absorbCluster(newCluster);
                } else {
                    // cluster with no new candidate
                    Log.e(TAG, "semantic cluster with no new location clusters: " +
                          semanticCluster);
                }
            }
        }
        saveSemanticClusters();
    }

    private void loadSemanticClusters() {
        List<Map<String, String> > allData = mStorage.getAllData();
        HashMap<String, Long> histogram = new HashMap<String, Long>();

        synchronized (mSemanticClusters) {
            mSemanticClusters.clear();
            for (Map<String, String> map : allData) {
                String semanticId = map.get(SEMANTIC_ID);
                double longitude = Double.valueOf(map.get(SEMANTIC_LONGITUDE));
                double latitude = Double.valueOf(map.get(SEMANTIC_LATITUDE));
                SemanticCluster cluster =
                    new SemanticCluster(semanticId, longitude, latitude);

                histogram.clear();
                for (int i = mFeatureValueStart; i <= mFeatureValueEnd; i++) {
                    String featureValue = SEMANTIC_COLUMNS[i];
                    if (map.containsKey(featureValue)) {
                      histogram.put(featureValue, Long.valueOf(map.get(featureValue)));
                    }
                }
                cluster.setHistogram(histogram);
                mSemanticClusters.add(cluster);
            }
            mSemanticClusterCount = mSemanticClusters.size();
            Log.e(TAG, "load " + mSemanticClusterCount + " semantic clusters.");
        }
    }

    private void saveSemanticClusters() {
        HashMap<String, String> rowFeatures = new HashMap<String, String>();

        mStorage.removeAllData();
        synchronized (mSemanticClusters) {
            for (SemanticCluster cluster : mSemanticClusters) {
                rowFeatures.clear();
                rowFeatures.put(SEMANTIC_ID, cluster.getSemanticId());

                rowFeatures.put(SEMANTIC_LONGITUDE,
                            String.valueOf(cluster.getCenterLongitude()));
                rowFeatures.put(SEMANTIC_LATITUDE,
                            String.valueOf(cluster.getCenterLatitude()));

                HashMap<String, Long> histogram = cluster.getHistogram();
                for (Map.Entry<String, Long> entry : histogram.entrySet()) {
                    rowFeatures.put(entry.getKey(), String.valueOf(entry.getValue()));
                }
                mStorage.addData(rowFeatures);
                Log.e(TAG, "saving semantic cluster: " + rowFeatures);
            }
        }
    }

    public String getSemanticLocation() {
        String label = LocationStatsAggregator.UNKNOWN_LOCATION;

        // instead of using the last location, try acquiring the latest location.
        if (mLastLocation != null) {
            // TODO: use fast neatest neighbor search speed up location search
            synchronized (mSemanticClusters) {
                for (SemanticCluster cluster: mSemanticClusters) {
                    if (cluster.distanceToCenter(mLastLocation) < SEMANTIC_CLUSTER_RADIUS) {
                        return cluster.getSemanticId();
                    }
                }
            }
        }
        return label;
    }

    public List<String> getClusterNames() {
        ArrayList<String> clusters = new ArrayList<String>();
        synchronized (mSemanticClusters) {
            for (SemanticCluster cluster: mSemanticClusters) {
                clusters.add(cluster.getSemanticId());
            }
        }
        return clusters;
    }
}
