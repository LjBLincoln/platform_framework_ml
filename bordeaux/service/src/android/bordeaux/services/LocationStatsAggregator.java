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
import android.location.Criteria;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.location.LocationProvider;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.os.Process;
import android.util.Log;
import java.util.HashMap;
import java.util.Map;

public class LocationStatsAggregator extends Aggregator {
    final String TAG = "LocationStatsAggregator";
    public static final String CURRENT_LOCATION = "Current Location";

    private static final long MINIMUM_TIME = 30000; // milliseconds
    private static final float MINIMUM_DISTANCE = 0f; // meter
    private static final int LOCATION_CHANGE = 1;

    private static final int BEST_PROVIDER_DURATION = 120000;

    private long mProviderSetTime;

    private final Criteria mCriteria = new Criteria();

    private Handler mHandler;
    private HandlerThread mHandlerThread;
    private LocationManager mLocationManager;
    private ClusterManager mClusterManager;

    public LocationStatsAggregator(final Context context) {

        Log.e(TAG, "initialize location manager");

        mLocationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
        setClusteringThread();

        requestLocationUpdate();
    }

    public String[] getListOfFeatures(){
        String [] list = new String[1];
        list[0] = CURRENT_LOCATION;
        return list;
    }

    public Map<String,String> getFeatureValue(String featureName) {
        HashMap<String,String> feature = new HashMap<String,String>();
        if (featureName.equals(CURRENT_LOCATION)) {
            feature.put(CURRENT_LOCATION, mClusterManager.getSemanticLocation());
        }
        return (Map) feature;
    }

    private void setClusteringThread() {
        mClusterManager = new ClusterManager();

        mHandlerThread = new HandlerThread("Location Handler",
                Process.THREAD_PRIORITY_BACKGROUND);
        mHandlerThread.start();
        mHandler = new Handler(mHandlerThread.getLooper()) {

            @Override
            public void handleMessage(Message msg) {
                if (!(msg.obj instanceof Location)) {
                    return;
                }
                Location location = (Location) msg.obj;
                switch(msg.what) {
                    case LOCATION_CHANGE:
                        mClusterManager.addSample(location);
                        break;
                    default:
                        super.handleMessage(msg);
                }
            }
        };
    }

    private void requestLocationUpdate() {
        Criteria criteria = new Criteria();
        criteria.setAccuracy(Criteria.ACCURACY_COARSE);
        criteria.setPowerRequirement(Criteria.POWER_LOW);
        criteria.setAltitudeRequired(false);
        criteria.setBearingRequired(false);
        criteria.setSpeedRequired(false);
        criteria.setCostAllowed(true);

        String bestProvider = mLocationManager.getBestProvider(criteria, true);
        Log.i(TAG, "Best Location Provider: " + bestProvider);

        mProviderSetTime = System.currentTimeMillis();
        if (bestProvider != null) {
            mLocationManager.requestLocationUpdates(
                bestProvider, MINIMUM_TIME, MINIMUM_DISTANCE, mLocationListener);
        }
    }

    private final LocationListener mLocationListener = new LocationListener() {
        public void onLocationChanged(Location location) {
            long currentTime = location.getTime();
            if (currentTime - mProviderSetTime < MINIMUM_TIME) {
                return;
            }
            mHandler.sendMessage(mHandler.obtainMessage(LOCATION_CHANGE, location));
            // search again for the location service
            if (currentTime - mProviderSetTime > BEST_PROVIDER_DURATION) {
                mLocationManager.removeUpdates(this);
                Log.e(TAG, "reselect best location provider");
                requestLocationUpdate();
            }
        }

        public void onStatusChanged(String provider, int status, Bundle extras) { }

        public void onProviderEnabled(String provider) { }

        public void onProviderDisabled(String provider) { }
    };
}
