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

// TODO: add functionality to detect speed (use GPS) when needed
// withouth draining the battery quickly
public class LocationStatsAggregator extends Aggregator {
    final String TAG = "LocationStatsAggregator";
    public static final String CURRENT_LOCATION = "Current Location";
    public static final String CURRENT_SPEED = "Current Speed";
    public static final String UNKNOWN_LOCATION = "Unknown Location";

    // TODO: Collect location on every minute
    private static final long MINIMUM_TIME = 30000; // milliseconds

    // reset best location provider on every 5 minutes
    private static final int BEST_PROVIDER_DURATION = 300000;

    private static final float MINIMUM_DISTANCE = 0f; // meter

    private static final int LOCATION_CHANGE = 1;

    // record time when the location provider is set
    private long mProviderSetTime;

    private Handler mHandler;
    private HandlerThread mHandlerThread;
    private LocationManager mLocationManager;
    private ClusterManager mClusterManager;

    public LocationStatsAggregator(final Context context) {
        mLocationManager =
            (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
        setClusteringThread(context);
        requestLocationUpdate();
    }

    public String[] getListOfFeatures(){
        String[] list = { CURRENT_LOCATION } ;
        return list;
    }

    public Map<String,String> getFeatureValue(String featureName) {
        HashMap<String,String> feature = new HashMap<String,String>();

        if (featureName.equals(CURRENT_LOCATION)) {

          // TODO: check last known location first before sending out location request.
          /*
            Location location =
                mLocationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
          */

            String location = mClusterManager.getSemanticLocation();
            if (!location.equals(UNKNOWN_LOCATION)) {
                feature.put(CURRENT_LOCATION, location);
            }
        }
        return (Map) feature;
    }

    private void setClusteringThread(Context context) {
        mClusterManager = new ClusterManager(context);

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
        /*
        criteria.setAltitudeRequired(false);
        criteria.setBearingRequired(false);
        criteria.setSpeedRequired(true);
        */
        criteria.setCostAllowed(true);

        String bestProvider = mLocationManager.getBestProvider(criteria, false);
        Log.i(TAG, "Best Location Provider: " + bestProvider);

        String bestAvailableProvider = mLocationManager.getBestProvider(criteria, true);
        Log.i(TAG, "Best Available Location Provider: " + bestAvailableProvider);

        mProviderSetTime = System.currentTimeMillis();
        if (bestAvailableProvider != null) {
            mLocationManager.requestLocationUpdates(
                bestAvailableProvider, MINIMUM_TIME, MINIMUM_DISTANCE, mLocationListener);
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
