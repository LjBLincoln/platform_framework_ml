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

import android.text.format.Time;
import android.util.Log;

import java.util.HashMap;
import java.util.Map;

// import java.util.Date;

// TODO: use build in functions in
// import android.text.format.Time;
public class TimeStatsAggregator extends Aggregator {
    final String TAG = "TimeStatsAggregator";

    public static final String TIME_OF_WEEK = "Time of Week";
    public static final String DAY_OF_WEEK = "Day of Week";
    public static final String TIME_OF_DAY = "Time of Day";
    public static final String PERIOD_OF_DAY = "Period of Day";

    static final String WEEKEND = "Weekend";
    static final String WEEKDAY = "Weekday";
    static final String MONDAY = "Monday";
    static final String TUESDAY = "Tuesday";
    static final String WEDNESDAY = "Wednesday";
    static final String THURSDAY = "Tuesday";
    static final String FRIDAY = "Friday";
    static final String SATURDAY = "Saturday";
    static final String SUNDAY = "Sunday";
    static final String MORNING = "Morning";
    static final String NOON = "Noon";
    static final String AFTERNOON = "AfterNoon";
    static final String EVENING = "Evening";
    static final String NIGHT = "Night";
    static final String LATENIGHT = "LateNight";
    static final String DAYTIME = "Daytime";
    static final String NIGHTTIME = "Nighttime";

    final Time mTime = new Time();
    final HashMap<String, String> mFeatures = new HashMap<String, String>();

    public String[] getListOfFeatures(){
        String [] list = new String[4];
        list[0] = TIME_OF_WEEK;
        list[1] = DAY_OF_WEEK;
        list[2] = TIME_OF_DAY;
        list[3] = PERIOD_OF_DAY;
        return list;
    }

    public Map<String,String> getFeatureValue(String featureName) {
        HashMap<String,String> feature = new HashMap<String,String>();

        updateFeatures();
        if (mFeatures.containsKey(featureName)) {
          feature.put(featureName, mFeatures.get(featureName));
        } else {
            Log.e(TAG, "There is no Time feature called " + featureName);
        }
        return (Map)feature;
    }

    private void updateFeatures() {
        mFeatures.clear();
        mTime.set(System.currentTimeMillis());

        switch (mTime.weekDay) {
            case Time.SATURDAY:
                mFeatures.put(DAY_OF_WEEK, SATURDAY);
                break;
            case Time.SUNDAY:
                mFeatures.put(DAY_OF_WEEK, SUNDAY);
                break;
            case Time.MONDAY:
                mFeatures.put(DAY_OF_WEEK, MONDAY);
                break;
            case Time.TUESDAY:
                mFeatures.put(DAY_OF_WEEK, TUESDAY);
                break;
            case Time.WEDNESDAY:
                mFeatures.put(DAY_OF_WEEK, WEDNESDAY);
                break;
            case Time.THURSDAY:
                mFeatures.put(DAY_OF_WEEK, THURSDAY);
                break;
            default:
                mFeatures.put(DAY_OF_WEEK, FRIDAY);
        }

        if (mTime.hour > 6 && mTime.hour < 19) {
            mFeatures.put(PERIOD_OF_DAY, DAYTIME);
        } else {
            mFeatures.put(PERIOD_OF_DAY, NIGHTTIME);
        }

        if (mTime.hour >= 5 && mTime.hour < 12) {
            mFeatures.put(TIME_OF_DAY, MORNING);
        } else if (mTime.hour >= 12 && mTime.hour < 14) {
            mFeatures.put(TIME_OF_DAY, NOON);
        } else if (mTime.hour >= 14 && mTime.hour < 18) {
            mFeatures.put(TIME_OF_DAY, AFTERNOON);
        } else if (mTime.hour >= 18 && mTime.hour < 22) {
            mFeatures.put(TIME_OF_DAY, EVENING);
        } else if ((mTime.hour >= 22 && mTime.hour < 24) ||
                   (mTime.hour >= 0 && mTime.hour < 1))  {
            mFeatures.put(TIME_OF_DAY, NIGHT);
        } else {
            mFeatures.put(TIME_OF_DAY, LATENIGHT);
        }

        if (mTime.weekDay == Time.SUNDAY || mTime.weekDay == Time.SATURDAY ||
                (mTime.weekDay == Time.FRIDAY &&
                mFeatures.get(PERIOD_OF_DAY).equals(NIGHTTIME))) {
            mFeatures.put(TIME_OF_WEEK, WEEKEND);
        } else {
            mFeatures.put(TIME_OF_WEEK, WEEKDAY);
        }
    }
}
