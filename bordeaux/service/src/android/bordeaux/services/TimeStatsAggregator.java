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
    static final String THURSDAY = "Thursday";
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

        HashMap<String, String> features =
            getAllTimeFeatures(System.currentTimeMillis());
        if (features.containsKey(featureName)) {
            feature.put(featureName, features.get(featureName));
        } else {
            Log.e(TAG, "There is no Time feature called " + featureName);
        }
        return (Map)feature;
    }

    private static String getTimeOfDay(int hour) {
        if (hour >= 5 && hour < 11) {
            return MORNING;
        } else if (hour >= 11 && hour < 14) {
            return NOON;
        } else if (hour >= 14 && hour < 18) {
            return AFTERNOON;
        } else if (hour >= 18 && hour < 21) {
            return EVENING;
        } else if ((hour >= 21 && hour < 24) ||
                   (hour >= 0 && hour < 1))  {
            return NIGHT;
        } else {
            return LATENIGHT;
        }
    }

    private static String getDayOfWeek(int day) {
        switch (day) {
            case Time.SATURDAY:
                return SATURDAY;
            case Time.SUNDAY:
                return SUNDAY;
            case Time.MONDAY:
                return MONDAY;
            case Time.TUESDAY:
                return TUESDAY;
            case Time.WEDNESDAY:
                return WEDNESDAY;
            case Time.THURSDAY:
                return THURSDAY;
            default:
                return FRIDAY;
        }
    }

    private static String getPeriodOfDay(int hour) {
        if (hour > 6 && hour < 19) {
            return DAYTIME;
        } else {
            return NIGHTTIME;
        }
    }

    static HashMap<String, String> getAllTimeFeatures(long utcTime) {
        HashMap<String, String> features = new HashMap<String, String>();
        Time time = new Time();
        time.set(utcTime);

        features.put(DAY_OF_WEEK, getDayOfWeek(time.weekDay));
        features.put(PERIOD_OF_DAY, getPeriodOfDay(time.hour));
        features.put(TIME_OF_DAY, getTimeOfDay(time.hour));

        if (time.weekDay == Time.SUNDAY || time.weekDay == Time.SATURDAY ||
                (time.weekDay == Time.FRIDAY &&
                features.get(PERIOD_OF_DAY).equals(NIGHTTIME))) {
            features.put(TIME_OF_WEEK, WEEKEND);
        } else {
            features.put(TIME_OF_WEEK, WEEKDAY);
        }
        return features;
    }
}
