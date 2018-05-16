/*
 * Copyright (C) 2018 The Android Open Source Project
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

package com.example.android.nn.benchmark;

import android.os.Parcel;
import android.os.Parcelable;

public class BenchmarkResult implements Parcelable {
    public float mTotalTimeSec;
    public float mTotalError;
    public int mIterations;
    public float mTimeStdDeviation;
    public String mTestInfo;

    public BenchmarkResult(float totalTimeSec, int iterations, float timeVarianceSec,
            float totalError, String testInfo) {
        mTotalTimeSec = totalTimeSec;
        mTotalError = totalError;
        mIterations = iterations;
        mTimeStdDeviation = timeVarianceSec;
        mTestInfo = testInfo;
    }

    protected BenchmarkResult(Parcel in) {
        mTotalTimeSec = in.readFloat();
        mTotalError = in.readFloat();
        mIterations = in.readInt();
        mTimeStdDeviation = in.readFloat();
        mTestInfo = in.readString();
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeFloat(mTotalTimeSec);
        dest.writeFloat(mTotalError);
        dest.writeInt(mIterations);
        dest.writeFloat(mTimeStdDeviation);
        dest.writeString(mTestInfo);
    }

    @SuppressWarnings("unused")
    public static final Parcelable.Creator<BenchmarkResult> CREATOR =
            new Parcelable.Creator<BenchmarkResult>() {
                @Override
                public BenchmarkResult createFromParcel(Parcel in) {
                    return new BenchmarkResult(in);
                }

                @Override
                public BenchmarkResult[] newArray(int size) {
                    return new BenchmarkResult[size];
                }
            };

    public float getMeanTimeSec() {
        return mTotalTimeSec / mIterations;
    }

    @Override
    public String toString() {
        return "BenchmarkResult{" +
                "mTestInfo='" + mTestInfo + '\'' +
                ", getMeanTimeSec()=" + getMeanTimeSec() +
                ", mTotalTimeSec=" + mTotalTimeSec +
                ", mTotalError=" + mTotalError +
                ", mIterations=" + mIterations +
                ", mTimeStdDeviation=" + mTimeStdDeviation +
                '}';
    }
}

