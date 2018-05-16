/*
 * Copyright (C) 2017 The Android Open Source Project
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


import android.app.Activity;
import android.os.Bundle;
import android.test.ActivityInstrumentationTestCase2;
import android.test.suitebuilder.annotation.MediumTest;
import android.util.Log;

import com.example.android.nn.benchmark.NNTestList.TestName;

import java.io.IOException;

/**
 * NNAPI benchmark test.
 * To run the test, please use command
 *
 * adb shell am instrument -w com.example.android.nn.benchmark/android.support.test.runner.AndroidJUnitRunner
 *
 */
public class NNTest extends ActivityInstrumentationTestCase2<NNBenchmark> {
    // Only run 1 iteration now to fit the MediumTest time requirement.
    // One iteration means running the tests continuous for 1s.
    private NNBenchmark mActivity;

    public NNTest() {
        super(NNBenchmark.class);
    }

    // Initialize the parameter for ImageProcessingActivityJB.
    protected void prepareTest() {
        mActivity = getActivity();
        mActivity.prepareInstrumentationTest();
    }

    @Override
    public void setUp() throws Exception {
        super.setUp();
        prepareTest();
        setActivityInitialTouchMode(false);
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
    }

    class TestAction implements Runnable {
        TestName mTestName;
        BenchmarkResult mResult;
        public TestAction(TestName testName) {
            mTestName = testName;
        }
        public void run() {
            try {
                mResult = mActivity.mProcessor.getInstrumentationResult(mTestName);
            } catch (IOException | BenchmarkException e) {
                e.printStackTrace();
            }
            Log.v(NNBenchmark.TAG,
                    "Benchmark for test \"" + mTestName.toString() + "\" is: " + mResult);
            synchronized(this) {
                this.notify();
            }
        }

        public BenchmarkResult getBenchmark() {
            return mResult;
        }
    }

    // Set the benchmark thread to run on ui thread
    // Synchronized the thread such that the test will wait for the benchmark thread to finish
    public void runOnUiThread(Runnable action) {
        synchronized(action) {
            mActivity.runOnUiThread(action);
            try {
                action.wait();
            } catch (InterruptedException e) {
                Log.v(NNBenchmark.TAG, "waiting for action running on UI thread is interrupted: " +
                        e.toString());
            }
        }
    }

    public void runTest(TestAction ta, String testName) {
        float sum = 0;
        runOnUiThread(ta);
        BenchmarkResult bmValue = ta.getBenchmark();

        // post result to INSTRUMENTATION_STATUS
        Bundle results = new Bundle();
        // Reported in ms
        results.putFloat(testName + "_avg", bmValue.getMeanTimeSec() * 1000.0f);
        getInstrumentation().sendStatus(Activity.RESULT_OK, results);
    }

    // Test case 0: MobileNet float32
    @MediumTest
    public void testMobileNetFloat() {
        TestAction ta = new TestAction(TestName.MobileNet_FLOAT);
        runTest(ta, TestName.MobileNet_FLOAT.name());
    }

    // Test case 1: MobileNet quantized
    @MediumTest
    public void testMobileNetQuantized() {
        TestAction ta = new TestAction(TestName.MobileNet_QUANT8);
        runTest(ta, TestName.MobileNet_QUANT8.name());
    }
}