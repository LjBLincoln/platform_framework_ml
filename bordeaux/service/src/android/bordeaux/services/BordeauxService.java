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

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.RemoteException;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.os.Process;
import android.os.RemoteCallbackList;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.bordeaux.R;
import android.bordeaux.learning.MulticlassPA;
import android.bordeaux.learning.StochasticLinearRanker;
import android.util.Log;
import java.util.List;
import java.util.ArrayList;
import java.io.*;
import java.util.Scanner;
import java.util.HashMap;

import android.content.pm.PackageManager;

/**
 * Machine Learning service that runs in a remote process.
 * The application doesn't use this class directly.
 *
 */
public class BordeauxService extends Service {
    private final String TAG = "BordeauxService";
    /**
     * This is a list of callbacks that have been registered with the
     * service.
     * It's a place holder for future communications with all registered
     * clients.
     */
    final RemoteCallbackList<IBordeauxServiceCallback> mCallbacks =
            new RemoteCallbackList<IBordeauxServiceCallback>();

    int mValue = 0;
    NotificationManager mNotificationManager;

    MulticlassPA mMulticlassPA_Learner = null;

    // All saved learning session data
    // TODO: backup to the storage
    HashMap<String, IBinder> mMulticlassPA_sessions = new HashMap<String, IBinder>();
    HashMap<String, IBinder> mStochasticLinearRanker_sessions = new HashMap<String, IBinder>();

    @Override
    public void onCreate() {
        mNotificationManager = (NotificationManager)getSystemService(NOTIFICATION_SERVICE);

        // Display a notification about us starting.
        // TODO: don't display the notification after the service is
        // automatically started by the system, currently it's useful for
        // debugging.
        showNotification();
    }

    @Override
    public void onDestroy() {
        // Cancel the persistent notification.
        mNotificationManager.cancel(R.string.remote_service_started);

        // Tell the user we stopped.
        Toast.makeText(this, R.string.remote_service_stopped, Toast.LENGTH_SHORT).show();

        // Unregister all callbacks.
        mCallbacks.kill();
    }

    @Override
    public IBinder onBind(Intent intent) {
        // Return the requested interface.
        if (IBordeauxService.class.getName().equals(intent.getAction())) {
            return mBinder;
        }
        return null;
    }

    // The main interface implemented by the service.
    private final IBordeauxService.Stub mBinder = new IBordeauxService.Stub() {
        public IBinder getClassifier(String name) {
            PackageManager pm = getPackageManager();
            String uidname = pm.getNameForUid(getCallingUid());
            Log.i(TAG,"Name for uid: " + uidname);
            // internal unique key that identifies the learning instance.
            // Composed by the unique id of the package plus the user requested
            // name.
            String key = name + "_MulticlassPA_" + getCallingUid();
            Log.i(TAG, "request classifier session: " + key);
            if (mMulticlassPA_sessions.containsKey(key)) {
                return mMulticlassPA_sessions.get(key);
            }
            IBinder classifier = new Learning_MulticlassPA();
            mMulticlassPA_sessions.put(key, classifier);
            Log.i(TAG, "create a new classifier session: " + key);
            return classifier;
        }

        public IBinder getRanker(String name) {
            // internal unique key that identifies the learning instance.
            // Composed by the unique id of the package plus the user requested
            // name.
            String key = name + "_Ranker_" + getCallingUid();
            Log.i(TAG, "request ranker session: " + key);
            if (mStochasticLinearRanker_sessions.containsKey(key)) {
                return mStochasticLinearRanker_sessions.get(key);
            }
            IBinder ranker = new Learning_StochasticLinearRanker(BordeauxService.this);
            mStochasticLinearRanker_sessions.put(key, ranker);
            Log.i(TAG, "create a new ranker session: " + key);
            return ranker;
        }

        public void registerCallback(IBordeauxServiceCallback cb) {
            if (cb != null) mCallbacks.register(cb);
        }

        public void unregisterCallback(IBordeauxServiceCallback cb) {
            if (cb != null) mCallbacks.unregister(cb);
        }
    };

    /**
     * A MulticlassPA learning interface.
     */
    private final ILearning_MulticlassPA.Stub mMulticlassPABinder = new Learning_MulticlassPA();
    /**
     * StochasticLinearRanker interface
     */
    private final Learning_StochasticLinearRanker mStochasticLinearRankerBinder = new
            Learning_StochasticLinearRanker(this);

    @Override
    public void onTaskRemoved(Intent rootIntent) {
        Toast.makeText(this, "Task removed: " + rootIntent, Toast.LENGTH_LONG).show();
    }

    /**
     * Show a notification while this service is running.
     * TODO: remove the code after production (when service is loaded
     * automatically by the system).
     */
    private void showNotification() {
        // In this sample, we'll use the same text for the ticker and the expanded notification
        CharSequence text = getText(R.string.remote_service_started);

        // The PendingIntent to launch our activity if the user selects this notification
        PendingIntent contentIntent =
                PendingIntent.getActivity(this, 0,
                                          new Intent("android.bordeaux.DEBUG_CONTROLLER"), 0);

       // // Set the info for the views that show in the notification panel.

        Notification.Builder builder = new Notification.Builder(this);
        builder.setSmallIcon(R.drawable.ic_bordeaux);
        builder.setWhen(System.currentTimeMillis());
        builder.setTicker(text);
        builder.setContentTitle(text);
        builder.setContentIntent(contentIntent);
        Notification notification = builder.getNotification();
        // Send the notification.
        // We use a string id because it is a unique number.  We use it later to cancel.
        mNotificationManager.notify(R.string.remote_service_started, notification);
    }

}
