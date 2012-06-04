package android.bordeaux.applauncherwidget;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import java.util.Comparator;
import java.util.ArrayList;
import java.io.*;
import android.app.PendingIntent;
import android.app.Service;
import android.app.AlarmManager;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.RemoteViews;
import android.widget.Button;
import android.widget.TextView;
import android.content.Intent;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.content.pm.ApplicationInfo;
import android.content.ComponentName;
import android.content.BroadcastReceiver;
import android.graphics.drawable.Drawable;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.IBinder;
import android.os.Bundle;
import android.os.RemoteException;
import android.os.ServiceManager;
import android.os.DropBoxManager;
import com.android.internal.os.PkgUsageStats;
import com.android.internal.app.IUsageStats;
// libraries for bordeaux service
import android.bordeaux.services.BordeauxPredictor;
import android.bordeaux.services.BordeauxAggregatorManager;
import android.bordeaux.services.BordeauxManagerService;
// libraries for using protobuffer
import com.google.common.io.protocol.ProtoBuf;
import com.google.common.io.protocol.ProtoBufType;
import com.google.protobuf.micro.*;
import android.bordeaux.applauncherwidget.proto.ActivityRecordProto;
import android.bordeaux.applauncherwidget.proto.ActivityRecordProto.ActivityRecordProtoBuff;

public class AppLauncherWidgetProvider extends AppWidgetProvider {
    String TAG = "AppLauncherWidgetProvider";
    final static int STAT_SAMPLING_TIME = 3000; // milliseconds
    final static int WIDGET_UPDATE_TIME = 8000; // milliseconds
    // These numbers are low for debugging purpose and should be changed for final version
    final static String PROTO_FILE_NAME = "ActivityLogProto";
    final static String PREDICTOR_NAME ="SmartAppLauncher";
    final static String CURRENT_TIME = "Current Time";
    final static String CURRENT_LOCATION = "Current Location";
    final static String CURRENT_MOTION = "Current Motion";
    final static String EXP_TIME_STRING = "120";

    static private PendingIntent sWidgetUpdateService = null;
    static private PendingIntent sGetStatService = null;
    static private AlarmManager sAlarmManager;
    static PkgUsageStats[] sStatsNew;
    static PkgUsageStats[] sStatsOld;
    static PackageManager sPackageManager;
    static IUsageStats sUsageStatsService;
    static BordeauxPredictor sPredictor;
    static BordeauxAggregatorManager sAggregatorManager;
    static volatile boolean sGetPredictor = false;
    static volatile boolean sSetPredictor = false;
    static volatile boolean sNewAppLaunched = false;
    static boolean sWidgetFirstRun;
    static boolean sWasScreenOn = true;

    public void onEnabled(Context context) {
        sWidgetFirstRun = true;
        sPackageManager = context.getPackageManager();
        sAlarmManager = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        sUsageStatsService = IUsageStats.Stub.asInterface(ServiceManager.getService("usagestats"));
        if (sUsageStatsService == null) {
            Log.e(TAG, "Failed to retrieve usagestats service");
            return;
        }
    }

    public void onDisabled(Context context) {
        sAlarmManager.cancel(sWidgetUpdateService);
        sAlarmManager.cancel(sGetStatService);
    }

    public void onUpdate(Context context, AppWidgetManager appWidgetManager, int[] appWidgetIds) {
        //Log.i(TAG, "onUpdate widget");
        setServices(context);
    }

    public static void setServices(Context context) {
        Intent i = new Intent(context, UpdateService.class);
        if (sWidgetUpdateService == null)
        {
            sWidgetUpdateService = PendingIntent.getService(context, 0, i,
                                                            PendingIntent.FLAG_CANCEL_CURRENT);
        }
        sAlarmManager.set(AlarmManager.RTC, System.currentTimeMillis(), sWidgetUpdateService);
        i = new Intent(context, GetStatService.class);
        if (sGetStatService == null)
        {
            sGetStatService = PendingIntent.getService(context, 1, i,
                                                       PendingIntent.FLAG_CANCEL_CURRENT);
        }
        sAlarmManager.set(AlarmManager.RTC, System.currentTimeMillis(), sGetStatService);
    }
    //broadcast reciever for screen on/off
    public static class ScreenReceiver extends BroadcastReceiver {
        String TAG = "ScreenReceiver";

        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(Intent.ACTION_SCREEN_OFF)) {
                if (sWasScreenOn) {
                    //Log.i(TAG,"Screen is off");
                    sAlarmManager.cancel(sWidgetUpdateService);
                    sAlarmManager.cancel(sGetStatService);
                    sWasScreenOn = false;
                }
            } else if (intent.getAction().equals(Intent.ACTION_SCREEN_ON)) {
                if (!sWasScreenOn) {
                    //Log.i(TAG,"Screen is on");
                    setServices(context);
                    sWasScreenOn = true;
                }
            }
        }
    }
    // Service for updating widget icons
    public static class UpdateService extends Service {
        String TAG = "updating Service";

        @Override
        public void onStart(Intent intent, int startId) {
            Log.i(TAG,"update widget layout ");
            //if ((mNewAppLaunched)|| (mWidgetFirstRun)) {
                RemoteViews updateViews = buildUpdate(this);
                ComponentName thisWidget = new ComponentName(this, AppLauncherWidgetProvider.class);
                AppWidgetManager manager = AppWidgetManager.getInstance(this);
                manager.updateAppWidget(thisWidget, updateViews);
                sNewAppLaunched = false;
                sWidgetFirstRun = false;
            //}
            sAlarmManager.set(AlarmManager.RTC, System.currentTimeMillis() + WIDGET_UPDATE_TIME,
                              sWidgetUpdateService);
        }

        @Override
        public IBinder onBind(Intent intent) {
            return null;
        }

        public RemoteViews buildUpdate(Context context) {
            final int APP_NUM = 4;
            ApplicationInfo[] appInfo = new ApplicationInfo[APP_NUM];
            PendingIntent[] pendingIntent = new PendingIntent[APP_NUM];
            CharSequence[] labels = new CharSequence[APP_NUM];
            Bitmap[] icons = new Bitmap[APP_NUM];
            String[] appList = new String [APP_NUM];
            RemoteViews  views = new RemoteViews(context.getPackageName(), R.layout.widget);
            // get the app list from the Bordeaux Service
            if (!sGetPredictor) {
                sAggregatorManager = new BordeauxAggregatorManager(this);
                sPredictor = new BordeauxPredictor(this, PREDICTOR_NAME);
                sGetPredictor = true;
            }

            if (!sPredictor.retrievePredictor()) {
                Log.i(TAG,"Predictor is not availible yet");
                appList[0] = "com.google.android.gm";
                appList[1] = "com.google.android.talk";
                appList[2] = "com.google.android.browser";
                appList[3] = "com.google.android.deskclock";
            } else {
                if (!sSetPredictor) {
                    sPredictor.setParameter("Set Feature", CURRENT_TIME);
                    sPredictor.setParameter("Set Feature", CURRENT_LOCATION);
                    sPredictor.setParameter("Set Feature", CURRENT_MOTION);
                    sPredictor.setParameter("SetExpireTime", EXP_TIME_STRING);
                    //sPredictor.setParameter("UseHistory", "true");
                    sSetPredictor = true;
                }
                appList = getSortedAppList(sPredictor, APP_NUM);
            }
            // Get intent, icon and label for each app
            for (int j = 0; j < APP_NUM; j++){
                try {
                    appInfo[j] = sPackageManager.getApplicationInfo(appList[j],
                                                                    sPackageManager.GET_META_DATA);
                    labels[j] = sPackageManager.getApplicationLabel(appInfo[j]);
                    Intent intent = sPackageManager.getLaunchIntentForPackage(appList[j]);
                    pendingIntent[j] = PendingIntent.getActivity(context, 0, intent, 0);
                    icons[j] = ((BitmapDrawable) sPackageManager
                            .getApplicationIcon(appInfo[j])).getBitmap();
                } catch (NameNotFoundException e ) {
                    Log.e(TAG,"package name is not found");
                }
            }
            // Set Application names
            views.setTextViewText(R.id.text_app0, labels[0]);
            views.setTextViewText(R.id.text_app1, labels[1]);
            views.setTextViewText(R.id.text_app2, labels[2]);
            views.setTextViewText(R.id.text_app3, labels[3]);
            // Set Application Icons
            views.setImageViewBitmap(R.id.button_app0, icons[0]);
            views.setImageViewBitmap(R.id.button_app1, icons[1]);
            views.setImageViewBitmap(R.id.button_app2, icons[2]);
            views.setImageViewBitmap(R.id.button_app3, icons[3]);
            // Set Application Intents
            views.setOnClickPendingIntent(R.id.button_app0, pendingIntent[0]);
            views.setOnClickPendingIntent(R.id.button_app1, pendingIntent[1]);
            views.setOnClickPendingIntent(R.id.button_app2, pendingIntent[2]);
            views.setOnClickPendingIntent(R.id.button_app3, pendingIntent[3]);
            return views;
        }

        private String[] getSortedAppList(BordeauxPredictor predictor, int n) {
            class TComp implements Comparator{
               public int compare(Object o1, Object o2) {
                    float f1 = ((Float)o1).floatValue();
                    float f2 = ((Float)o2).floatValue();
                    if (f1 < f2)
                        return +1;
                    return -1;
               }
            }
            TreeMap<Float, String> tMap = new TreeMap<Float, String>(new TComp());
            for (PackageInfo info : sPackageManager.getInstalledPackages(0)) {
                float f = predictor.getProbability(info.packageName);
                tMap.put( f, info.packageName);
            }
            String[] sortedlist = new String[n];
            ArrayList<String> sortedApps = new ArrayList<String>(tMap.values());
            int l = sortedApps.size();
            int j = 0;
            for (int i = 0; i < n ; i++) {
                while ((sPackageManager.getLaunchIntentForPackage(sortedApps.get(j))==null)&(j<l)){
                    j++;
                    if (j>= l)
                        break;
                }
                if (j>= l)
                    break;
                sortedlist[i] = sortedApps.get(j);
                j++;
            }
            return sortedlist;
        }
    }
    // Service for getting stats about application usage.
    public static class GetStatService extends Service {
        String TAG = "get stat Service";
        private class appRecord {
            String pkgName;
            int launchNum;
            float duration;
            public void set(appRecord r){
                this.pkgName = r.pkgName;
                this.launchNum = r.launchNum;
                this.duration = r.duration;
            }
        }

        public GetStatService() {
            super();
            try {
                sStatsOld = sUsageStatsService.getAllPkgUsageStats();
            } catch (RemoteException e) {
                Log.e(TAG, "Failed to get usage stats of applications");
            }
        }

        @Override
        public void onStart(Intent intent, int startId) {
            // registering broadcast reciever for screen on/off
            IntentFilter filter = new IntentFilter(Intent.ACTION_SCREEN_ON);
            filter.addAction(Intent.ACTION_SCREEN_OFF);
            BroadcastReceiver receiver = new ScreenReceiver();
            this.registerReceiver(receiver, filter);

            try {
                sStatsNew = sUsageStatsService.getAllPkgUsageStats();
            } catch (RemoteException e) {
                Log.e(TAG, "Failed to get usage stats of applications");
            }
            ArrayList<appRecord> newLaunchedApps = extractLaunchedApps(sStatsNew, sStatsOld);
            Log.i(TAG, "Serivce got the stats " + newLaunchedApps.size());
            if (!sGetPredictor) {
                sAggregatorManager = new BordeauxAggregatorManager(this);
                sPredictor = new BordeauxPredictor(this, PREDICTOR_NAME);
                sGetPredictor = true;
            }

            if ((newLaunchedApps.size() > 0 ) && sPredictor.retrievePredictor()) {
                sNewAppLaunched = true;
                if (!sSetPredictor) {
                    sPredictor.setParameter("Set Feature", CURRENT_TIME);
                    sPredictor.setParameter("Set Feature", CURRENT_LOCATION);
                    sPredictor.setParameter("Set Feature", CURRENT_MOTION);
                    sPredictor.setParameter("SetExpireTime", EXP_TIME_STRING);
                    //sPredictor.setParameter("UseHistory", "true");
                    sSetPredictor = true;
                }
                for (int i = 0 ; i < newLaunchedApps.size(); i ++ ) {
                    sPredictor.pushSample(newLaunchedApps.get(i).pkgName);
                }
                sStatsOld = (PkgUsageStats[]) sStatsNew.clone();
                //put activity logs in DropBox
                putLogsInDropBox(this, newLaunchedApps);
                // TODO Maybe Wait for sometime and then put the logs in DropBox
            }
            sAlarmManager.set(AlarmManager.RTC, System.currentTimeMillis() + STAT_SAMPLING_TIME,
                              sGetStatService);
        }

        private void putLogsInDropBox(Context con, ArrayList<appRecord> activityList) {
            // write new activities in protobuffer
            ActivityRecordProtoBuff activityRecordProto = new ActivityRecordProtoBuff();
            String feaNum = CURRENT_TIME;
            String currTime = sAggregatorManager.GetData(feaNum).get(feaNum);
            feaNum = CURRENT_LOCATION;
            String currLocation = sAggregatorManager.GetData(feaNum).get(feaNum);
            feaNum = CURRENT_MOTION;
            String currMotion = sAggregatorManager.GetData(feaNum).get(feaNum);
            for (int i = 0 ; i < activityList.size(); i ++ ) {
                ActivityRecordProtoBuff.activityInfo appInfo =
                        new ActivityRecordProtoBuff.activityInfo();
                appInfo.setPkgName(activityList.get(i).pkgName);
                appInfo.setTime(currTime);
                appInfo.setLocation(currLocation);
                appInfo.setMotion(currMotion);
                activityRecordProto.addActivityLog(appInfo);
            }
            // SEND TO DropBox
            DropBoxManager db = (DropBoxManager) con.getSystemService(Context.DROPBOX_SERVICE);
            try {
                File file = File.createTempFile(PROTO_FILE_NAME, "proto", con.getFilesDir());
                FileOutputStream fos = new FileOutputStream(file);
                CodedOutputStreamMicro cos = CodedOutputStreamMicro.newInstance(fos);
                activityRecordProto.writeTo(cos);
                cos.flush();
                fos.close();
                db.addFile(PROTO_FILE_NAME, file, 0);
            } catch (IOException e) {
                Log.e(TAG, "Couldn't write log file.", e);
            }
        }

        @Override
        public IBinder onBind(Intent intent) {
            return null;
        }

        private ArrayList<appRecord> extractLaunchedApps(PkgUsageStats[] statsNew,
                                                         PkgUsageStats[] statsOld) {
            ArrayList<appRecord> tmpRecord = new ArrayList<appRecord>();
            for (int i = 0; i < statsNew.length; i++) {
                if ( statsNew[i].launchCount > 0 ) {
                    boolean found = false;
                    String tmpName = statsNew[i].packageName;
                    if (sPackageManager.getLaunchIntentForPackage(tmpName) == null )
                        continue;
                    if (statsOld != null) {
                        for (int j =0; j < statsOld.length; j++) {
                            if  (tmpName.equals(statsOld[j].packageName)) {
                                if (statsNew[i].launchCount > statsOld[j].launchCount) {
                                    appRecord r = new appRecord();
                                    r.pkgName = tmpName;
                                    r.launchNum = statsNew[i].launchCount - statsOld[j].launchCount;
                                    r.duration = statsNew[i].usageTime - statsOld[j].usageTime;
                                    tmpRecord.add(r);
                                }
                                found = true;
                                break;
                            }
                        }
                    }
                    if (!found){
                        appRecord r = new appRecord();
                        r.pkgName = tmpName;
                        r.launchNum = statsNew[i].launchCount;
                        r.duration = statsNew[i].usageTime;
                        tmpRecord.add(r);
                    }
                }
            }
            return tmpRecord;
        }
    }
}
