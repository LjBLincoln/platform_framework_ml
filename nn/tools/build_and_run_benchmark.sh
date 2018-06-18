#!/bin/bash
#
# Build benchmark app and run it, mimicking a user-initiated run
#
# Output is logged to $HOME/nnapi-logs/perf-cpu.txt and can be
# parsed by tools/parse_bm.py
#

if [[ -z "$ANDROID_BUILD_TOP" ]]; then
  echo ANDROID_BUILD_TOP not set, bailing out
  echo you must run lunch before running this script
  exit 1
fi

set -e
cd $ANDROID_BUILD_TOP

LOGDIR=$(mktemp -d)/nnapi-logs
mkdir -p $LOGDIR
echo creating logs in $LOGDIR

adb -d root

# Skip setup wizard and remount (read-write)
if ! adb -d shell test -f /data/local.prop; then
  adb -d shell 'echo ro.setupwizard.mode=DISABLED > /data/local.prop'
  adb -d shell 'chmod 644 /data/local.prop'
  adb -d shell 'settings put global device_provisioned 1*'
  adb -d shell 'settings put secure user_setup_complete 1'
  adb -d disable-verity
  adb -d reboot
  adb wait-for-usb-device remount
fi

# Build and install NNAPI runtime shared library
make libneuralnetworks
adb -d shell stop
adb -d sync
adb -d shell start

# Build and install benchmark app
make NeuralNetworksApiBenchmark
adb -d install $OUT/data/app/NeuralNetworksApiBenchmark/NeuralNetworksApiBenchmark.apk

# Enable menu key press through adb
adb -d shell 'echo testing > /data/local/enable_menu_key'
# Leave screen on (affects scheduling)
adb -d shell settings put system screen_off_timeout 86400000
# Stop background apps, seem to take ~10% CPU otherwise
adb -d shell 'pm disable com.google.android.googlequicksearchbox'
adb -d shell 'pm disable com.breel.wallpapers'

LOGFILE=$LOGDIR/perf-cpu.txt
echo "CPU only" | tee $LOGFILE
adb -d shell setprop debug.nn.cpuonly 1
for i in $(seq 0 9); do
  echo "Run $((i+1)) / 10" | tee -a $LOGFILE
  # Menukey - make sure screen is on
  adb shell "input keyevent 82"
  # Set the shell pid as a top-app and run tests
  adb shell 'echo $$ > /dev/stune/top-app/tasks; am instrument -w -e size large com.example.android.nn.benchmark/android.support.test.runner.AndroidJUnitRunner' | tee -a $LOGFILE
  sleep 10  # let CPU cool down
done

cd $ANDROID_BUILD_TOP/frameworks/ml/nn
./tools/parse_benchmark.py $LOGFILE
echo
echo full log of CPU execution in $LOGFILE
echo

# TODO(mikie): test with driver
#echo "Driver allowed"
#adb -d shell setprop debug.nn.cpuonly 0
#for i in 0 1 2 3 4; do
#  adb shell am instrument -w com.example.android.nn.benchmark/android.support.test.runner.AndroidJUnitRunner
#done
