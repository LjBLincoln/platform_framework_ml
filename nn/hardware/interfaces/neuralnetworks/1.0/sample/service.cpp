#define LOG_TAG "android.hardware.neuralnetworks@1.0-service-sample"

#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <hidl/LegacySupport.h>

// Generated HIDL files
using android::hardware::defaultPassthroughServiceImplementation;
using android::hardware::neuralnetworks::V1_0::IDevice;

int main() {
    return defaultPassthroughServiceImplementation<IDevice>("sample");
}
