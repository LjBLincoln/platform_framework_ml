#define LOG_TAG "android.hardware.neuralnetworks@1.0-impl-sample"

#include "SampleDriver.h"

#include <android-base/logging.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {

const std::string kDescriptor = "sample";

extern "C" IDevice* HIDL_FETCH_IDevice(const char* name);

IDevice* HIDL_FETCH_IDevice(const char* name) {
    LOG(DEBUG) << "HIDL_FETCH_IDevice";
    return kDescriptor == name ? new nn::sample_driver::SampleDriver() : nullptr;
}

} // namespace implementation
} // namespace V1_0
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
