Copyright 2017 The Android Open Source Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------

This directory contains files for the Android Neural Networks API.

The design document for the NN API can be found at
https://goto.google.com/android-nnapi-design-doc

CONTENTS OF THIS DIRECTORY

./runtime: Implementation of the NN API runtime.
           Includes source code and internal header files.
./runtime/include: The header files that an external developer would use.
                   These will be packaged with the NDK.  Includes a
                   C++ wrapper around the C API to make it easier to use.
./runtime/doc: Documentation.
./runtime/test: Test files.

./sample_driver: Sample driver that uses the CPU to execute queries.
                 NOT TO BE SHIPPED.  Only to be used as a testing and learning tool.

./common: Contains files that can be useful for the multiple components (runtime/driver/api)
          Includes source code and internal header files.
./common/include: The files that can be included by the components.
./common/test: Test files.

./java_api: In future versions, this will include the Java version of the NN API.
./java_api/android/ml/nn: The Java files.
./java_api/jni/: The JNI files that interface to the C/C++ API.

./tools: Tools used to develop the API, i.e. not external developer tools
./tools/benchmark: To test performance.

./models: Contains definition and tests for the baseline models.

./hardware/interfaces/neuralnetworks/1.0: Definition of the HAL and VTS tests.
    TODO: This location is temporary.  Needs to be moved to
    /hardware/interfaces/neuralnetworks/1.0 when we have tests for all the
    HAL entry points. This is required by the Treble team.
./hardware/interfaces/neuralnetworks/1.0/vts: The VTS tests.

RELATED DIRECTORIES

/test/vts-testcase/hal/ml/nn: Configures the VTS tests

/cts/tests/tests/ml/nn: The CTS tests
