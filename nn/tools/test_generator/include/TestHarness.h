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

#ifndef ANDROID_ML_NN_TOOLS_TEST_GENERATOR_TEST_HARNESS_H
#define ANDROID_ML_NN_TOOLS_TEST_GENERATOR_TEST_HARNESS_H

#include <functional>
#include <map>
#include <tuple>
#include <vector>

namespace generated_tests {
typedef std::map<int, std::vector<float>> Float32Operands;
typedef std::map<int, std::vector<int32_t>> Int32Operands;
typedef std::map<int, std::vector<uint8_t>> Quant8Operands;
typedef std::tuple<Float32Operands,  // ANEURALNETWORKS_TENSOR_FLOAT32
                   Int32Operands,    // ANEURALNETWORKS_TENSOR_INT32
                   Quant8Operands    // ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
                   >
    MixedTyped;
typedef std::pair<MixedTyped, MixedTyped> MixedTypedExampleType;

// Helper template - go through a given type of input/output
template <typename T>
void for_each(MixedTyped &idx_and_data,
              std::function<void(int, std::vector<T> &)> execute_this) {
    for (auto &i : std::get<std::map<int, std::vector<T>>>(idx_and_data)) {
        execute_this(i.first, i.second);
    }
}

// Helper template - go through all index-value pairs
// expects a functor that takes (int index, void *raw data, size_t sz)
inline void for_all(MixedTyped &idx_and_data,
             std::function<void(int, void *, size_t)> execute_this) {
    #define FOR_EACH_TYPE(ty)                                           \
        for_each<ty>(idx_and_data, [&execute_this](int idx, auto &m) {  \
            execute_this(idx, (void *)m.data(), m.size() * sizeof(ty)); \
        });
        FOR_EACH_TYPE(float);
        FOR_EACH_TYPE(int32_t);
        FOR_EACH_TYPE(uint8_t);
    #undef FOR_EACH_TYPE
}

// Const variants of the helper
// Helper template - go through a given type of input/output
template <typename T>
void for_each(const MixedTyped &idx_and_data,
              std::function<void(int, const std::vector<T> &)> execute_this) {
    for (auto &i : std::get<std::map<int, std::vector<T>>>(idx_and_data)) {
        execute_this(i.first, i.second);
    }
}

// Const variants of the helper
// Helper template - go through all index-value pairs
// expects a functor that takes (int index, void *raw data, size_t sz)
inline void for_all(const MixedTyped &idx_and_data,
             std::function<void(int, const void *, size_t)> execute_this) {
    #define FOR_EACH_TYPE(ty)                                           \
        for_each<ty>(idx_and_data, [&execute_this](int idx, auto &m) {  \
            execute_this(idx, (const void *)m.data(),                   \
                         m.size() * sizeof(ty));                        \
        });
        FOR_EACH_TYPE(float);
        FOR_EACH_TYPE(int32_t);
        FOR_EACH_TYPE(uint8_t);
    #undef FOR_EACH_TYPE
}

// Helper template - resize test output per golden
template <typename ty>
void resize_accordingly(const MixedTyped &golden, MixedTyped &test) {
    for_each<ty>(golden, [&test](int index, auto &m) {
        auto &t = std::get<std::map<int, std::vector<ty>>>(test);
        t[index].resize(m.size());
    });
}

template <typename ty>
void filter(const MixedTyped &golden, MixedTyped *filtered,
                  std::function<bool(int)> is_ignored) {
    for_each<ty>(golden, [filtered, &is_ignored](int index, auto &m) {
        auto &g = std::get<std::map<int, std::vector<ty>>>(*filtered);
        if (!is_ignored(index))
            g[index] = m;
    });
}
};      // namespace generated_tests
#endif  // ANDROID_ML_NN_TOOLS_TEST_GENERATOR_TEST_HARNESS_H
