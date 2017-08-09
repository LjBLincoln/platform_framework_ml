/*
 * Copyright (C) 2011 The Android Open Source Project
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

#define LOG_TAG "nnCache_test"
//#define LOG_NDEBUG 0

#include <gtest/gtest.h>

#include <utils/Log.h>

#include <android-base/test_utils.h>

#include "nnCache.h"

#include <memory>

namespace android {

class NNCacheTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        mCache = NNCache::get();
    }

    virtual void TearDown() {
        mCache->setCacheFilename("");
        mCache->terminate();
    }

    NNCache* mCache;
};

TEST_F(NNCacheTest, UninitializedCacheAlwaysMisses) {
    uint8_t buf[4] = { 0xee, 0xee, 0xee, 0xee };
    mCache->setBlob("abcd", 4, "efgh", 4);
    ASSERT_EQ(0, mCache->getBlob("abcd", 4, buf, 4));
    ASSERT_EQ(0xee, buf[0]);
    ASSERT_EQ(0xee, buf[1]);
    ASSERT_EQ(0xee, buf[2]);
    ASSERT_EQ(0xee, buf[3]);
}

TEST_F(NNCacheTest, InitializedCacheAlwaysHits) {
    uint8_t buf[4] = { 0xee, 0xee, 0xee, 0xee };
    mCache->initialize();
    mCache->setBlob("abcd", 4, "efgh", 4);
    ASSERT_EQ(4, mCache->getBlob("abcd", 4, buf, 4));
    ASSERT_EQ('e', buf[0]);
    ASSERT_EQ('f', buf[1]);
    ASSERT_EQ('g', buf[2]);
    ASSERT_EQ('h', buf[3]);
}

TEST_F(NNCacheTest, TerminatedCacheAlwaysMisses) {
    uint8_t buf[4] = { 0xee, 0xee, 0xee, 0xee };
    mCache->initialize();
    mCache->setBlob("abcd", 4, "efgh", 4);
    mCache->terminate();
    ASSERT_EQ(0, mCache->getBlob("abcd", 4, buf, 4));
    ASSERT_EQ(0xee, buf[0]);
    ASSERT_EQ(0xee, buf[1]);
    ASSERT_EQ(0xee, buf[2]);
    ASSERT_EQ(0xee, buf[3]);
}

class NNCacheSerializationTest : public NNCacheTest {

protected:

    virtual void SetUp() {
        NNCacheTest::SetUp();
        mTempFile.reset(new TemporaryFile());
    }

    virtual void TearDown() {
        mTempFile.reset(nullptr);
        NNCacheTest::TearDown();
    }

    std::unique_ptr<TemporaryFile> mTempFile;
};

TEST_F(NNCacheSerializationTest, ReinitializedCacheContainsValues) {
    uint8_t buf[4] = { 0xee, 0xee, 0xee, 0xee };
    mCache->setCacheFilename(&mTempFile->path[0]);
    mCache->initialize();
    mCache->setBlob("abcd", 4, "efgh", 4);
    mCache->terminate();
    mCache->initialize();
    ASSERT_EQ(4, mCache->getBlob("abcd", 4, buf, 4));
    ASSERT_EQ('e', buf[0]);
    ASSERT_EQ('f', buf[1]);
    ASSERT_EQ('g', buf[2]);
    ASSERT_EQ('h', buf[3]);
}

}
