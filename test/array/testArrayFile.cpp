#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/ArrayFile.h>

using molpro::linalg::array::ArrayFile;

TEST(ArrayFile, constructor_size) {
  auto a = ArrayFile(100);
  EXPECT_EQ(a.size(), 100);
}

// FIXME more rigorous testing
