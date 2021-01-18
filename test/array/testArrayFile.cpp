#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

#include <molpro/linalg/array/ArrayFile.h>

using molpro::linalg::array::ArrayFile;
using molpro::linalg::array::Span;
using ::testing::DoubleEq;
using ::testing::Each;
using ::testing::Pointwise;

TEST(ArrayFile, constructor_size) {
  auto a = ArrayFile(100);
  EXPECT_EQ(a.size(), 100);
}

TEST(ArrayFile, fill_get) {
  const size_t size = 10;
  const double value = 3.;
  auto a = ArrayFile(size);
  a.fill(value);
  auto buffer = std::vector<double>(size);
  a.get(0, size, Span<double>(&buffer[0], size));
  ASSERT_THAT(buffer, Each(DoubleEq(value)));
}

TEST(ArrayFile, fill_put) {
  const size_t size = 10;
  const double value = 3.;
  auto a = ArrayFile(size);
  a.fill(value);
  auto buffer_put = std::vector<double>(size);
  std::iota(buffer_put.begin(), buffer_put.end(), 0.);
  a.put(0, buffer_put.size(), Span<double>(&buffer_put[0], size));
  auto buffer_get = std::vector<double>(size);
  a.get(0, buffer_get.size(), Span<double>(&buffer_get[0], size));
  ASSERT_THAT(buffer_get, Pointwise(DoubleEq(), buffer_put));
}
