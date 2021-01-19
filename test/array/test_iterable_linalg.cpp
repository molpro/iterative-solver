#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/util/iterable_lingalg.h>

using molpro::linalg::array::util::axpy;
using molpro::linalg::array::util::axpy_sparse;
using molpro::linalg::array::util::dot;
using molpro::linalg::array::util::dot_sparse;
using molpro::linalg::array::util::fill;
using molpro::linalg::array::util::scal;
using ::testing::DoubleEq;
using ::testing::Each;
using ::testing::Pointwise;

TEST(iterable_linalg, scal) {
  const size_t size = 5;
  const double value = 0.1;
  const double alpha = 2.0;
  auto x = std::vector<double>(size, value);
  scal(alpha, x);
  ASSERT_EQ(x.size(), size);
  ASSERT_THAT(x, Each(DoubleEq(alpha * value)));
}

TEST(iterable_linalg, fill) {
  const size_t size = 5;
  const double value = 0.1;
  const double alpha = 2.0;
  auto x = std::vector<double>(size, value);
  fill(alpha, x);
  ASSERT_EQ(x.size(), size);
  ASSERT_THAT(x, Each(DoubleEq(alpha)));
}

TEST(iterable_linalg, axpy) {
  const size_t size = 5;
  const double value_x = 0.1;
  const double value_y = 0.2;
  const double alpha = 2.0;
  const auto x = std::vector<double>(size, value_x);
  auto y = std::vector<double>(size, value_y);
  axpy(alpha, x, y);
  ASSERT_EQ(y.size(), size);
  ASSERT_THAT(y, Each(DoubleEq(value_y + alpha * value_x)));
}

TEST(iterable_linalg, axpy_sparse) {
  const size_t size = 5;
  const double value_x = 0.1;
  const double value_y = 0.2;
  const double alpha = 2.0;
  const auto x = std::map<size_t, double>{{1, value_x}, {3, value_x}, {4, value_x}};
  auto y = std::vector<double>(size, value_y);
  const auto axpy_value = value_y + alpha * value_x;
  auto reference_y = std::vector<double>{value_y, axpy_value, value_y, axpy_value, axpy_value};
  axpy_sparse(alpha, x, y);
  ASSERT_EQ(y.size(), size);
  ASSERT_THAT(y, Pointwise(DoubleEq(), reference_y));
}

TEST(iterable_linalg, dot) {
  const size_t size = 5;
  const double value_x = 0.1;
  const double value_y = 0.2;
  const auto x = std::vector<double>(size, value_x);
  const auto y = std::vector<double>(size, value_y);
  auto result = dot(x, y);
  ASSERT_DOUBLE_EQ(result, value_x * value_y * size);
}

TEST(iterable_linalg, dot_sparse) {
  const size_t size = 5;
  const double value_x = 0.1;
  const double value_y = 0.2;
  const auto x = std::vector<double>(size, value_x);
  const auto y = std::map<size_t, double>{{1, value_y}, {3, value_y}, {4, value_y}};
  auto result = dot_sparse(x, y);
  ASSERT_DOUBLE_EQ(result, value_x * value_y * y.size());
}
