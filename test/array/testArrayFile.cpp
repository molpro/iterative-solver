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

TEST(ArrayFile, scal) {
  auto a = ArrayFile(10, 3);
  const auto value = 2.;
  const auto scal = 3.;
  a.fill(value);
  a.scal(scal);
  auto buffer = a.get(0, a.size());
  ASSERT_THAT(buffer, Each(DoubleEq(value * scal)));
}

TEST(ArrayFile, axpy) {
  size_t const size = 11;
  size_t const block_size = 3;
  double const value_x = 2.;
  double const value_y = 1.;
  double const alpha = 3.;
  auto x = ArrayFile(size, block_size);
  auto y = ArrayFile(size, block_size);
  x.fill(value_x);
  y.fill(value_y);
  y.axpy(alpha, x);
  auto buffer_y = y.get(0, size);
  ASSERT_THAT(buffer_y, Each(DoubleEq(alpha * value_x + value_y)));
}

TEST(ArrayFile, axpy_vector_and_span) {
  size_t const size = 11;
  size_t const block_size = 3;
  double const value_x = 2.;
  double const value_y = 1.;
  double const alpha = 3.;
  auto x = std::vector<double>(size);
  auto y = ArrayFile(size, block_size);
  std::fill(x.begin(), x.end(), value_x);
  y.fill(value_y);
  y.axpy(alpha, x);
  auto buffer_y = y.get(0, size);
  ASSERT_THAT(buffer_y, Each(DoubleEq(alpha * value_x + value_y)));
  y.fill(value_y);
  y.axpy(alpha, molpro::linalg::array::util::vector_to_span(x));
  ASSERT_THAT(buffer_y, Each(DoubleEq(alpha * value_x + value_y)));
}

TEST(ArrayFile, axpy_map) {
  size_t const size = 7;
  size_t const block_size = 3;
  double const value_x = 2.;
  double const value_y = 1.;
  double const alpha = 3.;
  double const axpy_result = value_x * alpha + value_y;
  auto const indices = std::vector<size_t>{0, 3, 4, 6};
  auto x = std::map<size_t, double>{};
  for (auto i : indices)
    x[i] = value_x;
  auto ref_buffer = std::vector<double>{};
  ref_buffer.assign(size, value_y);
  for (auto i : indices)
    ref_buffer[i] = axpy_result;
  auto y = ArrayFile(size, block_size);
  y.fill(value_y);
  y.axpy(alpha, x);
  auto result = y.get(0, size);
  ASSERT_THAT(result, Pointwise(DoubleEq(), ref_buffer));
}

TEST(ArrayFile, dot) {
  size_t const size = 7;
  size_t const block_size = 3;
  double const value_x = 2.;
  double const value_y = 1.;
  auto x = ArrayFile(size, block_size);
  auto y = ArrayFile(size, block_size);
  x.fill(value_x);
  y.fill(value_y);
  auto const& xref = x;
  auto const& yref = y;
  auto result_x = xref.dot(yref);
  auto result_y = yref.dot(xref);
  ASSERT_EQ(result_x, result_y);
  ASSERT_EQ(result_x, value_x * value_y * size);
}

TEST(ArrayFile, dot_vector_and_span) {
  size_t const size = 7;
  size_t const block_size = 3;
  double const value_x = 2.;
  double const value_y = 1.;
  auto const x = std::vector<double>(size, value_x);
  auto y = ArrayFile(size, block_size);
  y.fill(value_y);
  auto const& yref = y;
  auto result = yref.dot(x);
  ASSERT_EQ(result, value_x * value_y * size);
}

TEST(ArrayFile, dot_map) {
  size_t const size = 7;
  size_t const block_size = 3;
  double const value_x = 2.;
  double const value_y = 1.;
  auto const indices = std::vector<size_t>{0, 3, 4, 6};
  auto x = std::map<size_t, double>{};
  for (auto i : indices)
    x[i] = value_x;
  auto ref_buffer = std::vector<double>{};
  ref_buffer.assign(size, value_y);
  auto y = ArrayFile(size, block_size);
  y.fill(value_y);
  auto const& yref = y;
  auto result = yref.dot(x);
  ASSERT_EQ(result, value_x * value_y * indices.size());
}
