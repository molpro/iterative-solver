#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

#include <molpro/linalg/array/ArrayFile.h>
#include <molpro/linalg/array/util/iterable_lingalg.h>

using molpro::linalg::array::ArrayFile;
using molpro::linalg::array::Span;
using molpro::linalg::array::util::vector_to_span;
using ::testing::DoubleEq;
using ::testing::Each;
using ::testing::Pointwise;

TEST(ArrayFile, constructor_size) {
  auto a = ArrayFile(100);
  EXPECT_EQ(a.size(), 100);
}

TEST(ArrayFile, constructor_0) {
  auto a = ArrayFile(0);
  EXPECT_EQ(a.size(), 0);
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

TEST(ArrayFile, add) {
  auto a = ArrayFile(10, 3);
  const auto value = 2.;
  const auto scal = 3.;
  a.fill(value);
  a.add(scal);
  auto buffer = a.get(0, a.size());
  ASSERT_THAT(buffer, Each(DoubleEq(value + scal)));
}

TEST(ArrayFile, block_size) {
  const size_t block_size = 3;
  auto a = ArrayFile(10, block_size);
  ASSERT_EQ(a.block_size(), block_size);
}

TEST(ArrayFile, times) {
  auto a = ArrayFile(10, 3);
  auto b = ArrayFile(a.size(), 3);
  const double value_x = 1.;
  const double value_y = 2.;
  a.fill(value_x);
  b.fill(value_y);
  a.times(b);
  auto buffer = a.get(0, a.size());
  ASSERT_THAT(buffer, Each(DoubleEq(value_x * value_y)));
}

TEST(ArrayFile, times_vector) {
  auto a = ArrayFile(10, 3);
  auto b = std::vector<double>(a.size());
  const double value_x = 1.;
  const double value_y = 2.;
  a.fill(value_x);
  molpro::linalg::array::util::fill(value_y, b);
  a.times(b);
  auto buffer = a.get(0, a.size());
  ASSERT_THAT(buffer, Each(DoubleEq(value_x * value_y)));
}

TEST(ArrayFile, times_xyz_vector) {
  auto a = ArrayFile(10, 3);
  auto b = std::vector<double>(a.size());
  auto c = std::vector<double>(a.size());
  const double value_x = 1.;
  const double value_y = 2.;
  a.fill(0.);
  molpro::linalg::array::util::fill(value_x, b);
  molpro::linalg::array::util::fill(value_y, c);
  a.times(b, c);
  auto buffer = a.get(0, a.size());
  ASSERT_THAT(buffer, Each(DoubleEq(value_x * value_y)));
}

TEST(ArrayFile, times_xyz) {
  auto a = ArrayFile(10, 3);
  auto b = ArrayFile(10, 3);
  auto c = ArrayFile(10, 3);
  const double value_x = 1.;
  const double value_y = 2.;
  a.fill(0.);
  b.fill(value_x);
  c.fill(value_y);
  a.times(b, c);
  auto buffer = a.get(0, a.size());
  ASSERT_THAT(buffer, Each(DoubleEq(value_x * value_y)));
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

class ArrayFileSelectMaxDotF : public ::testing::Test {
public:
  ArrayFileSelectMaxDotF() : x(size, block_size) { x.put(0, size, vector_to_span(x_values)); }

  std::map<size_t, double> max_dot(size_t n) {
    auto selection = std::map<size_t, double>{};
    for (size_t i = 0; i < n; ++i)
      selection.emplace(dot[i]);
    return selection;
  }

  std::vector<double> x_values{0, -1, 2, 1, -3, 3, -2};
  std::vector<double> y_values{9, 1, 2, 1, 3, 3, 2};
  std::vector<std::pair<size_t, double>> dot{{4, 9}, {5, 9}, {2, 4}, {6, 4}, {1, 1}, {3, 1}, {0, 0}};
  size_t size{x_values.size()};
  size_t block_size{3};
  ArrayFile x;
};

TEST_F(ArrayFileSelectMaxDotF, ArrayFile) {
  auto y = ArrayFile(size, block_size);
  y.put(0, size, vector_to_span(y_values));
  for (size_t n = 0; n < size; n += 2) {
    auto ref_selection = max_dot(n);
    auto selection = x.select_max_dot(n, y);
    ASSERT_THAT(selection, Pointwise(::testing::Eq(), ref_selection));
  }
}

TEST_F(ArrayFileSelectMaxDotF, Span) {
  const auto y = vector_to_span(y_values);
  for (size_t n = 0; n < size; n += 2) {
    auto ref_selection = max_dot(n);
    auto selection = x.select_max_dot(n, y);
    ASSERT_THAT(selection, Pointwise(::testing::Eq(), ref_selection));
  }
}

TEST_F(ArrayFileSelectMaxDotF, vector) {
  const auto y = y_values;
  for (size_t n = 0; n < size; n += 2) {
    auto ref_selection = max_dot(n);
    auto selection = x.select_max_dot(n, y);
    ASSERT_THAT(selection, Pointwise(::testing::Eq(), ref_selection));
  }
}

TEST_F(ArrayFileSelectMaxDotF, map) {
  auto y = std::map<size_t, double>{};
  for (size_t i = 0; i < size; ++i)
    y.emplace(i, y_values[i]);
  for (size_t n = 0; n < size; n += 2) {
    auto ref_selection = max_dot(n);
    auto selection = x.select_max_dot(n, y);
    ASSERT_THAT(selection, Pointwise(::testing::Eq(), ref_selection));
  }
}
