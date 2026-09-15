/*!
 * @file
 * @brief Tests for a complex scalar type.
 *
 * The library uses the hermitian inner product, so the subspace overlap and a hermitian operator are
 * hermitian matrices rather than complex symmetric ones.
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/ArrayHandlerIterable.h>
#include <molpro/linalg/array/ArrayHandlerIterableSparse.h>
#include <molpro/linalg/array/ArrayHandlerSparse.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/subspace/util.h>

#include <Eigen/Dense>

#include <complex>
#include <deque>
#include <map>
#include <vector>

using scalar = std::complex<double>;

//! The overlap matrix the array handlers build must be hermitian, not complex symmetric
TEST(helper_complex, inner_product_is_hermitian) {
  molpro::linalg::array::ArrayHandlerIterable<std::vector<scalar>, std::vector<scalar>> handler;
  const std::vector<scalar> x{{1, 2}, {3, -1}}, y{{0, 1}, {2, 5}};
  // <x|y> = conj(x).y
  const scalar expected = std::conj(x[0]) * y[0] + std::conj(x[1]) * y[1];
  EXPECT_NEAR(handler.dot(x, y).real(), expected.real(), 1e-14);
  EXPECT_NEAR(handler.dot(x, y).imag(), expected.imag(), 1e-14);
  // <y|x> is its conjugate, and <x|x> is real and positive
  EXPECT_NEAR(handler.dot(y, x).imag(), -handler.dot(x, y).imag(), 1e-14);
  EXPECT_NEAR(handler.dot(x, x).imag(), 0.0, 1e-14);
  EXPECT_GT(handler.dot(x, x).real(), 0.0);
}

TEST(helper_complex, overlap_matrix_is_hermitian) {
  molpro::linalg::array::ArrayHandlerIterable<std::vector<scalar>, std::vector<scalar>> handler;
  std::vector<std::vector<scalar>> params{{{1, 2}, {3, -1}}, {{0, 1}, {2, 5}}, {{-1, 0}, {1, 1}}};
  const auto s = molpro::linalg::itsolv::subspace::util::overlap(molpro::linalg::itsolv::cwrap(params), handler);
  for (size_t i = 0; i < s.rows(); ++i) {
    EXPECT_NEAR(s(i, i).imag(), 0.0, 1e-14) << "diagonal element " << i << " must be real";
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NEAR(s(i, j).real(), s(j, i).real(), 1e-14);
      EXPECT_NEAR(s(i, j).imag(), -s(j, i).imag(), 1e-14);
    }
  }
}


//! real_part and imaginary_part decompose a scalar, and are the identity and zero for a real type
TEST(helper_complex, real_and_imaginary_parts) {
  using molpro::linalg::imaginary_part;
  using molpro::linalg::real_part;
  const scalar z{1.5, -2.25};
  EXPECT_EQ(real_part(z), 1.5);
  EXPECT_EQ(imaginary_part(z), -2.25);
  EXPECT_EQ(real_part(3.25), 3.25);
  EXPECT_EQ(imaginary_part(3.25), 0.0);
  static_assert(std::is_same_v<decltype(imaginary_part(z)), double>);
  static_assert(std::is_same_v<decltype(imaginary_part(3.25)), double>);
}

//! Each array handler that implements dot itself must conjugate its left operand, not only the iterable one
TEST(helper_complex, sparse_inner_products_are_hermitian) {
  using sparse = std::map<size_t, scalar>;
  const sparse xs{{0, {1, 2}}, {2, {3, -1}}, {5, {0, 4}}};
  const sparse ys{{0, {0, 1}}, {2, {2, 5}}, {7, {9, 9}}}; // key 5 and key 7 do not overlap
  // only the shared keys 0 and 2 contribute
  const scalar expected = std::conj(xs.at(0)) * ys.at(0) + std::conj(xs.at(2)) * ys.at(2);

  molpro::linalg::array::ArrayHandlerSparse<sparse, sparse> sparse_handler;
  EXPECT_NEAR(std::abs(sparse_handler.dot(xs, ys) - expected), 0.0, 1e-14);
  EXPECT_NEAR(std::abs(sparse_handler.dot(ys, xs) - std::conj(expected)), 0.0, 1e-14);

  // iterable on the left, sparse on the right: the conjugation still falls on the left operand
  const std::vector<scalar> xv{{1, 2}, {5, 5}, {3, -1}};
  const sparse y_on_iterable{{0, {0, 1}}, {2, {2, 5}}};
  molpro::linalg::array::ArrayHandlerIterableSparse<std::vector<scalar>, sparse> mixed_handler;
  const scalar expected_mixed = std::conj(xv[0]) * y_on_iterable.at(0) + std::conj(xv[2]) * y_on_iterable.at(2);
  EXPECT_NEAR(std::abs(mixed_handler.dot(xv, y_on_iterable) - expected_mixed), 0.0, 1e-14);
}

//! The relation between the two off-diagonal blocks of a hermitian matrix
TEST(helper_complex, conjugate_transpose_copy) {
  using molpro::linalg::itsolv::subspace::Matrix;
  Matrix<scalar> source({2, 3});
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      source(i, j) = scalar(double(i + 1), double(j - i));
  Matrix<scalar> target({3, 2});
  molpro::linalg::itsolv::subspace::conjugate_transpose_copy(target, source);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_EQ(target(i, j), std::conj(source(j, i))) << "element " << i << "," << j;
}

/*!
 * @brief The overlap between two different parameter sets, computed through the transposed path.
 *
 * When the handler's operand order is the reverse of the sets being overlapped, subspace::util
 * computes the matrix the other way round and transposes it. With a hermitian inner product that
 * transpose has to conjugate as well, because <l|r> is the conjugate of <r|l>. This is the subtle
 * part of the change: a plain transpose would still produce a matrix of the right shape and
 * plausible values, just the complex conjugate of the correct one.
 */
TEST(helper_complex, overlap_through_transposed_path_is_conjugated) {
  using left_type = std::vector<scalar>;
  using right_type = std::deque<scalar>;
  std::vector<left_type> left{{{1, 2}, {3, -1}}, {{0, 1}, {2, 5}}};
  std::vector<right_type> right{{{-1, 0}, {1, 1}}, {{2, -3}, {0, 1}}, {{1, 1}, {1, -1}}};
  // handler operands in the reverse order of (left, right), which selects the transposed specialisation
  molpro::linalg::array::ArrayHandlerIterable<right_type, left_type> handler;
  const auto m = molpro::linalg::itsolv::subspace::util::overlap(molpro::linalg::itsolv::cwrap(left),
                                                                 molpro::linalg::itsolv::cwrap(right), handler);
  ASSERT_EQ(m.rows(), left.size());
  ASSERT_EQ(m.cols(), right.size());
  for (size_t a = 0; a < left.size(); ++a)
    for (size_t b = 0; b < right.size(); ++b) {
      scalar expected = 0;
      for (size_t k = 0; k < left[a].size(); ++k)
        expected += std::conj(left[a][k]) * right[b][k];
      EXPECT_NEAR(std::abs(m(a, b) - expected), 0.0, 1e-14) << "<left_" << a << "|right_" << b << ">";
    }
}
