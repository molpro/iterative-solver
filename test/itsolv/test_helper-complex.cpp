/*!
 * @file
 * @brief Tests of the subspace solvers for a complex scalar type.
 *
 * The library uses the hermitian inner product, so the subspace overlap and a hermitian operator are
 * hermitian matrices rather than complex symmetric ones. Each test below therefore checks the defining
 * equation directly -- H c = lambda S c, H x = b -- rather than comparing against a transcribed
 * reference, so that a wrong conjugation cannot pass unnoticed.
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/helper-implementation.h>
#include <molpro/linalg/array/ArrayHandlerIterable.h>
#include <molpro/linalg/array/ArrayHandlerIterableSparse.h>
#include <molpro/linalg/array/ArrayHandlerSparse.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/subspace/util.h>

#include <Eigen/Dense>

#include <complex>
#include <deque>
#include <map>
#include <limits>
#include <vector>

using scalar = std::complex<double>;
using molpro::linalg::itsolv::eigenproblem;
using molpro::linalg::itsolv::solve_DIIS;
using molpro::linalg::itsolv::solve_LinearEquations;
using molpro::linalg::itsolv::svd_system;

namespace {

using MatrixC = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>;
using VectorC = Eigen::Vector<scalar, Eigen::Dynamic>;

//! Row-major buffer, the layout subspace::Matrix uses and the subspace solvers expect
std::vector<scalar> row_major(const MatrixC& m) {
  std::vector<scalar> buffer(m.size());
  for (Eigen::Index i = 0; i < m.rows(); ++i)
    for (Eigen::Index j = 0; j < m.cols(); ++j)
      buffer[i * m.cols() + j] = m(i, j);
  return buffer;
}

//! A hermitian matrix with a distinct, well separated spectrum
MatrixC hermitian(int n) {
  MatrixC h(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      h(i, j) = i == j ? scalar(i + 1, 0) : scalar(0.05 * (i + j), 0.05 * (j - i));
  return h;
}

//! A hermitian positive definite metric, close to but not equal to the identity
MatrixC metric(int n) {
  MatrixC s = MatrixC::Identity(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      s(i, j) = scalar(0.01 * (i - j), 0.02);
      s(j, i) = std::conj(s(i, j));
    }
  return s;
}

//! A matrix that is not hermitian, for the general eigenproblem
MatrixC non_hermitian(int n) {
  MatrixC h(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      h(i, j) = i == j ? scalar(i + 1, 0.1) : scalar(0.05 * (i + 1), 0.03 * (j + 1));
  return h;
}

//! Eigenvector k of the returned column-major dimension x nsolution buffer
VectorC eigenvector(const std::vector<scalar>& evecs, int dimension, int k) {
  VectorC v(dimension);
  for (int i = 0; i < dimension; ++i)
    v(i) = evecs[k * dimension + i];
  return v;
}

constexpr int n = 6;

} // namespace

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

//! H c = lambda S c, for a hermitian H: the eigenvalues must come out real and correctly ordered
TEST(helper_complex, eigenproblem_hermitian) {
  const auto H = hermitian(n);
  const auto S = metric(n);
  std::vector<scalar> evecs, evals;
  eigenproblem(evecs, evals, row_major(H), row_major(S), n, true, 1e-14, 0);
  ASSERT_EQ(evals.size(), size_t(n));

  Eigen::GeneralizedSelfAdjointEigenSolver<MatrixC> reference(H, S);
  ASSERT_EQ(reference.info(), Eigen::Success);
  for (int k = 0; k < n; ++k) {
    EXPECT_NEAR(evals[k].imag(), 0.0, 1e-12) << "root " << k << " of a hermitian problem must be real";
    EXPECT_NEAR(evals[k].real(), reference.eigenvalues()(k), 1e-10) << "root " << k;
    if (k > 0)
      EXPECT_LE(evals[k - 1].real(), evals[k].real()) << "eigenvalues must be in ascending order";
    // the defining equation, which a mistaken conjugation anywhere would break
    const auto c = eigenvector(evecs, n, k);
    EXPECT_LT((H * c - evals[k] * (S * c)).norm(), 1e-10) << "residual of root " << k;
    // eigenvectors of a hermitian problem are S-orthonormal
    EXPECT_NEAR(std::abs(c.dot(S * c)), 1.0, 1e-10) << "normalisation of root " << k;
  }
}

//! The same for an H that is not hermitian, where the eigenvalues are genuinely complex
TEST(helper_complex, eigenproblem_non_hermitian) {
  const auto H = non_hermitian(n);
  const auto S = metric(n);
  std::vector<scalar> evecs, evals;
  eigenproblem(evecs, evals, row_major(H), row_major(S), n, false, 1e-14, 0);
  ASSERT_EQ(evals.size(), size_t(n));
  for (int k = 0; k < n; ++k) {
    const auto c = eigenvector(evecs, n, k);
    EXPECT_LT((H * c - evals[k] * (S * c)).norm(), 1e-10) << "residual of root " << k;
    if (k > 0)
      EXPECT_LE(evals[k - 1].real(), evals[k].real()) << "eigenvalues must be ordered by real part";
  }
}

//! A rank-deficient metric must have its null directions projected out, not produce NaNs
TEST(helper_complex, eigenproblem_rank_deficient_metric) {
  auto S = metric(n);
  S.col(n - 1) = S.col(0); // duplicate a direction, making the metric singular
  S.row(n - 1) = S.row(0);
  S(n - 1, n - 1) = S(0, 0);
  const auto H = hermitian(n);
  std::vector<scalar> evecs, evals;
  eigenproblem(evecs, evals, row_major(H), row_major(S), n, true, 1e-10, 0);
  EXPECT_EQ(evals.size(), size_t(n - 1)) << "the singular direction should have been dropped";
  for (const auto& e : evals) {
    EXPECT_FALSE(std::isnan(e.real()));
    EXPECT_NEAR(e.imag(), 0.0, 1e-10);
  }
  for (const auto& e : evecs)
    EXPECT_FALSE(std::isnan(e.real()) || std::isnan(e.imag()));
}

//! svd_system finds the null space of a rank-deficient hermitian overlap
TEST(helper_complex, svd_system_hermitian) {
  const int dim = 4;
  MatrixC s = MatrixC::Identity(dim, dim);
  // make the last vector a phase-rotated copy of the first, so the overlap is singular
  const scalar phase{0.0, 1.0};
  s(0, dim - 1) = std::conj(phase);
  s(dim - 1, 0) = phase;
  auto buffer = row_major(s);
  auto svds = svd_system<scalar>(dim, dim, molpro::linalg::array::Span<scalar>(buffer.data(), buffer.size()), 1e-12,
                                 true);
  ASSERT_EQ(svds.size(), 1u);
  EXPECT_LT(std::abs(svds.front().value), 1e-12);
  // the null vector is (1, 0, ..., 0, -phase)/sqrt(2), up to an overall phase
  const auto& v = svds.front().v;
  ASSERT_EQ(v.size(), size_t(dim));
  EXPECT_NEAR(std::abs(v.front()), std::sqrt(0.5), 1e-10);
  EXPECT_NEAR(std::abs(v.back()), std::sqrt(0.5), 1e-10);
  // check it really is annihilated by the overlap
  VectorC null(dim);
  for (int i = 0; i < dim; ++i)
    null(i) = v[i];
  EXPECT_LT((s * null).norm(), 1e-10);
}

//! Straight solution of H x = b
TEST(helper_complex, solve_linear_equations) {
  const int nroot = 2;
  const auto H = non_hermitian(n);
  const auto S = metric(n);
  MatrixC b(n, nroot);
  for (int i = 0; i < n; ++i)
    for (int r = 0; r < nroot; ++r)
      b(i, r) = scalar(0.5 * (i + 1), 0.25 * (r + 1) - 0.1 * i);

  std::vector<scalar> solution, eigenvalues;
  solve_LinearEquations(solution, eigenvalues, row_major(H), row_major(S), row_major(b), n, nroot, 0.0, 1e-14, 0);
  ASSERT_EQ(solution.size(), size_t(n * nroot));
  for (int r = 0; r < nroot; ++r) {
    VectorC x(n);
    for (int k = 0; k < n; ++k)
      x(k) = solution[k + n * r];
    EXPECT_LT((H * x - b.col(r)).norm(), 1e-10) << "residual of right-hand side " << r;
  }
}

//! The augmented-Hessian variant, which for a complex scalar cannot use Eigen's real-only solver
TEST(helper_complex, solve_linear_equations_augmented_hessian) {
  const int nroot = 1;
  const auto H = hermitian(n);
  const auto S = metric(n);
  MatrixC b(n, nroot);
  for (int i = 0; i < n; ++i)
    b(i, 0) = scalar(0.5 * (i + 1), -0.1 * i);

  std::vector<scalar> solution, eigenvalues;
  const double alpha = 0.1;
  solve_LinearEquations(solution, eigenvalues, row_major(H), row_major(S), row_major(b), n, nroot, alpha, 1e-14, 0);
  ASSERT_EQ(solution.size(), size_t(n));
  ASSERT_EQ(eigenvalues.size(), size_t(nroot));
  VectorC x(n);
  for (int k = 0; k < n; ++k)
    x(k) = solution[k];
  // the augmented Hessian solves the shifted equation (H - lambda S) x = b
  const scalar lambda = eigenvalues[0];
  EXPECT_LT(((H - lambda * S) * x - b.col(0)).norm(), 1e-8);
}

//! DIIS extrapolation coefficients for a diagonal residual overlap are known in closed form
TEST(helper_complex, solve_DIIS) {
  const size_t dim = 4;
  std::vector<scalar> b(dim * dim, scalar(0, 0));
  double norm = 0;
  for (size_t i = 0; i < dim; ++i) {
    b[i * dim + i] = scalar(i + 1, 0); // a residual overlap is hermitian, so its diagonal is real
    norm += 1.0 / (i + 1);
  }
  std::vector<scalar> solution;
  solve_DIIS(solution, b, dim, 1e-14, 0);
  ASSERT_EQ(solution.size(), dim);
  scalar sum{0, 0};
  for (size_t i = 0; i < dim; ++i) {
    EXPECT_NEAR(solution[i].real(), 1.0 / ((i + 1) * norm), 1e-12) << "coefficient " << i;
    EXPECT_NEAR(solution[i].imag(), 0.0, 1e-12) << "coefficient " << i;
    sum += solution[i];
  }
  EXPECT_NEAR(sum.real(), 1.0, 1e-12) << "DIIS coefficients must sum to one";
}

/*!
 * @brief Complex and extended precision compose: nothing in the complex path is tied to double.
 *
 * std::complex<long double> has no LAPACK kernel, so this also exercises the Eigen branch of the
 * dispatch with a complex scalar.
 */
TEST(helper_complex, extended_precision_complex) {
  // on arm64 macOS, for instance, long double is IEEE binary64 and there is no extra precision to test
  if constexpr (std::numeric_limits<long double>::epsilon() >= std::numeric_limits<double>::epsilon())
    GTEST_SKIP() << "long double is not wider than double on this platform";
  using scalar_long = std::complex<long double>;
  using MatrixL = Eigen::Matrix<scalar_long, Eigen::Dynamic, Eigen::Dynamic>;
  static_assert(!molpro::linalg::itsolv::has_lapack_kernel_v<scalar_long>);
  static_assert(std::is_same_v<molpro::linalg::itsolv::real_type_t<scalar_long>, long double>);

  const int dim = 5;
  MatrixL H(dim, dim), S = MatrixL::Identity(dim, dim);
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      H(i, j) = i == j ? scalar_long(i + 1, 0) : scalar_long(0.05L * (i + j), 0.05L * (j - i));

  std::vector<scalar_long> h(dim * dim), s(dim * dim), evecs, evals;
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      h[i * dim + j] = H(i, j);
      s[i * dim + j] = S(i, j);
    }
  eigenproblem(evecs, evals, h, s, dim, true, 1e-18L, 0);
  ASSERT_EQ(evals.size(), size_t(dim));
  for (int k = 0; k < dim; ++k) {
    EXPECT_LT(std::abs(evals[k].imag()), 1e-16L) << "root " << k << " must be real";
    Eigen::Vector<scalar_long, Eigen::Dynamic> c(dim);
    for (int i = 0; i < dim; ++i)
      c(i) = evecs[k * dim + i];
    // below the double-precision floor, so only genuine extended-precision arithmetic gets here
    EXPECT_LT((H * c - evals[k] * (S * c)).norm(), 1e-17L) << "residual of root " << k;
  }
}
