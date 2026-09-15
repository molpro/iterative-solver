/*!
 * @file
 * @brief Tests that the dense subspace kernels are precision-generic.
 *
 * The kernels dispatch on the scalar type: LAPACK for the four types it provides kernels for, Eigen
 * for everything else. These tests check that both branches agree where they overlap, and that the
 * Eigen branch really delivers the extended precision it is given -- an accuracy that an
 * implementation secretly rounding through double could not reach.
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/helper-implementation.h>

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

using molpro::linalg::itsolv::eigensolver_hermitian;
using molpro::linalg::itsolv::get_rank;
using molpro::linalg::itsolv::has_lapack_kernel_v;
using molpro::linalg::itsolv::precision_scaled;
using molpro::linalg::itsolv::real_type_t;
using molpro::linalg::itsolv::SVD;

namespace {

/*!
 * @brief Whether long double is genuinely wider than double on this platform.
 *
 * It is not everywhere: on arm64 macOS, for instance, both are IEEE binary64. Tests that compare the
 * two precisions, or that assert an accuracy below the double-precision floor, are meaningless there,
 * and tolerances have to be stated relative to the working epsilon rather than as absolute constants.
 */
constexpr bool long_double_is_extended =
    std::numeric_limits<long double>::epsilon() < std::numeric_limits<double>::epsilon();

//! A generous multiple of the working epsilon, for a residual that ought to vanish
template <typename T>
constexpr T rounding_error_bound(int elements) {
  return T(100 * elements) * std::numeric_limits<T>::epsilon();
}

//! Deterministic orthogonal matrix of dimension n, obtained from the QR decomposition of a fixed matrix
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> orthogonal_matrix(int n, int seed) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> a(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      a(i, j) = T(1) / T(1 + i + j + seed) + T((i * 7 + j * 3 + seed) % 5) / T(11);
  return Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(a).householderQ();
}

/*!
 * @brief An ill-conditioned generalised eigenproblem H C = S C E with known eigenvalues.
 *
 * With S = V diag(s) V^T and H = S^{1/2} W diag(d) W^T S^{1/2} the eigenvalues of the pencil are
 * exactly d, however badly conditioned S is made. Everything is built in long double, so that the
 * double-precision version of the problem is the same problem, merely rounded.
 */
template <typename T>
struct GeneralisedProblem {
  int n;
  std::vector<T> h, s, expected_eigenvalues;
};

template <typename T>
GeneralisedProblem<T> make_problem(int n, long double condition_number) {
  using LD = long double;
  using MatrixLD = Eigen::Matrix<LD, Eigen::Dynamic, Eigen::Dynamic>;
  const auto V = orthogonal_matrix<LD>(n, 0);
  const auto W = orthogonal_matrix<LD>(n, 3);
  Eigen::Vector<LD, Eigen::Dynamic> sv(n), d(n), sv_half(n);
  for (int i = 0; i < n; ++i) {
    sv(i) = std::pow(condition_number, -LD(i) / LD(n - 1)); // spans [1/condition_number, 1]
    sv_half(i) = std::sqrt(sv(i));
    d(i) = LD(i + 1);
  }
  const MatrixLD S = V * sv.asDiagonal() * V.transpose();
  const MatrixLD S_half = V * sv_half.asDiagonal() * V.transpose();
  const MatrixLD H = S_half * W * d.asDiagonal() * W.transpose() * S_half;

  GeneralisedProblem<T> problem{n, std::vector<T>(n * n), std::vector<T>(n * n), std::vector<T>(n)};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      problem.h[i * n + j] = T(H(i, j));
      problem.s[i * n + j] = T(S(i, j));
    }
    problem.expected_eigenvalues[i] = T(d(i));
  }
  return problem;
}

//! Largest absolute deviation of the computed eigenvalues from the exact ones
template <typename T>
long double eigenvalue_error(int n, long double condition_number) {
  const auto problem = make_problem<T>(n, condition_number);
  std::vector<T> evecs, evals;
  molpro::linalg::itsolv::eigenproblem(evecs, evals, problem.h, problem.s, problem.n, true,
                                       precision_scaled<T>(1e-14), 0);
  EXPECT_EQ(evals.size(), size_t(n));
  long double error = 0;
  for (int i = 0; i < n; ++i)
    error = std::max(error, std::abs(static_cast<long double>(evals[i] - problem.expected_eigenvalues[i])));
  return error;
}

//! A symmetric test matrix in the requested precision, identical for every precision up to rounding
template <typename T>
std::vector<T> symmetric_matrix(int n) {
  std::vector<T> m(n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      m[i * n + j] = m[j * n + i] = T(i == j ? 2 + i : 1) / T(1 + i + j);
  return m;
}

} // namespace

TEST(helper_precision, dispatch_selects_lapack_only_for_lapack_types) {
  // long double and beyond must never end up in LAPACK, whatever the build found
  EXPECT_FALSE(has_lapack_kernel_v<long double>);
  EXPECT_FALSE(has_lapack_kernel_v<std::complex<long double>>);
#if defined(HAVE_LAPACKE) || defined(MOLPRO)
  EXPECT_TRUE(has_lapack_kernel_v<double>);
#endif
#ifdef HAVE_LAPACKE
  EXPECT_TRUE(has_lapack_kernel_v<float>);
  EXPECT_TRUE(has_lapack_kernel_v<std::complex<float>>);
  EXPECT_TRUE(has_lapack_kernel_v<std::complex<double>>);
#endif
}

TEST(helper_precision, precision_scaled_keeps_double_constants) {
  EXPECT_EQ(precision_scaled<double>(1e-10), 1e-10);
  EXPECT_EQ(precision_scaled<std::complex<double>>(1e-10), 1e-10);
  // a tolerance stays the same number of machine epsilons wide at any precision
  EXPECT_GT(precision_scaled<long double>(1e-10), 0.0L);
  EXPECT_NEAR(double(precision_scaled<long double>(1e-10) / std::numeric_limits<long double>::epsilon()),
              1e-10 / std::numeric_limits<double>::epsilon(), 1e-4);
  if constexpr (long_double_is_extended)
    EXPECT_LT(precision_scaled<long double>(1e-10), 1e-10L);
  else
    EXPECT_EQ(precision_scaled<long double>(1e-10), 1e-10L);
  EXPECT_GT(precision_scaled<float>(1e-10), 1e-10f);
}

//! The Eigen branch must return the same decomposition as the LAPACK branch, to double accuracy
TEST(helper_precision, eigensolver_hermitian_agrees_across_branches) {
  constexpr int n = 7;
  const auto m_double = symmetric_matrix<double>(n);
  const auto m_long = symmetric_matrix<long double>(n);

  std::vector<double> evecs_double(n * n), evals_double(n);
  ASSERT_EQ(eigensolver_hermitian<double>(m_double, evecs_double, evals_double, n), 0);
  std::vector<long double> evecs_long(n * n), evals_long(n);
  ASSERT_EQ(eigensolver_hermitian<long double>(m_long, evecs_long, evals_long, n), 0);

  for (int i = 0; i < n; ++i) {
    // eigenvalues in ascending order in both branches
    if (i > 0) {
      EXPECT_LE(evals_double[i - 1], evals_double[i]);
      EXPECT_LE(evals_long[i - 1], evals_long[i]);
    }
    EXPECT_NEAR(evals_double[i], double(evals_long[i]), 1e-12);
    // eigenvector i occupies elements [i*n, (i+1)*n) in both branches; the overall sign is arbitrary
    for (int j = 0; j < n; ++j)
      EXPECT_NEAR(std::abs(evecs_double[i * n + j]), std::abs(double(evecs_long[i * n + j])), 1e-10);
  }
}

//! Whichever branch ran, the result must actually solve the eigenproblem to the working precision
TEST(helper_precision, eigensolver_hermitian_residual_at_working_precision) {
  constexpr int n = 7;
  const auto m = symmetric_matrix<long double>(n);
  std::vector<long double> evecs(n * n), evals(n);
  ASSERT_EQ(eigensolver_hermitian<long double>(m, evecs, evals, n), 0);
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < n; ++i) {
      long double residual = -evals[k] * evecs[k * n + i];
      for (int j = 0; j < n; ++j)
        residual += m[j * n + i] * evecs[k * n + j];
      // stated in units of the working epsilon: where long double is genuinely wider this is below
      // the double-precision floor, so an implementation rounding through double could not reach it
      EXPECT_LT(std::abs(residual), rounding_error_bound<long double>(n)) << "root " << k << ", component " << i;
    }
  }
}

/*!
 * @brief The Eigen fallback must be usable for a LAPACK type too, since a build that finds no LAPACKE
 * takes that branch for every scalar type. Compare the two branches on identical double input.
 */
TEST(helper_precision, eigen_fallback_matches_lapack_for_double) {
  constexpr int n = 6;
  const auto m = symmetric_matrix<double>(n);

  std::vector<double> via_dispatch(n * n), evals_dispatch(n);
  ASSERT_EQ(eigensolver_hermitian<double>(m, via_dispatch, evals_dispatch, n), 0);

  // force the Eigen branch, which is what a build without LAPACKE would use
  std::vector<double> via_eigen(m.begin(), m.end());
  std::vector<double> evals_eigen(n);
  ASSERT_EQ(molpro::linalg::itsolv::detail::eigensolver_hermitian_kernel<double>(
                std::false_type{}, std::span<double>{via_eigen}, std::span<double>{evals_eigen}, n),
            0);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(evals_dispatch[i], evals_eigen[i], 1e-12);
    for (int j = 0; j < n; ++j)
      EXPECT_NEAR(std::abs(via_dispatch[i * n + j]), std::abs(via_eigen[i * n + j]), 1e-10);
  }
}

TEST(helper_precision, eigensolver_hermitian_single_precision) {
  constexpr int n = 5;
  const auto m = symmetric_matrix<float>(n);
  std::vector<float> evecs(n * n), evals(n);
  ASSERT_EQ(eigensolver_hermitian<float>(m, evecs, evals, n), 0);
  const auto m_double = symmetric_matrix<double>(n);
  std::vector<double> evecs_double(n * n), evals_double(n);
  ASSERT_EQ(eigensolver_hermitian<double>(m_double, evecs_double, evals_double, n), 0);
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(double(evals[i]), evals_double[i], 1e-5);
}

TEST(helper_precision, eigensolver_hermitian_complex) {
  constexpr int n = 4;
  // a hermitian matrix: real symmetric part plus an antisymmetric imaginary part
  std::vector<std::complex<double>> m(n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      m[i * n + j] = {double(i == j ? 2 + i : 1) / double(1 + i + j), 0.1 * (j - i)};
  std::vector<std::complex<double>> evecs(n * n);
  std::vector<double> evals(n); // eigenvalues of a hermitian matrix are real: real_type_t<value_type>
  static_assert(std::is_same_v<real_type_t<std::complex<double>>, double>);
  ASSERT_EQ(eigensolver_hermitian<std::complex<double>>(m, evecs, evals, n), 0);
  for (int i = 1; i < n; ++i)
    EXPECT_LE(evals[i - 1], evals[i]);
  // residual of the eigenpairs; both the input and the eigenvectors are column-major, so the
  // element in row i and column j of the matrix is m[j * n + i]
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i) {
      std::complex<double> residual = -evals[k] * evecs[k * n + i];
      for (int j = 0; j < n; ++j)
        residual += m[j * n + i] * evecs[k * n + j];
      EXPECT_LT(std::abs(residual), 1e-12);
    }
}

TEST(helper_precision, eigensolver_hermitian_complex_single_precision) {
  constexpr int n = 3;
  std::vector<std::complex<float>> m(n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      m[i * n + j] = {float(i == j ? 2 + i : 1) / float(1 + i + j), 0.1f * (j - i)};
  std::vector<std::complex<float>> evecs(n * n);
  std::vector<float> evals(n);
  ASSERT_EQ(eigensolver_hermitian<std::complex<float>>(m, evecs, evals, n), 0);
  for (int i = 1; i < n; ++i)
    EXPECT_LE(evals[i - 1], evals[i]);
}

TEST(helper_precision, eigensolver_rejects_inconsistent_sizes) {
  std::vector<long double> m(9), evecs(4), evals(3);
  EXPECT_THROW(eigensolver_hermitian<long double>(m, evecs, evals, 3), std::runtime_error);
}

//! get_rank over a std::vector used to be declared but never defined, so it could not be linked
TEST(helper_precision, get_rank_of_vector) {
  const std::vector<double> eigenvalues{1e-14, 1e-3, 0.5, 1.0};
  EXPECT_EQ(get_rank(eigenvalues, 1e-2), 2u);
  const std::vector<long double> eigenvalues_long{1e-14L, 1e-3L, 0.5L, 1.0L};
  EXPECT_EQ(get_rank(eigenvalues_long, 1e-2L), 2u);
}

TEST(helper_precision, svd_system_hermitian_extended_precision) {
  constexpr int n = 5;
  // a rank-deficient overlap matrix: the last row/column duplicates the first
  std::vector<long double> m(n * n, 0);
  for (int i = 0; i < n; ++i)
    m[i * n + i] = 1;
  m[0 * n + (n - 1)] = m[(n - 1) * n + 0] = 1;
  auto svds = molpro::linalg::itsolv::svd_system<long double>(
      n, n, molpro::linalg::array::Span<long double>(m.data(), m.size()), 1e-12L, true);
  ASSERT_EQ(svds.size(), 1u);
  EXPECT_LT(std::abs(svds.front().value), 1e-18L);
  // the null vector is (1, 0, ..., 0, -1)/sqrt(2)
  EXPECT_NEAR(std::abs(double(svds.front().v.front())), std::sqrt(0.5), 1e-12);
  EXPECT_NEAR(std::abs(double(svds.front().v.back())), std::sqrt(0.5), 1e-12);
}

TEST(helper_precision, svd_system_general_extended_precision) {
  constexpr int n = 4;
  std::vector<long double> m(n * n, 0);
  for (int i = 0; i < n; ++i)
    m[i * n + i] = 1;
  m[(n - 1) * n + (n - 1)] = 0; // one null direction
  auto svds = molpro::linalg::itsolv::svd_system<long double>(
      n, n, molpro::linalg::array::Span<long double>(m.data(), m.size()), 1e-12L, false);
  ASSERT_EQ(svds.size(), 1u);
  EXPECT_LT(std::abs(svds.front().value), 1e-18L);
}

TEST(helper_precision, solve_DIIS_extended_precision) {
  // A DIIS residual overlap matrix whose exact solution is known: for a diagonal B the
  // extrapolation coefficients are c_i = (1/B_ii) / sum_j (1/B_jj)
  constexpr size_t n = 4;
  std::vector<long double> b(n * n, 0);
  long double norm = 0;
  for (size_t i = 0; i < n; ++i) {
    b[i * n + i] = static_cast<long double>(i + 1);
    norm += 1 / b[i * n + i];
  }
  std::vector<long double> solution;
  molpro::linalg::itsolv::solve_DIIS<long double>(solution, b, n, precision_scaled<long double>(1e-14), 0);
  ASSERT_EQ(solution.size(), n);
  for (size_t i = 0; i < n; ++i)
    EXPECT_LT(std::abs(solution[i] - 1 / (b[i * n + i] * norm)), rounding_error_bound<long double>(int(n)))
        << "coefficient " << i;
}

/*!
 * @brief The headline test: on an ill-conditioned generalised eigenproblem, the long double
 * instantiation reaches an accuracy that double precision cannot.
 */
TEST(helper_precision, eigenproblem_extended_precision_is_more_accurate) {
  if constexpr (!long_double_is_extended)
    GTEST_SKIP() << "long double is not wider than double on this platform, so there is nothing to compare";
  constexpr int n = 6;
  constexpr long double condition_number = 1e7L;
  const auto error_double = eigenvalue_error<double>(n, condition_number);
  const auto error_long_double = eigenvalue_error<long double>(n, condition_number);
  // both must solve the problem ...
  EXPECT_LT(error_double, 1e-5L);
  // ... but only extended precision can do so this accurately at this condition number
  EXPECT_LT(error_long_double, 1e-10L);
  EXPECT_LT(error_long_double, error_double);
}
