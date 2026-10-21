#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/ArrayHandlers.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/MetricChecker.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <vector>

using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::Logger;
using molpro::linalg::itsolv::MetricKind;
using molpro::linalg::itsolv::MetricStorage;
using molpro::linalg::itsolv::subspace::Matrix;
using molpro::linalg::itsolv::subspace::XSpace;

namespace {
using R = std::vector<double>;
using Q = R;
using P = std::map<size_t, double>;
using Checker = molpro::linalg::itsolv::detail::MetricChecker<R, Q, P>;

Matrix<double> matrix(std::vector<double> elements, size_t n) { return Matrix<double>(std::move(elements), {n, n}); }

//! A metric on vectors of length 3, Hermitian unless non_hermiticity is set
R apply_metric(const R& x, double non_hermiticity = 0) {
  R mx(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    for (size_t j = 0; j < x.size(); ++j)
      mx[i] += (i == j ? 2.0 + i : 0.3 * (j < i ? 1 - non_hermiticity : 1)) * x[j];
  return mx;
}
} // namespace

// A positive definite metric passes, also on nearly or exactly linearly dependent subspace vectors, where the Cholesky
// factorisation fails and the Rayleigh quotients on the span of the subspace vectors decide
TEST(MetricChecker, positive) {
  EXPECT_NO_THROW(Checker::check_positive(matrix({2, 0.5, 0.5, 1}, 2), matrix({1, 0, 0, 1}, 2)));
  // two copies of the same vector: the metric matrix is singular
  EXPECT_NO_THROW(Checker::check_positive(matrix({2, 2, 2, 2}, 2), matrix({1, 1, 1, 1}, 2)));
  // a metric matrix that is slightly indefinite through rounding on nearly dependent vectors
  EXPECT_NO_THROW(
      Checker::check_positive(matrix({1, 1 + 1e-13, 1 + 1e-13, 1}, 2), matrix({1, 1 - 1e-14, 1 - 1e-14, 1}, 2)));
  // the metric itself is indefinite
  EXPECT_THROW(Checker::check_positive(matrix({1, 0, 0, -1}, 2), matrix({1, 0, 0, 1}, 2)), std::runtime_error);
  EXPECT_THROW(Checker::check_positive(matrix({-1, 0, 0, -2}, 2), matrix({1, 0, 0, 1}, 2)), std::runtime_error);
}

// A positive definite metric with a large condition number passes on nearly linearly dependent subspace vectors that
// differ only along its smallest directions, where orthonormalising amplifies the rounding errors of the metric matrix
TEST(MetricChecker, positive_ill_conditioned_basis) {
  const size_t n = 400, k = 4;
  std::vector<double> metric(n);
  for (size_t i = 0; i < n; ++i)
    metric[i] = std::pow(1e-10, double(i) / (n - 1));
  std::mt19937 generator(7);
  std::normal_distribution<double> distribution;
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<std::vector<double>> b(k, std::vector<double>(n));
    for (size_t j = 0; j < k; ++j) {
      for (size_t i = 0; i < n; ++i) {
        const auto random = distribution(generator);
        b[j][i] = j == 0 ? random : b[j - 1][i] + (i >= n - n / 40 ? 2e-5 * random : 0.);
      }
      const auto norm = std::sqrt(std::inner_product(b[j].begin(), b[j].end(), b[j].begin(), 0.));
      for (auto& x : b[j])
        x /= norm;
    }
    Matrix<double> m({k, k}), s({k, k});
    for (size_t i = 0; i < k; ++i)
      for (size_t j = 0; j < k; ++j)
        for (size_t l = 0; l < n; ++l) {
          m(i, j) += b[i][l] * metric[l] * b[j][l];
          s(i, j) += b[i][l] * b[j][l];
        }
    EXPECT_NO_THROW(Checker::check_positive(m, s)) << "trial " << trial;
  }
}

// The metric matrix in the P space has to be Hermitian
TEST(MetricChecker, hermitian_pspace) {
  std::vector<double> hermitian{2, 0.5, 0.5, 1}, non_hermitian{2, 0.5, 0.4, 1};
  EXPECT_NO_THROW(Checker::check_hermitian(molpro::linalg::array::Span<double>(hermitian.data(), 4), 2));
  EXPECT_THROW(Checker::check_hermitian(molpro::linalg::array::Span<double>(non_hermitian.data(), 4), 2),
               std::runtime_error);
}

// A metric that is not Hermitian is noticed on new parameters by comparison with a vector of an earlier update,
// whether the subspace keeps the actions of the metric or not
TEST(MetricChecker, hermitian) {
  for (const auto storage : {MetricStorage::Stored, MetricStorage::OnTheFly}) {
    for (const double non_hermiticity : {0., 0.5}) {
      SCOPED_TRACE("on the fly=" + std::to_string(storage == MetricStorage::OnTheFly) +
                   " non-hermiticity=" + std::to_string(non_hermiticity));
      auto handlers = std::make_shared<ArrayHandlers<R, Q, P>>();
      XSpace<R, Q, P> xspace(handlers, std::make_shared<Logger>());
      xspace.set_metric(MetricKind::HermitianPositiveDefinite, storage);
      Checker checker;
      const std::vector<R> x1{{1, 0.2, 0}}, x2{{0, 1, 0.5}};
      const std::vector<R> m1{apply_metric(x1[0], non_hermiticity)}, m2{apply_metric(x2[0], non_hermiticity)};
      // nothing to compare with yet
      EXPECT_NO_THROW(checker.check_hermitian(cwrap(x1), cwrap(m1), xspace, *handlers));
      xspace.update_qspace(cwrap(x1), cwrap(x1), cwrap(m1));
      checker.remember(cwrap(x1), cwrap(m1), xspace, *handlers);
      if (non_hermiticity == 0)
        EXPECT_NO_THROW(checker.check_hermitian(cwrap(x2), cwrap(m2), xspace, *handlers));
      else
        EXPECT_THROW(checker.check_hermitian(cwrap(x2), cwrap(m2), xspace, *handlers), std::runtime_error);
    }
  }
}
