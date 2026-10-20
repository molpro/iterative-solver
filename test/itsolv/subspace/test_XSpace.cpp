#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/subspace/XSpace.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::CVecRef;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::Logger;
using molpro::linalg::itsolv::MetricKind;
using molpro::linalg::itsolv::wrap;
using molpro::linalg::itsolv::subspace::Dimensions;
using molpro::linalg::itsolv::subspace::EqnData;
using molpro::linalg::itsolv::subspace::Matrix;
using molpro::linalg::itsolv::subspace::XSpace;
using ::testing::ElementsAre;

namespace {
using R = std::vector<double>;
using Q = R;
using P = std::map<size_t, double>;

R scaled_unit(size_t i, double length) {
  R v(4, 0);
  v[i] = length;
  return v;
}

//! The diagonal of the overlap, i.e. the squared lengths of the vectors in the subspace, in subspace order
std::vector<double> squared_lengths(const XSpace<R, Q, P>& xs) {
  std::vector<double> d;
  for (size_t i = 0; i < xs.size(); ++i)
    d.push_back(xs.data.at(EqnData::S)(i, i));
  return d;
}
} // namespace

// erase(i) takes an index into the whole subspace (P, then Q, then D) and removes that vector
TEST(XSpace, erase_by_subspace_index) {
  XSpace<R, Q, P> xs(std::make_shared<ArrayHandlers<R, Q, P>>(), std::make_shared<Logger>());
  const std::vector<R> q{scaled_unit(0, 1), scaled_unit(1, 2), scaled_unit(2, 3)};
  xs.update_qspace(cwrap(q), cwrap(q));
  std::vector<Q> d{scaled_unit(3, 5)}, d_actions{scaled_unit(3, 5)};
  auto wd = wrap(d);
  auto wd_actions = wrap(d_actions);
  xs.update_dspace(wd, wd_actions);
  ASSERT_EQ(xs.dimensions().nQ, 3);
  ASSERT_EQ(xs.dimensions().nD, 1);
  ASSERT_THAT(squared_lengths(xs), ElementsAre(1, 4, 9, 25));

  xs.erase(xs.dimensions().oQ() + 1); // the middle Q vector
  EXPECT_EQ(xs.dimensions().nQ, 2);
  EXPECT_EQ(xs.dimensions().nD, 1);
  EXPECT_THAT(squared_lengths(xs), ElementsAre(1, 9, 25));

  xs.erase(xs.dimensions().oD()); // the D vector
  EXPECT_EQ(xs.dimensions().nQ, 2);
  EXPECT_EQ(xs.dimensions().nD, 0);
  EXPECT_THAT(squared_lengths(xs), ElementsAre(1, 9));

  xs.erase(xs.size()); // out of range: no change
  EXPECT_THAT(squared_lengths(xs), ElementsAre(1, 9));
}

namespace {
//! A symmetric, positive definite, non-diagonal metric on vectors of length 4
R apply_metric(const R& x) {
  R mx(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    for (size_t j = 0; j < x.size(); ++j)
      mx[i] += (i == j ? 2.0 + i : 0.1 / (1.0 + i + j)) * x[j];
  return mx;
}

double dot(const R& x, const R& y) { return std::inner_product(x.begin(), x.end(), y.begin(), 0.); }

std::vector<Q> metric_actions(const std::vector<R>& params) {
  std::vector<Q> m;
  for (const auto& p : params)
    m.push_back(apply_metric(p));
  return m;
}

//! The stored actions of the metric and the metric matrix must match the parameters, in subspace order
void check_metric_consistent(const XSpace<R, Q, P>& xs) {
  ASSERT_TRUE(xs.has_metric());
  auto params = xs.cparamsq();
  auto metric = xs.cmetric_actionsq();
  for (const auto& p : xs.cparamsd())
    params.push_back(p);
  for (const auto& m : xs.cmetric_actionsd())
    metric.push_back(m);
  ASSERT_EQ(metric.size(), params.size());
  ASSERT_EQ(params.size(), xs.size());
  const auto& m = xs.data.at(EqnData::M);
  ASSERT_EQ(m.rows(), xs.size());
  ASSERT_EQ(m.cols(), xs.size());
  for (size_t i = 0; i < params.size(); ++i) {
    EXPECT_THAT(metric[i].get(), ::testing::Pointwise(::testing::DoubleEq(), apply_metric(params[i].get())));
    for (size_t j = 0; j < params.size(); ++j)
      EXPECT_NEAR(m(i, j), dot(params[i], apply_metric(params[j])), 1e-14) << "M(" << i << "," << j << ")";
  }
}
} // namespace

// The metric matrix follows the subspace through Q space updates, including removal of a redundant new parameter, D
// space updates, which transform it, and erasure
TEST(XSpace, metric_matrix_consistent) {
  XSpace<R, Q, P> xs(std::make_shared<ArrayHandlers<R, Q, P>>(), std::make_shared<Logger>());
  xs.set_metric(MetricKind::HermitianPositiveDefinite);
  const std::vector<R> q1{{1, 0.5, 0, 0}, {0, 1, 0.2, 0}};
  const auto m1 = metric_actions(q1);
  xs.update_qspace(cwrap(q1), cwrap(q1), cwrap(m1));
  check_metric_consistent(xs);
  const std::vector<R> q2{{0, 0, 1, 0.3}, {0, 0, 1, 0.3}};
  const auto m2 = metric_actions(q2);
  xs.update_qspace(cwrap(q2), cwrap(q2), cwrap(m2));
  EXPECT_EQ(xs.dimensions().nQ, 3);
  check_metric_consistent(xs);
  // D vectors replacing deleted Q vectors and old D vectors get their metric matrix elements from the coefficients
  auto replace = [&xs](const std::vector<size_t>& q_delete, const std::vector<double>& coefficients) {
    const auto qparams = xs.cparamsq(), dparams = xs.cparamsd();
    std::vector<Q> d(1, Q(4));
    for (size_t j = 0; j < coefficients.size(); ++j) {
      const auto& b = j < q_delete.size() ? qparams[q_delete[j]].get() : dparams[j - q_delete.size()].get();
      for (size_t k = 0; k < 4; ++k)
        d[0][k] += coefficients[j] * b[k];
    }
    std::vector<Q> d_actions = d, d_metric = metric_actions(d);
    auto wd = wrap(d);
    auto wd_actions = wrap(d_actions);
    EXPECT_THROW(xs.update_dspace(wd, wd_actions, wrap(d_metric)), std::invalid_argument);
    xs.update_dspace(wd, wd_actions, wrap(d_metric), q_delete, Matrix<double>(coefficients, {1, coefficients.size()}));
  };
  replace({1}, {0.7});
  EXPECT_EQ(xs.dimensions().nQ, 2);
  EXPECT_EQ(xs.dimensions().nD, 1);
  check_metric_consistent(xs);
  replace({0}, {0.5, 0.3});
  EXPECT_EQ(xs.dimensions().nQ, 1);
  EXPECT_EQ(xs.dimensions().nD, 1);
  check_metric_consistent(xs);
  xs.update_qspace(cwrap(q1), cwrap(q1), cwrap(m1));
  check_metric_consistent(xs);
  xs.erase(xs.dimensions().oQ() + 1);
  check_metric_consistent(xs);
  xs.erase(xs.dimensions().oD());
  check_metric_consistent(xs);
}

// Without a declared metric, no metric is stored
TEST(XSpace, no_metric) {
  XSpace<R, Q, P> xs(std::make_shared<ArrayHandlers<R, Q, P>>(), std::make_shared<Logger>());
  const std::vector<R> q{scaled_unit(0, 1), scaled_unit(1, 2)};
  xs.update_qspace(cwrap(q), cwrap(q));
  EXPECT_FALSE(xs.has_metric());
  EXPECT_EQ(xs.data.count(EqnData::M), 0);
  EXPECT_TRUE(xs.cmetric_actionsq().empty());
}

// Actions of the metric are given for every update exactly if a metric was declared, which is possible only on an
// empty subspace
TEST(XSpace, metric_given_consistently) {
  const std::vector<R> q1{scaled_unit(0, 1)}, q2{scaled_unit(1, 1)};
  const auto m1 = metric_actions(q1), m2 = metric_actions(q2);
  {
    XSpace<R, Q, P> xs(std::make_shared<ArrayHandlers<R, Q, P>>(), std::make_shared<Logger>());
    xs.update_qspace(cwrap(q1), cwrap(q1));
    EXPECT_THROW(xs.update_qspace(cwrap(q2), cwrap(q2), cwrap(m2)), std::runtime_error);
    EXPECT_THROW(xs.set_metric(MetricKind::HermitianPositiveDefinite), std::logic_error);
    EXPECT_NO_THROW(xs.set_metric(MetricKind::Identity));
  }
  {
    XSpace<R, Q, P> xs(std::make_shared<ArrayHandlers<R, Q, P>>(), std::make_shared<Logger>());
    xs.set_metric(MetricKind::HermitianPositiveDefinite);
    xs.update_qspace(cwrap(q1), cwrap(q1), cwrap(m1));
    EXPECT_THROW(xs.set_metric(MetricKind::Identity), std::logic_error);
    // adding no vectors needs no actions of the metric either
    EXPECT_NO_THROW(xs.update_qspace(CVecRef<R>{}, CVecRef<R>{}));
    EXPECT_THROW(xs.update_qspace(cwrap(q2), cwrap(q2)), std::runtime_error);
    std::vector<Q> d{scaled_unit(2, 1)}, d_actions = d;
    auto wd = wrap(d);
    auto wd_actions = wrap(d_actions);
    EXPECT_THROW(xs.update_dspace(wd, wd_actions), std::invalid_argument);
  }
}

namespace {
// The metric matrix of a subspace whose deleted Q vectors and old D vectors are replaced by new D vectors, transformed
// from the old metric matrix, must be that of the new vectors
template <typename T>
void check_metric_with_new_dspace() {
  using R = std::vector<T>;
  const size_t n = 6;
  T phase = 1;
  if constexpr (!std::is_same_v<T, double>)
    phase = std::polar(1.0, 0.4);
  auto metric = [phase](size_t i, size_t j) {
    return i == j ? T(2.0 + i) : (i < j ? phase : molpro::linalg::conjugate(phase)) * T(0.1 / (1.0 + i + j));
  };
  auto element = [&](const R& x, const R& y) {
    T m = 0;
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        m += molpro::linalg::conjugate(x[i]) * metric(i, j) * y[j];
    return m;
  };
  auto matrix = [&](const std::vector<R>& vectors) {
    Matrix<T> m({vectors.size(), vectors.size()});
    for (size_t i = 0; i < vectors.size(); ++i)
      for (size_t j = 0; j < vectors.size(); ++j)
        m(i, j) = element(vectors[i], vectors[j]);
    return m;
  };
  // the subspace P+Q+D with nP = 1, nQ = 3, nD = 1
  std::vector<R> x(5, R(n));
  for (size_t k = 0; k < x.size(); ++k)
    for (size_t i = 0; i < n; ++i)
      x[k][i] = T(std::cos(1.0 + k + 0.7 * i)) + T(0.1 * k) * phase;
  const auto dims = Dimensions(1, 3, 1);
  const std::vector<size_t> q_delete{2, 0};
  // two new D vectors from Q vectors 2 and 0 and the old D vector
  Matrix<T> coefficients({2, 3});
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      coefficients(i, j) = T(0.3 + i - 0.2 * j) + T(0.1 * j) * phase;
  std::vector<R> x_new{x[dims.oP()], x[dims.oQ() + 1]};
  for (size_t i = 0; i < 2; ++i) {
    R d(n);
    const std::vector<size_t> from{dims.oQ() + 2, dims.oQ(), dims.oD()};
    for (size_t j = 0; j < 3; ++j)
      for (size_t k = 0; k < n; ++k)
        d[k] += coefficients(i, j) * x[from[j]][k];
    x_new.push_back(d);
  }
  const auto expected = matrix(x_new);
  const auto result =
      molpro::linalg::itsolv::subspace::xspace::metric_with_new_dspace(matrix(x), dims, q_delete, coefficients);
  ASSERT_EQ(result.rows(), expected.rows());
  ASSERT_EQ(result.cols(), expected.cols());
  for (size_t i = 0; i < expected.rows(); ++i)
    for (size_t j = 0; j < expected.cols(); ++j)
      EXPECT_NEAR(std::abs(result(i, j) - expected(i, j)), 0, 1e-13) << "M(" << i << "," << j << ")";
}
} // namespace

TEST(XSpace, metric_with_new_dspace) { check_metric_with_new_dspace<double>(); }

TEST(XSpace, metric_with_new_dspace_complex) { check_metric_with_new_dspace<std::complex<double>>(); }
