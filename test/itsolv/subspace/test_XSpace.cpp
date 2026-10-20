#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/subspace/XSpace.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <map>
#include <memory>
#include <numeric>
#include <vector>

using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::CVecRef;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::Logger;
using molpro::linalg::itsolv::wrap;
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
// space updates and erasure
TEST(XSpace, metric_matrix_consistent) {
  XSpace<R, Q, P> xs(std::make_shared<ArrayHandlers<R, Q, P>>(), std::make_shared<Logger>());
  xs.set_metric(true);
  const std::vector<R> q1{{1, 0.5, 0, 0}, {0, 1, 0.2, 0}};
  const auto m1 = metric_actions(q1);
  xs.update_qspace(cwrap(q1), cwrap(q1), cwrap(m1));
  check_metric_consistent(xs);
  const std::vector<R> q2{{0, 0, 1, 0.3}, {0, 0, 1, 0.3}};
  const auto m2 = metric_actions(q2);
  xs.update_qspace(cwrap(q2), cwrap(q2), cwrap(m2));
  EXPECT_EQ(xs.dimensions().nQ, 3);
  check_metric_consistent(xs);
  std::vector<Q> d{{0.1, 0.2, 0.3, 1}}, d_actions = d, d_metric = metric_actions(d);
  // the metric matrix with the new D space is given, rather than calculated from the vectors
  auto params = std::vector<R>{};
  for (const auto& q : xs.cparamsq())
    params.push_back(q.get());
  params.push_back(d[0]);
  Matrix<double> metric({params.size(), params.size()});
  for (size_t i = 0; i < params.size(); ++i)
    for (size_t j = 0; j < params.size(); ++j)
      metric(i, j) = dot(params[i], apply_metric(params[j]));
  auto wd = wrap(d);
  auto wd_actions = wrap(d_actions);
  EXPECT_THROW(xs.update_dspace(wd, wd_actions, wrap(d_metric)), std::runtime_error);
  xs.update_dspace(wd, wd_actions, wrap(d_metric), metric);
  EXPECT_EQ(xs.dimensions().nD, 1);
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
    EXPECT_THROW(xs.set_metric(true), std::logic_error);
    EXPECT_NO_THROW(xs.set_metric(false));
  }
  {
    XSpace<R, Q, P> xs(std::make_shared<ArrayHandlers<R, Q, P>>(), std::make_shared<Logger>());
    xs.set_metric(true);
    xs.update_qspace(cwrap(q1), cwrap(q1), cwrap(m1));
    EXPECT_THROW(xs.set_metric(false), std::logic_error);
    // adding no vectors needs no actions of the metric either
    EXPECT_NO_THROW(xs.update_qspace(CVecRef<R>{}, CVecRef<R>{}));
    EXPECT_THROW(xs.update_qspace(cwrap(q2), cwrap(q2)), std::runtime_error);
    std::vector<Q> d{scaled_unit(2, 1)}, d_actions = d;
    auto wd = wrap(d);
    auto wd_actions = wrap(d_actions);
    EXPECT_THROW(xs.update_dspace(wd, wd_actions), std::runtime_error);
  }
}
