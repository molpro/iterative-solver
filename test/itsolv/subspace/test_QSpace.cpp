#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/subspace/QSpace.h>
#include <molpro/linalg/itsolv/subspace/SubspaceData.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <map>
#include <memory>
#include <numeric>
#include <vector>

using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::Logger;
using molpro::linalg::itsolv::subspace::Dimensions;
using molpro::linalg::itsolv::subspace::EqnData;
using molpro::linalg::itsolv::subspace::null_data;
using molpro::linalg::itsolv::subspace::QSpace;
using molpro::linalg::itsolv::subspace::SubspaceData;

namespace {
using R = std::vector<double>;
using Q = R;
using P = std::map<size_t, double>;

double dot(const R& x, const R& y) { return std::inner_product(x.begin(), x.end(), y.begin(), 0.); }

//! Diagonal model operator, so that actions are distinguishable from parameters
R apply(const R& x) {
  auto ax = x;
  for (size_t i = 0; i < ax.size(); ++i)
    ax[i] *= double(i + 1);
  return ax;
}

struct QSpaceF : ::testing::Test {
  std::shared_ptr<ArrayHandlers<R, Q, P>> handlers = std::make_shared<ArrayHandlers<R, Q, P>>();
  QSpace<R, Q, P> qspace{handlers, std::make_shared<Logger>()};
  SubspaceData<double> data = null_data<double, EqnData::H, EqnData::S, EqnData::rhs>();

  //! Adds new parameters, building the qq/qx/xq blocks against the current Q space as XSpace would
  void update(const std::vector<R>& params) {
    std::vector<R> actions;
    for (const auto& p : params)
      actions.push_back(apply(p));
    const auto old_params = qspace.cparams();
    const auto old_actions = qspace.cactions();
    const size_t nnew = params.size(), nX = old_params.size();
    auto qq = null_data<double, EqnData::H, EqnData::S, EqnData::rhs>();
    auto qx = null_data<double, EqnData::H, EqnData::S>();
    auto xq = null_data<double, EqnData::H, EqnData::S>();
    for (auto d : {EqnData::H, EqnData::S}) {
      qq[d].resize({nnew, nnew});
      qx[d].resize({nnew, nX});
      xq[d].resize({nX, nnew});
    }
    for (size_t i = 0; i < nnew; ++i) {
      for (size_t j = 0; j < nnew; ++j) {
        qq[EqnData::S](i, j) = dot(params[i], params[j]);
        qq[EqnData::H](i, j) = dot(params[i], actions[j]);
      }
      for (size_t j = 0; j < nX; ++j) {
        qx[EqnData::S](i, j) = dot(params[i], old_params[j]);
        qx[EqnData::H](i, j) = dot(params[i], old_actions[j]);
        xq[EqnData::S](j, i) = dot(old_params[j], params[i]);
        xq[EqnData::H](j, i) = dot(old_params[j], actions[i]);
      }
    }
    qspace.update(cwrap(params), cwrap(actions), qq, qx, xq, Dimensions(0, nX, 0), data);
  }

  //! The subspace data must be consistent with the parameters and actions actually stored, in the same order
  void check_consistent() {
    const auto params = qspace.cparams();
    const auto actions = qspace.cactions();
    const auto n = params.size();
    ASSERT_EQ(data.at(EqnData::S).rows(), n);
    ASSERT_EQ(data.at(EqnData::S).cols(), n);
    ASSERT_EQ(data.at(EqnData::H).rows(), n);
    ASSERT_EQ(data.at(EqnData::H).cols(), n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j) {
        EXPECT_DOUBLE_EQ(data.at(EqnData::S)(i, j), dot(params[i], params[j])) << "S(" << i << "," << j << ")";
        EXPECT_DOUBLE_EQ(data.at(EqnData::H)(i, j), dot(params[i], actions[j])) << "H(" << i << "," << j << ")";
      }
  }
};
} // namespace

// A redundant parameter that is not the last of the new ones
TEST_F(QSpaceF, update_prunes_leading_redundant_parameter) {
  const R a{1, 0, 0}, b{0, 1, 0};
  update({a, a, b});
  EXPECT_EQ(qspace.size(), 2);
  check_consistent();
  const auto params = qspace.cparams();
  EXPECT_THAT(params[0].get(), ::testing::Pointwise(::testing::DoubleEq(), a));
  EXPECT_THAT(params[1].get(), ::testing::Pointwise(::testing::DoubleEq(), b));
}

// Pruning new parameters must not discard existing Q space history
TEST_F(QSpaceF, update_prunes_without_losing_history) {
  const R a{1, 0, 0}, c{0, 0, 1};
  update({c});
  update({a, a});
  EXPECT_EQ(qspace.size(), 2);
  check_consistent();
  const auto params = qspace.cparams();
  EXPECT_THAT(params[0].get(), ::testing::Pointwise(::testing::DoubleEq(), a));
  EXPECT_THAT(params[1].get(), ::testing::Pointwise(::testing::DoubleEq(), c));
}
