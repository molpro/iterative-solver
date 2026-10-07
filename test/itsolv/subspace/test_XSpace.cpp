#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/subspace/XSpace.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <map>
#include <memory>
#include <vector>

using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::cwrap;
using molpro::linalg::itsolv::Logger;
using molpro::linalg::itsolv::wrap;
using molpro::linalg::itsolv::subspace::EqnData;
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
