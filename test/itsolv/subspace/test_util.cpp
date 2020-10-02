#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/subspace/check_conditioning.h>
#include <molpro/linalg/itsolv/subspace/gram_schmidt.h>
#include <molpro/linalg/itsolv/subspace/util.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <numeric>

using molpro::linalg::itsolv::ArrayHandlers;
using molpro::linalg::itsolv::wrap;
using molpro::linalg::itsolv::subspace::Matrix;
using molpro::linalg::itsolv::subspace::detail::gram_schmidt;
using molpro::linalg::itsolv::subspace::util::eye_order;
using molpro::linalg::itsolv::subspace::util::overlap;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Pointwise;

struct OverlapF : ::testing::Test {
  using R = std::vector<double>;
  using Q = R;
  using P = std::map<size_t, double>;
  OverlapF() : alphas({1, 2, 3}) {
    x.resize(alphas.size());
    for (size_t i = 0; i < alphas.size(); ++i) {
      x[i].assign(nx, alphas[i]);
    }
    ref_overlap.resize({x.size(), x.size()});
    for (size_t i = 0; i < alphas.size(); ++i) {
      for (size_t j = 0; j < alphas.size(); ++j) {
        ref_overlap(i, j) = nx * alphas[i] * alphas[j];
      }
    }
  }
  std::vector<R> x;
  const size_t nx = 5;
  const std::vector<double> alphas;
  Matrix<double> ref_overlap;
  ArrayHandlers<R, Q, P> handler;
};

TEST_F(OverlapF, null_vectors) {
  auto m = overlap<R, R>({}, {}, handler.rr());
  ASSERT_TRUE(m.empty());
}

TEST_F(OverlapF, overlap_one_param) {
  auto m = overlap(wrap(x), handler.rr());
  ASSERT_THAT(m.data(), Pointwise(DoubleEq(), ref_overlap.data()));
}

TEST_F(OverlapF, overlap_two_params) {
  auto m = overlap(wrap(x), wrap(x), handler.rr());
  ASSERT_THAT(m.data(), Pointwise(DoubleEq(), ref_overlap.data()));
}

TEST(EyeOrder, null) {
  auto m = Matrix<double>{};
  auto order = eye_order(m);
  ASSERT_TRUE(order.empty());
}

TEST(EyeOrder, identity) {
  auto&& data = std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  auto m = Matrix<double>{std::move(data), {3, 3}};
  auto order = eye_order(m);
  const auto reference = std::vector<size_t>{0, 1, 2};
  ASSERT_FALSE(order.empty());
  ASSERT_THAT(order, Pointwise(Eq(), reference)) << "matrix is already identity";
}

TEST(EyeOrder, identity_circular_shift) {
  auto&& data = std::vector<double>{0, 1, 0, 0, 0, 1, 1, 0, 0};
  auto m = Matrix<double>{std::move(data), {3, 3}};
  auto order = eye_order(m);
  const auto reference = std::vector<size_t>{2, 0, 1};
  ASSERT_FALSE(order.empty());
  ASSERT_THAT(order, Pointwise(Eq(), reference));
}

TEST(EyeOrder, real_scenario) {
  auto&& data = std::vector<double>{0.1, 0.5, 0.2, 0.2, 0.1, 0.5, 0.5, 0.2, 0.1};
  auto m = Matrix<double>{std::move(data), {3, 3}};
  auto order = eye_order(m);
  const auto reference = std::vector<size_t>{2, 0, 1};
  ASSERT_FALSE(order.empty());
  ASSERT_THAT(order, Pointwise(Eq(), reference));
}

TEST(gram_schmidt, null) {
  auto s = Matrix<double>{};
  auto t = Matrix<double>{};
  auto result = gram_schmidt(s, t);
  ASSERT_TRUE(result.empty());
}

TEST(gram_schmidt, s_3x3) {
  const size_t n = 3;
  auto s = Matrix<double>{std::vector<double>{14, 25, 31, 25, 45, 56, 31, 56, 70}, {n, n}};
  auto t = Matrix<double>{};
  auto tref = Matrix<double>{std::vector<double>{1., 0., 0., -25. / 14., 1., 0., 1., -9. / 5., 1.}, {n, n}};
  auto norm_ref = std::vector<double>{std::sqrt(14.), std::sqrt(5. / 14.), std::sqrt(1. / 5.)};
  auto result = gram_schmidt(s, t);
  ASSERT_EQ(result.size(), n);
  ASSERT_EQ(t.rows(), n);
  ASSERT_EQ(t.cols(), n);
  for (size_t i = 0; i < t.rows(); ++i)
    for (size_t j = i + 1; j < t.cols(); ++j)
      ASSERT_DOUBLE_EQ(t(i, j), 0.) << " Uppert triangular elements must be zero , i=" << std::to_string(i) << " "
                                    << std::to_string(j);
  ASSERT_THAT(t.data(), Pointwise(DoubleNear(1.0e-14), tref.data()));
  ASSERT_THAT(result, Pointwise(DoubleNear(1.0e-13), norm_ref));
}
