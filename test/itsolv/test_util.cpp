#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/util.h>

using molpro::linalg::itsolv::util::construct_zeroed_copy;
using molpro::linalg::itsolv::util::is_iota;

TEST(itsolv_util, is_iota_null) {
  auto vec = std::vector<int>{};
  ASSERT_TRUE(is_iota(vec.begin(), vec.end(), 0));
}

TEST(itsolv_util, is_iota_true) {
  auto vec = std::vector<int>{1, 2, 3, 4, 5};
  ASSERT_TRUE(is_iota(vec.begin(), vec.end(), 1));
}

TEST(itsolv_util, is_iota_false) {
  auto vec = std::vector<int>{1, 2, 3, 4, 5};
  ASSERT_FALSE(is_iota(vec.begin(), vec.end(), 0));
  auto vec2 = std::vector<int>{1, 3, 4, 5};
  ASSERT_FALSE(is_iota(vec2.begin(), vec2.end(), vec2[0]));
  auto vec3 = std::vector<int>{3, 2, 1};
  ASSERT_FALSE(is_iota(vec3.begin(), vec3.end(), vec3[0]));
}

TEST(itsolv_util, construct_zeroed_copy) {
  using R = std::vector<double>;
  using Q = std::list<double>;
  auto handler = molpro::linalg::array::create_default_handler<Q, R>();
  auto r = R(10);
  std::iota(begin(r), end(r), 0);
  auto q = construct_zeroed_copy(r, *handler);
  ASSERT_EQ(q.size(), r.size());
  ASSERT_THAT(q, ::testing::Each(::testing::DoubleEq(0)));
}