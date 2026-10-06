#include <gtest/gtest.h>

#include <molpro/linalg/itsolv/subspace/Dimensions.h>

using molpro::linalg::itsolv::subspace::Dimensions;

TEST(Dimensions, derived_from_sizes) {
  const auto dims = Dimensions(2, 3, 4);
  EXPECT_EQ(dims.nX(), 9);
  EXPECT_EQ(dims.oP(), 0);
  EXPECT_EQ(dims.oQ(), 2);
  EXPECT_EQ(dims.oD(), 5);
}

// Changing a block size after construction must be reflected in the total size and offsets
TEST(Dimensions, derived_follow_mutation) {
  auto dims = Dimensions(2, 3, 4);
  dims.nP = 1;
  dims.nQ = 5;
  EXPECT_EQ(dims.nX(), 10);
  EXPECT_EQ(dims.oQ(), 1);
  EXPECT_EQ(dims.oD(), 6);
}
