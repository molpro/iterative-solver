#include "testDistrArray.h"

#include <molpro/linalg/array/DistrArrayGA.h>

using DistrArrayGA = molpro::linalg::array::DistrArrayGA<double>;

using ArrayTypes = ::testing::Types<DistrArrayGA>;
INSTANTIATE_TYPED_TEST_SUITE_P(GA, TestDistrArray, ArrayTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(GA, DistArrayInitializationF, ArrayTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(GA, DistrArrayRangeF, ArrayTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(GA, DistrArrayCollectiveOpF, ArrayTypes);
