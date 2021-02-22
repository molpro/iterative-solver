#include "testDistrArray.h"

#include <molpro/linalg/array/DistrArrayMPI3.h>

using molpro::linalg::array::DistrArrayMPI3;
using molpro::linalg::array::util::LockMPI3;

using ArrayTypes = ::testing::Types<DistrArrayMPI3<double>>;
INSTANTIATE_TYPED_TEST_SUITE_P(MPI3, TestDistrArray, ArrayTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(MPI3, DistArrayInitializationF, ArrayTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(MPI3, DistrArrayRangeF, ArrayTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(MPI3, DistrArrayCollectiveOpF, ArrayTypes);

TEST(DistrArrayMPI3, allocate_buffer_external) {
  const size_t dim = 30;
  auto buffer = std::vector<double>(dim);
  LockMPI3 lock(mpi_comm);
  auto a = DistrArrayMPI3<double>(dim, mpi_comm);
  a.allocate_buffer({&buffer[0], buffer.size()});
  {
    auto l = lock.scope();
    ASSERT_FALSE(a.empty());
    auto loc_buffer = a.local_buffer();
    ASSERT_EQ(&(*loc_buffer)[0], &buffer[0]);
  }
  a.free_buffer();
  {
    auto l = lock.scope();
    ASSERT_TRUE(a.empty());
  }
}