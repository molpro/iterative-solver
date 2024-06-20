#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "parallel_util.h"
#include <molpro/linalg/array/util.h>
#include <molpro/linalg/array/util/DistrFlags.h>

#include <mpi.h>

using molpro::linalg::array::util::DistrFlags;
using molpro::linalg::array::util::ScopeLock;
using molpro::linalg::test::mpi_comm;

namespace {
int mpi_size() {
  int size;
  MPI_Comm_size(mpi_comm, &size);
  return size;
}
int mpi_rank() {
  int rank;
  MPI_Comm_rank(mpi_comm, &rank);
  return rank;
}
} // namespace

TEST(DistrFlags, MPI_RMA) {

  std::cout << "mpi_size: "<<mpi_size()<<std::endl;
  std::cout << "mpi_rank: "<<mpi_rank()<<std::endl;
  int* a;
  MPI_Win win;
  /* collectively create remote accessible memory in a window */
  MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, mpi_comm, &a, &win);
  /* Array ‘a’ is now accessible from all processes in
   * mpi_comm */
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
  int val1 = 42;
  MPI_Put(&val1, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
  int val2 = 12345;
  MPI_Get(&val2, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
  ASSERT_EQ(val1, val2);
  MPI_Get(&val2, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
  ASSERT_EQ(val1, val2);
  int val3 = 84, res = 9999;
  auto code = MPI_Fetch_and_op(&val3, &res, MPI_INT, 0, 0, MPI_REPLACE, win); // this seems not always to work
  ASSERT_EQ(code, MPI_SUCCESS);
  ASSERT_EQ(val1, res);
  MPI_Win_unlock(0, win);
  MPI_Win_free(&win);
}

TEST(DistrFlags, constructor_default) {
  DistrFlags df{};
  ScopeLock l{mpi_comm};
  ASSERT_TRUE(df.empty());
}

TEST(DistrFlags, constructor_comm) {
  DistrFlags df{mpi_comm};
  ScopeLock l{mpi_comm};
  ASSERT_FALSE(df.empty());
}

TEST(DistrFlags, constructor_comm_value) {
  DistrFlags df{mpi_comm, 0};
  ScopeLock l{mpi_comm};
  ASSERT_FALSE(df.empty());
}

TEST(DistrFlags, constructor_copy) {
  DistrFlags df{mpi_comm};
  DistrFlags g{df};
  ScopeLock l{mpi_comm};
  ASSERT_EQ(g.empty(), df.empty());
  ASSERT_EQ(g.communicator(), df.communicator());
}

TEST(DistrFlags, constructor_move) {
  DistrFlags&& df{mpi_comm};
  DistrFlags g{std::move(df)};
  ScopeLock l{mpi_comm};
  ASSERT_TRUE(df.empty());
  ASSERT_FALSE(g.empty());
  ASSERT_EQ(g.communicator(), mpi_comm);
}

TEST(DistrFlags, copy_assignment) {
  DistrFlags df{mpi_comm};
  DistrFlags g{};
  g = df;
  ScopeLock l{mpi_comm};
  ASSERT_EQ(g.empty(), df.empty());
  ASSERT_EQ(g.communicator(), df.communicator());
}

TEST(DistrFlags, move_assignment) {
  DistrFlags&& df{mpi_comm};
  DistrFlags g{};
  g = std::move(df);
  ScopeLock l{mpi_comm};
  ASSERT_TRUE(df.empty());
  ASSERT_FALSE(g.empty());
  ASSERT_EQ(g.communicator(), mpi_comm);
}

TEST(DistrFlags, access) {
  DistrFlags df{mpi_comm};
  auto p = df.access();
}

TEST(DistrFlags, access_rank) {
  auto rank = mpi_rank();
  auto size = mpi_size();
  DistrFlags df{mpi_comm};
  auto r = rank != size - 1 ? rank + 1 : 0;
  auto p = df.access(r);
}

TEST(DistrFlags, get) {
  auto rank = mpi_rank();
  int test_value{42};
  DistrFlags df{mpi_comm, test_value};
  auto v = df.access().get();
  ScopeLock l{mpi_comm};
  ASSERT_EQ(v, test_value);
}

TEST(DistrFlags, replace) {
  const int alpha = 42;
  auto rank = mpi_rank();
  DistrFlags df{mpi_comm, alpha};
  auto a = df.access(0);
  //  MPI_Win_flush(m_rank, m_win);
  ASSERT_EQ(a.get(), alpha);
  auto v = a.replace(rank);
  ASSERT_EQ(v, alpha);
  v = a.replace(alpha);
  ASSERT_EQ(v, rank);
}

struct DistrFlagsF : DistrFlags, ::testing::Test {
  DistrFlagsF() : DistrFlags{mpi_comm, 0} {}
};

TEST_F(DistrFlagsF, counter) {
  ASSERT_EQ(*m_counter, 0);
  {
    auto a = access();
    ASSERT_EQ(*m_counter, 1);
    {
      auto a = access();
      ASSERT_EQ(*m_counter, 2);
      {
        auto a = access();
        ASSERT_EQ(*m_counter, 3);
      }
      ASSERT_EQ(*m_counter, 2);
    }
    ASSERT_EQ(*m_counter, 1);
  }
  ASSERT_EQ(*m_counter, 0);
}

// struct DistrFlagsF : DistrFlags, ::testing::Test {
//  DistrFlagsF() : DistrFlags(mpi_comm, value) {}
//
//  static constexpr int value = 1;
//};
