#include "DistrArrayDisk.h"

namespace molpro::linalg::array {
using util::Task;
namespace {

int mpi_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

} // namespace
} // namespace molpro::linalg::array

#include "DistrArrayDisk-implementation.h"
template class molpro::linalg::array::DistrArrayDisk<double>;
