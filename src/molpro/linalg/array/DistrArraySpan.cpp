#include "DistrArraySpan.h"

namespace molpro::linalg::array {

namespace {
int mpi_size(MPI_Comm comm) {
  int rank;
  MPI_Comm_size(comm, &rank);
  return rank;
}
} // namespace

} // namespace molpro::linalg::array
#include "DistrArraySpan-implementation.h"
template class molpro::linalg::array::DistrArraySpan<double>;
template void molpro::linalg::array::swap(molpro::linalg::array::DistrArraySpan<double> &a1, molpro::linalg::array::DistrArraySpan<double> &a2);
