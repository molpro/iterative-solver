#include "DistrArrayHDF5.h"
namespace molpro::linalg::array {
namespace {
int mpi_size(MPI_Comm comm) {
  int rank;
  MPI_Comm_size(comm, &rank);
  return rank;
}
} // namespace
} // namespace molpro::linalg::array

#include "DistrArrayHDF5-implementation.h"
template class molpro::linalg::array::DistrArrayHDF5<double>;
template void molpro::linalg::array::swap(molpro::linalg::array::DistrArrayHDF5<double> &a1, molpro::linalg::array::DistrArrayHDF5<double> &a2);
