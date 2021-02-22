#include "DistrArrayMPI3.h"
namespace molpro::linalg::array {

namespace {
int comm_size(MPI_Comm comm) {
  int res;
  MPI_Comm_size(comm, &res);
  return res;
}

int comm_rank(MPI_Comm comm) {
  int res;
  MPI_Comm_rank(comm, &res);
  return res;
}
} // namespace
} // namespace molpro::linalg::array

#include "DistrArrayMPI3-implementation.h"
template class molpro::linalg::array::DistrArrayMPI3<double>;
template void molpro::linalg::array::swap(molpro::linalg::array::DistrArrayMPI3<double> &a1, molpro::linalg::array::DistrArrayMPI3<double> &a2);

