#ifdef LINEARALGEBRA_ARRAY_GA
#include "DistrArrayGA.h"

namespace molpro::linalg::array {

namespace {
int get_communicator_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

int get_communicator_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

} // namespace
} // namespace molpro::linalg::array
#include "DistrArrayGA-implementation.h"
template class molpro::linalg::array::DistrArrayGA<double>;
template void molpro::linalg::array::swap(molpro::linalg::array::DistrArrayGA<double> &a1, molpro::linalg::array::DistrArrayGA<double> &a2);

#endif