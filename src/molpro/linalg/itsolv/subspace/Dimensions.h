#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_DIMENSIONS_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_DIMENSIONS_H

#include <cstddef>

namespace molpro::linalg::itsolv::subspace {
//! Stores partitioning of XSpace into P, Q and D blocks with sizes and offsets for each one
//! The total size and the offsets are computed from the block sizes, so they cannot go stale when a size changes.
struct Dimensions {
  Dimensions() = default;
  Dimensions(size_t np, size_t nq, size_t nc) : nP(np), nQ(nq), nD(nc) {}
  size_t nP = 0;
  size_t nQ = 0;
  size_t nD = 0;
  size_t nRHS = 0; //!< number of rigt-hand-side vectors in the system of linear equations
  size_t nX() const { return nP + nQ + nD; } //!< total size of the X space
  size_t oP() const { return 0; }            //!< offset of the P block
  size_t oQ() const { return nP; }           //!< offset of the Q block
  size_t oD() const { return nP + nQ; }      //!< offset of the D block
};
} // namespace molpro::linalg::itsolv::subspace

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_DIMENSIONS_H
