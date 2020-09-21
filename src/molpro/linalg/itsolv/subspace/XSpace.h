#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_XSPACE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_XSPACE_H
#include <molpro/linalg/itsolv/subspace/SubspaceData.h>

namespace molpro {
namespace linalg {
namespace itsolv {
namespace subspace {

//! Base class for XSpace solvers
template <class Rs, class Qs, class Ps>
class XSpace {
public:
  using R = typename Rs::R;
  using Q = typename Qs::Q;
  using P = typename Ps::P;
  using RS = Rs;
  using QS = Qs;
  using PS = Ps;

  SubspaceData data = null_data<EqnData::H, EqnData::S>();

  //! Ensures that the subspace is well conditioned.
  virtual void check_conditioning(RS& rs, QS& qs, PS& ps) = 0;

  //! Solves the underlying problem in the subspace. solver can be used to pass extra parameters defining the problem
  virtual void solve(const IterativeSolver<R, Q, P>& solver) = 0;

  //! Access solution corresponding to root i
  virtual const std::vector<double>& solution(size_t i) const = 0;

  //! Number of vectors forming the subspace
  size_t size() { return data.at(EqnData::H).rows(); }

  //! Build the subspace matrices H, S etc.
  virtual void build_subspace(RS& rs, QS& qs, PS& ps) = 0;
};

} // namespace subspace
} // namespace itsolv
} // namespace linalg
} // namespace molpro

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_XSPACE_H