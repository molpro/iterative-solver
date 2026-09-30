#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_SUBSPACEDATA_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_SUBSPACEDATA_H
#include <map>
#include <molpro/linalg/itsolv/subspace/Matrix.h>

namespace molpro::linalg::itsolv::subspace {
enum class EqnData { H, S, rhs, value };

//! Equation data blocks of the subspace, in the scalar type of the problem
template <typename T = double>
using SubspaceData = std::map<EqnData, Matrix<T>>;

template <typename T = double, EqnData... DataTypes>
auto null_data() {
  return SubspaceData<T>{std::make_pair<EqnData, Matrix<T>>(DataTypes, {})...};
}
} // namespace molpro::linalg::itsolv::subspace

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_SUBSPACEDATA_H