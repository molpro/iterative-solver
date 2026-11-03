#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_SUBSPACEDATA_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_SUBSPACEDATA_H
#include <map>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <vector>

namespace molpro::linalg::itsolv::subspace {
/*!
 * @brief Kinds of equation data in the subspace
 *
 * H is the kernel matrix and S the overlap of the subspace vectors. For a generalised eigenproblem, M is the metric
 * matrix, which then replaces S in the subspace eigenproblem; S remains the overlap used for managing the subspace.
 */
enum class EqnData { H, S, rhs, value, M };

//! Equation data blocks of the subspace, in the scalar type of the problem
template <typename T = double>
using SubspaceData = std::map<EqnData, Matrix<T>>;

//! The square matrices, H, S and, if present, M, among the equation data
template <typename T>
std::vector<EqnData> square_blocks(const SubspaceData<T>& data) {
  auto blocks = std::vector<EqnData>{EqnData::H, EqnData::S};
  if (data.count(EqnData::M))
    blocks.push_back(EqnData::M);
  return blocks;
}

template <typename T = double, EqnData... DataTypes>
auto null_data() {
  return SubspaceData<T>{std::make_pair<EqnData, Matrix<T>>(DataTypes, {})...};
}
} // namespace molpro::linalg::itsolv::subspace

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_SUBSPACEDATA_H