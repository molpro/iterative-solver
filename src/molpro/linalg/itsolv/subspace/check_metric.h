#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_CHECK_METRIC_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_CHECK_METRIC_H
#include <Eigen/Dense>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/scalar_traits.h>

#include <format>
#include <stdexcept>

namespace molpro::linalg::itsolv::subspace {

/*!
 * @brief Throws if the metric matrix of the subspace shows that the metric is not positive (semi)definite
 *
 * The metric matrix M_ij = <b_i|M b_j> of a positive definite metric M is positive definite for linearly independent
 * subspace vectors b_i, which a Cholesky factorisation confirms cheaply. Where it fails, which nearly linearly dependent
 * subspace vectors can cause as well, the Rayleigh quotients of M on the span of the subspace vectors decide. They are
 * the eigenvalues of the metric matrix in the basis orthonormalised with the overlap S_ij = <b_i|b_j>, leaving out
 * nearly linearly dependent directions, and so do not depend on how well conditioned the basis is. The metric fails
 * the check if one of them is negative by more than a small fraction of the largest.
 *
 * An indefinite metric is noticed only once one of its negative directions enters the subspace.
 * @param metric the metric matrix of the subspace
 * @param overlap the overlap matrix of the subspace vectors
 */
template <typename T>
void check_metric_positive(const Matrix<T>& metric, const Matrix<T>& overlap) {
  using real_t = real_type_t<T>;
  using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using row_major_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const auto n = Eigen::Index(metric.rows());
  if (n == 0)
    return;
  const matrix_type m_rows = Eigen::Map<const row_major_type>(metric.data().data(), n, n);
  const matrix_type m = (m_rows + m_rows.adjoint()) / T(2);
  if (Eigen::LLT<matrix_type>(m).info() == Eigen::Success)
    return;
  const matrix_type s = Eigen::Map<const row_major_type>(overlap.data().data(), n, n);
  const Eigen::SelfAdjointEigenSolver<matrix_type> s_eigen(s);
  const real_t s_max = s_eigen.eigenvalues().maxCoeff();
  const real_t s_threshold = precision_scaled<T>(1e-10) * s_max;
  Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < n; ++i)
    if (s_eigen.eigenvalues()(i) > s_threshold)
      ++rank;
  if (rank == 0)
    return;
  // the eigenvalues come in ascending order, so the retained directions are the trailing ones
  const auto retained = s_eigen.eigenvectors().rightCols(rank);
  const matrix_type orthonormal =
      retained * s_eigen.eigenvalues().tail(rank).cwiseSqrt().cwiseInverse().template cast<T>().asDiagonal();
  const Eigen::SelfAdjointEigenSolver<matrix_type> m_eigen(orthonormal.adjoint() * m * orthonormal,
                                                           Eigen::EigenvaluesOnly);
  const real_t lowest = m_eigen.eigenvalues().minCoeff();
  const real_t largest = m_eigen.eigenvalues().cwiseAbs().maxCoeff();
  if (lowest < -precision_scaled<T>(1e-8) * largest)
    throw std::runtime_error(std::format("The metric is not positive definite: it has the Rayleigh quotient {:.3e} on "
                                         "the subspace, against a largest one of {:.3e}",
                                         double(lowest), double(largest)));
}

} // namespace molpro::linalg::itsolv::subspace

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_CHECK_METRIC_H
