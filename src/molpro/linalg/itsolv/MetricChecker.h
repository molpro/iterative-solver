#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_METRICCHECKER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_METRICCHECKER_H
#include <Eigen/Dense>
#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/itsolv/ArrayHandlers.h>
#include <molpro/linalg/itsolv/subspace/IXSpace.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/linalg/scalar_traits.h>

#include <cmath>
#include <format>
#include <optional>
#include <stdexcept>
#include <utility>

namespace molpro::linalg::itsolv::detail {

/*!
 * @brief Checks that the metric of a generalised problem has the properties that its kind declares, as far as the
 * subspace shows them
 *
 * All checks throw std::runtime_error if they fail. They compare against tolerance() relative to the size of the
 * quantities involved.
 */
template <class R, class Q, class P>
class MetricChecker {
public:
  using value_type = typename array::ArrayHandler<R, R>::value_type;
  using value_type_abs = typename array::ArrayHandler<R, R>::value_type_abs;

  //! The relative tolerance of all checks, calibrated for double precision and rescaled to the working precision
  static value_type_abs tolerance() { return precision_scaled<value_type>(1e-8); }

  //! Forgets the vectors kept for the check of Hermiticity, as when the subspace starts anew
  void reset() { m_reference.reset(); }

  /*!
   * @brief Checks that the metric is Hermitian on new parameters before they are added to the subspace
   *
   * The metric matrix is calculated from the actions of the metric on new parameters only, assuming that it is
   * Hermitian. This compares <x|M q> with <M x|q> for the new parameters x and a vector q of an earlier update: the
   * most recent Q vector if the subspace keeps the actions of the metric, otherwise the first parameter of the
   * previous update, kept by remember().
   * @param params the new parameters x
   * @param metric_actions the actions of the metric on params
   * @param xspace the subspace, before the update
   * @param handlers array handlers
   */
  void check_hermitian(const CVecRef<R>& params, const CVecRef<R>& metric_actions,
                       const subspace::IXSpace<R, Q, P>& xspace, ArrayHandlers<R, Q, P>& handlers) const {
    const Q* q = nullptr;
    const Q* metric_action_q = nullptr;
    if (xspace.stores_metric_actions()) {
      if (xspace.dimensions().nQ == 0)
        return;
      q = &xspace.cparamsq().front().get();
      metric_action_q = &xspace.cmetric_actionsq().front().get();
    } else if (m_reference) {
      q = &m_reference->first;
      metric_action_q = &m_reference->second;
    } else
      return;
    const auto norm_q = std::sqrt(std::abs(handlers.qq().dot(*q, *q)));
    const auto norm_metric_action_q = std::sqrt(std::abs(handlers.qq().dot(*metric_action_q, *metric_action_q)));
    for (size_t i = 0; i < params.size(); ++i) {
      const auto xmq = handlers.rq().dot(params[i], *metric_action_q);
      const auto mxq = handlers.rq().dot(metric_actions[i], *q);
      const auto norm_x = std::sqrt(std::abs(handlers.rr().dot(params[i], params[i])));
      const auto norm_metric_action_x = std::sqrt(std::abs(handlers.rr().dot(metric_actions[i], metric_actions[i])));
      const auto scale = (norm_x * norm_metric_action_q + norm_metric_action_x * norm_q) / 2;
      if (std::abs(xmq - mxq) > tolerance() * scale)
        throw std::runtime_error(std::format("The metric is not Hermitian: <x|M q> = {:.6e} differs from <M x|q> = "
                                             "{:.6e} for a new parameter x and a subspace vector q",
                                             double(std::abs(xmq)), double(std::abs(mxq))));
    }
  }

  /*!
   * @brief Keeps the first new parameter and its action of the metric for the next check_hermitian(), if the subspace
   * does not keep the actions of the metric
   */
  void remember(const CVecRef<R>& params, const CVecRef<R>& metric_actions, const subspace::IXSpace<R, Q, P>& xspace,
                ArrayHandlers<R, Q, P>& handlers) {
    if (xspace.stores_metric_actions() || params.empty())
      return;
    m_reference.emplace(handlers.qr().copy(params.front()), handlers.qr().copy(metric_actions.front()));
  }

  /*!
   * @brief Checks that the metric matrix in the P space is Hermitian
   * @param matrix the matrix, with the second index running fastest
   * @param n its dimension
   */
  static void check_hermitian(const array::Span<value_type>& matrix, const size_t n) {
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < i; ++j)
        if (std::abs(matrix[i * n + j] - conjugate(matrix[j * n + i])) >
            tolerance() * std::sqrt(std::abs(matrix[i * n + i] * matrix[j * n + j])))
          throw std::runtime_error("The metric is not Hermitian: its matrix in the P space is not");
  }

  /*!
   * @brief Checks that the metric matrix of the subspace is consistent with a positive (semi)definite metric
   *
   * The metric matrix M_ij = <b_i|M b_j> of a positive definite metric M is positive definite for linearly independent
   * subspace vectors b_i, which a Cholesky factorisation confirms cheaply. Where it fails, which nearly linearly
   * dependent subspace vectors can cause as well, the Rayleigh quotients of M on the span of the subspace vectors
   * decide. They are the eigenvalues of the metric matrix in the basis orthonormalised with the overlap
   * S_ij = <b_i|b_j>, leaving out nearly linearly dependent directions. Orthonormalising amplifies the rounding errors
   * of the metric matrix by up to the inverse of the smallest eigenvalue of the overlap that is kept, so the metric
   * fails the check only if a Rayleigh quotient is negative by more than tolerance() times the largest plus these
   * amplified rounding errors.
   *
   * An indefinite metric is noticed only once one of its negative directions enters the subspace.
   * @param metric the metric matrix of the subspace
   * @param overlap the overlap matrix of the subspace vectors
   */
  static void check_positive(const subspace::Matrix<value_type>& metric, const subspace::Matrix<value_type>& overlap) {
    using matrix_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;
    using row_major_type = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const auto n = Eigen::Index(metric.rows());
    if (n == 0)
      return;
    const matrix_type m_rows = Eigen::Map<const row_major_type>(metric.data().data(), n, n);
    const matrix_type m = (m_rows + m_rows.adjoint()) / value_type(2);
    if (Eigen::LLT<matrix_type>(m).info() == Eigen::Success)
      return;
    const matrix_type s = Eigen::Map<const row_major_type>(overlap.data().data(), n, n);
    const Eigen::SelfAdjointEigenSolver<matrix_type> s_eigen(s);
    const value_type_abs s_threshold = precision_scaled<value_type>(1e-10) * s_eigen.eigenvalues().maxCoeff();
    Eigen::Index rank = 0;
    for (Eigen::Index i = 0; i < n; ++i)
      if (s_eigen.eigenvalues()(i) > s_threshold)
        ++rank;
    if (rank == 0)
      return;
    // the eigenvalues come in ascending order, so the retained directions are the trailing ones
    const auto retained = s_eigen.eigenvectors().rightCols(rank);
    const matrix_type orthonormal =
        retained * s_eigen.eigenvalues().tail(rank).cwiseSqrt().cwiseInverse().template cast<value_type>().asDiagonal();
    const Eigen::SelfAdjointEigenSolver<matrix_type> m_eigen(orthonormal.adjoint() * m * orthonormal,
                                                             Eigen::EigenvaluesOnly);
    const value_type_abs lowest = m_eigen.eigenvalues().minCoeff();
    const value_type_abs largest = m_eigen.eigenvalues().cwiseAbs().maxCoeff();
    // the rounding errors of the elements of the metric matrix, amplified by the orthonormalisation
    const value_type_abs rounding = precision_scaled<value_type>(1e-13) * m.norm() / s_eigen.eigenvalues()(n - rank);
    if (lowest < -tolerance() * largest - rounding)
      throw std::runtime_error(std::format("The metric is not positive definite: it has the Rayleigh quotient {:.3e} "
                                           "on the subspace, against a largest one of {:.3e}",
                                           double(lowest), double(largest)));
  }

private:
  std::optional<std::pair<Q, Q>> m_reference; //!< a parameter and its action of the metric, see remember()
};

} // namespace molpro::linalg::itsolv::detail

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_METRICCHECKER_H
