#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_METRIC_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_METRIC_H

namespace molpro::linalg::itsolv {

/*!
 * @brief The kind of metric M of a generalised problem, such as the eigenproblem A v = lambda M v
 *
 * Which kinds are supported depends on the solver, see IterativeSolver::supports_metric().
 */
enum class MetricKind {
  Identity,                      //!< M = 1, i.e. no metric
  HermitianPositiveDefinite,     //!< Hermitian and positive definite, e.g. the overlap matrix of a non-orthogonal basis
  HermitianPositiveSemidefinite, //!< Hermitian and positive semidefinite, i.e. possibly singular
  HermitianIndefinite,           //!< Hermitian and indefinite, e.g. in RPA, TDHF or Bethe-Salpeter equations
};

//! How a solver obtains the actions of a metric on its subspace vectors, see IterativeSolver::set_metric()
enum class MetricStorage {
  Stored,   //!< the actions of the metric on all subspace vectors are kept
  OnTheFly, //!< the metric is applied to the solutions whenever residuals are constructed
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_METRIC_H
