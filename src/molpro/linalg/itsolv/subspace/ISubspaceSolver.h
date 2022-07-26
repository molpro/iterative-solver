#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_ISUBSPACESOLVER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_ISUBSPACESOLVER_H
#include <molpro/linalg/array/ArrayHandler.h>
#include <molpro/linalg/itsolv/subspace/IXSpace.h>

namespace molpro::linalg::itsolv::subspace {

/*!
 * @brief Manages solution of the subspace problem and storage of those solutions
 *
 * Iterative solvers can have different ways to solve the subspace problem, e.g. dense diagonalisation for eigenvalue
 * problem, and residual minimisation in DIIS. They should inherit from this class to have consistent interface.
 */
template <class RT, class QT, class PT>
struct ISubspaceSolver {
  using R = RT;
  using Q = QT;
  using P = PT;
  using value_type = typename array::ArrayHandler<R, R>::value_type;
  using value_type_abs = typename array::ArrayHandler<R, R>::value_type_abs;
  virtual ~ISubspaceSolver() = default;

  /*!
   * @brief Solve the subspace problem
   * @param xspace definition of the subspace
   * @param nroots_max maximum number of roots to calculate
   */
  virtual void solve(IXSpace<R, Q, P>& xspace, size_t nroots_max) = 0;

  /*!
   * @brief Update the error associated with a given root
   * @param root solution index
   * @param error value of the error
   */
  virtual void set_error(int root, value_type_abs error) = 0;
  /*!
   * @brief Update errors for a group of roots
   * @param roots group of roots
   * @param errors errors corresponding to each root
   */
  virtual void set_error(const std::vector<int>& roots, const std::vector<value_type_abs>& errors) = 0;
  //! Access solutions from the last solve() call
  virtual const Matrix<value_type>& solutions() const = 0;
  //! Access eigenvalues from the last solve() call
  virtual const std::vector<value_type>& eigenvalues() const = 0;
  //! Access errors corresponding to each solution
  virtual const std::vector<value_type_abs>& errors() const = 0;
  //! Number of solutions stored in this object
  virtual size_t size() const = 0;
};

} // namespace molpro::linalg::itsolv::subspace
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_ISUBSPACESOLVER_H
