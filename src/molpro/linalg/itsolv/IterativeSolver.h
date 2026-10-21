#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
#include <molpro/linalg/array/ArrayHandler.h>
#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/itsolv/Options.h>
#include <molpro/linalg/itsolv/Statistics.h>
#include <molpro/linalg/itsolv/helper.h>
#include <molpro/linalg/itsolv/subspace/Dimensions.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/Metric.h>

#include <molpro/linalg/array/DistrArray.h>
#include <molpro/linalg/array/util/Distribution.h>

#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <vector>

namespace molpro::profiler {
class Profiler;
}
namespace molpro::linalg::itsolv {

template <typename T>
struct has_iterator {
  template <typename C>
  constexpr static std::true_type test(typename C::iterator*);

  template <typename>
  constexpr static std::false_type test(...);

  constexpr static bool value =
      std::is_same<std::true_type, decltype(test<typename std::remove_reference<T>::type>(0))>::value;
};

namespace detail {
// Divides each action[k] elementwise by (diagonals - shift[k] * metric_diagonals), where metric_diagonals == nullptr
// stands for a unit metric; if regularise, a small constant is added to the denominator
template <typename T, typename S,
          typename = std::enable_if_t<std::is_base_of<molpro::linalg::array::DistrArray, T>::value>>
void precondition_default(const VecRef<T>& action, const std::vector<S>& shift, const T& diagonals,
                          const T* metric_diagonals, const bool regularise) {
  // regularisation of the denominator, calibrated for double precision and rescaled to the working precision
  const auto regulariser = regularise ? precision_scaled<S>(1e-15) : decltype(precision_scaled<S>(1e-15))(0);
  auto diagonals_local_buffer = diagonals.local_buffer();
  auto metric_local_buffer = metric_diagonals ? metric_diagonals->local_buffer() : nullptr;
  for (size_t k = 0; k < action.size(); k++) {
    auto action_local_buffer = action[k].get().local_buffer();
    auto& distribution = action[k].get().distribution();
    auto range = distribution.range(molpro::mpi::rank_global());
    for (auto i = range.first; i < range.second; i++) {
      const auto j = i - range.first;
      const auto scaled_shift = metric_local_buffer ? shift[k] * (*metric_local_buffer)[j] : shift[k];
      (*action_local_buffer)[j] /= ((*diagonals_local_buffer)[j] - scaled_shift + regulariser);
    }
  }
}

template <class T, typename S>
void precondition_default(const VecRef<T>& action, const std::vector<S>& shift, const T& diagonals,
                          const T* metric_diagonals, const bool regularise,
                          typename T::iterator* = nullptr // SFINAE
) {
  const auto regulariser = regularise ? precision_scaled<S>(1e-15) : decltype(precision_scaled<S>(1e-15))(0);
  for (size_t k = 0; k < action.size(); k++) {
    auto a = action[k].get().begin();
    auto m = metric_diagonals ? metric_diagonals->begin() : decltype(diagonals.begin()){};
    for (auto d = diagonals.begin(); d != diagonals.end(); ++d, ++a) {
      const auto scaled_shift = metric_diagonals ? shift[k] * *m++ : shift[k];
      *a /= (*d - scaled_shift + regulariser);
    }
  }
}

template <typename T, typename S,
          typename = std::enable_if_t<!std::is_base_of<molpro::linalg::array::DistrArray, T>::value>, class = void>
void precondition_default(const VecRef<T>& action, const std::vector<S>& shift, const T& diagonals,
                          const T* metric_diagonals, const bool regularise,
                          typename std::enable_if<!has_iterator<T>::value, void*>::type = nullptr // SFINAE
) {
  throw std::logic_error("Unimplemented preconditioner");
}
} // namespace detail

/*!
 * @brief Davidson preconditioner: divides each action[k] elementwise by (diagonals - shift[k])
 * @param action On entry, the residuals; on exit, the preconditioned residuals
 * @param shift The shift for each of the residuals
 * @param diagonals The diagonal elements of the kernel
 */
template <typename T, typename S>
void precondition_default(const VecRef<T>& action, const std::vector<S>& shift, const T& diagonals) {
  detail::precondition_default(action, shift, diagonals, static_cast<const T*>(nullptr), true);
}

/*!
 * @brief Davidson preconditioner for a generalised eigenproblem: divides each action[k] elementwise by
 * (diagonals - shift[k] * metric_diagonals)
 * @param action On entry, the residuals; on exit, the preconditioned residuals
 * @param shift The shift for each of the residuals
 * @param diagonals The diagonal elements of the kernel
 * @param metric_diagonals The diagonal elements of the metric
 */
template <typename T, typename S>
void precondition_default(const VecRef<T>& action, const std::vector<S>& shift, const T& diagonals,
                          const T& metric_diagonals) {
  detail::precondition_default(action, shift, diagonals, &metric_diagonals, true);
}

/*!
 * @example ExampleProblem.h
 * Simple example of a problem-defining class
 * @example ExampleProblemDistrArray.h
 * Example of a problem-defining class with distributed data
 */
/*!
 * @brief Abstract class defining the problem-specific interface for the simplified solver
 * interface to IterativeSolver
 * @tparam R the type of container for solutions and residuals
 */
template <typename R, typename P = std::map<size_t, typename R::value_type>>
class Problem {
public:
  Problem() = default;
  virtual ~Problem() = default;
  using container_t = R;
  using p_container_t = P;
  using value_t = typename R::value_type;

  /*!
   * @brief Calculate the residual vector. Used by non-linear solvers (NonLinearEquations,
   * Optimize) only.
   * @param parameters The trial solution for which the residual is to be calculated
   * @param residual The residual vector
   * @return In the case where the residual is an exact differential, the corresponding
   * function value. Used by Optimize but not NonLinearEquations.
   */
  virtual value_t residual(const R& parameters, R& residual) const { return 0; }

  /*!
   * @brief Calculate the action of the kernel matrix on a set of parameters. Used by
   * linear solvers, but not by the non-linear solvers (NonLinearEquations, Optimize).
   * @param parameters The trial solutions for which the action is to be calculated
   * @param action The action vectors
   */
  virtual void action(const CVecRef<R>& parameters, const VecRef<R>& action) const { return; }

  /*!
   * @brief Optionally provide the diagonal elements of the underlying kernel. If
   * implemented and returning true, the provided diagonals will be used by
   * IterativeSolver for preconditioning (and therefore the precondition() function does
   * not need to be implemented), and, in the case of linear problems, for selection of
   * the P space. Otherwise, preconditioning will be done with precondition(), and any P
   * space has to be provided manually.
   * @param d On exit, contains the diagonal elements
   * @return Whether diagonals have been provided.
   */
  virtual bool diagonals(container_t& d) const { return false; }

  /*!
   * @brief Apply preconditioning to a residual vector in order to predict a step towards
   * the solution
   * @param residual On entry, assumed to be the residual. On exit, the negative of the
   * predicted step.
   * @param shift When called from LinearEigensystem, contains the corresponding current
   * eigenvalue estimates for each of the parameter vectors in the set. All other solvers
   * pass a vector of zeroes.
   *
   * Used if diagonals() is not provided. For a generalised eigenproblem A v = lambda M v, the preconditioner should
   * approximate the inverse of A - shift M.
   */
  virtual void precondition(const VecRef<R>& residual, const std::vector<value_t>& shift) const { return; }

  /*!
   * @brief Apply preconditioning to a residual vector in order to predict a step towards
   * the solution
   * @param residual On entry, assumed to be the residual. On exit, the negative of the
   * predicted step.
   * @param shift When called from LinearEigensystem, contains the corresponding current
   * eigenvalue estimates for each of the parameter vectors in the set. All other solvers
   * pass a vector of zeroes.
   * @param diagonals The diagonal elements of the underlying kernel
   */
  virtual void precondition(const VecRef<R>& residual, const std::vector<value_t>& shift, const R& diagonals) const {
    precondition_default(residual, shift, diagonals);
  }

  /*!
   * @brief Apply preconditioning to a residual vector of a generalised eigenproblem. Used instead of precondition()
   * when both diagonals() and metric_diagonals() are provided.
   * @param residual On entry, assumed to be the residual. On exit, the negative of the
   * predicted step.
   * @param shift Contains the current eigenvalue estimates for each of the parameter vectors in the set.
   * @param diagonals The diagonal elements of the underlying kernel
   * @param metric_diagonals The diagonal elements of the metric
   */
  virtual void precondition_with_metric(const VecRef<R>& residual, const std::vector<value_t>& shift,
                                        const R& diagonals, const R& metric_diagonals) const {
    precondition_default(residual, shift, diagonals, metric_diagonals);
  }

  /*!
   * @brief The kind of metric M of the problem. Anything other than MetricKind::Identity turns the eigenproblem
   * A v = lambda v into the generalised eigenproblem A v = lambda M v, which requires metric_action() and solving
   * with the IterativeSolver::solve() overload that takes buffers for the actions of the metric. Which kinds are
   * supported depends on the solver (IterativeSolver::supports_metric()); solve() throws for the others.
   */
  virtual MetricKind metric_kind() const { return MetricKind::Identity; }

  /*!
   * @brief How IterativeSolver::solve() obtains the actions of the metric on its subspace vectors, see
   * IterativeSolver::set_metric(). With MetricStorage::Stored (the default) it keeps the action of the metric on
   * every subspace vector; with MetricStorage::OnTheFly it keeps none, and calls metric_action() on the solutions
   * whenever it constructs residuals instead. OnTheFly saves the memory or disk space of one vector per subspace vector
   * at the cost of applying the metric once more per root and iteration, which pays off where the metric is cheap to
   * apply, e.g. diagonal or sparse.
   */
  virtual MetricStorage metric_storage() const { return MetricStorage::Stored; }

  /*!
   * @brief Calculate the action of the metric on a set of parameters. Called only if metric_kind() is not
   * MetricKind::Identity.
   * @param parameters The trial solutions for which the action is to be calculated
   * @param metric_actions The action vectors of the metric
   */
  virtual void metric_action(const CVecRef<R>& parameters, const VecRef<R>& metric_actions) const {
    throw std::logic_error("Metric unavailable: unimplemented metric_action() in Problem class");
  }

  /*!
   * @brief Calculate the actions of the kernel and of the metric on a set of parameters together. Used by
   * IterativeSolver::solve() for problems with a metric; the default calls action() and metric_action(), and can be
   * overridden where both are cheaper to calculate together.
   * @param parameters The trial solutions for which the actions are to be calculated
   * @param actions The action vectors of the kernel
   * @param metric_actions The action vectors of the metric
   */
  virtual void action_with_metric(const CVecRef<R>& parameters, const VecRef<R>& actions,
                                  const VecRef<R>& metric_actions) const {
    action(parameters, actions);
    metric_action(parameters, metric_actions);
  }

  /*!
   * @brief Provide the diagonal elements of the metric. For a problem with a metric that provides diagonals(), this
   * is required too, so that the default preconditioner divides by diag(A) - lambda diag(M); IterativeSolver::solve()
   * throws otherwise. Approximate values are fine. The diagonal elements of the kernel and of the metric are then passed to
   * precondition_with_metric(), and an initial guess is generated from the lowest ratios of the diagonal elements of
   * the kernel and the metric, which needs a container supported by precondition_default().
   * @param d On exit, contains the diagonal elements of the metric
   * @return Whether the diagonal elements of the metric have been provided.
   */
  virtual bool metric_diagonals(R& d) const { return false; }

  /*!
   * @brief Return the inhomogeneous part of a linear equation system.
   * @param RHS On return, will contain the requested right-hand-side of equations if available.
   * @param instance Which RHS is required. If there are N sets of equations to solve, then 0,1,...N-1 number the instances.
   * The solver will call this function repeatedly with instance=0,1,2,... until it receives a negative response.
   * @return whether the requested RHS is available and has been provided
   */
  virtual bool RHS(R& RHS, unsigned int instance) const { return false;}

  /*!
   * @brief Calculate the kernel matrix in the P space
   * @param pparams Specification of the P space
   * @return
   */
  virtual std::vector<value_t> pp_action_matrix(const std::vector<P>& pparams) const {
    if (not pparams.empty())
      throw std::logic_error("P-space unavailable: unimplemented pp_action_matrix() in Problem class");
    return std::vector<value_t>(0);
  }

  /*!
   * @brief Calculate the action of the kernel matrix on a set of vectors in the P space
   * @param p_coefficients The projection of the vectors onto to the P space
   * @param pparams Specification of the P space
   * @param actions On exit, the computed action has been added to the original contents
   */
  virtual void p_action(const std::vector<std::vector<value_t>>& p_coefficients, const CVecRef<P>& pparams,
                        const VecRef<container_t>& actions) const {
    if (not pparams.empty())
      throw std::logic_error("P-space unavailable: unimplemented p_action() in Problem class");
  }

  /*!
   * @brief Calculate the metric matrix in the P space, for a problem with a metric (metric_kind())
   * @param pparams Specification of the P space
   * @return the matrix <p_i|M p_j>, with j running fastest
   */
  virtual std::vector<value_t> pp_metric_matrix(const std::vector<P>& pparams) const {
    if (not pparams.empty())
      throw std::logic_error("P-space unavailable: unimplemented pp_metric_matrix() in Problem class");
    return std::vector<value_t>(0);
  }

  /*!
   * @brief Calculate the action of the metric on a set of vectors in the P space, for a problem with a metric
   * (metric_kind()). Not used if the metric is applied on the fly (metric_storage()).
   * @param p_coefficients The projection of the vectors onto to the P space
   * @param pparams Specification of the P space
   * @param metric_actions On exit, the computed action of the metric has been added to the original contents
   *
   * IterativeSolver::solve() calls it once with no vectors (empty p_coefficients and metric_actions) before it changes
   * the solver, to make sure that it is available.
   */
  virtual void p_metric_action(const std::vector<std::vector<value_t>>& p_coefficients, const CVecRef<P>& pparams,
                               const VecRef<container_t>& metric_actions) const {
    if (not pparams.empty())
      throw std::logic_error("P-space unavailable: unimplemented p_metric_action() in Problem class");
  }

  /*!
   * @brief Provide values of R vectors for testing the problem class.
   * For use in a non-linear solver, the first vector (instance=0) should be a reference point, and the remainder
   * (instance>0) should be close to it, such that meaningful numerical differentation can be done to test the residual
   * function.
   * @param instance
   * @param parameters
   * @return true if a vector has been provided
   */
  virtual bool test_parameters(unsigned int instance, R& parameters) const { return false; }
};

/*!
 * @brief The metric of a problem in the P space, for IterativeSolver::add_p()
 * @tparam R the type of container for solutions and residuals
 * @tparam P the type of a P-space vector
 */
template <class R, class P = std::map<size_t, typename R::value_type>>
struct PSpaceMetric {
  using value_type = typename R::value_type;
  //! the metric matrix in the P space, in the layout of the kernel matrix given to add_p(), which copies it
  array::Span<value_type> pp_matrix;
  //! adds the action of the metric on vectors in the P space, like the function for the kernel given to add_p(); needed
  //! with MetricStorage::Stored, ignored with MetricStorage::OnTheFly, and kept by the solver
  std::function<void(const std::vector<std::vector<value_type>>&, const CVecRef<P>&, const VecRef<R>&)> apply;
};

/*!
 * @brief Base class defining the interface common to all iterative solvers
 *
 * As well as through the interface, some behaviour (profiling, and tuning of BLAS operations)
 * is influenced by the contents of
 * a global molpro::Options object, which defaults to molpro::Options("ITERATIVE-SOLVER", "").
 *
 * @tparam R container for "working-set" vectors. These are typically implemented in
 * memory, and are created by the client program. R vectors are never created inside
 * IterativeSolver.
 * @tparam Q container for other vectors. These are typically implemented on backing store
 * and/or distributed across processors.  IterativeSolver constructs a number of instances
 * of Q containers to store history.
 * @tparam P a class that specifies the definition of a single P-space vector, which is a
 * strictly sparse vector in the underlying space.
 */
template <class R, class Q, class P>
class IterativeSolver {
public:
  using value_type = typename R::value_type;                          ///< The underlying type of elements of vectors
  using scalar_type = typename array::ArrayHandler<R, Q>::value_type; ///< The type of scalar products of
                                                                      ///< vectors
  using value_type_abs = typename array::ArrayHandler<R, R>::value_type_abs;
  using VectorP = std::vector<value_type>; //!< type for vectors projected on to P space,
                                           //!< each element is a coefficient for the
                                           //!< corresponding P space parameter
  //! Function type for applying matrix to the P space vectors and accumulating result in
  //! a residual
  using fapply_on_p_type = std::function<void(const std::vector<VectorP>&, const CVecRef<P>&, const VecRef<R>&)>;
  //! Function type for applying the metric to a set of vectors (first argument), writing the results into the second
  using fapply_metric_type = std::function<void(const CVecRef<R>&, const VecRef<R>&)>;
  using PSpaceMetric = itsolv::PSpaceMetric<R, P>;

  virtual ~IterativeSolver() = default;
  IterativeSolver() = default;
  IterativeSolver(const IterativeSolver<R, Q, P>&) = delete;
  IterativeSolver<R, Q, P>& operator=(const IterativeSolver<R, Q, P>&) = delete;
  IterativeSolver(IterativeSolver<R, Q, P>&&) noexcept = default;
  IterativeSolver<R, Q, P>& operator=(IterativeSolver<R, Q, P>&&) noexcept = default;

  /*!
   * @example LinearEigensystemExample.cpp
   * Example for solving linear eigensystem
   * @example LinearEigensystemMultirootExample.cpp
   * Example for solving linear eigensystem, simultaneously tracking multiple roots
   * @example LinearEigensystemDistrArrayExample.cpp
   * Example for solving linear eigensystem, history vectors on disk, optional MPI
   * @example GeneralizedEigensystemExample.cpp
   * Example for solving a generalised eigensystem A v = lambda M v, with solve() and stepwise
   * @example LinearEquationsExample.cpp
   * Example for solving inhomogeneous linear equations
   * @example NonLinearEquationsExample.cpp
   * Example for solving non-linear equations
   * @example OptimizeExample.cpp
   * Example for minimising a function
   * @example OptimizeDistrArrayExample.cpp
   * Example for minimising a function, history vectors on disk, optional MPI
   */
  /*!
   * @brief Simplified one-call solver
   * @param parameters A set of scratch vectors. On entry, these vectors should be filled
   * with starting guesses. Where possible, the number of vectors should be equal to the
   * number of solutions sought, but a smaller array is permitted.
   * @param actions A set of scratch vectors. It should have the same size as parameters.
   * @param problem A Problem object defining the problem to be solved
   * @param generate_initial_guess Whether to start with a guess based on diagonal
   * elements (true) or on the contents of parameters (false)
   * @return true if the solution was found
   *
   * If a P space is used (max_p > 0, diagonal elements are available and the selected P space is at least as large as
   * the number of roots), the initial subspace is the P space alone, and neither the contents of parameters on entry
   * nor a generated initial guess are used. If max_p > 0 and diagonal elements are available but the selected P space
   * is smaller than the number of roots, no P space is used and the initial guess is generated from the diagonal
   * elements, irrespective of generate_initial_guess; the contents of parameters on entry are not used in this case
   * either.
   *
   * Starting vectors, whether generated or taken from parameters, are only used for the roots in the working set, at
   * most one per root. On a solver that is used again after an earlier solve(), roots that have already converged are
   * not in the working set, so fewer guesses, or none at all, are generated.
   *
   * A problem with a metric (Problem::metric_kind() other than MetricKind::Identity) has to be solved with the
   * overload taking buffers for the actions of the metric; this one throws for it.
   */
  virtual bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem,
                     bool generate_initial_guess = false) = 0;
  //! @copydoc solve(const VecRef<R>&, const VecRef<R>&, const Problem<R>&, bool)
  virtual bool solve(R& parameters, R& actions, const Problem<R>& problem, bool generate_initial_guess = false) = 0;
  //! @copydoc solve(const VecRef<R>&, const VecRef<R>&, const Problem<R>&, bool)
  virtual bool solve(std::vector<R>& parameters, std::vector<R>& actions, const Problem<R>& problem,
                     bool generate_initial_guess = false) = 0;

  /*!
   * @brief Simplified one-call solver for problems that may have a metric, e.g. the generalised eigenproblem
   * A v = lambda M v
   * @param parameters A set of scratch vectors, as for solve(const VecRef<R>&, const VecRef<R>&, const Problem<R>&,
   * bool)
   * @param actions A set of scratch vectors. It should have the same size as parameters.
   * @param metric_actions A set of scratch vectors, of the same size as parameters. The solver uses them for the actions
   * of the metric on new expansion vectors, which it needs also when the metric is applied on the fly, as scratch space,
   * and to hand the diagonal elements of the metric to the preconditioner.
   * @param problem A Problem object defining the problem to be solved
   * @param generate_initial_guess Whether to start with a guess based on diagonal elements (true) or on the contents
   * of parameters (false)
   * @return true if the solution was found
   *
   * Behaves like solve(const VecRef<R>&, const VecRef<R>&, const Problem<R>&, bool), and for a problem without a
   * metric exactly so. For a problem with a metric, set_metric() is called with Problem::metric_kind() (throwing if
   * the solver does not support it) and Problem::metric_storage(), with, for MetricStorage::OnTheFly, a function
   * calling Problem::metric_action(). The actions of kernel and metric are calculated together by
   * Problem::action_with_metric(). If the problem provides diagonals(), it has to provide metric_diagonals()
   * too; both are passed to the preconditioner, and a generated initial guess and the P space are selected from the
   * lowest ratios of the diagonal elements of the kernel and the metric. A P space additionally needs
   * Problem::pp_metric_matrix() and, unless the metric is applied on the fly, Problem::p_metric_action().
   *
   * The solver keeps references to problem for later calls such as solution(), so problem has to outlive them; a
   * further call of solve() replaces them.
   */
  virtual bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const VecRef<R>& metric_actions,
                     const Problem<R>& problem, bool generate_initial_guess = false) {
    throw std::invalid_argument("This solver does not support solving a problem with a metric");
  }
  //! @copydoc solve(const VecRef<R>&, const VecRef<R>&, const VecRef<R>&, const Problem<R>&, bool)
  virtual bool solve(R& parameters, R& actions, R& metric_actions, const Problem<R>& problem,
                     bool generate_initial_guess = false) {
    return solve(wrap_arg(parameters), wrap_arg(actions), wrap_arg(metric_actions), problem, generate_initial_guess);
  }
  //! @copydoc solve(const VecRef<R>&, const VecRef<R>&, const VecRef<R>&, const Problem<R>&, bool)
  virtual bool solve(std::vector<R>& parameters, std::vector<R>& actions, std::vector<R>& metric_actions,
                     const Problem<R>& problem, bool generate_initial_guess = false) {
    return solve(wrap(parameters), wrap(actions), wrap(metric_actions), problem, generate_initial_guess);
  }

  /*!
   * @brief Whether the solver supports a metric of the given kind. MetricKind::Identity is always supported.
   */
  virtual bool supports_metric(MetricKind kind) const { return kind == MetricKind::Identity; }

  /*!
   * @brief Declares the metric M of a generalised problem, e.g. the eigenproblem A v = lambda M v, for the stepwise
   * interface. solve() calls it itself.
   *
   * It has to be called before the first add_p() or add_vector(). On a non-empty subspace it may only be called again
   * with the same kind and storage, replacing apply_metric. Throws if the solver does not support kind.
   *
   * Once a metric other than the identity is declared, every add_vector() has to give the actions of the metric.
   * Solutions are then normalised such that |<x|M|x>| = 1, and their residuals are A x - lambda M x. The errors are
   * the norms of the residuals of the solutions scaled to unit length, |A x - lambda M x| / |x|, which do not depend
   * on the scale of M and are the same as without a metric if M is the identity.
   *
   * add_p() and add_vector() throw std::runtime_error if the actions of the metric show that it is not Hermitian or,
   * for the positive (semi)definite kinds, that it is not positive (semi)definite, unless switched off with
   * set_check_metric(). These checks see the metric only on the subspace, so that, e.g., an indefinite metric is
   * noticed only once one of its negative directions enters the subspace; test_problem() checks the metric of a
   * Problem more directly.
   * @param kind the kind of metric
   * @param storage how the solver obtains the actions of the metric on its subspace vectors: with
   * MetricStorage::Stored it keeps those given to add_vector(); with MetricStorage::OnTheFly, which is worthwhile where
   * the metric is cheap to apply, it keeps none, and calls apply_metric on the solutions whenever it constructs
   * residuals instead. Irrelevant for MetricKind::Identity.
   * @param apply_metric applies the metric, given exactly for MetricStorage::OnTheFly; it has to remain valid as long
   * as residuals are constructed
   */
  virtual void set_metric(MetricKind kind, MetricStorage storage = MetricStorage::Stored,
                          fapply_metric_type apply_metric = {}) {
    if (kind != MetricKind::Identity)
      throw std::invalid_argument("This solver does not support a metric");
  }
  //! The kind of metric declared with set_metric()
  virtual MetricKind metric_kind() const { return MetricKind::Identity; }
  //! How the actions of the metric declared with set_metric() are obtained
  virtual MetricStorage metric_storage() const { return MetricStorage::Stored; }
  /*!
   * @brief Sets whether add_p() and add_vector() check that the metric is Hermitian and, for the positive
   * (semi)definite kinds, positive (semi)definite on the subspace (default: true). The checks cost a few operations on
   * vectors in every iteration.
   */
  virtual void set_check_metric(bool check) {}
  //! Whether the metric is checked, see set_check_metric()
  virtual bool get_check_metric() const { return false; }

  /*!
   * \brief Take, typically, a current solution and residual, and add it to the solution
   * space. \param parameters On input, the current solution or expansion vector. On exit,
   * undefined. \param actions On input, the residual for parameters (non-linear), or
   * action of matrix on parameters (linear). On exit, a vector set that should be
   * preconditioned before returning to end_iteration(). \param value The value of the
   * objective function for parameters. Used only in Optimize classes. \return The size of
   * the new working set. In non-linear optimisation, the special value -1 can also be
   * returned, indicating that preconditioning should not be carried out on action.
   */
  virtual int add_vector(const VecRef<R>& parameters, const VecRef<R>& actions) = 0;

  // FIXME this should be removed in favour of VecRef interface
  virtual int add_vector(std::vector<R>& parameters, std::vector<R>& action) = 0;
  virtual int add_vector(R& parameters, R& action, value_type value = 0) = 0;

  /*!
   * @brief Like add_vector(const VecRef<R>&, const VecRef<R>&), for a problem with a metric declared with
   * set_metric()
   * @param parameters On input, the current solution or expansion vector. On exit, undefined.
   * @param actions On input, the action of the kernel on parameters. On exit, a vector set that should be
   * preconditioned before returning to end_iteration().
   * @param metric_actions On input, the action of the metric on parameters. On exit, undefined.
   * @return The size of the new working set.
   */
  virtual int add_vector(const VecRef<R>& parameters, const VecRef<R>& actions, const VecRef<R>& metric_actions) {
    throw std::invalid_argument("This solver does not support a metric");
  }
  //! @copydoc add_vector(const VecRef<R>&, const VecRef<R>&, const VecRef<R>&)
  virtual int add_vector(std::vector<R>& parameters, std::vector<R>& actions, std::vector<R>& metric_actions) {
    return add_vector(wrap(parameters), wrap(actions), wrap(metric_actions));
  }
  //! @copydoc add_vector(const VecRef<R>&, const VecRef<R>&, const VecRef<R>&)
  virtual int add_vector(R& parameters, R& actions, R& metric_actions) {
    return add_vector(wrap_arg(parameters), wrap_arg(actions), wrap_arg(metric_actions));
  }

  /*!
   * \brief Add P-space vectors to the expansion set for linear methods.
   * \note the apply_p function is stored and used by the solver internally.
   * \param Pparams the vectors to add. Each Pvector specifies a sparse vector in the
   * underlying space. The size of the P space must be at least the number of roots to be sought.
   * \param pp_action_matrix Matrix projected onto the existing+new, new
   * P space. It should be provided as a 1-dimensional array, with the existing+new index
   * running fastest.
   * \param parameters Used as scratch working space
   * \param action  On exit, the  residual of the interpolated solution. The contribution from the new, and
   * any existing, P parameters is missing, and should be added in subsequently.
   * \param apply_p A function that evaluates the action of the matrix on vectors in the P space
   * \return The number of vectors contained in parameters, action, parametersP
   */
  virtual size_t add_p(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix,
                       const VecRef<R>& parameters, const VecRef<R>& action, fapply_on_p_type apply_p) = 0;

  /*!
   * \brief Like add_p(const CVecRef<P>&, const array::Span<value_type>&, const VecRef<R>&, const VecRef<R>&,
   * fapply_on_p_type), for a problem with a metric declared with set_metric(), which this overload requires.
   * \param pparams the vectors to add, as for the other overload
   * \param pp_action_matrix the kernel matrix in the P space, as for the other overload
   * \param parameters Used as scratch working space
   * \param action On exit, the residual of the interpolated solution, missing the contributions from the P space
   * \param apply_p A function that adds the action of the kernel on vectors in the P space
   * \param metric the metric in the P space
   * \return The number of vectors contained in parameters, action, parametersP
   */
  virtual size_t add_p(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix,
                       const VecRef<R>& parameters, const VecRef<R>& action, fapply_on_p_type apply_p,
                       const PSpaceMetric& metric) {
    throw std::invalid_argument("This solver does not support a metric");
  }

  // FIXME Is this needed?
  virtual void clearP() = 0;

  /*!
   * @brief Construct solution and residual for a given set of roots
   *
   * For a problem solved with solve() whose metric is applied on the fly (Problem::metric_storage()), this applies the
   * metric through the problem, which therefore has to be still alive.
   */
  virtual void solution(const std::vector<int>& roots, const VecRef<R>& parameters, const VecRef<R>& residual) = 0;

  //! Constructs parameters of selected roots
  virtual void solution_params(const std::vector<int>& roots, const VecRef<R>& parameters) = 0;

  //! Behaviour depends on the solver
  virtual size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& residual) = 0;

  /*!
   * Callback invoked once the solver iterations are done but before the solve function returns
   * to the caller. Behaviour depends on the solver.
   */
  virtual void finalize() = 0;

  /*!
   * @brief signal whether end_iteration should be called
   */
  virtual bool end_iteration_needed() = 0;

  /*!
   * \brief Get the solver's suggestion of which degrees of freedom would be best
   * to add to the P-space.
   * \param solution Current solution
   * \param residual Current residual
   * \param max_number Suggest no more than this number
   * \param threshold Suggest only axes for which the current residual and update
   * indicate an energy improvement in the next iteration of this amount or more.
   * \return
   */
  virtual std::vector<size_t> suggest_p(const CVecRef<R>& solution, const CVecRef<R>& residual, size_t max_number,
                                        value_type_abs threshold) = 0;

  virtual void solution(const std::vector<int>& roots, std::vector<R>& parameters, std::vector<R>& residual) = 0;
  virtual void solution(R& parameters, R& residual) = 0;
  virtual void solution_params(const std::vector<int>& roots, std::vector<R>& parameters) = 0;
  virtual void solution_params(R& parameters) = 0;
  virtual size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) = 0;
  virtual size_t end_iteration(R& parameters, R& action) = 0;

  /*!
   * @brief Working set of roots that are not yet converged
   */
  virtual const std::vector<int>& working_set() const = 0;
  //! The calculated eigenvalues for roots in the working set (eigenvalue problems) or
  //! zero (otherwise)
  virtual std::vector<scalar_type> working_set_eigenvalues() const {
    return std::vector<scalar_type>(working_set().size(), 0);
  }
  //! Total number of roots we are solving for, including the ones that are already
  //! converged
  virtual size_t n_roots() const = 0;
  virtual void set_n_roots(size_t nroots) = 0;
  //! Errors are norms of the residuals, so they are real even when the problem is complex
  virtual const std::vector<value_type_abs>& errors() const = 0;
  virtual const Statistics& statistics() const = 0;
  //! Writes a report to cout output stream
  virtual void report(std::ostream& cout, bool endl = true) const = 0;
  //! Writes a report to std::cout
  virtual void report() const = 0;

  //! Sets the convergence threshold
  virtual void set_convergence_threshold(value_type_abs thresh) = 0;
  //! Reports the convergence threshold
  virtual value_type_abs convergence_threshold() const = 0;
  //! Sets the value convergence threshold
  virtual void set_convergence_threshold_value(value_type_abs thresh) = 0;
  //! Reports the value convergence threshold
  virtual value_type_abs convergence_threshold_value() const = 0;
  [[deprecated("Set the verbosity on the logger directly")]]
  virtual void set_verbosity(Verbosity v) = 0;
  virtual void set_verbosity(int v) = 0;
  [[deprecated("Query the logger directly")]]
  virtual Verbosity get_verbosity() const = 0;
  virtual void set_max_iter(int n) = 0;
  virtual int get_max_iter() const = 0;
  virtual void set_max_p(int n) = 0;
  virtual int get_max_p() const = 0;
  virtual void set_p_threshold(value_type_abs thresh) = 0;
  virtual value_type_abs get_p_threshold() const = 0;
  virtual const subspace::Dimensions& dimensions() const = 0;
  // FIXME Missing parameters: SVD threshold
  //! Set all spcecified options. This is no different than using setters, but can be used
  //! with forward declaration.
  virtual void set_options(const Options& options) = 0;
  //! Return all options. This is no different than using getters, but can be used with
  //! forward declaration.
  virtual std::shared_ptr<Options> get_options() const = 0;
  /*!
   * @brief Report the function value for the current optimum solution
   * @return
   */
  virtual scalar_type value() const = 0;
  /*!
   * @brief Report whether the class is a non-linear solver
   * @return
   */
  virtual bool nonlinear() const = 0;
  /*!
   * @brief Attach a profiler in order to collect performance data
   * @param profiler The profiler to use. It is not owned by the solver and must outlive it.
   */
  virtual void set_profiler(molpro::profiler::Profiler& profiler) = 0;
  /*!
   * @brief Attach a profiler in order to collect performance data
   * @param profiler The profiler to use. The solver shares ownership of it, so it stays valid for the solver's lifetime.
   */
  virtual void set_profiler(std::shared_ptr<molpro::profiler::Profiler> profiler) = 0;
  virtual const std::shared_ptr<molpro::profiler::Profiler>& profiler() const = 0;
  /*!
   * @brief Set the logger instance that shall be used
   * @param logger
   */
  virtual void set_logger(std::shared_ptr<Logger> logger) = 0;
  virtual Logger &logger() = 0;
  /*!
   * @brief Test a supplied problem class
   *
   * For a linear problem, checks that the actions of the kernel and, if there is one, of the metric are linear, and
   * that the metric is Hermitian and, for the positive (semi)definite kinds, positive (semi)definite on the test
   * parameters.
   * @param problem
   * @param v0
   * @param v1
   * @param verbosity
   * @param threshold
   * @return false if errors were found, otherwise true
   */
  virtual bool test_problem(const Problem<R>& problem, R& v0, R& v1, int verbosity = 0,
                            value_type_abs threshold = 1e-5) const = 0;
};

/*!
 * @brief Interface for a specific iterative solver, it can add special member functions
 * or variables.
 */
template <class R, class Q, class P>
class LinearEigensystem : public IterativeSolver<R, Q, P> {
public:
  using typename IterativeSolver<R, Q, P>::scalar_type;
  //! The calculated eigenvalues of the subspace matrix
  virtual std::vector<scalar_type> eigenvalues() const = 0;
  //! Sets hermiticity of kernel
  virtual void set_hermiticity(bool hermitian) = 0;
  //! Gets hermiticity of kernel, if true than it is hermitian, otherwise it is not
  virtual bool get_hermiticity() const = 0;
};

template <class R, class Q, class P>
class LinearEquations : public IterativeSolver<R, Q, P> {
public:
  using typename IterativeSolver<R, Q, P>::scalar_type;
  virtual void add_equations(const CVecRef<R>& rhs) = 0;
  virtual void add_equations(const std::vector<R>& rhs) = 0;
  virtual void add_equations(const R& rhs) = 0;
  virtual CVecRef<Q> rhs() const = 0;
  //! Sets hermiticity of kernel
  virtual void set_hermiticity(bool hermitian) = 0;
  //! Gets hermiticity of kernel, if true than it is hermitian, otherwise it is not
  virtual bool get_hermiticity() const = 0;
};

//! Optimises to a stationary point using methods such as L-BFGS. The objective is minimised; to maximise a function,
//! minimise its negative. (The C, Fortran and Python interfaces do that when asked to maximise.)
template <class R, class Q, class P>
class Optimize : public IterativeSolver<R, Q, P> {};

//! Solves non-linear system of equations using methods such as DIIS
template <class R, class Q, class P>
class NonLinearEquations : public IterativeSolver<R, Q, P> {};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
