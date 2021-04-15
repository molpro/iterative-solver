#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
#include <molpro/linalg/array/ArrayHandler.h>
#include <molpro/linalg/array/Span.h>
#include <molpro/linalg/itsolv/Options.h>
#include <molpro/linalg/itsolv/Statistics.h>
#include <molpro/linalg/itsolv/subspace/Dimensions.h>
#include <molpro/linalg/itsolv/wrap.h>

#include <memory>
#include <ostream>
#include <vector>

namespace molpro::linalg::itsolv {

/*!
 * @example ExampleProblem.h
 * Example of a problem-defining class
 */
/*!
 * @brief Abstract class defining the problem-specific interface for the simplified solver interface to IterativeSolver
 * @tparam R the type of container for solutions and residuals
 */
template <typename R>
class Problem {
public:
  Problem() = default;
  virtual ~Problem() = default;
  using container_t = R;

  /*!
   * @brief Calculate the residual vector. Used by non-linear solvers (NonLinearEquations, Optimize) only.
   * @param parameters The trial solution for which the residual is to be calculated
   * @param residual The residual vector
   * @return In the case where the residual is an exact differential, the corresponding function value. Used by Optimize
   * but not NonLinearEquations.
   */
  virtual double residual(const R& parameters, R& residual) const { return 0; }

  /*!
   * @brief Calculate the action of the kernel matrix on a set of parameters. Used by linear solvers, but not by the
   * non-linear solvers (NonLinearEquations, Optimize).
   * @param parameters The trial solutions for which the action is to be calculated
   * @param action The action vectors
   */
  virtual void action(const CVecRef<R>& parameters, const VecRef<R>& action) const { return; }

  /*!
   * @brief Apply preconditioning to a residual vector in order to predict a step towards the solution
   * @param residual On entry, assumed to be the residual. On exit, the negative of the predicted step.
   * @param shift When called from LinearEigensystem, contains the corresponding current eigenvalue estimates for each
   * of the parameter vectors in the set. All other solvers pass a vector of zeroes.
   */
  virtual void precondition(const VecRef<R>& residual, const std::vector<typename R::value_type>& shift) const {
    return;
  }
};

/*!
 * @brief Base class defining the interface common to all iterative solvers
 *
 * @tparam R container for "working-set" vectors. These are typically implemented in memory, and are created by the
 * client program. R vectors are never created inside IterativeSolver.
 * @tparam Q container for other vectors. These are typically implemented on backing store and/or distributed across
 * processors.  IterativeSolver constructs a number of instances of Q containers to store history.
 * @tparam P a class that specifies the definition of a single P-space vector, which is a strictly sparse vector in the
 * underlying space.
 */
template <class R, class Q, class P>
class IterativeSolver {
public:
  using value_type = typename R::value_type;                          ///< The underlying type of elements of vectors
  using scalar_type = typename array::ArrayHandler<R, Q>::value_type; ///< The type of scalar products of vectors
  using value_type_abs = typename array::ArrayHandler<R, R>::value_type_abs;
  using VectorP = std::vector<value_type>; //!< type for vectors projected on to P space, each element is a coefficient
                                           //!< for the corresponding P space parameter
  //! Function type for applying matrix to the P space vectors and accumulating result in a residual
  using fapply_on_p_type = std::function<void(const std::vector<VectorP>&, const CVecRef<P>&, const VecRef<R>&)>;

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
   * @example LinearEquationsExample.cpp
   * Example for solving inhomogeneous linear equations
   * @example NonLinearEquationsExample.cpp
   * Example for solving non-linear equations
   * @example OptimizeExample.cpp
   * Example for minimising a function
   */
  /*!
   * @brief Simplified one-call solver
   * @param parameters A set of scratch vectors. On entry, these vectors should be filled with starting guesses.
   * Where possible, the number of vectors should be equal to the number of solutions sought, but a smaller array is
   * permitted.
   * @param actions A set of scratch vectors. It should have the same size as parameters.
   * @param problem A Problem object defining the problem to be solved
   * @return true if the solution was found
   */
  //  virtual bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem) = 0; // TODO
  //  make abstract when all concretizations are done
  virtual bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem) { return false; }
  virtual bool solve(R& parameters, R& actions, const Problem<R>& problem) {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return solve(wparams, wactions, problem);
  }
  virtual bool solve(std::vector<R>& parameters, std::vector<R>& actions, const Problem<R>& problem) {
    return solve(wrap(parameters), wrap(actions), problem);
  }

  /*!
   * \brief Take, typically, a current solution and residual, and add it to the solution space.
   * \param parameters On input, the current solution or expansion vector. On exit, undefined.
   * \param actions On input, the residual for parameters (non-linear), or action of matrix
   * on parameters (linear). On exit, a vector set that should be preconditioned before returning to end_iteration().
   * \param value The value of the objective function for parameters. Used only in Optimize classes.
   * \return The size of the new working set. In non-linear optimisation, the special value -1 can also be returned,
   * indicating that preconditioning should not be carried out on action.
   */
  virtual int add_vector(const VecRef<R>& parameters, const VecRef<R>& actions) = 0;

  // FIXME this should be removed in favour of VecRef interface
  virtual int add_vector(std::vector<R>& parameters, std::vector<R>& action) = 0;
  virtual int add_vector(R& parameters, R& action, value_type value = 0) = 0;

  /*!
   * \brief Add P-space vectors to the expansion set for linear methods.
   * \note the apply_p function is stored and used by the solver internally.
   * \param Pparams the vectors to add. Each Pvector specifies a sparse vector in the underlying space
   * \param pp_action_matrix Matrix projected onto the existing+new, new P space. It should be provided as a
   * 1-dimensional array, with the existing+new index running fastest.
   * \param parameters Used as scratch working space
   * \param action  On exit, the  residual of the interpolated solution.
   * The contribution from the new, and any existing, P parameters is missing, and should be added in subsequently.
   * \param apply_p A function that evaluates the action of the matrix on vectors in the P space
   * \return The number of vectors contained in parameters, action, parametersP
   */
  virtual size_t add_p(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix,
                       const VecRef<R>& parameters, const VecRef<R>& action, fapply_on_p_type apply_p) = 0;

  // FIXME Is this needed?
  virtual void clearP() = 0;

  //! Construct solution and residual for a given set of roots
  virtual void solution(const std::vector<int>& roots, const VecRef<R>& parameters, const VecRef<R>& residual) = 0;

  //! Constructs parameters of selected roots
  virtual void solution_params(const std::vector<int>& roots, const VecRef<R>& parameters) = 0;

  //! Behaviour depends on the solver
  virtual size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& residual) = 0;

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
                                        double threshold) = 0;

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
  //! The calculated eigenvalues for roots in the working set (eigenvalue problems) or zero (otherwise)
  virtual std::vector<scalar_type> working_set_eigenvalues() const {
    return std::vector<scalar_type>(working_set().size(), 0);
  }
  //! Total number of roots we are solving for, including the ones that are already converged
  virtual size_t n_roots() const = 0;
  virtual void set_n_roots(size_t nroots) = 0;
  virtual const std::vector<scalar_type>& errors() const = 0;
  virtual const Statistics& statistics() const = 0;
  //! Writes a report to cout output stream
  virtual void report(std::ostream& cout) const = 0;
  //! Writes a report to std::cout
  virtual void report() const = 0;

  //! Sets the convergence threshold
  virtual void set_convergence_threshold(double thresh) = 0;
  //! Reports the convergence threshold
  virtual double convergence_threshold() const = 0;
  //! Sets the value convergence threshold
  virtual void set_convergence_threshold_value(double thresh) = 0;
  //! Reports the value convergence threshold
  virtual double convergence_threshold_value() const = 0;
  virtual void set_verbosity(Verbosity v) = 0;
  virtual Verbosity get_verbosity() const = 0;
  virtual void set_max_iter(int n) = 0;
  virtual int get_max_iter() const = 0;
  virtual const subspace::Dimensions& dimensions() const = 0;
  // FIXME Missing parameters: SVD threshold
  //! Set all spcecified options. This is no different than using setters, but can be used with forward declaration.
  virtual void set_options(const Options& options) = 0;
  //! Return all options. This is no different than using getters, but can be used with forward declaration.
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
};

/*!
 * @brief Interface for a specific iterative solver, it can add special member functions or variables.
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

//! Optimises to a stationary point using methods such as L-BFGS
template <class R, class Q, class P>
class Optimize : public IterativeSolver<R, Q, P> {};

//! Solves non-linear system of equations using methods such as DIIS
template <class R, class Q, class P>
class NonLinearEquations : public IterativeSolver<R, Q, P> {};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVER_H
