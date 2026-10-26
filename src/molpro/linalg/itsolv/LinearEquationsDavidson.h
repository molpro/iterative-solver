#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSON_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSON_H
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DavidsonCommon.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverLinEig.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

namespace molpro::linalg::itsolv {
/*!
 * @brief Solves a system of linear equation, A x = b
 *
 * The equations are solved using a Krylov subspace projection method with the P and D space. This is the same approach
 * as LinearEigenvalue.
 *
 * Residual
 * --------
 * The residual is scaled down by the norm of the RHS vector so that thresholds are consistent.
 *
 * Value
 * -----
 * The value of solution i is \f$\chi_{ii}=\langle b_i|x_i\rangle\f$, the diagonal element of the susceptibility
 * (second-order property) when the equations are for first-order perturbed wavefunctions. The value error used with
 * the value convergence threshold is the absolute change in \f$\chi_{ii}\f$ since the previous iteration, without
 * the scaling applied to the residual. With the augmented Hessian it is instead the change in the subspace eigenvalue.
 *
 * Preconditioner
 * -------------
 * @todo Explain some of the theory
 *
 * Augmented Hessian
 * -----------------
 * @todo Explain some of the theory
 */
template <class R, class Q = R, class P = std::map<size_t, typename R::value_type>>
class LinearEquationsDavidson : public DavidsonCommon<LinearEquations, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<LinearEquations, R, Q, P>;
  using SolverTemplate ::report;
  using typename SolverTemplate::value_type;
  using typename SolverTemplate::value_type_abs;

  explicit LinearEquationsDavidson(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers,
                                   const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : DavidsonCommon<LinearEquations, R, Q, P>(handlers, logger_, true) {}

  bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem,
             bool generate_initial_guess = false) override {
    R& vector = actions[0];
    for (unsigned int instance = 0; problem.RHS(vector, instance); ++instance)
      add_equations(vector);
    return IterativeSolverTemplate<LinearEquations, R, Q, P>::solve(parameters, actions, problem,
                                                                    generate_initial_guess);
  }

  void add_equations(const CVecRef<R>& rhs) override {
    auto prof = this->profiler()->push("itsolv::add_equations");
    auto xspace = std::static_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    xspace->add_rhs_equations(rhs);
    this->set_n_roots(xspace->dimensions().nRHS);
  }

  void add_equations(const R& rhs) override { add_equations(cwrap_arg(rhs)); }
  void add_equations(const std::vector<R>& rhs) override { add_equations(cwrap(rhs)); }

  CVecRef<Q> rhs() const override {
    auto xspace = std::static_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    return xspace->rhs();
  }

  //!@copydoc subspace::SubspaceSolverLinEig::set_augmented_hessian()
  void set_augmented_hessian(const value_type_abs parameter) {
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    subspace_solver->set_augmented_hessian(parameter);
  }
  //!@copydoc subspace::SubspaceSolverLinEig::get_augmented_hessian()
  value_type_abs get_augmented_hessian() const {
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    return subspace_solver->get_augmented_hessian();
  }

  void set_options(const Options& options) override {
    const auto opt = CastOptions::LinearEquations(options);
    this->set_davidson_options(options, opt);
    if (opt.augmented_hessian)
      set_augmented_hessian(opt.augmented_hessian.value());
  }

  std::shared_ptr<Options> get_options() const override {
    auto opt = this->template get_davidson_options<LinearEquationsDavidsonOptions>();
    opt->augmented_hessian = get_augmented_hessian();
    return opt;
  }

  void report(std::ostream& cout, bool endl = true) const override {
    const util::StreamFormatGuard format_guard(cout);
    SolverTemplate::report(cout, false);
    cout << ", errors " << std::scientific;
    auto& err = this->m_errors;
    std::copy(begin(err), end(err), std::ostream_iterator<value_type_abs>(cout, ", "));
    if (endl)
      cout << std::endl;
  }

protected:
  //! The values chi_ii = <b_i|x_i> of the current solutions, or with the augmented Hessian the subspace eigenvalues
  std::vector<value_type> values() const override {
    if (get_augmented_hessian() > 0)
      return this->m_subspace_solver->eigenvalues();
    const auto& solutions = this->m_subspace_solver->solutions(); // c(i, k): root i, subspace vector k
    const auto& rhs = this->m_xspace->data.at(subspace::EqnData::rhs); // <v_k|b_i>
    auto chi = std::vector<value_type>(std::min(solutions.rows(), rhs.cols()), value_type{0});
    for (size_t i = 0; i < chi.size(); ++i)
      for (size_t k = 0; k < std::min(solutions.cols(), rhs.rows()); ++k)
        chi[i] += solutions(i, k) * molpro::linalg::conjugate(rhs(k, i));
    return chi;
  }

  // FIXME The scale is fixed by the norm of RHS, but if RHS=0 there is no reference. We could use the norm of params
  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {
    assert(params.size() >= roots.size());
    const auto& norm = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace)->rhs_norm();
    // With the augmented Hessian the subspace solution satisfies (H - lambda) x = rhs, with lambda the subspace
    // eigenvalue; without it lambda is zero
    const auto& eigenvalues = this->m_subspace_solver->eigenvalues();
    for (size_t i = 0; i < roots.size(); ++i) {
      const auto ii = roots[i];
      this->m_handlers->rq().axpy(-1, rhs().at(ii), actions.at(i));
      if (size_t(ii) < eigenvalues.size() and eigenvalues[ii] != value_type(0))
        this->m_handlers->rr().axpy(-eigenvalues[ii], params.at(i), actions.at(i));
      if (norm.at(ii) != 0) {
        auto scal = 1 / norm[ii];
        this->m_handlers->rr().scal(scal, actions.at(i));
      }
    }
  }
};

} // namespace molpro::linalg::itsolv
#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREQUATIONSDAVIDSON_H
