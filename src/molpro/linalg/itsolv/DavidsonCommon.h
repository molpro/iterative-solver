#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_DAVIDSONCOMMON_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_DAVIDSONCOMMON_H
#include <molpro/linalg/itsolv/DSpaceResetter.h>
#include <molpro/linalg/itsolv/DavidsonOptions.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/Options.h>
#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/qspace_options.h>
#include <molpro/linalg/itsolv/rspace_options.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverLinEig.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace molpro::linalg::itsolv {

/*!
 * @brief Implementation shared by the linear Davidson solvers, LinearEigensystemDavidson and LinearEquationsDavidson
 *
 * Holds the management of the subspace between iterations: proposing new parameters from the preconditioned residuals,
 * limiting the size of the Q space by moving deleted Q vectors into the D space, and periodically resetting the D
 * space, together with the options that control it.
 *
 * @tparam Solver the solver interface, LinearEigensystem or LinearEquations
 */
template <template <class, class, class> class Solver, class R, class Q, class P>
class DavidsonCommon : public IterativeSolverTemplate<Solver, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<Solver, R, Q, P>;
  using typename SolverTemplate::value_type;
  using typename SolverTemplate::value_type_abs;

  bool nonlinear() const override { return false; }

  /*!
   * \brief Proposes new parameters for the subspace from the preconditioned residuals.
   *
   * After add_vector solves the subspace problem it returns the new solution and residual. The residual should be
   * preconditioned, i.e. using Davidson method, to accelerate convergence. The updated residual is used in this
   * function to propose new Q space parameters orthonormal to the old space. They are returned in parameters so that
   * corresponding actions can be calculated and used in add_vector in the next iteration.
   *
   * Every reset_D iterations, unless reset_D is 0 (the default) or there is no D space, the D space is reset instead
   * (see detail::DSpaceResetter): the current solutions are returned in parameters as the new parameters, the D space
   * is cleared, and for each returned solution the Q vector with the largest overlap with it is deleted. The Q space is
   * then reduced to at most reset_D_max_Q_size vectors, including the returned solutions. If there are more solutions
   * than parameters, the reset is spread over several iterations. Until the next add_vector() takes up the returned
   * solutions, solution() is not available, as the subspace left may be too small to hold them; eigenvalues() and
   * errors() still refer to the solutions before the reset.
   *
   * Once all roots have converged, the subspace is left unchanged.
   *
   * @param parameters output new parameters for the subspace.
   * @param action preconditioned residuals.
   * @return number of significant parameters to calculate the action for
   */
  size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& action) override {
    auto prof = this->profiler();
    auto m_prof = prof->push("itsolv::end_iteration");
    if (this->m_working_set.empty() && !m_dspace_resetter.in_progress()) {
      // all roots have converged: the subspace is left as it is
      m_resetting_in_progress = false;
    } else if (m_dspace_resetter.do_reset(this->m_stats->iterations, this->m_xspace->dimensions())) {
      m_resetting_in_progress = true;
      this->m_working_set =
          m_dspace_resetter.run(parameters, *this->m_xspace, this->m_subspace_solver->retained_solutions(),
                                rspace_opts.norm_thresh, rspace_opts.svd_thresh, *this->m_handlers, *this->m_logger);
      this->m_solutions_outdated = true;
    } else {
      m_resetting_in_progress = false;
      prof->start("end_iteration (propose_rspace)");
      this->m_working_set =
          detail::propose_rspace(*this, parameters, action, *this->m_xspace, *this->m_subspace_solver,
                                 *this->m_handlers, *this->m_logger, rspace_opts, qspace_opts, *this->profiler().get());
      prof->stop();
    }
    this->m_stats->iterations++;
    read_handler_counts(this->m_stats, this->m_handlers);
    this->m_end_iteration_needed = false;
    return this->working_set().size();
  }
  //! @copydoc end_iteration(const VecRef<R>&, const VecRef<R>&)
  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override {
    return end_iteration(wrap(parameters), wrap(action));
  }
  //! @copydoc end_iteration(const VecRef<R>&, const VecRef<R>&)
  size_t end_iteration(R& parameters, R& actions) override {
    return end_iteration(wrap_arg(parameters), wrap_arg(actions));
  }

  //! Set threshold on the norm of parameters that should be considered null
  void set_norm_thresh(value_type_abs thresh) { rspace_opts.norm_thresh = thresh; }
  value_type_abs get_norm_thresh() const { return rspace_opts.norm_thresh; }
  //! Set the smallest singular value in the subspace that can be allowed when
  //! constructing the working set. Smaller singular values will lead to deletion of parameters
  void set_svd_thresh(value_type_abs thresh) { rspace_opts.svd_thresh = thresh; }
  value_type_abs get_svd_thresh() const { return rspace_opts.svd_thresh; }
  //! Set the period in iterations for resetting the D space, 0 for never
  void set_reset_D(size_t n) { m_dspace_resetter.set_nreset(n); }
  size_t get_reset_D() const { return m_dspace_resetter.get_nreset(); }
  //! Set the maximum size of Q space after resetting the D space
  void set_reset_D_maxQ_size(size_t n) { m_dspace_resetter.set_max_Qsize(n); }
  size_t get_reset_D_maxQ_size() const { return m_dspace_resetter.get_max_Qsize(); }
  //! Set a limit on the maximum size of Q space. This does not include the size of the working space (R) and the D
  //! space. It must not be less than the minimum size
  void set_max_size_qspace(std::size_t n) { set_qspace_size_limits(get_min_size_qspace(), n); }
  std::size_t get_max_size_qspace() const { return qspace_opts.max_size; }
  //! Set the minimum size of Q space when it is truncated, which must not exceed the maximum size
  void set_min_size_qspace(std::size_t n) { set_qspace_size_limits(n, get_max_size_qspace()); }
  std::size_t get_min_size_qspace() const { return qspace_opts.min_size; }

  void set_hermiticity(bool hermitian) override {
    m_hermiticity = hermitian;
    auto xspace = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    xspace->set_hermiticity(hermitian);
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    subspace_solver->set_hermiticity(hermitian);
  }
  bool get_hermiticity() const override { return m_hermiticity; }

protected:
  DavidsonCommon(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers, const std::shared_ptr<Logger>& logger,
                 bool hermiticity)
      : SolverTemplate(std::make_shared<subspace::XSpace<R, Q, P>>(handlers, logger),
                       std::static_pointer_cast<subspace::ISubspaceSolver<R, Q, P>>(
                           std::make_shared<subspace::SubspaceSolverLinEig<R, Q, P>>(logger)),
                       handlers, std::make_shared<Statistics>(), logger) {
    set_hermiticity(hermiticity);
  }

  /*!
   * @brief Sets the general options and those of the Davidson solvers, after checking them
   * @param options all options
   * @param opt the Davidson options among them
   */
  void set_davidson_options(const Options& options, const DavidsonOptions& opt) {
    const auto min_size_qspace = opt.min_size_qspace.value_or(get_min_size_qspace());
    const auto max_size_qspace = opt.max_size_qspace.value_or(get_max_size_qspace());
    check_qspace_size_limits(min_size_qspace, max_size_qspace);
    SolverTemplate::set_options(options);
    if (opt.reset_D)
      set_reset_D(opt.reset_D.value());
    if (opt.reset_D_max_Q_size)
      set_reset_D_maxQ_size(opt.reset_D_max_Q_size.value());
    if (opt.max_size_qspace or opt.min_size_qspace)
      set_qspace_size_limits(min_size_qspace, max_size_qspace);
    if (opt.contrib_thresh)
      qspace_opts.contrib_thresh = opt.contrib_thresh.value();
    if (opt.norm_thresh)
      set_norm_thresh(opt.norm_thresh.value());
    if (opt.svd_thresh)
      set_svd_thresh(opt.svd_thresh.value());
    if (opt.hermiticity)
      set_hermiticity(opt.hermiticity.value());
  }

  //! Returns the general options and those of the Davidson solvers in an options object of type SolverOptions
  template <class SolverOptions>
  std::shared_ptr<SolverOptions> get_davidson_options() const {
    auto opt = std::make_shared<SolverOptions>();
    opt->copy(*SolverTemplate::get_options());
    opt->reset_D = get_reset_D();
    opt->reset_D_max_Q_size = get_reset_D_maxQ_size();
    opt->max_size_qspace = get_max_size_qspace();
    opt->min_size_qspace = get_min_size_qspace();
    opt->contrib_thresh = qspace_opts.contrib_thresh;
    opt->norm_thresh = get_norm_thresh();
    opt->svd_thresh = get_svd_thresh();
    opt->hermiticity = get_hermiticity();
    return opt;
  }

  //! Throws std::invalid_argument if the minimum size of the Q space exceeds its maximum size
  static void check_qspace_size_limits(std::size_t min_size, std::size_t max_size) {
    if (min_size > max_size)
      throw std::invalid_argument("minimum size of the Q space (" + std::to_string(min_size) +
                                  ") exceeds its maximum size (" + std::to_string(max_size) + ")");
  }
  //! Sets the minimum and maximum size of the Q space, after checking that the minimum does not exceed the maximum
  void set_qspace_size_limits(std::size_t min_size, std::size_t max_size) {
    check_qspace_size_limits(min_size, max_size);
    qspace_opts.min_size = min_size;
    qspace_opts.max_size = max_size;
    if (m_dspace_resetter.get_max_Qsize() > qspace_opts.max_size)
      m_dspace_resetter.set_max_Qsize(qspace_opts.max_size);
  }

  //! The values of the current solutions, whose changes between iterations are the value errors
  virtual std::vector<value_type> values() const = 0;

  void set_value_errors() override {
    const auto current_values = values();
    this->m_value_errors.assign(current_values.size(), std::numeric_limits<value_type_abs>::max());
    for (size_t i = 0; i < std::min(m_last_values.size(), current_values.size()); i++)
      this->m_value_errors[i] = std::abs(current_values[i] - m_last_values[i]);
    // resetting the D space does not change the solutions, so the change measured across it would be meaningless
    if (!m_resetting_in_progress)
      m_last_values = current_values;
  }

  RSpaceOptions<value_type_abs> rspace_opts;   //!< Options concerning R-space handling
  QSpaceOptions qspace_opts;                   //!< Options concerning Q-space handling
  detail::DSpaceResetter<Q> m_dspace_resetter; //!< resets D space
  bool m_hermiticity = false;                  //!< whether the problem is hermitian or not
  std::vector<value_type> m_last_values;       //!< the values from the previous iteration
  bool m_resetting_in_progress = false;        //!< whether D space resetting is in progress
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_DAVIDSONCOMMON_H
