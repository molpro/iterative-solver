#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVERTEMPLATE_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVERTEMPLATE_H
#include <molpro/Profiler.h>
#include <molpro/iostream.h>
#include <molpro/linalg/itsolv/IterativeSolver.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/subspace/ISubspaceSolver.h>
#include <molpro/linalg/itsolv/subspace/IXSpace.h>
#include <molpro/linalg/itsolv/subspace/Matrix.h>
#include <molpro/linalg/itsolv/subspace/check_metric.h>
#include <molpro/linalg/itsolv/subspace/util.h>
#include <molpro/linalg/itsolv/util.h>
#include <molpro/linalg/itsolv/wrap.h>
#include <molpro/profiler/Profiler.h>

#include <cassert>
#include <cmath>
#include <format>
#include <iostream>
#include <optional>
#include <sstream>

namespace molpro::linalg::itsolv {
namespace detail {

inline std::vector<std::pair<size_t, size_t>> parameter_batches(const size_t nsol, const size_t nparam) {
  auto batches = std::vector<std::pair<size_t, size_t>>{};
  if (nparam && nsol) {
    auto n_batch = nsol / nparam + (nsol % nparam ? 1 : 0);
    for (size_t ib = 0, start_sol = 0, end_sol = 0; ib < n_batch; ++ib, start_sol = end_sol) {
      end_sol = std::min(start_sol + nparam, nsol);
      batches.emplace_back(start_sol, end_sol);
    }
  }
  return batches;
}

template <class R, class Q, class P, typename value_type>
void construct_solution(const VecRef<R>& params, const std::vector<int>& roots,
                        const subspace::Matrix<value_type>& solutions,
                        const std::vector<std::reference_wrapper<P>>& pparams,
                        const std::vector<std::reference_wrapper<Q>>& qparams,
                        const std::vector<std::reference_wrapper<Q>>& dparams, size_t oP, size_t oQ, size_t oD,
                        ArrayHandlers<R, Q, P>& handlers) {
  if (roots.empty())
    return;
  auto prof = molpro::Profiler::single();
  prof->start("get rd_mat");
  assert(params.size() >= roots.size());
  for (size_t i = 0; i < roots.size(); ++i) {
    handlers.rr().fill(0, params.at(i));
  }
  subspace::Matrix<value_type> rp_mat(std::make_pair(pparams.size(), roots.size())),
      rq_mat(std::make_pair(qparams.size(), roots.size())), rd_mat(std::make_pair(dparams.size(), roots.size()));
  for (size_t i = 0; i < roots.size(); ++i) {
    for (size_t j = 0; j < pparams.size(); ++j) {
      rp_mat(j, i) = solutions(roots[i], oP + j);
    }
    for (size_t j = 0; j < qparams.size(); ++j) {
      rq_mat(j, i) = solutions(roots[i], oQ + j);
    }
    for (size_t j = 0; j < dparams.size(); ++j) {
      rd_mat(j, i) = solutions(roots[i], oD + j);
    }
  }
  prof->stop();
  handlers.rp().gemm_outer(rp_mat, cwrap(pparams), params);
  handlers.rq().gemm_outer(rq_mat, cwrap(qparams), params);
  handlers.rq().gemm_outer(rd_mat, cwrap(dparams), params);
}

//! The actions of a metric applied on the fly on solutions, in R vectors or, if there are not enough of them, Q vectors
template <class R, class Q>
struct MetricSolutions {
  CVecRef<R> r;
  CVecRef<Q> q;
  bool empty() const { return r.empty() && q.empty(); }
};

template <typename T>
std::vector<std::vector<T>> construct_vectorP(const std::vector<int>& roots, const subspace::Matrix<T>& solutions,
                                              const size_t oP, const size_t nP) {
  auto vectorP = std::vector<std::vector<T>>{};
  for (auto root : roots) {
    vectorP.emplace_back();
    for (size_t j = 0; j < nP; ++j)
      vectorP.back().push_back(solutions(root, oP + j));
  }
  return vectorP;
}

template <class R, typename T>
void update_errors(std::vector<T>& errors, const CVecRef<R>& residual, array::ArrayHandler<R, R>& handler) {
  assert(residual.size() >= errors.size());
  for (size_t i = 0; i < errors.size(); ++i) {
    auto a = handler.dot(residual[i], residual[i]);
    errors[i] = std::sqrt(std::abs(a));
  }
}

template <typename T>
std::vector<int> select_working_set(const size_t nw, const std::vector<T>& errors, const T threshold,
                                    const std::vector<T>& value_errors, const T value_threshold) {
  auto ordered_errors = std::multimap<T, size_t, std::greater<T>>{};
  for (size_t i = 0; i < errors.size(); ++i) {
    if (errors[i] > threshold or (i < value_errors.size() and value_errors[i] > value_threshold))
      ordered_errors.emplace(errors[i], i);
  }
  auto working_set = std::vector<int>{};
  auto end = (ordered_errors.size() < nw ? ordered_errors.end() : next(begin(ordered_errors), nw));
  std::transform(begin(ordered_errors), end, std::back_inserter(working_set), [](const auto& el) { return el.second; });
  std::sort(working_set.begin(), working_set.end());
  return working_set;
}

} // namespace detail

namespace log {

/*!
 * Context used when logging about the beginning of a new iteration
 */
struct NewIteration : ContextBase<NewIteration, true, int, std::vector<double>> {
  static constexpr const char *name = "NewIteration";
  static constexpr std::size_t iter = 0;
  static constexpr std::size_t errors = 1;
};
static_assert(context<NewIteration>);

/*!
 * Context used when logging a per-iteration report message
 */
struct IterationReport : ContextBase<IterationReport, true> {
  static constexpr const char *name = "IterationReport";
};
static_assert(context<IterationReport>);

/*!
 * Context used when logging a summary report message
 */
struct SummaryReport : ContextBase<SummaryReport, true> {
  static constexpr const char *name = "SummaryReport";
};
static_assert(context<SummaryReport>);

}

/*!
 * @brief Implements IterativeSolver interface that is common to all solvers
 *
 * @tparam Solver one of the solvers that inherits from IterativeSolver.
 */
template <template <class, class, class> class Solver, class R, class Q, class P>
class IterativeSolverTemplate : public Solver<R, Q, P> {
public:
  using typename Solver<R, Q, P>::fapply_on_p_type;
  using typename Solver<R, Q, P>::fapply_metric_type;
  using typename Solver<R, Q, P>::scalar_type;
  using typename Solver<R, Q, P>::value_type;
  using typename Solver<R, Q, P>::value_type_abs;
  using typename Solver<R, Q, P>::VectorP;

  IterativeSolverTemplate() = delete;
  IterativeSolverTemplate(const IterativeSolverTemplate<Solver, R, Q, P>&) = delete;
  IterativeSolverTemplate(IterativeSolverTemplate<Solver, R, Q, P>&&) noexcept = default;
  IterativeSolverTemplate<Solver, R, Q, P>& operator=(const IterativeSolverTemplate<Solver, R, Q, P>&) = delete;
  IterativeSolverTemplate<Solver, R, Q, P>& operator=(IterativeSolverTemplate<Solver, R, Q, P>&&) noexcept = default;

  void set_logger(std::shared_ptr<Logger> logger) override {
    assert(logger);
    m_logger = std::move(logger);
    m_xspace->set_logger(m_logger);
    m_subspace_solver->set_logger(m_logger);
  }

  Logger &logger() override { return *m_logger; }

  using Solver<R, Q, P>::add_vector;
  using Solver<R, Q, P>::solve;

  int add_vector(const VecRef<R>& parameters, const VecRef<R>& actions) override {
    if (m_metric_kind != MetricKind::Identity)
      throw std::logic_error("The actions of the metric declared with set_metric() have to be given to add_vector()");
    return add_vector_impl(parameters, actions, {});
  }

  int add_vector(const VecRef<R>& parameters, const VecRef<R>& actions, const VecRef<R>& metric_actions) override {
    if (m_metric_kind == MetricKind::Identity)
      throw std::logic_error("Actions of a metric given to add_vector(), but no metric was declared with set_metric()");
    const auto nW = n_added_vectors(parameters);
    if (metric_actions.size() < nW)
      throw std::runtime_error("IterativeSolver::add_vector(): fewer actions of the metric than parameters");
    return add_vector_impl(parameters, actions, wrap(begin(metric_actions), begin(metric_actions) + nW));
  }

  bool supports_metric(MetricKind kind) const override { return kind == MetricKind::Identity; }

  void set_metric(MetricKind kind, fapply_metric_type apply_metric = {}) override {
    if (!supports_metric(kind))
      throw std::invalid_argument("This solver does not support the requested kind of metric");
    if (kind == MetricKind::Identity && apply_metric)
      throw std::invalid_argument("A function applying the metric was given for the identity");
    const bool store_actions = !apply_metric;
    if (m_xspace->dimensions().nX() != 0 &&
        (kind != m_metric_kind || (kind != MetricKind::Identity && store_actions != !m_apply_metric)))
      throw std::logic_error("The metric, or whether its actions are stored, cannot be changed once the subspace is not "
                             "empty");
    m_metric_kind = kind;
    m_apply_metric = std::move(apply_metric);
    m_xspace->set_metric(kind != MetricKind::Identity, store_actions);
  }

  MetricKind metric_kind() const override { return m_metric_kind; }

  int add_vector(std::vector<R>& parameters, std::vector<R>& actions) override {
    return add_vector(wrap(parameters), wrap(actions));
  }
  int add_vector(R& parameters, R& actions, value_type value = 0) override {
    return add_vector(wrap_arg(parameters), wrap_arg(actions));
  }

  size_t add_p(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix, const VecRef<R>& parameters,
               const VecRef<R>& actions, fapply_on_p_type apply_p) override {
    if (m_metric_kind != MetricKind::Identity && !pparams.empty())
      throw std::logic_error("With a metric, the P space has to be added with the overload of add_p() taking the metric "
                             "matrix");
    return add_p_impl(pparams, pp_action_matrix, {}, parameters, actions, std::move(apply_p));
  }

  size_t add_p(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix,
               const array::Span<value_type>& pp_metric_matrix, const VecRef<R>& parameters, const VecRef<R>& actions,
               fapply_on_p_type apply_p, fapply_on_p_type apply_metric_p) override {
    if (m_metric_kind == MetricKind::Identity)
      throw std::logic_error("A metric matrix in the P space was given, but no metric was declared with set_metric()");
    if (!pparams.empty() && m_xspace->stores_metric_actions() && !apply_metric_p)
      throw std::invalid_argument("A function applying the metric in the P space is needed unless the metric is "
                                  "applied on the fly");
    if (apply_metric_p)
      m_apply_metric_p = std::move(apply_metric_p);
    return add_p_impl(pparams, pp_action_matrix, pp_metric_matrix, parameters, actions, std::move(apply_p));
  }

  void clearP() override {}

  void solution(const std::vector<int>& roots, const VecRef<R>& parameters, const VecRef<R>& residual) override {
    solution_impl(roots, parameters, residual, {});
  }

  /*!
   * @brief Implements solution()
   * @param scratch R vectors that may be overwritten, for the actions of a metric applied on the fly; if there are
   * fewer than roots, Q vectors are used instead
   */
  void solution_impl(const std::vector<int>& roots, const VecRef<R>& parameters, const VecRef<R>& residual,
                     const VecRef<R>& scratch) {
    // auto prof = profiler()->push("itsolv::solution"); // FIXME two profilers
    auto prof = molpro::Profiler::single();
    check_consistent_number_of_roots_and_solutions(roots, parameters.size());
    prof->start("construct_solution (parameters)");
    detail::construct_solution(parameters, roots, m_subspace_solver->solutions(), m_xspace->paramsp(),
                               m_xspace->paramsq(), m_xspace->paramsd(), m_xspace->dimensions().oP(),
                               m_xspace->dimensions().oQ(), m_xspace->dimensions().oD(), *m_handlers);
    prof->stop();
    auto metric_solutions = detail::MetricSolutions<R, Q>{};
    auto metric_solutions_q = std::vector<Q>{};
    if (m_apply_metric && !roots.empty()) {
      prof->start("apply metric");
      const auto solutions = cwrap(parameters.begin(), parameters.begin() + roots.size());
      const bool use_scratch = scratch.size() >= roots.size();
      // without scratch vectors, the residuals hold the actions of the metric until they are copied
      const auto metric_actions = use_scratch ? wrap(scratch.begin(), scratch.begin() + roots.size())
                                              : wrap(residual.begin(), residual.begin() + roots.size());
      m_apply_metric(solutions, metric_actions);
      if (use_scratch) {
        metric_solutions.r = cwrap(metric_actions);
      } else {
        for (const auto& m : metric_actions)
          metric_solutions_q.emplace_back(m_handlers->qr().copy(m));
        m_stats->q_creations += metric_solutions_q.size();
        metric_solutions.q = cwrap(metric_solutions_q);
      }
      prof->stop();
    }
    prof->start("construct_solution (residual)");
    detail::construct_solution(residual, roots, m_subspace_solver->solutions(), {}, m_xspace->actionsq(),
                               m_xspace->actionsd(), m_xspace->dimensions().oP(), m_xspace->dimensions().oQ(),
                               m_xspace->dimensions().oD(), *m_handlers);
    prof->stop();
    prof->start("apply");
    auto pvectors = detail::construct_vectorP(roots, m_subspace_solver->solutions(), m_xspace->dimensions().oP(),
                                              m_xspace->dimensions().nP);
    if (m_apply_p)
      m_apply_p(pvectors, m_xspace->cparamsp(), residual);
    construct_residual(roots, cwrap(parameters), residual, metric_solutions);
    read_handler_counts(m_stats, m_handlers);
    prof->stop();
  }

  void solution(const std::vector<int>& roots, std::vector<R>& parameters, std::vector<R>& residual) override {
    return solution(roots, wrap(parameters), wrap(residual));
  }
  void solution(R& parameters, R& residual) override {
    return solution(std::vector<int>(1, 0), wrap_arg(parameters), wrap_arg(residual));
  }

  void solution_params(const std::vector<int>& roots, std::vector<R>& parameters) override {
    return solution_params(roots, wrap(parameters));
  }

  void solution_params(const std::vector<int>& roots, const VecRef<R>& parameters) override {
    check_consistent_number_of_roots_and_solutions(roots, parameters.size());
    detail::construct_solution(parameters, roots, m_subspace_solver->solutions(), m_xspace->paramsp(),
                               m_xspace->paramsq(), m_xspace->paramsd(), m_xspace->dimensions().oP(),
                               m_xspace->dimensions().oQ(), m_xspace->dimensions().oD(), *m_handlers);
  };

  void solution_params(R& parameters) override { return solution_params(std::vector<int>(1, 0), wrap_arg(parameters)); }

  // TODO Implement this
  std::vector<size_t> suggest_p(const CVecRef<R>& solution, const CVecRef<R>& residual, size_t max_number,
                                value_type_abs threshold) override {
    return {};
  }

  const std::vector<int>& working_set() const override { return m_working_set; }

  size_t n_roots() const override { return m_nroots; }

  void set_n_roots(size_t roots) override {
    m_nroots = roots;
    m_working_set.resize(roots);
    std::iota(begin(m_working_set), end(m_working_set), (int)0);
  }

  void set_options(const Options& options) override {
    if (options.n_roots)
      set_n_roots(options.n_roots.value());
    if (options.convergence_threshold)
      set_convergence_threshold(options.convergence_threshold.value());
    if (options.verbosity)
      set_verbosity(options.verbosity.value());
    if (options.max_iter)
      set_max_iter(options.max_iter.value());
    if (options.max_p)
      set_max_p(options.max_p.value());
    if (options.p_threshold)
      set_p_threshold(options.p_threshold.value());
  }

  std::shared_ptr<Options> get_options() const override {
    auto options = std::make_shared<Options>();
    options->n_roots = n_roots();
    options->convergence_threshold = convergence_threshold();
    options->verbosity = get_verbosity();
    options->max_iter = get_max_iter();
    options->max_p = get_max_p();
    options->p_threshold = get_p_threshold();
    return options;
  }

  const std::vector<value_type_abs>& errors() const override { return m_errors; }

  const Statistics& statistics() const override { return *m_stats; }

  void report(std::ostream& cout, bool endl = true) const override {
    cout << "iteration " << m_stats->iterations;
    if (not m_errors.empty()) {
      auto it_max_error = std::max_element(m_errors.cbegin(), m_errors.cend());
      if (n_roots() > 1)
        cout << ", |residual[" << std::distance(m_errors.cbegin(), it_max_error) << "]| = ";
      else
        cout << ", |residual| = ";
      cout << std::scientific << *it_max_error << std::defaultfloat;
    }
    if (endl)
      cout << std::endl;
  }

  void iteration_report() const {
    std::stringstream sstream;
    report(sstream, false);
    this->m_logger->template info<log::IterationReport>(sstream.str());
  }

  void summary_report() const {
    std::stringstream sstream;
    report(sstream, false);
    this->m_logger->template info<log::SummaryReport>(sstream.str());
  }

  void report() const override {
    // We don't know what kind of report this is supposed to be -> default to summary
    summary_report();
  }

  void set_convergence_threshold(value_type_abs thresh) override { m_convergence_threshold = thresh; }
  value_type_abs convergence_threshold() const override { return m_convergence_threshold; }
  void set_convergence_threshold_value(value_type_abs thresh) override { m_convergence_threshold_value = thresh; }
  value_type_abs convergence_threshold_value() const override { return m_convergence_threshold_value; }
  void set_verbosity(Verbosity v) override { m_verbosity = v; }
  void set_verbosity(int v) override {
    if (v == 0) {
      m_logger->set_verbosity(log::Verbosity::None);
      m_logger->set_min_severity(log::Severity::Error);
    } else if (v == 1) {
      m_logger->set_verbosity(log::Verbosity::None);
      m_logger->set_min_severity(log::Severity::Warning);
    } else if (v == 2) {
      m_logger->set_verbosity(log::Verbosity::Info);
      m_logger->set_min_severity(log::Severity::Normal);
    } else if (v == 3) {
      m_logger->set_verbosity(log::Verbosity::Debug);
      m_logger->set_min_severity(log::Severity::Normal);
    } else {
      m_logger->set_verbosity(log::Verbosity::Trace);
      m_logger->set_min_severity(log::Severity::Normal);
    }
  }
  Verbosity get_verbosity() const override { return m_verbosity.value_or(Verbosity::None); }
  void set_max_iter(int n) override { m_max_iter = n; }
  int get_max_iter() const override { return m_max_iter; }
  void set_max_p(int n) override { m_max_p = n; }
  int get_max_p() const override { return m_max_p; }
  void set_p_threshold(value_type_abs threshold) override { m_p_threshold = threshold; }
  value_type_abs get_p_threshold() const override { return m_p_threshold; }
  //! Access dimensions of the subspace
  const subspace::Dimensions& dimensions() const override { return m_xspace->dimensions(); }
  scalar_type value() const override {
    return m_xspace->data.count(subspace::EqnData::value) > 0 ? m_xspace->data[subspace::EqnData::value](0, 0)
                                                              : nan("molpro::linalg::itsolv::IterativeSolver::value");
  }
  void set_profiler(molpro::profiler::Profiler& profiler) override {
    // non-owning: the profiler belongs to the caller and must not be deleted by the solver
    m_profiler = std::shared_ptr<molpro::profiler::Profiler>(std::shared_ptr<molpro::profiler::Profiler>{}, &profiler);
  }
  void set_profiler(std::shared_ptr<molpro::profiler::Profiler> profiler) override {
    if (!profiler)
      throw std::invalid_argument("IterativeSolver::set_profiler: null profiler");
    m_profiler = std::move(profiler);
  }
  const std::shared_ptr<molpro::profiler::Profiler>& profiler() const override { return m_profiler; }

  bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const Problem<R>& problem,
             bool generate_initial_guess = false) override {
    if (problem.metric_kind() != MetricKind::Identity)
      throw std::invalid_argument("A problem with a metric has to be solved with the overload of "
                                  "IterativeSolver::solve() taking buffers for the actions of the metric");
    return solve_impl(parameters, actions, {}, problem, generate_initial_guess);
  }

  bool solve(const VecRef<R>& parameters, const VecRef<R>& actions, const VecRef<R>& metric_actions,
             const Problem<R>& problem, bool generate_initial_guess = false) override {
    if (parameters.size() != metric_actions.size())
      throw std::runtime_error("Inconsistent container sizes in IterativeSolver::solve()");
    return solve_impl(parameters, actions, metric_actions, problem, generate_initial_guess);
  }

  //! Implements solve(), with empty metric_actions for a problem without a metric
  bool solve_impl(const VecRef<R>& parameters, const VecRef<R>& actions, const VecRef<R>& metric_actions,
                  const Problem<R>& problem, bool generate_initial_guess) {
    if (parameters.empty())
      throw std::runtime_error("Empty container passed to IterativeSolver::solve()");
    if (parameters.size() != actions.size())
      throw std::runtime_error("Inconsistent container sizes in IterativeSolver::solve()");
    if (this->m_verbosity == Verbosity::None) {
      // Backwards compatibility
      this->m_logger->set_verbosity(log::Verbosity::None);
    }
    if (this->m_verbosity == Verbosity::Detailed) {
      this->m_logger->set_verbosity(log::Verbosity::Trace);
      this->m_logger->enable_data_dumps(true);
    }
    const bool metric = problem.metric_kind() != MetricKind::Identity;
    if (metric and problem.metric_is_cheap())
      set_metric(problem.metric_kind(), [&problem](const CVecRef<R>& parameters, const VecRef<R>& metric_actions) {
        problem.metric_action(parameters, metric_actions);
      });
    else
      set_metric(problem.metric_kind());
    bool use_diagonals = problem.diagonals(actions.at(0));
    std::unique_ptr<Q> diagonals;
    if (use_diagonals)
      diagonals.reset(new Q{m_handlers->qr().copy(actions.at(0))});
    std::unique_ptr<Q> metric_diagonals;
    if (metric and use_diagonals) {
      if (not problem.metric_diagonals(actions.at(0)))
        throw std::runtime_error("The problem provides diagonals() but not metric_diagonals(), without which the "
                                 "default preconditioner would ignore the metric");
      metric_diagonals.reset(new Q{m_handlers->qr().copy(actions.at(0))});
    }
    if (generate_initial_guess and not use_diagonals)
      throw std::runtime_error("Default initial guess requested, but diagonal elements are not available");
    // the initial guess and the P space are selected from the lowest of these values
    const Q* selection = diagonals.get();
    std::unique_ptr<Q> ratios;
    if (metric_diagonals and (generate_initial_guess or m_max_p > 0)) {
      // the lowest eigenvalues of a generalised eigenproblem are expected at the lowest ratios of the diagonals
      m_handlers->rq().copy(parameters.at(0), *diagonals);
      precondition_default(wrap_arg(parameters.at(0).get()), std::vector<value_type>{0}, actions.at(0).get());
      ratios.reset(new Q{m_handlers->qr().copy(parameters.at(0))});
      selection = ratios.get();
    }
    int nwork = parameters.size();
    std::vector<P> pspace;
    if (use_diagonals and m_max_p > 0)
      pspace = select_pspace(*selection);
    // a P space replaces any initial guess; without one, max_p > 0 still means the contents of parameters are not used
    if (pspace.empty() and (generate_initial_guess or (use_diagonals and m_max_p > 0)))
      generate_initial_guess_from_diagonals(parameters, *selection);
    if (!pspace.empty())
      nwork = initialise_pspace(pspace, problem, parameters, actions);
    for (auto iter = 0; iter < this->m_max_iter && nwork > 0; iter++) {
      // the logging context carries the errors as doubles: a log line only needs their magnitude
      m_logger->template info<log::NewIteration>("Iteration, Current errors", iter,
                                        std::vector<double>(m_errors.begin(), m_errors.end()));
      value_type value;
      if (this->nonlinear()) {
        value = problem.residual(*parameters.begin(), *actions.begin());
        nwork = this->add_vector(*parameters.begin(), *actions.begin(), value);
      } else if (iter > 0 or pspace.empty()) {
        // add_vector() takes up no more vectors than there are in the working set
        const auto naction = std::min(size_t(nwork), m_working_set.size());
        const auto wparameters = cwrap(parameters.begin(), parameters.begin() + naction);
        const auto wactions = wrap(actions.begin(), actions.begin() + naction);
        if (naction > 0 and metric)
          problem.action_with_metric(wparameters, wactions,
                                     wrap(metric_actions.begin(), metric_actions.begin() + naction));
        else if (naction > 0)
          problem.action(wparameters, wactions);
        nwork = metric ? this->add_vector(parameters, actions, metric_actions) : this->add_vector(parameters, actions);
      }
      while (this->end_iteration_needed()) {
        if (nwork > 0) {
          if (metric_diagonals) {
            // parameters[0] and metric_actions[0] are overwritten to hand the diagonals to the preconditioner
            m_handlers->rq().copy(parameters.at(0), *diagonals);
            m_handlers->rq().copy(metric_actions.at(0), *metric_diagonals);
            problem.precondition(wrap(actions.begin(), actions.begin() + nwork), this->working_set_eigenvalues(),
                                 parameters.at(0), metric_actions.at(0));
          } else if (use_diagonals) {
            // parameters[0] is overwritten to hand the diagonals to the preconditioner
            m_handlers->rq().copy(parameters.at(0), *diagonals);
            problem.precondition(wrap(actions.begin(), actions.begin() + nwork), this->working_set_eigenvalues(),
                                 parameters.at(0));
          } else
            problem.precondition(wrap(actions.begin(), actions.begin() + nwork), this->working_set_eigenvalues());
        }
        nwork = this->end_iteration(parameters, actions);
      }
      if (verbosity_at_least(Verbosity::Iteration)) {
        iteration_report();
      }
    }

    this->finalize();

    if (verbosity_at_least(Verbosity::Summary)) {
      summary_report();
    }
    // without any errors (e.g. no iteration was done) nothing is known to have converged
    const bool errors_converged =
        !m_errors.empty() && *std::max_element(m_errors.begin(), m_errors.end()) <= m_convergence_threshold;
    if (verbosity_at_least(Verbosity::Summary)) {
      if (!errors_converged) {
        this->m_logger->warn("Solver has not converged to threshold ", m_convergence_threshold);
      } else {
        this->m_logger->info("Solver converged");
      }
    }
    return nwork == 0 and errors_converged;
  }

  bool solve(R& parameters, R& actions, const Problem<R>& problem, bool generate_initial_guess = false) override {
    auto wparams = std::vector<std::reference_wrapper<R>>{std::ref(parameters)};
    auto wactions = std::vector<std::reference_wrapper<R>>{std::ref(actions)};
    return solve(wparams, wactions, problem, generate_initial_guess);
  }
  bool solve(std::vector<R>& parameters, std::vector<R>& actions, const Problem<R>& problem,
             bool generate_initial_guess = false) override {
    return solve(wrap(parameters), wrap(actions), problem, generate_initial_guess);
  }

  bool test_problem(const Problem<R>& problem, R& v0, R& v1, int verbosity, value_type_abs threshold) const override {
    bool success = true;
    value_type_abs step = 1e-4;
    if (this->nonlinear()) {
      if (!problem.test_parameters(0, v0))
        return true;
      auto value0 = problem.residual(v0, v1);
      if (verbosity > 1) {
        std::cout << "value0 " << value0 << std::endl;
        //        std::cout << "parameters0 " << v0[0] << "," << v0[1] << ",..." << std::endl;
        //        std::cout << "residual0 " << v1[0] << "," << v1[1] << ",..." << std::endl;
      }
      Q parameters0 = m_handlers->qr().copy(v0);
      Q residual0 = m_handlers->qr().copy(v1);
      for (int instance = 1; problem.test_parameters(instance, v0); ++instance) {
        m_handlers->rq().axpy(-1.0, parameters0, v0);
        m_handlers->rr().scal(1 / std::sqrt(std::abs(m_handlers->rr().dot(v0, v0))), v0);
        Q step1 = m_handlers->qr().copy(v0);
        auto residual_analytic = m_handlers->rq().dot(v0, residual0);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(-2*step, step1, v0);
        auto valuem2 = problem.residual(v0, v1);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(-1*step, step1, v0);
        auto valuem1 = problem.residual(v0, v1);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(+1*step, step1, v0);
        auto valuep1 = problem.residual(v0, v1);
        m_handlers->rq().copy(v0, parameters0);
        m_handlers->rq().axpy(+2*step, step1, v0);
        auto valuep2 = problem.residual(v0, v1);
        auto residual_numerical = (valuem2 - value_type(8) * valuem1 + value_type(8) * valuep1 - valuep2) /
                                  value_type(12 * step);
        if (verbosity > 1)
          std::cout << "testing problem class, instance: " << instance << ", numerical: " << residual_numerical <<", analytical: "<<residual_analytic <<", difference: "<<residual_numerical-residual_analytic << std::endl;
        success = success && std::abs(residual_numerical - residual_analytic) < threshold;
      }
    } else {
      // checks that the operator applied by apply(x, y) scales linearly; name describes its result
      auto check_scaling = [&](const auto& apply, const std::string& name) {
        apply(v0, v1);
        Q residual = m_handlers->qr().copy(v1);
        auto norm2_residual = std::sqrt(m_handlers->rr().dot(v1, v1));
        const value_type_abs scale_factor{10.0};
        m_handlers->rr().scal(scale_factor, v0);
        apply(v0, v1);
        m_handlers->rq().axpy(-scale_factor, residual, v1);
        auto norm2 = std::sqrt(m_handlers->rr().dot(v1, v1));
        success = success && std::abs(norm2 / norm2_residual) < threshold;
        if (verbosity > 0 or (verbosity > -1 and not success))
          std::cout << "Length of " << name << ": " << norm2_residual << ", scaling defect: " << norm2 << std::endl;
      };
      for (int instance = 0; problem.test_parameters(instance, v0); ++instance) {
        check_scaling([&problem](R& x, R& y) { problem.action({x}, {y}); }, "residual");
        if (problem.metric_kind() != MetricKind::Identity)
          check_scaling([&problem](R& x, R& y) { problem.metric_action({x}, {y}); }, "action of the metric");
      }
      if (problem.metric_kind() != MetricKind::Identity)
        success = test_metric(problem, v0, v1, verbosity, threshold) && success;
    }
    return success;
  }

protected:
  IterativeSolverTemplate(std::shared_ptr<subspace::IXSpace<R, Q, P>> xspace,
                          std::shared_ptr<subspace::ISubspaceSolver<R, Q, P>> solver,
                          std::shared_ptr<ArrayHandlers<R, Q, P>> handlers, std::shared_ptr<Statistics> stats,
                          std::shared_ptr<Logger> logger)
      : m_handlers(std::move(handlers)), m_xspace(std::move(xspace)), m_subspace_solver(std::move(solver)),
        m_stats(std::move(stats)), m_logger(std::move(logger)), m_profiler(molpro::Profiler::single()),
        m_profiler_saved_depth(m_profiler->get_max_depth()) {
    set_n_roots(1);
    m_profiler->set_max_depth(options()->parameter("PROFILER_DEPTH", 0));
  }

  // The destructor is implicitly noexcept, so any exception from writing the profiler output would terminate the
  // program. Report such failures and carry on.
  virtual ~IterativeSolverTemplate() {
    try {
      if (molpro::mpi::rank_global() == 0) {
        auto file = options()->parameter("PROFILER_OUTPUT", "");
        if (profiler()->get_max_depth() > 0 and
            std::find_if(file.begin(), file.end(), [](unsigned char ch) { return !std::isspace(ch); }) != file.end())
          std::ofstream(file) << *profiler() << std::endl;
      }
      if (molpro::mpi::rank_global() == 0) {
        auto file = options()->parameter("PROFILER_DOTGRAPH", "");
        if (profiler()->get_max_depth() > 0 and
            std::find_if(file.begin(), file.end(), [](unsigned char ch) { return !std::isspace(ch); }) != file.end())
          profiler()->dotgraph(file, options()->parameter("PROFILER_THRESHOLD", .01));
      }
    } catch (const std::exception& e) {
      std::cerr << "~IterativeSolverTemplate: failed to write profiler output: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "~IterativeSolverTemplate: failed to write profiler output" << std::endl;
    }
    molpro::Profiler::single()->set_max_depth(m_profiler_saved_depth);
  }

  /*!
   * @brief Tests that the metric of a problem is Hermitian, <u|M v> = <M u|v>, and, if its kind says so, positive
   * (semi)definite, on the first two test parameters
   * @return false if errors were found, otherwise true
   */
  bool test_metric(const Problem<R>& problem, R& v0, R& v1, int verbosity, value_type_abs threshold) const {
    bool success = true;
    const bool positive = problem.metric_kind() == MetricKind::HermitianPositiveDefinite ||
                          problem.metric_kind() == MetricKind::HermitianPositiveSemidefinite;
    std::optional<std::pair<Q, Q>> u; // the first test parameter and its action of the metric
    for (int instance = 0; instance < 2 && problem.test_parameters(instance, v0); ++instance) {
      problem.metric_action({v0}, {v1});
      const auto norm = std::sqrt(std::abs(m_handlers->rr().dot(v0, v0)));
      const auto norm_metric_action = std::sqrt(std::abs(m_handlers->rr().dot(v1, v1)));
      const auto vmv = m_handlers->rr().dot(v0, v1);
      if (std::abs(vmv - conjugate(vmv)) / 2 > threshold * norm * norm_metric_action or
          (positive and real_part(vmv) < -threshold * norm * norm_metric_action)) {
        success = false;
        if (verbosity > -1)
          std::cout << "<v|M v> = " << vmv << " is not real" << (positive ? " and non-negative" : "") << std::endl;
      }
      if (u) {
        const auto umv = conjugate(m_handlers->rq().dot(v1, u->first));
        const auto muv = m_handlers->rq().dot(v0, u->second);
        const auto scale = std::sqrt(std::abs(m_handlers->qq().dot(u->first, u->first)) *
                                     std::abs(m_handlers->qq().dot(u->second, u->second))) *
                           norm * norm_metric_action;
        const auto defect = std::abs(umv - conjugate(muv));
        if (defect > threshold * std::sqrt(scale)) {
          success = false;
          if (verbosity > -1)
            std::cout << "The metric is not Hermitian: <u|M v> = " << umv << ", <M u|v> = " << conjugate(muv)
                      << std::endl;
        } else if (verbosity > 0)
          std::cout << "Hermiticity defect of the metric: " << defect << std::endl;
      } else
        u.emplace(m_handlers->qr().copy(v0), m_handlers->qr().copy(v1));
    }
    return success;
  }

  //! Implementation class should overload this to set errors in the current values (e.g. change in eigenvalues)
  virtual void set_value_errors() {}
  /*!
   * @brief Constructs residual for given roots provided their parameters and actions
   * @param metric_solutions the actions of the metric on params if it is applied on the fly, otherwise empty
   */
  virtual void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions,
                                  const detail::MetricSolutions<R, Q>& metric_solutions) = 0;
  virtual bool linearEigensystem() const { return false; }

  // FIXME Currently only works if called on an empty subspace. Either enforce it or generalise.
  //! Implements add_p(), with an empty pp_metric_matrix for a problem without a metric
  size_t add_p_impl(const CVecRef<P>& pparams, const array::Span<value_type>& pp_action_matrix,
                    const array::Span<value_type>& pp_metric_matrix, const VecRef<R>& parameters,
                    const VecRef<R>& actions, fapply_on_p_type apply_p) {
    auto prof = profiler()->push("itsolv::add_p");
    if (not pparams.empty() and pparams.size() < n_roots())
      throw std::runtime_error("P space must be empty or at least as large as number of roots sought");
    if (apply_p)
      m_apply_p = std::move(apply_p);
    m_xspace->update_pspace(pparams, pp_action_matrix, pp_metric_matrix);
    check_metric_positive();
    auto working_set = solve_and_generate_working_set(parameters, actions);
    read_handler_counts(m_stats, m_handlers);
    return working_set;
  }

  //! For a positive (semi)definite metric, throws if the metric matrix shows that it is not
  void check_metric_positive() const {
    if (m_metric_kind == MetricKind::HermitianPositiveDefinite ||
        m_metric_kind == MetricKind::HermitianPositiveSemidefinite)
      subspace::check_metric_positive(m_xspace->data.at(subspace::EqnData::M), m_xspace->data.at(subspace::EqnData::S));
  }

  //! The number of the given parameters that add_vector() takes up: no more than there are in the working set
  size_t n_added_vectors(const VecRef<R>& parameters) const {
    return std::min(m_working_set.size(), parameters.size());
  }

  /*!
   * @brief Implements add_vector()
   * @param metric_actions For a problem with a metric, the actions of the metric on the first n_added_vectors() of
   * parameters, otherwise empty
   */
  int add_vector_impl(const VecRef<R>& parameters, const VecRef<R>& actions, const VecRef<R>& metric_actions) {
    auto outer = profiler()->push("itsolv::add_vector");
    auto prof = molpro::Profiler::single();
    m_logger->trace("IterativeSolverTemplate::add_vector  iteration = ", m_stats->iterations);
    m_logger->debug("IterativeSolverTemplate::add_vector  size of {params, actions, working_set} = ", parameters.size(),
                    actions.size(), m_working_set.size());
    if (m_xspace->dimensions().nP != 0 && !m_apply_p)
      throw std::runtime_error(
          "Solver contains P space but no valid apply_p function. Make sure add_p was called correctly.");
    if (m_xspace->dimensions().nP != 0 && m_xspace->stores_metric_actions() && !m_apply_metric_p)
      throw std::runtime_error("Solver contains P space but no valid function applying the metric to it. Make sure "
                               "add_p was called correctly.");
    auto nW = n_added_vectors(parameters);
    auto cwparams = cwrap(begin(parameters), begin(parameters) + nW);
    auto cwactions = cwrap(begin(actions), begin(actions) + nW);
    m_stats->r_creations += nW;
    prof->start("update_qspace");
    m_xspace->update_qspace(cwparams, cwactions, cwrap(metric_actions));
    m_stats->q_creations += (m_xspace->stores_metric_actions() ? 3 : 2) * nW;
    prof->stop();
    check_metric_positive();
    prof->start("solve_and_generate_working_set");
    // the actions of the metric are not needed any more and serve as scratch space
    auto working_set = solve_and_generate_working_set(parameters, actions, metric_actions);
    prof->stop();
    read_handler_counts(m_stats, m_handlers);
    this->m_end_iteration_needed = true;
    return working_set;
  }
  /*!
   * @brief Solves the subspace problems and selects the working set of roots, returning their parameters and residual
   * in parameters and action
   * @param parameters container for storing parameters of the working set
   * @param action container for storing the residual of the working set
   * @param scratch vectors that may be overwritten, see solution_impl()
   * @return size of the working set
   */
  size_t solve_and_generate_working_set(const VecRef<R>& parameters, const VecRef<R>& action,
                                        const VecRef<R>& scratch = {}) {
    auto prof = profiler();
    auto p = prof->push("itsolv::solve_and_generate_working_set");
    prof->start("itsolv::solve");
    m_subspace_solver->solve(*m_xspace, n_roots());
    prof->stop();
    prof->start("itsolv::temp_solutions");
    auto nsol = m_subspace_solver->size();
    std::vector<std::pair<Q, Q>> temp_solutions{};
    const auto batches = detail::parameter_batches(nsol, parameters.size());
    for (const auto& batch : batches) {
      auto [start_sol, end_sol] = batch;
      auto roots = std::vector<int>(end_sol - start_sol);
      std::iota(begin(roots), end(roots), start_sol);
      solution_impl(roots, parameters, action, scratch);
      auto errors = std::vector<value_type_abs>(roots.size(), 0);
      detail::update_errors(errors, cwrap(action), m_handlers->rr());
      if (m_xspace->has_metric()) {
        // The solutions are normalised in the metric, so their Euclidean length, and with it the length of the
        // residual, would depend on the scale of the metric. The error is that of the solution of unit length instead,
        // which is the same as without a metric.
        for (size_t i = 0; i < roots.size(); ++i)
          errors[i] /= subspace::util::solution_norm(m_subspace_solver->solutions(), roots[i],
                                                     m_xspace->data.at(subspace::EqnData::S));
      }
      if (batches.size() > 1) {
        for (size_t i = 0; i < roots.size(); ++i)
          temp_solutions.emplace_back(m_handlers->qr().copy(parameters[i]), m_handlers->qr().copy(action[i]));
        m_stats->q_creations += 2 * roots.size();
      }
      m_subspace_solver->set_error(roots, errors);
    }
    prof->stop();
    set_value_errors();
    m_errors = m_subspace_solver->errors();
    m_working_set = detail::select_working_set(parameters.size(), m_errors, m_convergence_threshold, m_value_errors,
                                               m_convergence_threshold_value);
    for (size_t i = 0; i < m_working_set.size(); ++i) {
      size_t root = m_working_set[i];
      if (batches.size() > 1) {
        m_handlers->rq().copy(parameters[i], temp_solutions.at(root).first);
        m_handlers->rq().copy(action[i], temp_solutions.at(root).second);
      } else {
        if (root < i)
          throw std::logic_error("incorrect ordering of roots");
        if (root > i && root < parameters.size()) {
          m_handlers->rr().copy(parameters[i], parameters[root]);
          m_handlers->rr().copy(action[i], action[root]);
        }
      }
    }
    m_logger->trace("add_vector::errors = ", m_errors);
    return m_working_set.size();
  }

  template <typename TTT>
  void check_consistent_number_of_roots_and_solutions(const std::vector<TTT>& roots, const size_t nparams) {
    if (roots.size() > nparams)
      throw std::runtime_error("asking for more roots than parameters");
    if (!roots.empty() &&
        size_t(*std::max_element(roots.begin(), roots.end())) >= m_subspace_solver->solutions().size())
      throw std::runtime_error("asking for more roots than there are solutions");
  }

  bool end_iteration_needed() override { return m_end_iteration_needed; }

  void finalize() override {}

  //! Whether output of verbosity v is printed in solve(); everything is printed if no verbosity was set
  bool verbosity_at_least(Verbosity v) const { return !m_verbosity.has_value() || m_verbosity >= v; }

  /*!
   * @brief Fills parameters with unit vectors at the positions of the lowest diagonal elements
   *
   * One vector is generated per root in the working set, as far as parameters can hold them; with an empty working
   * set nothing is generated.
   */
  void generate_initial_guess_from_diagonals(const VecRef<R>& parameters, const Q& diagonals) {
    // add_vector() takes up no more vectors than there are in the working set
    const auto nguess = std::min(parameters.size(), m_working_set.size());
    if (nguess == 0) {
      if (verbosity_at_least(Verbosity::Summary))
        this->m_logger->info("No initial guess generated, as no roots remain in the working set");
      return;
    }
    auto guess = m_handlers->qq().select(nguess, diagonals);
    size_t root = 0;
    if (verbosity_at_least(Verbosity::Summary))
      this->m_logger->info("Initial guess generated from diagonal elements for ", guess.size(), " vectors");
    for (const auto& g : guess) {
      m_handlers->rp().copy(parameters[root], P{{g.first, 1}});
      if (verbosity_at_least(Verbosity::Detailed)) {
        m_logger->debug("-> initial guess with index ", g.first);
      }
      root++;
    }
  }

  /*!
   * @brief Selects the P space from the lowest diagonal elements, subject to max_p and p_threshold
   * @param diagonals the diagonal elements or, for a generalised eigenproblem, their ratios with those of the metric
   * @return the P-space vectors, empty if fewer than n_roots() elements were selected
   */
  std::vector<P> select_pspace(const Q& diagonals) {
    auto selectp = m_handlers->qq().select(m_max_p, diagonals);
    if (!selectp.empty()) {
      // selectp is keyed by index, not value; find the smallest selected
      // diagonal explicitly before applying the threshold.
      // the diagonal of a hermitian operator is real, so it is orderable
      auto min_val = real_part(std::min_element(selectp.begin(), selectp.end(),
                                                [](const auto& a, const auto& b) {
                                                  return real_part(a.second) < real_part(b.second);
                                                })
                                   ->second);
      for (auto s = selectp.begin(); s != selectp.end();) {
        if (real_part(s->second) > min_val + m_p_threshold)
          s = selectp.erase(s);
        else
          ++s;
      }
    }
    if (!selectp.empty() && selectp.size() < n_roots()) {
      this->m_logger->warn("P space smaller than number of roots is not used, starting from the diagonal elements "
                           "instead; dimension and number of roots",
                           selectp.size(), n_roots());
      selectp.clear();
    }
    if (!selectp.empty() && verbosity_at_least(Verbosity::Summary)) {
      this->m_logger->info("P-space dimension, threshold and limit", selectp.size(), m_p_threshold, m_max_p);
    }
    if (verbosity_at_least(Verbosity::Detailed)) {
      for (const auto& s : selectp) {
        this->m_logger->debug(std::format("P space element {}: {:.6e}", s.first, double(real_part(s.second))));
      }
    }
    std::vector<P> pspace;
    for (const auto& s : selectp)
      pspace.emplace_back((P){{s.first, 1}});
    return pspace;
  }

  //! Adds the P space to the subspace using the P-space kernel provided by problem; returns the working-set size
  size_t initialise_pspace(const std::vector<P>& pspace, const Problem<R>& problem, const VecRef<R>& parameters,
                           const VecRef<R>& actions) {
    fapply_on_p_type apply_on_p = [&problem](const std::vector<std::vector<value_type>>& pcoeff,
                                             const CVecRef<P>& pparams, const VecRef<R>& actions) {
      problem.p_action(pcoeff, pparams, actions);
    };
    auto action_matrix = problem.pp_action_matrix(pspace);
    const auto action_span = array::Span<value_type>(action_matrix.data(), action_matrix.size());
    if (m_metric_kind == MetricKind::Identity)
      return add_p(cwrap(pspace), action_span, parameters, actions, apply_on_p);
    fapply_on_p_type apply_metric_on_p = [&problem](const std::vector<std::vector<value_type>>& pcoeff,
                                                    const CVecRef<P>& pparams, const VecRef<R>& metric_actions) {
      problem.p_metric_action(pcoeff, pparams, metric_actions);
    };
    auto metric_matrix = problem.pp_metric_matrix(pspace);
    return add_p(cwrap(pspace), action_span, array::Span<value_type>(metric_matrix.data(), metric_matrix.size()),
                 parameters, actions, apply_on_p, apply_metric_on_p);
  }


  std::shared_ptr<ArrayHandlers<R, Q, P>> m_handlers;                    //!< Array handlers
  std::shared_ptr<subspace::IXSpace<R, Q, P>> m_xspace;                  //!< manages the subspace and associated data
  std::shared_ptr<subspace::ISubspaceSolver<R, Q, P>> m_subspace_solver; //!< solves the subspace problem
  std::vector<value_type_abs> m_errors;       //!< errors from the most recent solution
  std::vector<value_type_abs> m_value_errors; //!< value errors from the most recent solution
  std::vector<int> m_working_set;             //!< indices of roots in the working set
  size_t m_nroots{0};                         //!< number of roots the solver is searching for
  value_type_abs m_convergence_threshold{1.0e-8}; //!< residual norms less than this mark a converged solution
  value_type_abs m_convergence_threshold_value{
      std::numeric_limits<value_type_abs>::max()}; //!< value changes less than this mark a converged solution
  std::shared_ptr<Statistics> m_stats;          //!< accumulates statistics of operations performed by the solver
  std::shared_ptr<Logger> m_logger;             //!< logger
  fapply_on_p_type m_apply_p = {};              //!< function that evaluates effect of action on the P space projection
  std::optional<Verbosity> m_verbosity = {};    //!< how much output to print in solve()
  int m_max_iter = 100;                         //!< maximum number of iterations in solve()
  size_t m_max_p = 0;                           //!< maximum size of P space
  MetricKind m_metric_kind = MetricKind::Identity; //!< the metric declared with set_metric()
  fapply_metric_type m_apply_metric = {};          //!< applies the metric on the fly, if its actions are not stored
  fapply_on_p_type m_apply_metric_p = {}; //!< adds the action of the metric on the P space, if its actions are stored
  value_type_abs m_p_threshold = std::numeric_limits<value_type_abs>::max(); //!< threshold for selecting P space
private:
  mutable std::shared_ptr<molpro::profiler::Profiler> m_profiler;
  int m_profiler_saved_depth; //!< max_depth of molpro::Profiler::single() before this object changed it
protected:
  bool m_end_iteration_needed = true; //!< whether end_iteration should be called after any preconditioner
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_ITERATIVESOLVERTEMPLATE_H
