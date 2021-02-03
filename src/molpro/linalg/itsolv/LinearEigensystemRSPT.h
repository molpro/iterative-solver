//
// Created by Peter Knowles on 18/01/2021.
//

#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_LINEAREIGENSYSTEMRSPT_H_
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_LINEAREIGENSYSTEMRSPT_H_

#include <iterator>
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DSpaceResetter.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/propose_rspace.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverRSPT.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>
#include <molpro/linalg/itsolv/LinearEigensystemRSPTOptions.h>

namespace molpro::linalg::itsolv {

/*!
 * @brief One specific implementation of LinearEigensystem using Davidson's algorithm
 * with modifications to manage near linear dependencies, and consequent numerical noise, in candidate expansion
 * vectors.
 *
 * TODO add more documentation and examples
 *
 * @tparam R
 * @tparam Q
 * @tparam P
 */
template <class R, class Q, class P>
class LinearEigensystemRSPT : public IterativeSolverTemplate<LinearEigensystem, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<LinearEigensystem, R, Q, P>;
  using typename SolverTemplate::scalar_type;
  using IterativeSolverTemplate<LinearEigensystem, R, Q, P>::report;

  explicit LinearEigensystemRSPT(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers,
                                 const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : SolverTemplate(std::make_shared<subspace::XSpace<R, Q, P>>(handlers, logger_),
                       std::static_pointer_cast<subspace::ISubspaceSolver<R, Q, P>>(
                           std::make_shared<subspace::SubspaceSolverRSPT<R, Q, P>>(logger_)),
                       handlers, std::make_shared<Statistics>(), logger_),
        logger(logger_) {
//    set_hermiticity(true);
            auto xspace = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
            xspace->set_hermiticity(true);
            auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverRSPT<R, Q, P>>(this->m_subspace_solver);
            subspace_solver->set_hermiticity(true);
    this->set_n_roots(1);
    this->m_normalise_solution = false;
  }

  /*!
   * \brief constructs next perturbed wavefunction
   *
   * @param parameters output new parameters for the subspace.
   * @param residual preconditioned residuals.
   * @return number of significant parameters to calculate the action for
   */
  size_t end_iteration(const VecRef<R>& parameters, const VecRef<R>& action) override {
    return end_iteration(parameters.front().get(), action.front().get());
  }
  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override {
    return end_iteration(parameters.front(), action.front());
  }
  size_t end_iteration(R& parameters, R& actions) override {
    // TODO implement RSPT next wavefunction
    std::cout << "xspace size "<<this->m_xspace->size()<<std::endl;
    // TODO implement RSPT energies
    return this->m_errors.front() < this->m_convergence_threshold ? 0 : 1;
  }

  //! Applies the Davidson preconditioner
  void precondition(std::vector<R>& parameters, std::vector<R>& action) const {}

  std::vector<scalar_type> eigenvalues() const override { return this->m_subspace_solver->eigenvalues(); }

  std::vector<scalar_type> working_set_eigenvalues() const override {
    auto eval = std::vector<scalar_type>{};
    for (auto i : this->working_set()) {
      eval.emplace_back(this->m_subspace_solver->eigenvalues().at(i));
    }
    return eval;
  }

//  void set_value_errors() override {
//    auto current_values = this->m_subspace_solver->eigenvalues();
//    this->m_value_errors.assign(current_values.size(), std::numeric_limits<scalar_type>::max());
//    for (size_t i = 0; i < std::min(m_last_values.size(), current_values.size()); i++)
//      this->m_value_errors[i] = std::abs(current_values[i] - m_last_values[i]);
//    if (!m_resetting_in_progress)
//      m_last_values = current_values;
//  }

  void report(std::ostream& cout) const override {
    SolverTemplate::report(cout);
//    cout << "errors " << std::scientific;
//    auto& err = this->m_errors;
//    std::copy(begin(err), end(err), std::ostream_iterator<scalar_type>(molpro::cout, ", "));
//    cout << std::endl;
//    cout << "eigenvalues ";
//    auto ev = eigenvalues();
    cout << "Perturbed energies ";
    cout << std::fixed << std::setprecision(14);
    std::copy(begin(m_rspt_values), end(m_rspt_values), std::ostream_iterator<scalar_type>(molpro::cout, ", "));
    cout << std::defaultfloat << std::endl;
  }

  void set_hermiticity(bool hermitian) override {
//    m_hermiticity = hermitian;
    auto xspace = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    xspace->set_hermiticity(hermitian);
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverRSPT<R, Q, P>>(this->m_subspace_solver);
    subspace_solver->set_hermiticity(hermitian);
  }

  bool get_hermiticity() const override { return true;}

  void set_options(const Options& options) override {
    SolverTemplate::set_options(options);
    auto opt = CastOptions::LinearEigensystemRSPT(options);
    if (opt.norm_thresh)
      propose_rspace_norm_thresh = opt.norm_thresh.value();
    if (opt.svd_thresh)
      propose_rspace_svd_thresh = opt.svd_thresh.value();
  }

  std::shared_ptr<Options> get_options() const override {
    auto opt = std::make_shared<LinearEigensystemRSPTOptions>();
    opt->copy(*SolverTemplate::get_options());
    opt->norm_thresh = propose_rspace_norm_thresh;
    opt->svd_thresh = propose_rspace_svd_thresh;
    return opt;
  }

  std::shared_ptr<Logger> logger;
  double propose_rspace_norm_thresh = 1e-10; //!< vectors with norm less than threshold can be considered null.
  double propose_rspace_svd_thresh = 1e-12;  //!< the smallest singular value in the subspace that can be allowed when
  //!< constructing the working set. Smaller singular values will lead to
  //!< deletion of parameters from the Q space
protected:
  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {
    auto xspace = std::dynamic_pointer_cast<subspace::XSpace<R, Q, P>>(this->m_xspace);
    const auto& q = xspace->paramsq();
    const auto& n = q.size();
    const auto& c = params.back();
    auto& hc = actions.back();
    if (n==0) m_rspt_values.assign(1,0);
    m_rspt_values.push_back(this->m_handlers->rr().dot(c,hc));
    this->m_handlers->rr().axpy(-m_rspt_values[0], c, hc);
    for (int k=0; k<q.size()-1; k++)
      this->m_handlers->rq().axpy(-m_rspt_values[n-k], q.at(k), hc);
  }

  std::vector<double> m_rspt_values; //!< perturbation series for the eigenvalue
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_SUBSPACE_LINEAREIGENSYSTEMRSPT_H_
