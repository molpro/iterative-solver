#ifndef LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREIGENSYSTEMDAVIDSON_H
#define LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREIGENSYSTEMDAVIDSON_H
#include <molpro/Profiler.h>
#include <molpro/linalg/itsolv/CastOptions.h>
#include <molpro/linalg/itsolv/DavidsonCommon.h>
#include <molpro/linalg/itsolv/IterativeSolverTemplate.h>
#include <molpro/linalg/itsolv/Logger.h>
#include <molpro/linalg/itsolv/helper.h>
#include <molpro/linalg/itsolv/subspace/SubspaceSolverLinEig.h>
#include <molpro/linalg/itsolv/subspace/XSpace.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace molpro::linalg::itsolv {

namespace log {

template< typename value_type >
struct ComplexRootsDavidson : ContextBase<ComplexRootsDavidson<value_type>, true, std::vector<std::pair<std::size_t, value_type>>> {
	static const char *name;
};
template<typename value_type>
const char *ComplexRootsDavidson<value_type>::name = "ComplexRootsDavidson";
static_assert(context<ComplexRootsDavidson<double>>);

}

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
template <class R, class Q = R, class P = std::map<size_t, typename R::value_type>>
class LinearEigensystemDavidson : public DavidsonCommon<LinearEigensystem, R, Q, P> {
public:
  using SolverTemplate = IterativeSolverTemplate<LinearEigensystem, R, Q, P>;
  using SolverTemplate::report;
  using typename SolverTemplate::scalar_type;
  using typename SolverTemplate::value_type;
  using typename SolverTemplate::value_type_abs;

  explicit LinearEigensystemDavidson(const std::shared_ptr<ArrayHandlers<R, Q, P>>& handlers =
                                         std::make_shared<molpro::linalg::itsolv::ArrayHandlers<R, Q, P>>(),
                                     const std::shared_ptr<Logger>& logger_ = std::make_shared<Logger>())
      : DavidsonCommon<LinearEigensystem, R, Q, P>(handlers, logger_, false) {}

  void finalize() override {
    if constexpr (is_complex<typename R::value_type>::value) {
      return;
    } else {
      auto subspace_solver =
          std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
      assert(subspace_solver);
      auto imag_eigval_components = subspace_solver->imag_eigval_components();
      if (imag_eigval_components.empty()) {
        return;
      }

      // Convert to 1-based indexing for printout
      for (auto& pair : imag_eigval_components) {
        pair.first += 1;
      }

      this->m_logger->template warn<log::ComplexRootsDavidson<typename R::value_type>>(
          "The following roots are complex-valued. Associated eigenvectors are the real and imaginary part "
          "of the pairs and the imaginary parts of the eigenvalues are ",
          imag_eigval_components);

      const auto nroots = subspace_solver->size();
      if (const auto partner = subspace_solver->complex_partner(nroots - 1); partner && *partner >= nroots)
        this->m_logger->warn(std::format("The complex conjugate partner of root {} is not among the requested roots, so "
                                         "only the real part of its eigenvector is available",
                                         nroots));
    }
  }

  std::vector<scalar_type> eigenvalues() const override { return this->m_subspace_solver->eigenvalues(); }

  virtual std::vector<scalar_type> working_set_eigenvalues() const override {
    auto eval = std::vector<scalar_type>{};
    for (auto i : this->working_set()) {
      eval.push_back(i < this->m_subspace_solver->eigenvalues().size() ? this->m_subspace_solver->eigenvalues().at(i)
                                                                       : 0);
    }
    return eval;
  }

  void report(std::ostream& cout, bool endl = true) const override {
    const util::StreamFormatGuard format_guard(cout);
    SolverTemplate::report(cout);
    cout << "errors " << std::scientific;
    auto& err = this->m_errors;
    std::copy(begin(err), end(err), std::ostream_iterator<scalar_type>(cout, ", "));
    cout << std::endl;
    cout << "eigenvalues ";
    auto ev = eigenvalues();
    cout << std::fixed << std::setprecision(14);
    std::copy(begin(ev), end(ev), std::ostream_iterator<scalar_type>(cout, ", "));
    if (endl)
      cout << std::endl;
  }

  void set_options(const Options& options) override {
    this->set_davidson_options(options, CastOptions::LinearEigensystem(options));
  }

  std::shared_ptr<Options> get_options() const override {
    return this->template get_davidson_options<LinearEigensystemDavidsonOptions>();
  }

protected:
  std::vector<value_type> values() const override { return this->m_subspace_solver->eigenvalues(); }

  void construct_residual(const std::vector<int>& roots, const CVecRef<R>& params, const VecRef<R>& actions) override {
    auto prof = this->profiler()->push("itsolv::construct_residual");
    assert(params.size() >= roots.size());
    const auto& eigvals = eigenvalues();

    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    assert(subspace_solver);
    const auto& imag_eigval_components = subspace_solver->imag_eigval_components();

    for (size_t i = 0; i < roots.size(); ++i) {
      this->m_handlers->rr().axpy(-eigvals.at(roots[i]), params.at(i), actions.at(i));

      auto it = std::ranges::find(imag_eigval_components, roots[i], [](const auto& pair) { return pair.first; });

      if (it != imag_eigval_components.end()) {
        // Ref.: https://doi.org/10.1063/1.2755681 (Appendix)
        // The idea is the following: We make use of the fact that the real and imaginary parts of the
        // eigenvectors belonging to the complex root pair are spanned by the same basis. This allows
        // us to split the associated residual vectors into a part corresponding to the real and a part
        // corresponding to the imaginary part as well. The actual (complex-valued) residual can then
        // always be reconstructed from this and therefore we don't lose any information. The formulas are
        // |(R_real)_i> = sum_j [ (|A_j> - (e_real)_i |b_j>) (c_real)_{ji} + (e_imag) |b_j> (c_imag)_{ji} ]
        // |(R_imag)_i> = sum_j [ (|A_j> - (e_real)_i |b_j>) (c_imag)_{ji} - (e_imag) |b_j> (c_real)_{ji} ]
        // where |A_j> is the j-th action and |b_j> the j-th trial vector. e are the complex conjugate
        // eigenvalues and c the associated eigenvectors.
        // Note: At this point, params and actions already contain the vectors transformed into the
        // eigenbasis (aka.: multiplied with c). Due to the way we process these complex-valued vectors,
        // the first vector in a pair is the one that has been transformed with the real and the second
        // with the imaginary part of c.
        // The above code has already dealt with the bulk of the necessary expression and we only need
        // to add the part with the imaginary components of the eigenvalue pair, for which the vector of
        // the partner root is used.
        // Note: The minus sign in above expression is implicitly taken into account as the
        // imaginary parts of the complex conjugate eigenvalue pair has flipped signs
        const auto partner = subspace_solver->complex_partner(roots[i]).value();
        const auto root_it = std::ranges::find(roots, int(partner));
        if (root_it != roots.end()) {
          this->m_handlers->rr().axpy(it->second, params.at(std::ranges::distance(roots.begin(), root_it)),
                                      actions.at(i));
        } else {
          // The partner is not among roots, or was truncated away, so its vector is assembled from the subspace
          if (partner >= subspace_solver->retained_solutions().rows())
            throw std::logic_error("The complex conjugate partner of a root is not among the retained solutions");
          auto partner_param = this->m_handlers->rr().copy(actions.at(i));
          construct_subspace_vector(subspace_solver->retained_solutions(), int(partner), partner_param);
          this->m_handlers->rr().axpy(it->second, partner_param, actions.at(i));
        }
      }
    }
  }

  /*!
   * @brief The error of both members of a complex pair is the norm of their complex residual
   *
   * The residuals of the two members are the real and imaginary part of the complex residual. A pair whose second
   * member was truncated away is handled by compute_errors().
   */
  void combine_errors() override {
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    assert(subspace_solver);
    const auto& errors = subspace_solver->errors();
    for (size_t root = 0; root < subspace_solver->size(); ++root) {
      const auto partner = subspace_solver->complex_partner(root);
      if (!partner || *partner <= root || *partner >= subspace_solver->size())
        continue;
      const auto error = std::sqrt(errors[root] * errors[root] + errors[*partner] * errors[*partner]);
      subspace_solver->set_error(int(root), error);
      subspace_solver->set_error(int(*partner), error);
    }
  }

  /*!
   * @brief The error of the last root, whose complex pair partner was truncated away, is the norm of its complex
   * residual
   *
   * The residual of that root is only the real part of the complex residual. The imaginary part is the residual of
   * the truncated partner, which is assembled from the subspace.
   */
  void compute_errors(const std::vector<int>& roots, const CVecRef<R>& params, const CVecRef<R>& residuals,
                      std::vector<value_type_abs>& errors) override {
    SolverTemplate::compute_errors(roots, params, residuals, errors);
    auto subspace_solver = std::dynamic_pointer_cast<subspace::SubspaceSolverLinEig<R, Q, P>>(this->m_subspace_solver);
    assert(subspace_solver);
    const auto nroots = subspace_solver->size();
    if (nroots == 0)
      return;
    const auto partner = subspace_solver->complex_partner(nroots - 1);
    const auto root_it = std::ranges::find(roots, int(nroots - 1));
    if (!partner || *partner < nroots || root_it == roots.end())
      return;
    const auto i = std::ranges::distance(roots.begin(), root_it);
    const auto& imag_eigval_components = subspace_solver->imag_eigval_components();
    const auto it = std::ranges::find(imag_eigval_components, nroots - 1, [](const auto& pair) { return pair.first; });
    if (it == imag_eigval_components.end())
      throw std::logic_error("The imaginary part of the eigenvalue of a complex pair member is missing");
    const auto& solutions = subspace_solver->retained_solutions();
    auto partner_param = this->m_handlers->rr().copy(params.at(i));
    auto partner_residual = this->m_handlers->rr().copy(params.at(i));
    construct_subspace_vector(solutions, int(*partner), partner_param);
    construct_subspace_action(solutions, int(*partner), partner_residual);
    // The imaginary part of the partner's eigenvalue is the negative of the one of the last root
    this->m_handlers->rr().axpy(-eigenvalues().at(nroots - 1), partner_param, partner_residual);
    this->m_handlers->rr().axpy(-it->second, params.at(i), partner_residual);
    errors.at(i) = std::sqrt(errors.at(i) * errors.at(i) +
                             std::abs(this->m_handlers->rr().dot(partner_residual, partner_residual)));
  }

  //! Constructs in param the subspace vector with the coefficients in the given row of solutions
  void construct_subspace_vector(const subspace::Matrix<scalar_type>& solutions, int row, R& param) {
    const auto& dims = this->m_xspace->dimensions();
    assert(solutions.cols() == dims.nX());
    util::construct_solutions(wrap_arg(param), {row}, solutions, this->m_xspace->cparamsp(), this->m_xspace->cparamsq(),
                              this->m_xspace->cparamsd(), dims.oP(), dims.oQ(), dims.oD(), this->m_handlers->rr(),
                              this->m_handlers->rp(), this->m_handlers->rq());
  }

  //! Constructs in action the action on the subspace vector with the coefficients in the given row of solutions
  void construct_subspace_action(const subspace::Matrix<scalar_type>& solutions, int row, R& action) {
    const auto& dims = this->m_xspace->dimensions();
    assert(solutions.cols() == dims.nX());
    util::construct_solutions(wrap_arg(action), {row}, solutions, {}, this->m_xspace->cactionsq(),
                              this->m_xspace->cactionsd(), dims.oP(), dims.oQ(), dims.oD(), this->m_handlers->rr(),
                              this->m_handlers->rp(), this->m_handlers->rq());
    if (this->m_apply_p)
      this->m_apply_p(detail::construct_vectorP({row}, solutions, dims.oP(), dims.nP), this->m_xspace->cparamsp(),
                      wrap_arg(action));
  }
};

} // namespace molpro::linalg::itsolv

#endif // LINEARALGEBRA_SRC_MOLPRO_LINALG_ITSOLV_LINEAREIGENSYSTEMDAVIDSON_H
