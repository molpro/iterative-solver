#ifndef LINEARALGEBRA_TEST_ITSOLV_SUBSPACE_DUMMYSOLVER_H
#define LINEARALGEBRA_TEST_ITSOLV_SUBSPACE_DUMMYSOLVER_H
#include <molpro/linalg/itsolv/IterativeSolver.h>

using molpro::linalg::itsolv::IterativeSolver;
using molpro::linalg::itsolv::Statistics;

namespace {
template <class R, class Q, class P>
struct DummySolver : IterativeSolver<R, Q, P> {
  using typename IterativeSolver<R, Q, P>::value_type;
  using typename IterativeSolver<R, Q, P>::scalar_type;
  using typename IterativeSolver<R, Q, P>::fapply_on_p_type;
  size_t add_vector(std::vector<R>& parameters, std::vector<R>& action) override { return 0; };
  size_t add_vector(std::vector<R>& parameters, std::vector<R>& action, std::vector<P>& parametersP) override {
    return 0;
  };
  size_t add_vector(std::vector<R>& parameters, std::vector<R>& action, fapply_on_p_type& aply_p) override {
    return 0;
  };
  size_t add_p(std::vector<P>& Pvectors, const value_type* PP, std::vector<R>& parameters, std::vector<R>& action,
               std::vector<P>& parametersP) override {
    return 0;
  };
  void solution(const std::vector<unsigned int>& roots, std::vector<R>& parameters,
                std::vector<R>& residual) override{};
  void solution(const std::vector<unsigned int>& roots, std::vector<R>& parameters, std::vector<R>& residual,
                std::vector<P>& parametersP) override{};
  std::vector<size_t> suggest_p(const std::vector<R>& solution, const std::vector<R>& residual, size_t maximumNumber,
                                double threshold) override {
    return {};
  };
  size_t end_iteration(std::vector<R>& parameters, std::vector<R>& action) override { return 0; }

  const std::vector<unsigned int>& working_set() const override { return ws; };
  size_t n_roots() const override { return nr; };
  void set_n_roots(size_t nroots) override { nr = nroots; };
  void report() const override{};
  const std::vector<scalar_type>& errors() const override { return er; };
  const Statistics& statistics() const override { return st; };
  std::vector<unsigned int> ws;
  std::vector<scalar_type> er;
  Statistics st;
  size_t nr{0};
};
} // namespace

#endif // LINEARALGEBRA_TEST_ITSOLV_SUBSPACE_DUMMYSOLVER_H
