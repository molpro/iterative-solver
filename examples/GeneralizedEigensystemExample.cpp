#include "ExampleProblem.h"
#include <cmath>
#include <iostream>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/mpi.h>

/*!
 * @brief The generalised eigenproblem A v = lambda M v, with the kernel A of ExampleProblem and a Hermitian, positive
 * definite metric M, e.g. the overlap matrix of a non-orthogonal basis
 *
 * Compared with a problem without a metric, it declares the kind of metric and provides its action and, as it
 * provides diagonals(), its diagonal elements.
 */
class GeneralizedExampleProblem : public ExampleProblem<> {
public:
  using ExampleProblem::ExampleProblem;
  //! Whether solve() applies the metric to the solutions in every iteration instead of keeping its actions
  bool apply_on_the_fly = false;

  double metric(size_t i, size_t j) const { return i == j ? 1.5 + 0.5 * std::cos(double(i)) : 0.01 / (1. + i + j); }

  molpro::linalg::itsolv::MetricKind metric_kind() const override {
    return molpro::linalg::itsolv::MetricKind::HermitianPositiveDefinite;
  }

  molpro::linalg::itsolv::MetricStorage metric_storage() const override {
    return apply_on_the_fly ? molpro::linalg::itsolv::MetricStorage::OnTheFly
                            : molpro::linalg::itsolv::MetricStorage::Stored;
  }

  void metric_action(const molpro::linalg::itsolv::CVecRef<container_t>& parameters,
                     const molpro::linalg::itsolv::VecRef<container_t>& metric_actions) const override {
    for (size_t k = 0; k < parameters.size(); k++) {
      const auto& v = parameters[k].get();
      auto& m = metric_actions[k].get();
      for (size_t i = 0; i < m.size(); i++) {
        m[i] = 0;
        for (size_t j = 0; j < m.size(); j++)
          m[i] += metric(i, j) * v[j];
      }
    }
  }

  bool metric_diagonals(container_t& d) const override {
    for (size_t i = 0; i < d.size(); i++)
      d[i] = metric(i, i);
    return true;
  }
};

int main(int argc, char* argv[]) {
  molpro::mpi::init();
  int status = 0;
  {
    using Rvector = GeneralizedExampleProblem::container_t;
    auto problem = GeneralizedExampleProblem(argc > 1 ? std::stoi(argv[1]) : 20);
    const size_t nroot = argc > 2 ? std::stoi(argv[2]) : 2;
    std::vector<double> eigenvalues;
    // solve(), which needs a third set of buffers for the actions of the metric
    for (const bool on_the_fly : {false, true}) {
      problem.apply_on_the_fly = on_the_fly;
      auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector>("Davidson");
      solver->set_n_roots(nroot);
      solver->set_hermiticity(true);
      std::vector<Rvector> c(nroot, Rvector(problem.n)), g = c, m = c;
      if (not solver->solve(c, g, m, problem, true)) {
        std::cout << "failed" << std::endl;
        status = 1;
      }
      std::cout << "metric " << (on_the_fly ? "applied on the fly" : "stored") << ", converged in "
                << solver->statistics().iterations << " iterations, eigenvalues";
      for (const auto& ev : solver->eigenvalues())
        std::cout << " " << ev;
      std::cout << std::endl;
      eigenvalues = solver->eigenvalues();
    }
    // the stepwise interface: declare the metric, give the actions of the metric to add_vector(), and precondition with
    // the diagonal elements of kernel and metric
    {
      auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector>("Davidson");
      solver->set_n_roots(nroot);
      solver->set_hermiticity(true);
      solver->set_metric(molpro::linalg::itsolv::MetricKind::HermitianPositiveDefinite);
      std::vector<Rvector> c(nroot, Rvector(problem.n)), g = c, m = c;
      Rvector diagonals(problem.n), metric_diagonals(problem.n);
      problem.diagonals(diagonals);
      problem.metric_diagonals(metric_diagonals);
      for (size_t k = 0; k < nroot; ++k)
        c[k][k] = 1;
      for (int iter = 0; iter < 100; ++iter) {
        problem.action_with_metric(molpro::linalg::itsolv::cwrap(c), molpro::linalg::itsolv::wrap(g),
                                   molpro::linalg::itsolv::wrap(m));
        const auto nwork = solver->add_vector(c, g, m);
        if (nwork == 0)
          break;
        problem.precondition_with_metric(molpro::linalg::itsolv::wrap(g.begin(), g.begin() + nwork),
                                         solver->working_set_eigenvalues(), diagonals, metric_diagonals);
        if (solver->end_iteration(c, g) == 0)
          break;
      }
      std::cout << "stepwise, converged in " << solver->statistics().iterations << " iterations, eigenvalues";
      for (size_t k = 0; k < nroot; ++k) {
        std::cout << " " << solver->eigenvalues()[k];
        if (std::abs(solver->eigenvalues()[k] - eigenvalues[k]) > 1e-8)
          status = 1;
      }
      std::cout << std::endl;
    }
  }
  molpro::mpi::finalize();
  return status;
}

#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
#include <vector>
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, std::vector<double>,
                                                     std::map<size_t, double>>;
