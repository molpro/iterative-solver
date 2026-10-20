#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/DistrArrayFile.h>
#include <molpro/linalg/itsolv/IterativeSolver.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/mpi.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace {
//! MPI is initialised by the tests that need it, and finalised once, after all tests have run
class MPIEnvironment : public ::testing::Environment {
public:
  void TearDown() override {
#ifdef HAVE_MPI_H
    int initialized, finalized;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized and not finalized)
      molpro::mpi::finalize();
#endif
  }
};
[[maybe_unused]] auto *const mpi_environment = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
} // namespace

template <typename Container = std::vector<double>>
class ComplexRootsProblem : public molpro::linalg::itsolv::Problem<Container> {
public:
  using container_t = Container;
  using value_t = typename Container::value_type;

  // clang-format off
  Eigen::Matrix<value_t, 4, 4, Eigen::RowMajor> matrix{
      { 2, -1, -2, 0},
      {-1,  1, -3, 1},
      { 0,  2,  3,  0},
	  { 3,  2,  1,  0},
  };
  // clang-format on
  using VecT = Eigen::Vector4d;

  std::size_t dim() const { return matrix.cols(); }

  bool diagonals(container_t &d) const override {
    for (std::size_t i = 0; i < dim(); ++i) {
      d.at(i) = matrix(i, i);
    }

    return true;
  }

  void action(const molpro::linalg::itsolv::CVecRef<container_t> &parameters,
              const molpro::linalg::itsolv::VecRef<container_t> &actions) const override {
    for (std::size_t i = 0; i < parameters.size(); i++) {
      Eigen::Map<const VecT> input(parameters.at(i).get().data(), parameters.at(i).get().size());
      Eigen::Map<VecT> output(actions.at(i).get().data(), actions.at(i).get().size());

      output = matrix * input;
    }
  }
};

TEST(LinearEigensystem, Davidson_complex_roots) {
  molpro::mpi::init();

  {
    ComplexRootsProblem<> problem;

    using Rvector = decltype(problem)::container_t;

    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<Rvector>("Davidson");
    solver->set_n_roots(4);
    solver->set_max_iter(100);

    std::vector<Rvector> params;
    std::vector<Rvector> actions;
    for (std::size_t i = 0; i < solver->n_roots(); ++i) {
      params.emplace_back(problem.dim());
      actions.emplace_back(problem.dim());
    }

    if (!solver->solve(params, actions, problem, true)) {
      FAIL() << "Solver did not converge";
    }

    std::vector<int> roots(solver->n_roots());
    std::iota(roots.begin(), roots.end(), 0);

    solver->solution(roots, params, actions);
    auto eigvals = solver->eigenvalues();

    ASSERT_EQ(eigvals.size(), solver->n_roots());
    const double tol = 1e-9;
    ASSERT_THAT(eigvals.at(0), ::testing::DoubleNear(-0.360430299, tol));
    ASSERT_THAT(eigvals.at(1), ::testing::DoubleNear(1.5975220393, tol));
    ASSERT_THAT(eigvals.at(2), ::testing::DoubleNear(2.3814541296, tol));
    ASSERT_THAT(eigvals.at(3), ::testing::DoubleNear(2.3814541296, tol));
  }
}

//! A symmetric, diagonally dominant matrix
class SymmetricProblem : public molpro::linalg::itsolv::Problem<std::vector<double>> {
public:
  static constexpr size_t n = 60;
  double element(size_t i, size_t j) const { return i == j ? 1. + i : 0.1 / (1. + std::abs(double(i) - double(j))); }

  bool diagonals(std::vector<double> &d) const override {
    for (size_t i = 0; i < n; ++i)
      d[i] = element(i, i);
    return true;
  }

  void action(const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
              const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const override {
    for (size_t k = 0; k < parameters.size(); ++k)
      for (size_t i = 0; i < n; ++i) {
        actions[k].get()[i] = 0;
        for (size_t j = 0; j < n; ++j)
          actions[k].get()[i] += element(i, j) * parameters[k].get()[j];
      }
  }
};

template <class Qvector>
std::vector<double> solve_symmetric_problem() {
  SymmetricProblem problem;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>, Qvector>("Davidson");
  solver->set_n_roots(3);
  solver->set_max_iter(100);
  solver->set_hermiticity(true);
  solver->set_convergence_threshold(1e-10);
  auto params = std::vector<std::vector<double>>(solver->n_roots(), std::vector<double>(problem.n));
  auto actions = params;
  EXPECT_TRUE(solver->solve(params, actions, problem, true));
  return solver->eigenvalues();
}

// The R vectors are contiguous in memory and the Q vectors on disk, as handled by ArrayHandlerIterableDDisk and
// ArrayHandlerDDiskIterable
TEST(LinearEigensystem, Davidson_vector_R_disk_Q) {
  molpro::mpi::init();
  const auto reference = solve_symmetric_problem<std::vector<double>>();
  const auto result = solve_symmetric_problem<molpro::linalg::array::DistrArrayFile>();
  ASSERT_EQ(result.size(), reference.size());
  for (size_t i = 0; i < reference.size(); ++i)
    EXPECT_NEAR(result[i], reference[i], 1e-9);
}

namespace {
/*!
 * @brief A generalised eigenproblem A v = lambda M v with a symmetric positive definite, non-diagonal metric M whose
 * diagonal varies strongly, and a diagonally dominant A that is symmetric unless non_hermiticity is set
 */
class GeneralizedProblem : public molpro::linalg::itsolv::Problem<std::vector<double>> {
public:
  static constexpr size_t n = 60;
  double non_hermiticity = 0;
  bool complex_pair = false; //!< if true, the two lowest eigenvalues are a complex conjugate pair
  double metric_scale = 1;   //!< factor applied to the whole metric
  bool provide_metric_diagonals = true;
  bool linear_metric = true; //!< if false, metric_action() adds a constant, which test_problem() must notice

  double kernel(size_t i, size_t j) const {
    if (complex_pair and i + j == 1)
      return i == 0 ? -0.8 : 0.8;
    return i == j ? 1. + i : 0.1 / (1. + std::abs(double(i) - double(j))) * (j < i ? 1 - non_hermiticity : 1);
  }
  bool hermitian() const { return non_hermiticity == 0 and not complex_pair; }
  double metric(size_t i, size_t j) const {
    return metric_scale * (i == j ? 1.5 + std::cos(double(i)) : 0.05 / (1. + i + j));
  }

  bool diagonals(std::vector<double> &d) const override {
    for (size_t i = 0; i < n; ++i)
      d[i] = kernel(i, i);
    return true;
  }
  molpro::linalg::itsolv::MetricKind metric_kind() const override {
    return molpro::linalg::itsolv::MetricKind::HermitianPositiveDefinite;
  }
  bool metric_diagonals(std::vector<double> &d) const override {
    for (size_t i = 0; i < n; ++i)
      d[i] = metric(i, i);
    return provide_metric_diagonals;
  }

  void action(const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
              const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const override {
    apply([this](size_t i, size_t j) { return kernel(i, j); }, parameters, actions);
  }
  void metric_action(const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
                     const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const override {
    apply([this](size_t i, size_t j) { return metric(i, j); }, parameters, actions);
    if (!linear_metric)
      for (auto &a : actions)
        a.get()[0] += 1;
  }

  bool test_parameters(unsigned int instance, std::vector<double> &parameters) const override {
    if (instance > 1)
      return false;
    for (size_t i = 0; i < n; ++i)
      parameters[i] = std::cos(1. + instance + 0.3 * i);
    return true;
  }

  //! The lowest eigenvalues, computed densely
  std::vector<double> reference(size_t nroot) const {
    Eigen::MatrixXd a(n, n), m(n, n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j) {
        a(i, j) = kernel(i, j);
        m(i, j) = metric(i, j);
      }
    std::vector<double> values;
    if (hermitian()) {
      Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(a, m);
      values.assign(es.eigenvalues().data(), es.eigenvalues().data() + n);
    } else {
      Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> es(a, m);
      std::vector<std::complex<double>> complex_values;
      for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i)
        complex_values.push_back(es.eigenvalues()(i));
      std::sort(complex_values.begin(), complex_values.end(),
                [](const auto &x, const auto &y) { return x.real() < y.real(); });
      for (size_t i = 0; i < complex_values.size(); ++i) {
        // a complex pair is expected exactly where it is built into the kernel
        if (complex_pair and i < 2)
          EXPECT_GT(std::abs(complex_values[i].imag()), 0.1);
        else
          EXPECT_NEAR(complex_values[i].imag(), 0, 1e-10);
        values.push_back(complex_values[i].real());
      }
    }
    values.resize(nroot);
    return values;
  }

private:
  template <class Element>
  void apply(const Element &element, const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
             const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const {
    for (size_t k = 0; k < parameters.size(); ++k)
      for (size_t i = 0; i < n; ++i) {
        actions[k].get()[i] = 0;
        for (size_t j = 0; j < n; ++j)
          actions[k].get()[i] += element(i, j) * parameters[k].get()[j];
      }
  }
};

template <class Qvector>
void solve_generalized_problem(const GeneralizedProblem &problem, const size_t nroot, const size_t nparams) {
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>, Qvector>("Davidson");
  solver->set_n_roots(nroot);
  solver->set_hermiticity(problem.hermitian());
  solver->set_convergence_threshold(1e-10);
  auto params = std::vector<std::vector<double>>(nparams, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
  EXPECT_THAT(solver->errors(), ::testing::Each(::testing::Le(1e-10)));
  EXPECT_THAT(solver->eigenvalues(), ::testing::Pointwise(::testing::DoubleNear(1e-9), problem.reference(nroot)));
}
} // namespace

// solve() for generalised eigenproblems, with Q vectors in memory and on disk
TEST(LinearEigensystem, Davidson_generalized) {
  molpro::mpi::init();
  GeneralizedProblem problem;
  for (const double non_hermiticity : {0., 0.2}) {
    SCOPED_TRACE("non-hermiticity=" + std::to_string(non_hermiticity));
    problem.non_hermiticity = non_hermiticity;
    solve_generalized_problem<std::vector<double>>(problem, 3, 3);
    solve_generalized_problem<molpro::linalg::array::DistrArrayFile>(problem, 3, 3);
  }
}

// A complex conjugate pair of eigenvalues of a real non-hermitian kernel: the residuals of the real and imaginary
// parts of the eigenvector are coupled through the metric
TEST(LinearEigensystem, Davidson_generalized_complex_pair) {
  GeneralizedProblem problem;
  problem.complex_pair = true;
  solve_generalized_problem<std::vector<double>>(problem, 3, 3);
}

// Scaling the metric scales the eigenvalues inversely, but changes neither the errors nor the course of the iterations,
// so that a convergence threshold has the same meaning whatever the scale of the metric. Powers of two keep the
// scaled problems exact in floating point.
TEST(LinearEigensystem, Davidson_generalized_metric_scale) {
  GeneralizedProblem problem;
  struct Outcome {
    std::vector<double> eigenvalues, errors;
    int iterations;
  };
  auto solve = [&problem](double metric_scale, int max_iter) {
    problem.metric_scale = metric_scale;
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
    solver->set_n_roots(3);
    solver->set_hermiticity(true);
    solver->set_convergence_threshold(1e-10);
    solver->set_max_iter(max_iter);
    auto params = std::vector<std::vector<double>>(3, std::vector<double>(problem.n));
    auto actions = params, metric_actions = params;
    solver->solve(params, actions, metric_actions, problem, true);
    return Outcome{solver->eigenvalues(), solver->errors(), solver->statistics().iterations};
  };
  // Stopped early, the errors are well above the rounding level, so that they can be compared closely. Errors that
  // depended on the scale of the metric would differ by the factor 1/sqrt(metric scale), i.e. 8 here.
  const auto reference_unconverged = solve(1, 3);
  ASSERT_THAT(reference_unconverged.errors, ::testing::Each(::testing::Gt(1e-8)));
  // Converged, the iterations and eigenvalues are compared
  const auto reference = solve(1, 100);
  ASSERT_THAT(reference.errors, ::testing::Each(::testing::Le(1e-10)));
  for (const double metric_scale : {64., 1. / 64}) {
    SCOPED_TRACE("metric scale=" + std::to_string(metric_scale));
    const auto unconverged = solve(metric_scale, 3);
    ASSERT_EQ(unconverged.errors.size(), reference_unconverged.errors.size());
    for (size_t i = 0; i < reference_unconverged.errors.size(); ++i)
      EXPECT_NEAR(unconverged.errors[i], reference_unconverged.errors[i], 1e-6 * reference_unconverged.errors[i])
          << "root " << i;
    const auto converged = solve(metric_scale, 100);
    EXPECT_EQ(converged.iterations, reference.iterations);
    ASSERT_EQ(converged.eigenvalues.size(), reference.eigenvalues.size());
    for (size_t i = 0; i < reference.eigenvalues.size(); ++i)
      EXPECT_NEAR(converged.eigenvalues[i] * metric_scale, reference.eigenvalues[i], 1e-12) << "root " << i;
  }
}

// The contributions of Q vectors to the solutions, which decide which Q vectors are deleted, are measured on solutions of
// unit length, so that trimming the Q space does not depend on the scale of the metric either
TEST(LinearEigensystem, Davidson_generalized_metric_scale_qspace_trimming) {
  GeneralizedProblem problem;
  struct Outcome {
    std::vector<double> eigenvalues;
    int iterations;
    size_t nQ;
  };
  auto solve = [&problem](double metric_scale) {
    problem.metric_scale = metric_scale;
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
    solver->set_n_roots(2);
    solver->set_hermiticity(true);
    solver->set_convergence_threshold(1e-10);
    auto options = molpro::linalg::itsolv::LinearEigensystemDavidsonOptions{};
    options.contrib_thresh = 1e-2;
    options.min_size_qspace = 2;
    solver->set_options(options);
    auto params = std::vector<std::vector<double>>(2, std::vector<double>(problem.n));
    auto actions = params, metric_actions = params;
    EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
    return Outcome{solver->eigenvalues(), solver->statistics().iterations, solver->dimensions().nQ};
  };
  const auto reference = solve(1);
  for (const double metric_scale : {64., 1. / 64}) {
    SCOPED_TRACE("metric scale=" + std::to_string(metric_scale));
    const auto scaled = solve(metric_scale);
    EXPECT_EQ(scaled.iterations, reference.iterations);
    EXPECT_EQ(scaled.nQ, reference.nQ);
    ASSERT_EQ(scaled.eigenvalues.size(), reference.eigenvalues.size());
    for (size_t i = 0; i < reference.eigenvalues.size(); ++i)
      EXPECT_NEAR(scaled.eigenvalues[i] * metric_scale, reference.eigenvalues[i], 1e-12) << "root " << i;
  }
}

// A single root is found with a single set of buffers, which also hand the diagonal elements of kernel and metric to
// the preconditioner
TEST(LinearEigensystem, Davidson_generalized_single_root) {
  GeneralizedProblem problem;
  solve_generalized_problem<std::vector<double>>(problem, 1, 1);
}

// solve() calculates the actions of kernel and metric through action_with_metric(), which a problem can override
TEST(LinearEigensystem, Davidson_generalized_combined_action) {
  struct CountingProblem : GeneralizedProblem {
    mutable int combined_calls = 0;
    void action_with_metric(const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
                            const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions,
                            const molpro::linalg::itsolv::VecRef<std::vector<double>> &metric_actions) const override {
      ++combined_calls;
      GeneralizedProblem::action_with_metric(parameters, actions, metric_actions);
    }
  } problem;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
  solver->set_n_roots(2);
  solver->set_hermiticity(true);
  auto params = std::vector<std::vector<double>>(2, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
  EXPECT_EQ(problem.combined_calls, solver->statistics().iterations);
}

// test_problem() checks the action of the metric as well
TEST(LinearEigensystem, Davidson_generalized_test_problem) {
  GeneralizedProblem problem;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
  std::vector<double> v0(problem.n), v1(problem.n);
  EXPECT_TRUE(solver->test_problem(problem, v0, v1, -1));
  problem.linear_metric = false;
  EXPECT_FALSE(solver->test_problem(problem, v0, v1, -1));
}

// Uses of a metric that solve() cannot support are reported
TEST(LinearEigensystem, Davidson_generalized_errors) {
  GeneralizedProblem problem;
  auto params = std::vector<std::vector<double>>(2, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  auto davidson = []() { return molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson"); };
  {
    auto solver = davidson();
    solver->set_max_p(10);
    EXPECT_THROW(solver->solve(params, actions, metric_actions, problem, true), std::runtime_error);
  }
  // without buffers for the actions of the metric
  EXPECT_THROW(davidson()->solve(params, actions, problem, true), std::invalid_argument);
  {
    auto too_few = std::vector<std::vector<double>>(1, std::vector<double>(problem.n));
    EXPECT_THROW(davidson()->solve(params, actions, too_few, problem, true), std::runtime_error);
  }
  // the default preconditioner would ignore the metric without its diagonal elements
  problem.provide_metric_diagonals = false;
  EXPECT_THROW(davidson()->solve(params, actions, metric_actions, problem, true), std::runtime_error);
  problem.provide_metric_diagonals = true;
  {
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("RSPT");
    EXPECT_THROW(solver->solve(params, actions, metric_actions, problem, true), std::invalid_argument);
  }
  {
    auto solver = molpro::linalg::itsolv::create_LinearEquations<std::vector<double>>("Davidson");
    EXPECT_THROW(solver->solve(params, actions, metric_actions, problem, true), std::invalid_argument);
  }
}

#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, std::vector<double>,
                                                     std::map<size_t, double>>;
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, molpro::linalg::array::DistrArrayFile,
                                                     std::map<size_t, double>>;
