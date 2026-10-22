#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/DistrArrayFile.h>
#include <molpro/linalg/itsolv/IterativeSolver.h>
#include <molpro/linalg/itsolv/LinearEigensystemDavidson.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/mpi.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
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

//! A non-symmetric, diagonally dominant matrix whose third eigenvalue (by real part) is a member of a complex pair
class SplitComplexPairProblem : public molpro::linalg::itsolv::Problem<std::vector<double>> {
public:
  static constexpr size_t n = 30;
  double coupling = 1.;      //!< couples the third and fourth diagonal element into the complex pair
  double offdiagonal = 0.01; //!< scale of the remaining off-diagonal elements
  double element(size_t i, size_t j) const {
    if (i == 2 and j == 3)
      return coupling;
    if (i == 3 and j == 2)
      return -coupling;
    return i == j ? 1. + i : offdiagonal / (1. + std::abs(double(i) - double(j)));
  }

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

//! The matrix of problem
Eigen::MatrixXd dense_matrix(const SplitComplexPairProblem &problem) {
  Eigen::MatrixXd matrix(problem.n, problem.n);
  for (size_t i = 0; i < problem.n; ++i)
    for (size_t j = 0; j < problem.n; ++j)
      matrix(i, j) = problem.element(i, j);
  return matrix;
}

//! The eigenvalues of problem in the order of the solver, i.e. by real part and then by imaginary part
std::vector<std::complex<double>> sorted_reference_eigenvalues(const SplitComplexPairProblem &problem) {
  const Eigen::VectorXcd eigenvalues = Eigen::EigenSolver<Eigen::MatrixXd>(dense_matrix(problem)).eigenvalues();
  std::vector<std::complex<double>> reference(eigenvalues.begin(), eigenvalues.end());
  std::ranges::sort(reference, {}, [](const auto &e) { return std::pair{e.real(), e.imag()}; });
  return reference;
}

//! A non-hermitian Davidson solver for n_roots roots, converging to 1e-8 within 100 iterations
auto create_split_pair_solver(size_t n_roots, const std::string &options = "") {
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson", options);
  solver->set_n_roots(n_roots);
  solver->set_max_iter(100);
  solver->set_hermiticity(false);
  solver->set_convergence_threshold(1e-8);
  return solver;
}

TEST(LinearEigensystem, Davidson_complex_pair_split_by_n_roots) {
  molpro::mpi::init();
  SplitComplexPairProblem problem;
  const auto reference = sorted_reference_eigenvalues(problem);
  ASSERT_NEAR(std::abs(reference.at(2) - std::conj(reference.at(3))), 0., 1e-12);
  ASSERT_GT(std::abs(reference.at(2).imag()), 0.5);

  auto solver = create_split_pair_solver(3);
  auto params = std::vector<std::vector<double>>(solver->n_roots(), std::vector<double>(problem.n));
  auto actions = params;
  ASSERT_TRUE(solver->solve(params, actions, problem, true));
  const auto eigvals = solver->eigenvalues();
  ASSERT_EQ(eigvals.size(), solver->n_roots());
  for (size_t i = 0; i < eigvals.size(); ++i)
    EXPECT_NEAR(eigvals[i], reference[i].real(), 1e-8);
}

//! The convergence of a root whose complex pair partner is truncated covers the imaginary part of its residual
TEST(LinearEigensystem, Davidson_complex_pair_split_by_n_roots_imaginary_residual) {
  molpro::mpi::init();
  SplitComplexPairProblem problem;
  problem.offdiagonal = 0.1;
  const Eigen::MatrixXd matrix = dense_matrix(problem);
  const auto reference = sorted_reference_eigenvalues(problem);
  ASSERT_NEAR(std::abs(reference.at(2) - std::conj(reference.at(3))), 0., 1e-12);

  auto solver = create_split_pair_solver(3);
  auto params = std::vector<std::vector<double>>(solver->n_roots(), std::vector<double>(problem.n));
  auto actions = params;
  ASSERT_TRUE(solver->solve(params, actions, problem, true));
  solver->solution({0, 1, 2}, params, actions);
  // With the residuals r = M x + e_imag y and r_imag = M y - e_imag x of the real and imaginary part x and y of the
  // complex eigenvector, where M = A - e_real, y and r_imag follow from x and r
  const Eigen::Map<const Eigen::VectorXd> x(params[2].data(), problem.n), r(actions[2].data(), problem.n);
  const Eigen::MatrixXd m = matrix - reference[2].real() * Eigen::MatrixXd::Identity(problem.n, problem.n);
  const double e_imag = reference[2].imag();
  const Eigen::VectorXd y = (r - m * x) / e_imag;
  const Eigen::VectorXd r_imag = m * y - e_imag * x;
  EXPECT_LT(std::sqrt((r.squaredNorm() + r_imag.squaredNorm()) / (x.squaredNorm() + y.squaredNorm())),
            solver->convergence_threshold());
}

//! Records the warnings issued through it
class WarningRecorder : public molpro::linalg::itsolv::Logger {
public:
  mutable std::vector<std::string> warnings;

protected:
  void default_message_handler(std::string_view ctx, molpro::linalg::itsolv::log::Severity severity,
                               molpro::linalg::itsolv::log::Verbosity verbosity, std::string_view msg) const override {
    if (severity == molpro::linalg::itsolv::log::Severity::Warning)
      warnings.emplace_back(msg);
  }
};

//! The user is told that only the real part of the eigenvector of a root with a truncated complex partner is available
TEST(LinearEigensystem, Davidson_complex_pair_split_by_n_roots_warning) {
  molpro::mpi::init();
  SplitComplexPairProblem problem;
  auto solver = create_split_pair_solver(3);
  auto logger = std::make_shared<WarningRecorder>();
  solver->set_logger(logger);
  auto params = std::vector<std::vector<double>>(solver->n_roots(), std::vector<double>(problem.n));
  auto actions = params;
  ASSERT_TRUE(solver->solve(params, actions, problem, true));
  EXPECT_THAT(logger->warnings, ::testing::Contains(::testing::HasSubstr(
                                    "partner of root 3 is not among the requested roots, so only the real part")));
}

//! A complex pair split by the number of roots with a reduced subspace, which has to keep the truncated partner
class Davidson_complex_pair_split_reduced_subspace : public ::testing::TestWithParam<std::string> {};

TEST_P(Davidson_complex_pair_split_reduced_subspace, converges) {
  molpro::mpi::init();
  SplitComplexPairProblem problem;
  const auto reference = sorted_reference_eigenvalues(problem);

  auto solver = create_split_pair_solver(3, GetParam());
  auto params = std::vector<std::vector<double>>(solver->n_roots(), std::vector<double>(problem.n));
  auto actions = params;
  ASSERT_TRUE(solver->solve(params, actions, problem, true));
  const auto eigvals = solver->eigenvalues();
  ASSERT_EQ(eigvals.size(), solver->n_roots());
  for (size_t i = 0; i < eigvals.size(); ++i)
    EXPECT_NEAR(eigvals[i], reference[i].real(), 1e-8);
}

INSTANTIATE_TEST_SUITE_P(LinearEigensystem, Davidson_complex_pair_split_reduced_subspace,
                         ::testing::Values("max_size_qspace=4", "contrib_thresh=1e-3", "max_size_qspace=4,reset_D=3"));

//! The two members of a complex pair end up in different batches of the solution when there are fewer parameter
//! vectors than roots
TEST(LinearEigensystem, Davidson_complex_pair_split_by_batches) {
  molpro::mpi::init();
  SplitComplexPairProblem problem;
  const auto reference = sorted_reference_eigenvalues(problem);
  ASSERT_NEAR(std::abs(reference.at(2) - std::conj(reference.at(3))), 0., 1e-12);

  auto solver = create_split_pair_solver(4);
  auto params = std::vector<std::vector<double>>(3, std::vector<double>(problem.n));
  auto actions = params;
  ASSERT_TRUE(solver->solve(params, actions, problem, true));
  const auto eigvals = solver->eigenvalues();
  ASSERT_EQ(eigvals.size(), solver->n_roots());
  for (size_t i = 0; i < eigvals.size(); ++i)
    EXPECT_NEAR(eigvals[i], reference[i].real(), 1e-8);
}

//! Both members of a complex pair among the roots have the norm of their complex residual as error, also when they end
//! up in different batches because there are fewer parameter vectors than roots
class Davidson_complex_pair_errors : public ::testing::TestWithParam<size_t> {};

TEST_P(Davidson_complex_pair_errors, are_norm_of_complex_residual) {
  molpro::mpi::init();
  SplitComplexPairProblem problem;
  problem.offdiagonal = 0.1;
  auto solver = create_split_pair_solver(4);
  auto params = std::vector<std::vector<double>>(GetParam(), std::vector<double>(problem.n));
  auto actions = params;
  ASSERT_TRUE(solver->solve(params, actions, problem, true));
  auto solutions = std::vector<std::vector<double>>(solver->n_roots(), std::vector<double>(problem.n));
  auto residuals = solutions;
  solver->solution({0, 1, 2, 3}, solutions, residuals);
  const Eigen::Map<const Eigen::VectorXd> r_real(residuals[2].data(), problem.n),
      r_imag(residuals[3].data(), problem.n);
  const auto errors = solver->errors();
  EXPECT_EQ(errors.at(2), errors.at(3));
  EXPECT_NEAR(errors.at(2), std::sqrt(r_real.squaredNorm() + r_imag.squaredNorm()), 1e-14);
}

INSTANTIATE_TEST_SUITE_P(LinearEigensystem, Davidson_complex_pair_errors, ::testing::Values(4, 3));

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
  double metric_non_hermiticity = 0; //!< factor reducing the elements of the metric below the diagonal
  double metric_condition = 1;       //!< roughly the condition number of the metric
  bool cheap_metric = false;         //!< whether the metric is applied on the fly rather than its actions stored

  molpro::linalg::itsolv::MetricStorage metric_storage() const override {
    return cheap_metric ? molpro::linalg::itsolv::MetricStorage::OnTheFly
                        : molpro::linalg::itsolv::MetricStorage::Stored;
  }
  bool provide_pp_metric_matrix = true;
  bool provide_p_metric_action = true;
  bool negative_pp_metric_matrix = false; //!< if true, pp_metric_matrix() is not positive definite

  using P = std::map<size_t, double>;
  std::vector<double> pp_action_matrix(const std::vector<P> &pparams) const override {
    return p_matrix([this](size_t i, size_t j) { return kernel(i, j); }, pparams);
  }
  void p_action(const std::vector<std::vector<double>> &p_coefficients,
                const molpro::linalg::itsolv::CVecRef<P> &pparams,
                const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const override {
    add_p_action([this](size_t i, size_t j) { return kernel(i, j); }, p_coefficients, pparams, actions);
  }
  std::vector<double> pp_metric_matrix(const std::vector<P> &pparams) const override {
    if (not provide_pp_metric_matrix)
      return Problem::pp_metric_matrix(pparams);
    const double sign = negative_pp_metric_matrix ? -1 : 1;
    return p_matrix([this, sign](size_t i, size_t j) { return sign * metric(i, j); }, pparams);
  }
  void p_metric_action(const std::vector<std::vector<double>> &p_coefficients,
                       const molpro::linalg::itsolv::CVecRef<P> &pparams,
                       const molpro::linalg::itsolv::VecRef<std::vector<double>> &metric_actions) const override {
    if (not provide_p_metric_action)
      return Problem::p_metric_action(p_coefficients, pparams, metric_actions);
    add_p_action([this](size_t i, size_t j) { return metric(i, j); }, p_coefficients, pparams, metric_actions);
  }

  double kernel(size_t i, size_t j) const {
    if (complex_pair and i + j == 1)
      return i == 0 ? -0.8 : 0.8;
    return i == j ? 1. + i : 0.1 / (1. + std::abs(double(i) - double(j))) * (j < i ? 1 - non_hermiticity : 1);
  }
  bool hermitian() const { return non_hermiticity == 0 and not complex_pair; }
  double metric(size_t i, size_t j) const {
    // the scaling keeps the metric positive definite whatever its condition
    auto scaling = [this](size_t k) { return std::pow(metric_condition, -0.5 * double(k) / (n - 1)); };
    return metric_scale * scaling(i) * scaling(j) *
           (i == j ? 1.5 + std::cos(double(i)) : 0.05 / (1. + i + j) * (j < i ? 1 - metric_non_hermiticity : 1));
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
  std::vector<double> p_matrix(const Element &element, const std::vector<P> &pparams) const {
    std::vector<double> result;
    for (const auto &pi : pparams)
      for (const auto &pj : pparams) {
        result.push_back(0);
        for (const auto &[i, ci] : pi)
          for (const auto &[j, cj] : pj)
            result.back() += ci * element(i, j) * cj;
      }
    return result;
  }
  template <class Element>
  void add_p_action(const Element &element, const std::vector<std::vector<double>> &p_coefficients,
                    const molpro::linalg::itsolv::CVecRef<P> &pparams,
                    const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const {
    for (size_t k = 0; k < p_coefficients.size(); ++k)
      for (size_t p = 0; p < pparams.size(); ++p)
        for (const auto &[j, cj] : pparams[p].get())
          for (size_t i = 0; i < n; ++i)
            actions[k].get()[i] += element(i, j) * cj * p_coefficients[k][p];
  }
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

//! Solves problem and checks the eigenvalues and, except for complex pairs, the residuals; returns the iterations
template <class Qvector>
int solve_generalized_problem(const GeneralizedProblem &problem, const size_t nroot, const size_t nparams,
                              const size_t max_size_qspace = std::numeric_limits<size_t>::max(), const int max_p = 0) {
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>, Qvector>("Davidson");
  solver->set_n_roots(nroot);
  solver->set_hermiticity(problem.hermitian());
  solver->set_convergence_threshold(1e-10);
  auto options = molpro::linalg::itsolv::LinearEigensystemDavidsonOptions{};
  options.max_size_qspace = max_size_qspace;
  options.max_p = max_p;
  solver->set_options(options);
  auto params = std::vector<std::vector<double>>(nparams, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
  EXPECT_THAT(solver->errors(), ::testing::Each(::testing::Le(1e-10)));
  EXPECT_THAT(solver->eigenvalues(), ::testing::Pointwise(::testing::DoubleNear(1e-9), problem.reference(nroot)));
  if (max_p > 0)
    EXPECT_EQ(solver->dimensions().nP, size_t(max_p));
  if (not problem.complex_pair) {
    // the residuals returned by solution() are A x - lambda M x
    auto x = std::vector<std::vector<double>>(nroot, std::vector<double>(problem.n));
    auto residuals = x, ax = x, mx = x;
    std::vector<int> roots(nroot);
    std::iota(roots.begin(), roots.end(), 0);
    solver->solution(roots, x, residuals);
    problem.action_with_metric(molpro::linalg::itsolv::cwrap(x), molpro::linalg::itsolv::wrap(ax),
                               molpro::linalg::itsolv::wrap(mx));
    for (size_t k = 0; k < nroot; ++k)
      for (size_t i = 0; i < problem.n; ++i)
        EXPECT_NEAR(residuals[k][i], ax[k][i] - solver->eigenvalues()[k] * mx[k][i], 1e-12) << "root " << k;
  }
  return solver->statistics().iterations;
}
} // namespace

// solve() for generalised eigenproblems, with Q vectors in memory and on disk, and with the actions of the metric
// stored or the metric applied on the fly
TEST(LinearEigensystem, Davidson_generalized) {
  molpro::mpi::init();
  GeneralizedProblem problem;
  for (const bool cheap_metric : {false, true})
    for (const double non_hermiticity : {0., 0.2}) {
      SCOPED_TRACE("non-hermiticity=" + std::to_string(non_hermiticity) + " cheap=" + std::to_string(cheap_metric));
      problem.non_hermiticity = non_hermiticity;
      problem.cheap_metric = cheap_metric;
      solve_generalized_problem<std::vector<double>>(problem, 3, 3);
      solve_generalized_problem<molpro::linalg::array::DistrArrayFile>(problem, 3, 3);
    }
}

// A complex conjugate pair of eigenvalues of a real non-hermitian kernel: the residuals of the real and imaginary
// parts of the eigenvector are coupled through the metric
TEST(LinearEigensystem, Davidson_generalized_complex_pair) {
  GeneralizedProblem problem;
  for (const bool cheap_metric : {false, true}) {
    SCOPED_TRACE("cheap=" + std::to_string(cheap_metric));
    problem.complex_pair = true;
    problem.cheap_metric = cheap_metric;
    solve_generalized_problem<std::vector<double>>(problem, 3, 3);
  }
}

// Applying a cheap metric on the fly gives the same iterations as storing its actions, also when Q vectors are moved
// into the D space, and also with fewer parameter vectors than roots, when the actions of the metric on the solutions
// are kept in Q vectors
TEST(LinearEigensystem, Davidson_generalized_cheap_metric) {
  GeneralizedProblem problem;
  for (const double non_hermiticity : {0., 0.2})
    for (const size_t nparams : {3, 2}) {
      SCOPED_TRACE("non-hermiticity=" + std::to_string(non_hermiticity) + " nparams=" + std::to_string(nparams));
      problem.non_hermiticity = non_hermiticity;
      problem.cheap_metric = false;
      const auto stored = solve_generalized_problem<std::vector<double>>(problem, 3, nparams, 6);
      problem.cheap_metric = true;
      EXPECT_EQ(solve_generalized_problem<std::vector<double>>(problem, 3, nparams, 6), stored);
    }
}

// A P space together with a metric, whose actions are stored or applied on the fly, also when Q vectors are moved into
// the D space
TEST(LinearEigensystem, Davidson_generalized_pspace) {
  GeneralizedProblem problem;
  for (const bool cheap_metric : {false, true})
    for (const size_t max_size_qspace : {std::numeric_limits<size_t>::max(), size_t(4)}) {
      SCOPED_TRACE("cheap=" + std::to_string(cheap_metric) + " max_size_qspace=" + std::to_string(max_size_qspace));
      problem.cheap_metric = cheap_metric;
      solve_generalized_problem<std::vector<double>>(problem, 3, 3, max_size_qspace, 10);
    }
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
  for (const double metric_scale : {64., 1. / 64, std::ldexp(1., -50), std::ldexp(1., 50)}) {
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

// With the metric applied on the fly, a P space needs no function applying the metric to it, also not when no residual
// is constructed
TEST(LinearEigensystem, Davidson_generalized_pspace_stepwise_on_the_fly) {
  using molpro::linalg::itsolv::cwrap;
  using molpro::linalg::itsolv::wrap;
  using P = GeneralizedProblem::P;
  GeneralizedProblem problem;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
  solver->set_n_roots(2);
  solver->set_hermiticity(true);
  solver->set_metric(
      molpro::linalg::itsolv::MetricKind::HermitianPositiveDefinite, molpro::linalg::itsolv::MetricStorage::OnTheFly,
      [&problem](const molpro::linalg::itsolv::CVecRef<std::vector<double>> &x,
                 const molpro::linalg::itsolv::VecRef<std::vector<double>> &mx) { problem.metric_action(x, mx); });
  std::vector<P> pspace;
  for (size_t i = 0; i < 4; ++i)
    pspace.push_back(P{{i, 1.}});
  auto pp = problem.pp_action_matrix(pspace), mm = problem.pp_metric_matrix(pspace);
  std::vector<std::vector<double>> x(2, std::vector<double>(problem.n)), r = x;
  using PSpaceMetric = molpro::linalg::itsolv::PSpaceMetric<std::vector<double>>;
  solver->add_p(
      cwrap(pspace), molpro::linalg::array::Span<double>(pp.data(), pp.size()), wrap(x), wrap(r),
      [&problem](const std::vector<std::vector<double>> &c, const molpro::linalg::itsolv::CVecRef<P> &p,
                 const molpro::linalg::itsolv::VecRef<std::vector<double>> &a) { problem.p_action(c, p, a); },
      PSpaceMetric{molpro::linalg::array::Span<double>(mm.data(), mm.size()), {}});
  EXPECT_NO_THROW(solver->solution(std::vector<int>{}, x, r));
  EXPECT_NO_THROW(solver->solution(std::vector<int>{0, 1}, x, r));
}

// The contributions of Q vectors to the solutions, which decide which Q vectors are deleted, and the norms of the
// projections of the solutions that make up the D space are measured on solutions of unit length, so that trimming the
// Q space, building the D space and resetting it do not depend on the scale of the metric either
TEST(LinearEigensystem, Davidson_generalized_metric_scale_dspace) {
  GeneralizedProblem problem;
  struct Outcome {
    std::vector<double> eigenvalues;
    int iterations;
    size_t nQ, nD;
  };
  auto solve = [&problem](double metric_scale,
                          const molpro::linalg::itsolv::LinearEigensystemDavidsonOptions &options) {
    problem.metric_scale = metric_scale;
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
    solver->set_n_roots(2);
    solver->set_hermiticity(true);
    solver->set_convergence_threshold(1e-10);
    solver->set_options(options);
    auto params = std::vector<std::vector<double>>(2, std::vector<double>(problem.n));
    auto actions = params, metric_actions = params;
    EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
    return Outcome{solver->eigenvalues(), solver->statistics().iterations, solver->dimensions().nQ,
                   solver->dimensions().nD};
  };
  auto trimming = molpro::linalg::itsolv::LinearEigensystemDavidsonOptions{};
  trimming.contrib_thresh = 1e-2;
  trimming.min_size_qspace = 2;
  auto resetting = molpro::linalg::itsolv::LinearEigensystemDavidsonOptions{};
  resetting.max_size_qspace = 4;
  resetting.reset_D = 3;
  resetting.reset_D_max_Q_size = 3;
  for (const auto &[name, options] : {std::make_pair("trimming", trimming), std::make_pair("resetting", resetting)}) {
    const auto reference = solve(1, options);
    for (const double metric_scale : {64., 1. / 64, std::ldexp(1., -50), std::ldexp(1., 50)}) {
      SCOPED_TRACE(std::string(name) + " metric scale=" + std::to_string(metric_scale));
      const auto scaled = solve(metric_scale, options);
      EXPECT_EQ(scaled.iterations, reference.iterations);
      EXPECT_EQ(scaled.nQ, reference.nQ);
      EXPECT_EQ(scaled.nD, reference.nD);
      ASSERT_EQ(scaled.eigenvalues.size(), reference.eigenvalues.size());
      for (size_t i = 0; i < reference.eigenvalues.size(); ++i)
        EXPECT_NEAR(scaled.eigenvalues[i] * metric_scale, reference.eigenvalues[i], 1e-12) << "root " << i;
    }
  }
}

// The initial guess is taken at the lowest ratio of the diagonal elements of kernel and metric, whatever the scale of
// the metric. For a diagonal problem the guess is an exact eigenvector, so a guess at another ratio would converge to
// a root that is not the lowest.
TEST(LinearEigensystem, Davidson_generalized_guess_metric_scale) {
  struct DiagonalProblem : molpro::linalg::itsolv::Problem<std::vector<double>> {
    const size_t n = 10;
    double scale = 1;
    double kernel(size_t i) const { return i == 0 ? 1. : i == 1 ? 3. : 10. + i; }
    double metric(size_t i) const { return scale * (i == 0 ? 0.5 : i == 1 ? 2.5 : 1.); }
    molpro::linalg::itsolv::MetricKind metric_kind() const override {
      return molpro::linalg::itsolv::MetricKind::HermitianPositiveDefinite;
    }
    bool diagonals(std::vector<double> &d) const override {
      for (size_t i = 0; i < n; ++i)
        d[i] = kernel(i);
      return true;
    }
    bool metric_diagonals(std::vector<double> &d) const override {
      for (size_t i = 0; i < n; ++i)
        d[i] = metric(i);
      return true;
    }
    void action(const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
                const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const override {
      for (size_t k = 0; k < parameters.size(); ++k)
        for (size_t i = 0; i < n; ++i)
          actions[k].get()[i] = kernel(i) * parameters[k].get()[i];
    }
    void metric_action(const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
                       const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions) const override {
      for (size_t k = 0; k < parameters.size(); ++k)
        for (size_t i = 0; i < n; ++i)
          actions[k].get()[i] = metric(i) * parameters[k].get()[i];
    }
  } problem;
  for (const double scale : {1., std::ldexp(1., -50), 1e-20}) {
    SCOPED_TRACE("metric scale=" + std::to_string(scale));
    problem.scale = scale;
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
    solver->set_n_roots(1);
    solver->set_hermiticity(true);
    std::vector<std::vector<double>> params(1, std::vector<double>(problem.n)), actions = params,
                                                                                metric_actions = params;
    EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
    EXPECT_NEAR(solver->eigenvalues()[0] * scale, 1.2, 1e-12);
  }
}

// solve() may keep references to the problem, so it does not accept a temporary problem, with or without buffers for
// the actions of a metric
template <class Solver, class Problem>
constexpr bool solves_temporary =
    requires(Solver &solver, std::vector<std::vector<double>> &v) { solver.solve(v, v, v, Problem{}); } or
    requires(Solver &solver, std::vector<std::vector<double>> &v) { solver.solve(v, v, Problem{}); } or
    requires(Solver &solver, std::vector<double> &r) { solver.solve(r, r, Problem{}); };
template <class Solver, class Problem>
constexpr bool solves_lvalue =
    requires(Solver &solver, std::vector<std::vector<double>> &v, std::vector<double> &r, const Problem &problem) {
      solver.solve(v, v, v, problem);
      solver.solve(v, v, problem);
      solver.solve(r, r, problem);
    };
using LinearEigensystem =
    molpro::linalg::itsolv::LinearEigensystem<std::vector<double>, std::vector<double>, std::map<size_t, double>>;
using LinearEigensystemDavidson = molpro::linalg::itsolv::LinearEigensystemDavidson<std::vector<double>>;
static_assert(not solves_temporary<LinearEigensystem, GeneralizedProblem>);
static_assert(not solves_temporary<LinearEigensystemDavidson, GeneralizedProblem>);
static_assert(solves_lvalue<LinearEigensystem, GeneralizedProblem>);
static_assert(solves_lvalue<LinearEigensystemDavidson, GeneralizedProblem>);

// A single root is found with a single set of buffers, which also hand the diagonal elements of kernel and metric to
// the preconditioner
TEST(LinearEigensystem, Davidson_generalized_single_root) {
  GeneralizedProblem problem;
  solve_generalized_problem<std::vector<double>>(problem, 1, 1);
}

// solve() calculates the actions of kernel and metric through action_with_metric() and preconditions through
// precondition_with_metric(), which a problem can override
TEST(LinearEigensystem, Davidson_generalized_combined_action) {
  struct CountingProblem : GeneralizedProblem {
    mutable int combined_calls = 0, precondition_calls = 0;
    void action_with_metric(const molpro::linalg::itsolv::CVecRef<std::vector<double>> &parameters,
                            const molpro::linalg::itsolv::VecRef<std::vector<double>> &actions,
                            const molpro::linalg::itsolv::VecRef<std::vector<double>> &metric_actions) const override {
      ++combined_calls;
      GeneralizedProblem::action_with_metric(parameters, actions, metric_actions);
    }
    void precondition_with_metric(const molpro::linalg::itsolv::VecRef<std::vector<double>> &residual,
                                  const std::vector<double> &shift, const std::vector<double> &diagonals,
                                  const std::vector<double> &metric_diagonals) const override {
      ++precondition_calls;
      GeneralizedProblem::precondition_with_metric(residual, shift, diagonals, metric_diagonals);
    }
  } problem;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
  solver->set_n_roots(2);
  solver->set_hermiticity(true);
  auto params = std::vector<std::vector<double>>(2, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
  EXPECT_EQ(problem.combined_calls, solver->statistics().iterations);
  EXPECT_EQ(problem.precondition_calls, solver->statistics().iterations - 1);
}

TEST(LinearEigensystem, Davidson_generalized_test_problem) {
  GeneralizedProblem problem;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
  std::vector<double> v0(problem.n), v1(problem.n);
  EXPECT_TRUE(solver->test_problem(problem, v0, v1, -1));
  problem.linear_metric = false;
  EXPECT_FALSE(solver->test_problem(problem, v0, v1, -1));
}

// A metric that is not positive definite, or not Hermitian, is reported by solve() and by test_problem()
TEST(LinearEigensystem, Davidson_generalized_invalid_metric) {
  GeneralizedProblem problem;
  auto params = std::vector<std::vector<double>>(2, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  std::vector<double> v0(problem.n), v1(problem.n);
  int max_p = 0;
  auto solve = [&]() {
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
    solver->set_n_roots(2);
    solver->set_hermiticity(true);
    solver->set_max_p(max_p);
    return solver->solve(params, actions, metric_actions, problem, true);
  };
  auto test_problem = [&]() {
    return molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson")
        ->test_problem(problem, v0, v1, -1);
  };
  auto solve_error = [&]() -> std::string {
    try {
      solve();
    } catch (const std::runtime_error &e) {
      return e.what();
    }
    return "";
  };
  problem.metric_scale = -1;
  EXPECT_FALSE(test_problem());
  EXPECT_THAT(solve_error(), ::testing::HasSubstr("not positive definite"));
  problem.metric_scale = 1;
  problem.metric_non_hermiticity = 0.5;
  EXPECT_FALSE(test_problem());
  EXPECT_THAT(solve_error(), ::testing::HasSubstr("not Hermitian"));
  // also in the P space
  max_p = 10;
  EXPECT_THAT(solve_error(), ::testing::HasSubstr("not Hermitian"));
  max_p = 0;
  // also without stored actions of the metric
  problem.cheap_metric = true;
  EXPECT_THAT(solve_error(), ::testing::HasSubstr("not Hermitian"));
  problem.metric_scale = -1;
  problem.metric_non_hermiticity = 0;
  EXPECT_THAT(solve_error(), ::testing::HasSubstr("not positive definite"));
  // unless the checks are switched off
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson", "check_metric=false");
  EXPECT_FALSE(solver->get_check_metric());
  solver->set_n_roots(2);
  solver->set_hermiticity(true);
  EXPECT_NO_THROW(solver->solve(params, actions, metric_actions, problem, true));
}

// An ill-conditioned, but positive definite, metric is not mistaken for one that is not positive definite
TEST(LinearEigensystem, Davidson_generalized_ill_conditioned_metric) {
  GeneralizedProblem problem;
  problem.metric_condition = 1e10;
  std::vector<double> v0(problem.n), v1(problem.n);
  EXPECT_TRUE(molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson")
                  ->test_problem(problem, v0, v1, -1));
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson");
  solver->set_n_roots(3);
  solver->set_hermiticity(true);
  solver->set_convergence_threshold(1e-8);
  auto params = std::vector<std::vector<double>>(3, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
  // The reference from the equivalent problem M v = (1 / lambda) A v. With the well conditioned A in place of the
  // metric, the dense solver gives the largest 1 / lambda, i.e. the lowest lambda, to full precision, whereas solving
  // A v = lambda M v densely loses about the condition number of M times the machine epsilon.
  Eigen::MatrixXd a(problem.n, problem.n), m(problem.n, problem.n);
  for (size_t i = 0; i < problem.n; ++i)
    for (size_t j = 0; j < problem.n; ++j) {
      a(i, j) = problem.kernel(i, j);
      m(i, j) = problem.metric(i, j);
    }
  const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> inverse(m, a, Eigen::EigenvaluesOnly);
  for (size_t i = 0; i < 3; ++i)
    EXPECT_NEAR(solver->eigenvalues()[i], 1 / inverse.eigenvalues()(problem.n - 1 - i),
                1e-7 * std::abs(solver->eigenvalues()[i]))
        << "root " << i;
}

// Uses of a metric that solve() cannot support are reported
TEST(LinearEigensystem, Davidson_generalized_errors) {
  GeneralizedProblem problem;
  auto params = std::vector<std::vector<double>>(2, std::vector<double>(problem.n));
  auto actions = params, metric_actions = params;
  auto davidson = []() { return molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("Davidson"); };
  // a P space needs the metric matrix in it and, where the actions of the metric are stored, its action, which are
  // checked before the solver changes
  for (const bool pp_metric_matrix : {false, true}) {
    problem.provide_pp_metric_matrix = pp_metric_matrix;
    problem.provide_p_metric_action = !pp_metric_matrix;
    auto solver = davidson();
    solver->set_max_p(10);
    EXPECT_THROW(solver->solve(params, actions, metric_actions, problem, true), std::logic_error);
    EXPECT_EQ(solver->metric_kind(), molpro::linalg::itsolv::MetricKind::Identity);
    EXPECT_EQ(solver->dimensions().nX(), 0);
  }
  problem.provide_pp_metric_matrix = true;
  problem.provide_p_metric_action = true;
  {
    // a metric that is not positive definite in the P space is rejected before the solver changes, so that the
    // solver can still solve the corrected problem
    problem.negative_pp_metric_matrix = true;
    auto solver = davidson();
    solver->set_n_roots(2);
    solver->set_hermiticity(true);
    solver->set_max_p(6);
    EXPECT_THROW(solver->solve(params, actions, metric_actions, problem, true), std::runtime_error);
    EXPECT_EQ(solver->metric_kind(), molpro::linalg::itsolv::MetricKind::Identity);
    EXPECT_EQ(solver->dimensions().nX(), 0);
    problem.negative_pp_metric_matrix = false;
    EXPECT_TRUE(solver->solve(params, actions, metric_actions, problem, true));
  }
  {
    // the P space of a problem with a metric is added with its metric matrix
    using P = GeneralizedProblem::P;
    auto solver = davidson();
    solver->set_metric(molpro::linalg::itsolv::MetricKind::HermitianPositiveDefinite);
    std::vector<P> pspace{P{{0, 1.}}, P{{1, 1.}}};
    auto pp = problem.pp_action_matrix(pspace);
    EXPECT_THROW(solver->add_p(molpro::linalg::itsolv::cwrap(pspace),
                               molpro::linalg::array::Span<double>(pp.data(), pp.size()),
                               molpro::linalg::itsolv::wrap(params), molpro::linalg::itsolv::wrap(actions), {}),
                 std::logic_error);
    // where the actions of the metric are stored, its action in the P space is needed
    auto mm = problem.pp_metric_matrix(pspace);
    using PSpaceMetric = molpro::linalg::itsolv::PSpaceMetric<std::vector<double>>;
    EXPECT_THROW(solver->add_p(molpro::linalg::itsolv::cwrap(pspace),
                               molpro::linalg::array::Span<double>(pp.data(), pp.size()),
                               molpro::linalg::itsolv::wrap(params), molpro::linalg::itsolv::wrap(actions), {},
                               PSpaceMetric{molpro::linalg::array::Span<double>(mm.data(), mm.size()), {}}),
                 std::invalid_argument);
    // and a metric in the P space needs a declared metric
    EXPECT_THROW(davidson()->add_p(molpro::linalg::itsolv::cwrap(pspace),
                                   molpro::linalg::array::Span<double>(pp.data(), pp.size()),
                                   molpro::linalg::itsolv::wrap(params), molpro::linalg::itsolv::wrap(actions), {},
                                   PSpaceMetric{molpro::linalg::array::Span<double>(mm.data(), mm.size()), {}}),
                 std::logic_error);
  }
  // without buffers for the actions of the metric
  EXPECT_THROW(davidson()->solve(params, actions, problem, true), std::invalid_argument);
  {
    auto too_few = std::vector<std::vector<double>>(1, std::vector<double>(problem.n));
    EXPECT_THROW(davidson()->solve(params, actions, too_few, problem, true), std::invalid_argument);
  }
  // the default preconditioner would ignore the metric without its diagonal elements
  problem.provide_metric_diagonals = false;
  {
    // the problem is rejected before the solver changes
    auto solver = davidson();
    EXPECT_THROW(solver->solve(params, actions, metric_actions, problem, true), std::invalid_argument);
    EXPECT_EQ(solver->metric_kind(), molpro::linalg::itsolv::MetricKind::Identity);
  }
  problem.provide_metric_diagonals = true;
  {
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<std::vector<double>>("RSPT");
    EXPECT_THROW(solver->solve(params, actions, metric_actions, problem, true), std::invalid_argument);
  }
  {
    // the right-hand sides of linear equations are not taken over from a problem that is rejected
    struct RHSProblem : GeneralizedProblem {
      bool RHS(std::vector<double> &rhs, unsigned int instance) const override {
        std::fill(rhs.begin(), rhs.end(), 1.);
        return instance == 0;
      }
    } rhs_problem;
    auto solver = molpro::linalg::itsolv::create_LinearEquations<std::vector<double>>("Davidson");
    EXPECT_THROW(solver->solve(params, actions, metric_actions, rhs_problem, true), std::invalid_argument);
    EXPECT_EQ(solver->dimensions().nRHS, 0);
  }
}

#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, std::vector<double>,
                                                     std::map<size_t, double>>;
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, molpro::linalg::array::DistrArrayFile,
                                                     std::map<size_t, double>>;
