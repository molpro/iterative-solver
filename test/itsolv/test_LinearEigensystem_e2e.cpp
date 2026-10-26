#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <molpro/linalg/array/DistrArrayFile.h>
#include <molpro/linalg/itsolv/IterativeSolver.h>
#include <molpro/linalg/itsolv/LinearEquationsDavidson.h>
#include <molpro/linalg/itsolv/SolverFactory.h>
#include <molpro/mpi.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <deque>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
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

#include <molpro/linalg/itsolv/SolverFactory-implementation.h>
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, std::vector<double>,
                                                     std::map<size_t, double>>;
template class molpro::linalg::itsolv::SolverFactory<std::vector<double>, molpro::linalg::array::DistrArrayFile,
                                                     std::map<size_t, double>>;

//! A hermitian, diagonally dominant matrix in any value type
template <typename scalar>
class HermitianProblem : public molpro::linalg::itsolv::Problem<std::vector<scalar>> {
public:
  using R = std::vector<scalar>;
  static constexpr size_t n = 60;
  scalar element(size_t i, size_t j) const {
    if (i == j)
      return scalar(1. + i);
    const double re = 0.1 / (1. + std::abs(double(i) - double(j)));
    if constexpr (molpro::linalg::itsolv::is_complex<scalar>{})
      return scalar(re, i < j ? 0.05 * re : -0.05 * re);
    else
      return scalar(re);
  }

  bool diagonals(R &d) const override {
    for (size_t i = 0; i < n; ++i)
      d[i] = element(i, i);
    return true;
  }

  void action(const molpro::linalg::itsolv::CVecRef<R> &parameters,
              const molpro::linalg::itsolv::VecRef<R> &actions) const override {
    for (size_t k = 0; k < parameters.size(); ++k)
      for (size_t i = 0; i < n; ++i) {
        actions[k].get()[i] = 0;
        for (size_t j = 0; j < n; ++j)
          actions[k].get()[i] += element(i, j) * parameters[k].get()[j];
      }
  }
};

template <typename scalar>
class Davidson_reset_D : public ::testing::Test {};
using value_types = ::testing::Types<double, std::complex<double>, long double>;
TYPED_TEST_SUITE(Davidson_reset_D, value_types);

//! A D-space reset that falls due once all roots have converged leaves the converged subspace alone
TYPED_TEST(Davidson_reset_D, not_done_after_convergence) {
  using R = std::vector<TypeParam>;
  HermitianProblem<TypeParam> problem;
  for (size_t nroot = 1; nroot <= 3; ++nroot) {
    auto run = [&](const std::string &options) {
      auto solver =
          molpro::linalg::itsolv::create_LinearEigensystem<R, std::deque<TypeParam>, std::map<size_t, TypeParam>>(
              "Davidson", options);
      solver->set_n_roots(nroot);
      solver->set_convergence_threshold(1e-9);
      solver->set_verbosity(molpro::linalg::itsolv::Verbosity::None);
      auto params = std::vector<R>(nroot, R(problem.n));
      for (size_t i = 0; i < nroot; ++i)
        params[i][i] = 1; // select() of the diagonal guess is not implemented for complex values
      auto actions = params;
      EXPECT_TRUE(solver->solve(params, actions, problem));
      return solver;
    };
    const auto reference = run("max_size_qspace=3");
    ASSERT_GT(reference->dimensions().nD, 0) << "no D space to reset";
    // the reset falls due in the iteration that converges
    const auto iterations = reference->statistics().iterations;
    const auto reset = run("max_size_qspace=3,reset_D=" + std::to_string(iterations));
    EXPECT_EQ(reset->statistics().iterations, iterations) << "nroot = " << nroot;
    EXPECT_EQ(reset->statistics().r_creations, reference->statistics().r_creations) << "nroot = " << nroot;
    EXPECT_EQ(reset->dimensions().nD, reference->dimensions().nD) << "nroot = " << nroot;
  }
}

//! Between an end_iteration() that resets the D space and the next add_vector(), solution() throws without touching its
//! arguments, while eigenvalues() are still those from before the reset; the next add_vector() takes up the solutions
TYPED_TEST(Davidson_reset_D, solution_after_reset) {
  using R = std::vector<TypeParam>;
  using molpro::linalg::itsolv::cwrap;
  using molpro::linalg::itsolv::wrap;
  HermitianProblem<TypeParam> problem;
  R diagonals(problem.n);
  problem.diagonals(diagonals);
  // a Q space of 3 is too small to keep vectors for the solutions after the reset
  for (size_t nroot = 2; nroot <= 3; ++nroot) {
    auto solver =
        molpro::linalg::itsolv::create_LinearEigensystem<R, std::deque<TypeParam>, std::map<size_t, TypeParam>>(
            "Davidson", "max_size_qspace=3,reset_D=4");
    solver->set_n_roots(nroot);
    solver->set_convergence_threshold(1e-9);
    auto roots = std::vector<int>(nroot);
    std::iota(roots.begin(), roots.end(), 0);
    auto params = std::vector<R>(nroot, R(problem.n));
    auto actions = params;
    for (size_t i = 0; i < nroot; ++i)
      params[i][i] = 1;
    int nreset = 0;
    std::vector<TypeParam> eigenvalues_before_reset;
    for (int iter = 0, nwork = nroot; iter < 20 && nwork > 0; ++iter) {
      problem.action(cwrap(params.begin(), params.begin() + nwork), wrap(actions.begin(), actions.begin() + nwork));
      nwork = solver->add_vector(params, actions);
      if (!eigenvalues_before_reset.empty()) {
        // the subspace holds the solutions from before the reset again
        const auto eigenvalues = solver->eigenvalues();
        ASSERT_EQ(eigenvalues.size(), nroot);
        for (size_t k = 0; k < nroot; ++k)
          EXPECT_LE(std::real(eigenvalues[k]), std::real(eigenvalues_before_reset[k]) + 1e-10)
              << "nroot " << nroot << ", root " << k;
        eigenvalues_before_reset.clear();
      }
      const auto nD = solver->dimensions().nD;
      const auto eigenvalues = solver->eigenvalues();
      if (nwork > 0)
        problem.precondition(wrap(actions.begin(), actions.begin() + nwork), solver->working_set_eigenvalues(),
                             diagonals);
      nwork = solver->end_iteration(params, actions);
      if (nD == 0 or solver->dimensions().nD > 0)
        continue;
      ++nreset;
      EXPECT_EQ(solver->eigenvalues(), eigenvalues);
      auto x = std::vector<R>(nroot, R(problem.n, TypeParam(7)));
      auto residual = x;
      EXPECT_THROW(solver->solution(roots, x, residual), std::logic_error);
      EXPECT_THROW(solver->solution_params(roots, x), std::logic_error);
      EXPECT_EQ(x, std::vector<R>(nroot, R(problem.n, TypeParam(7))));
      EXPECT_EQ(residual, x);
      eigenvalues_before_reset = eigenvalues;
    }
    EXPECT_GT(nreset, 0) << "nroot " << nroot;
  }
}

//! solve() does not end right after an iteration that resets the D space, so that it can return the solutions
TYPED_TEST(Davidson_reset_D, solve_ends_after_reset) {
  using R = std::vector<TypeParam>;
  using molpro::linalg::itsolv::cwrap;
  using molpro::linalg::itsolv::wrap;
  HermitianProblem<TypeParam> problem;
  for (size_t nroot = 2; nroot <= 3; ++nroot) {
    auto solver =
        molpro::linalg::itsolv::create_LinearEigensystem<R, std::deque<TypeParam>, std::map<size_t, TypeParam>>(
            "Davidson", "max_size_qspace=3,reset_D=4");
    solver->set_n_roots(nroot);
    solver->set_convergence_threshold(1e-9);
    solver->set_verbosity(molpro::linalg::itsolv::Verbosity::None);
    // the reset falls due in the last allowed iteration
    solver->set_max_iter(4);
    auto params = std::vector<R>(nroot, R(problem.n));
    auto actions = params;
    for (size_t i = 0; i < nroot; ++i)
      params[i][i] = 1;
    solver->solve(params, actions, problem);
    EXPECT_EQ(solver->statistics().iterations, 5) << "nroot " << nroot;
    const auto eigenvalues = solver->eigenvalues();
    ASSERT_EQ(eigenvalues.size(), nroot);
    auto roots = std::vector<int>(nroot);
    std::iota(roots.begin(), roots.end(), 0);
    auto x = std::vector<R>(nroot, R(problem.n));
    auto residual = x;
    solver->solution(roots, x, residual);
    problem.action(cwrap(x), wrap(residual));
    for (size_t k = 0; k < nroot; ++k) {
      TypeParam xr = 0, xx = 0;
      for (size_t i = 0; i < problem.n; ++i) {
        residual[k][i] -= eigenvalues[k] * x[k][i];
        xr += molpro::linalg::conjugate(x[k][i]) * residual[k][i];
        xx += molpro::linalg::conjugate(x[k][i]) * x[k][i];
      }
      EXPECT_NEAR(std::abs(xr / xx), 0, 1e-12) << "nroot " << nroot << ", root " << k;
    }
  }
}

//! A diagonal matrix in any value type
template <typename scalar>
class DiagonalProblem : public molpro::linalg::itsolv::Problem<std::vector<scalar>> {
public:
  using R = std::vector<scalar>;
  static constexpr size_t n = 10;
  bool diagonals(R &d) const override {
    for (size_t i = 0; i < n; ++i)
      d[i] = scalar(1. + i);
    return true;
  }
  void action(const molpro::linalg::itsolv::CVecRef<R> &parameters,
              const molpro::linalg::itsolv::VecRef<R> &actions) const override {
    for (size_t k = 0; k < parameters.size(); ++k)
      for (size_t i = 0; i < n; ++i)
        actions[k].get()[i] = scalar(1. + i) * parameters[k].get()[i];
  }
};

template <typename scalar>
class LinearEigensystem_convergence : public ::testing::Test {};
TYPED_TEST_SUITE(LinearEigensystem_convergence, value_types);

//! solve() does not report convergence while the subspace holds solutions for fewer than all roots
TYPED_TEST(LinearEigensystem_convergence, needs_all_roots) {
  using R = std::vector<TypeParam>;
  DiagonalProblem<TypeParam> problem;
  auto solver = molpro::linalg::itsolv::create_LinearEigensystem<R, std::deque<TypeParam>, std::map<size_t, TypeParam>>(
      "Davidson");
  solver->set_n_roots(2);
  solver->set_verbosity(molpro::linalg::itsolv::Verbosity::None);
  // an exact eigenvector, whose residual offers no direction to find the second root in
  R parameters(problem.n), actions(problem.n);
  parameters[0] = 1;
  EXPECT_FALSE(solver->solve(parameters, actions, problem));
  EXPECT_EQ(solver->errors().size(), 1);
}

//! report() leaves the format of the stream as it was
TEST(LinearEigensystem, report_keeps_stream_format) {
  using R = std::vector<double>;
  HermitianProblem<double> problem;
  for (const std::string method : {"Davidson", "RSPT"}) {
    auto solver = molpro::linalg::itsolv::create_LinearEigensystem<R>(method);
    solver->set_verbosity(molpro::linalg::itsolv::Verbosity::None);
    solver->set_max_iter(3);
    R parameters(problem.n), actions(problem.n);
    parameters[0] = 1;
    solver->solve(parameters, actions, problem);
    ASSERT_FALSE(solver->errors().empty()) << method;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3);
    const auto flags = stream.flags();
    solver->report(stream);
    EXPECT_EQ(stream.flags(), flags) << method;
    EXPECT_EQ(stream.precision(), 3) << method;
  }
}

//! The statistics of LinearEquationsDavidson include the array operations done in end_iteration(); the Q vectors are
//! on disk, as only their handlers count operations
TEST(LinearEquations, statistics_after_end_iteration) {
  molpro::mpi::init();
  using R = std::vector<double>;
  using Q = molpro::linalg::array::DistrArrayFile;
  using P = std::map<size_t, double>;
  SymmetricProblem problem;
  auto handlers = std::make_shared<molpro::linalg::itsolv::ArrayHandlers<R, Q, P>>();
  auto solver = molpro::linalg::itsolv::LinearEquationsDavidson<R, Q, P>(handlers);
  R rhs(problem.n, 1.), parameters(problem.n), actions(problem.n);
  solver.add_equations(rhs);
  parameters[0] = 1;
  problem.action(molpro::linalg::itsolv::cwrap_arg(parameters), molpro::linalg::itsolv::wrap_arg(actions));
  ASSERT_EQ(solver.add_vector(parameters, actions), 1);
  auto counts = [](const molpro::linalg::itsolv::Statistics &statistics) {
    return statistics.qr_ops + statistics.rq_ops + statistics.qq_ops;
  };
  const auto counts_before = counts(solver.statistics());
  ASSERT_EQ(solver.end_iteration(parameters, actions), 1);
  auto current = std::make_shared<molpro::linalg::itsolv::Statistics>();
  molpro::linalg::itsolv::read_handler_counts(current, handlers);
  ASSERT_NE(counts(*current), counts_before) << "end_iteration() did no counted operations";
  EXPECT_EQ(counts(solver.statistics()), counts(*current));
}
